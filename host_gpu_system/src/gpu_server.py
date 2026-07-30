#!/usr/bin/env python3
"""
GPU Inference Server for UR3 Grasping System

Runs local-bbox DQN grasp inference/training (YOLO detects a bounding box on
the warped board image, a fixed window around its center feeds a local DQN
cell picker). Also supports optional VLM-based box selection
(--use-vlm-select) and a YOLO-only debug mode (--yolo-locator-test).
"""

import torch
import numpy as np
import cv2
import socket
import json
import sys
import threading
import time
import yaml
import base64
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque

from yolo_locator import (
    LocalYoloLocator, load_yolo_locator_config, random_cell_in_bbox,
)
from local_bbox_dqn_module import LocalBBoxDQNModule, create_local_bbox_dqn_module
from local_bbox_window import (
    load_local_bbox_dqn_config, local_bbox_grid_params,
    bbox_center_world, crop_fixed_window, local_cell_to_world, world_to_local_cell,
)
from vlm_box_selector import (
    DEFAULT_OLLAMA_URL, DEFAULT_VLM_MODEL,
    VlmSelectFailedError, VlmUnavailableError, select_bbox_by_vlm_point,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "vm_simulation_system" / "src"))
from spawn_geometry import classify_demo_bucket  # noqa: E402
from grasp_geometry import compute_grasp_pose_from_object_world, DEFAULT_OBJECT_Y_M  # noqa: E402
from local_grid import epsilon_for_episode  # noqa: E402
from board_grid import (  # noqa: E402
    load_board_dqn_config, board_n, world_to_board_cell, board_cell_to_world,
    invalid_cell_mask,
)
from board_warp import (
    warp_rgb_to_board, warp_depth_to_board, world_xz_to_warp_pixel,
)  # noqa: E402

_PLATFORM_CENTER = {1: (-0.646, 0.841), 2: (-1.365, 0.850)}


class GPUInferenceServer:
    """
    Local-bbox DQN inference/training server. Optional VLM box selection and
    a YOLO-only debug mode (--yolo-locator-test) are also supported.
    """
    def __init__(self, config_path: str = "config/network_config.yaml",
                 yolo_locator_test: bool = False,
                 yolo_locator_config_path: str = None,
                 local_bbox_dqn: bool = False,
                 local_bbox_dqn_train: bool = False,
                 local_bbox_dqn_config_path: str = None,
                 local_bbox_model_path: str = None,
                 local_bbox_model_path_r2: str = None,
                 use_vlm_select: bool = False,
                 vlm_model: str = None,
                 ollama_url: str = None):
        self.config = self._load_config(config_path)
        self.yolo_locator_test = yolo_locator_test
        self.local_bbox_dqn = local_bbox_dqn
        self.local_bbox_dqn_train = local_bbox_dqn_train
        self.local_bbox_model_path = local_bbox_model_path
        self.local_bbox_model_path_r2 = local_bbox_model_path_r2
        self.use_vlm_select = bool(use_vlm_select)
        self.vlm_model = str(vlm_model or DEFAULT_VLM_MODEL)
        self.ollama_url = str(ollama_url or DEFAULT_OLLAMA_URL)
        self.yolo_locator_cfg: Dict = {}
        self.yolo_client: Optional[LocalYoloLocator] = None
        self.local_bbox_cfg: Dict = {}
        self.local_bbox_module: Optional[LocalBBoxDQNModule] = None
        self.local_bbox_module2: Optional[LocalBBoxDQNModule] = None
        self._local_bbox_n = 20
        self._local_bbox_window_m = 0.02
        self._local_bbox_crop_size = 224
        self._weak_regions: Dict = {}
        self._checkpoint_every = 100
        self._save_path_r1_local_bbox = "R1_local_bbox_dqn.pth"
        self._save_path_r2_local_bbox = "R2_local_bbox_dqn.pth"

        # =========================================================================
        # DEVICE & BOARD/WARP CONFIG (platform bounds shared by local-bbox + YOLO test)
        # =========================================================================
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        b_path = str(Path(__file__).resolve().parent.parent / "config" / "board_dqn_config.yaml")
        self.board_cfg = load_board_dqn_config(b_path)
        self._board_n = board_n(self.board_cfg)
        self._board_n_cells = self._board_n * self._board_n

        if yolo_locator_test:
            yl_path = yolo_locator_config_path or str(
                Path(__file__).resolve().parent.parent / "config" / "yolo_locator_config.yaml"
            )
            self.yolo_locator_cfg = load_yolo_locator_config(yl_path)
            self.yolo_client = LocalYoloLocator(self.yolo_locator_cfg)

        if local_bbox_dqn or local_bbox_dqn_train:
            lb_path = local_bbox_dqn_config_path or str(
                Path(__file__).resolve().parent.parent / "config" / "local_bbox_dqn_config.yaml"
            )
            self.local_bbox_cfg = load_local_bbox_dqn_config(lb_path)
            self._local_bbox_n, self._local_bbox_window_m = local_bbox_grid_params(self.local_bbox_cfg)
            self._local_bbox_crop_size = int(self.local_bbox_cfg.get("crop", {}).get("out_size", 224))
            yolo_cfg = dict(self.local_bbox_cfg.get("yolo") or {})
            if yolo_locator_config_path:
                yolo_cfg = load_yolo_locator_config(yolo_locator_config_path)
            elif not yolo_cfg.get("weights"):
                yolo_cfg = load_yolo_locator_config()
            self.yolo_locator_cfg = yolo_cfg
            if self.yolo_client is None:
                self.yolo_client = LocalYoloLocator(yolo_cfg)
            ckpt_cfg = self.local_bbox_cfg.get("checkpoints", {})
            save_l1 = ckpt_cfg.get("save_r1", "models/R1_local_bbox_dqn.pth")
            save_l2 = ckpt_cfg.get("save_r2", "models/R2_local_bbox_dqn.pth")
            self._save_path_r1_local_bbox = Path(save_l1).name
            self._save_path_r2_local_bbox = Path(save_l2).name
            if local_bbox_dqn and not local_bbox_dqn_train:
                if not self.local_bbox_model_path:
                    self.local_bbox_model_path = save_l1
                if not self.local_bbox_model_path_r2:
                    self.local_bbox_model_path_r2 = save_l2
            if local_bbox_dqn_train:
                self._checkpoint_every = int(
                    self.local_bbox_cfg.get("training", {}).get("checkpoint_every_steps", 100)
                )

        # =========================================================================
        # TRAINING BUFFERS
        # =========================================================================
        if local_bbox_dqn or local_bbox_dqn_train:
            self.batch_size = int(self.local_bbox_cfg.get("training", {}).get("batch_size", 8))
        else:
            self.batch_size = 16

        self.weak_buffer          = deque(maxlen=10000)   # Robot 1 shaping buffer
        self.normal_buffer        = deque(maxlen=10000)
        self.weak_buffer2         = deque(maxlen=10000)   # Robot 2 shaping buffer
        self.normal_buffer2       = deque(maxlen=10000)
        replay_cap = 5000
        if local_bbox_dqn_train:
            replay_cap = int(self.local_bbox_cfg.get("training", {}).get("replay_capacity", 5000))
        self.local_bbox_replay    = deque(maxlen=replay_cap)
        self.local_bbox_replay2   = deque(maxlen=replay_cap)
        self.local_bbox_training_step_count  = 0
        self.local_bbox_training_step_count2 = 0

        if local_bbox_dqn or local_bbox_dqn_train:
            self._init_local_bbox_modules()

        # =========================================================================
        # VISION PREPROCESSING CONFIGURATION
        # =========================================================================
        # Fractional crop boundaries (0.0-1.0) targeting the platform region.
        # R1 (Left Camera) and R2 (Right Camera) require asymmetric windows.

        # Robot 1: Left camera looking inward
        R1_CROP = dict(crop_y0=0.22, crop_y1=0.7, crop_x0=0.37,  crop_x1=0.60)

        # Robot 2: Right camera looking inward
        R2_CROP = dict(crop_y0=0.21, crop_y1=0.885, crop_x0=0.37,  crop_x1=0.71)

        self._r1_crop = R1_CROP
        self._r2_crop = R2_CROP

        # =========================================================================
        # EPISODE SYNCHRONIZATION BARRIER
        # =========================================================================
        # Ensures all active robots complete their current episode before triggering
        # global domain randomizations (e.g., lighting, floor textures).
        self._barrier_num_robots  = 1         # Set to 1 for single-robot deployments
        self._barrier_ready_count = 0
        self._barrier_event       = threading.Event()
        self._barrier_lock        = threading.Lock()
        # R2 waits here until R1 finishes domain randomization + spawn (dual-arm only).
        self._setup_r1_event      = threading.Event()
        self._setup_all_event     = threading.Event()
        self._setup_ready_count   = 0
        self._setup_lock          = threading.Lock()
        self._setup_wait_timeout_s = 120.0
        self._barrier_wait_timeout_s = 120.0

        # =========================================================================
        # NETWORKING & CONCURRENCY
        # =========================================================================
        self.is_running         = False
        self.server_socket      = None
        self.client_connections = []
        self.train_lock         = threading.Lock()
        self.train_lock2        = threading.Lock()

    def _load_config(self, config_path: str) -> Dict:
        """Loads server network configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return {'network': {'host_ip': '0.0.0.0', 'port': 8888}}

    @staticmethod
    def _resolve_checkpoint_path(path_override: Optional[str], default_name: str) -> Path:
        """
        Resolve a checkpoint path relative to host_gpu_system/, repo root, or cwd.

        Accepts ``models/R1_locator.pth`` (under host_gpu_system) or
        ``host_gpu_system/models/...`` when launched from the repo root.
        """
        host_root = Path(__file__).resolve().parent.parent
        repo_root = host_root.parent

        if path_override:
            raw = Path(path_override)
            if raw.is_absolute():
                return raw
            candidates: List[Path] = []
            if raw.exists():
                candidates.append(raw.resolve())
            parts = raw.parts
            if parts and parts[0].lower() == "host_gpu_system":
                candidates.append(host_root.joinpath(*parts[1:]))
            candidates.append(host_root / raw)
            candidates.append(repo_root / raw)
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()
            return (host_root / raw).resolve()

        for name in (default_name, "ur3_model.pth"):
            path = host_root / "models" / name
            if path.exists():
                return path.resolve()
        return (host_root / "models" / default_name).resolve()

    def _init_local_bbox_modules(self) -> None:
        tr = self.local_bbox_cfg.get("training", {})
        lr = float(tr.get("learning_rate", 1e-3))
        wd = float(tr.get("weight_decay", 8e-5))
        gamma = float(tr.get("gamma", 0.99))
        enc_ph = float(tr.get("encoder_lr_factor", 0.1))
        self.local_bbox_module = create_local_bbox_dqn_module(
            grid_n=self._local_bbox_n,
            learning_rate=lr,
            weight_decay=wd,
            encoder_lr_factor=enc_ph,
            gamma=gamma,
        ).to(self.device)
        self.local_bbox_module2 = create_local_bbox_dqn_module(
            grid_n=self._local_bbox_n,
            learning_rate=lr,
            weight_decay=wd,
            encoder_lr_factor=enc_ph,
            gamma=gamma,
        ).to(self.device)
        self._load_local_bbox_weights()

    def _load_local_bbox_weights(self) -> None:
        def _load(mod: Optional[LocalBBoxDQNModule], path_override: Optional[str],
                  default_name: str, step_attr: str):
            if mod is None:
                return
            path = self._resolve_checkpoint_path(path_override, default_name)
            print(f"[LOCAL-BBOX] Looking for checkpoint at: {path.resolve()}")
            if path.exists():
                step = mod.load_model(str(path))
                setattr(self, step_attr, max(getattr(self, step_attr), step))
                print(f"[LOCAL-BBOX] Loaded: {path.name}  (step {step})")
            elif self.local_bbox_dqn_train:
                print("   -> No local-bbox checkpoint yet — starting from pretrained MobileNet")
            elif self.local_bbox_dqn:
                print(f"[LOCAL-BBOX] WARNING: No checkpoint at {path.name}")

        _load(self.local_bbox_module, self.local_bbox_model_path,
              self._save_path_r1_local_bbox, "local_bbox_training_step_count")
        _load(self.local_bbox_module2, self.local_bbox_model_path_r2,
              self._save_path_r2_local_bbox, "local_bbox_training_step_count2")

    def _local_bbox_training_phase(self, robot_id: int) -> str:
        phase_cfg = str(self.local_bbox_cfg.get("training", {}).get("phase", "both")).lower()
        if phase_cfg == "shaping":
            return "shaping"
        if phase_cfg == "rl":
            return "rl"
        step = self.local_bbox_training_step_count2 if robot_id == 2 else self.local_bbox_training_step_count
        shaping_steps = int(self.local_bbox_cfg.get("training", {}).get("shaping_steps", 500))
        weak = self.weak_buffer2 if robot_id == 2 else self.weak_buffer
        normal = self.normal_buffer2 if robot_id == 2 else self.normal_buffer
        # Explore-only runs send RL samples, not shaping demos — don't deadlock in shaping.
        if step < shaping_steps and (len(weak) + len(normal)) > 0:
            return "shaping"
        return "rl"

    def _local_bbox_epsilon(self, session_episode: int) -> float:
        return epsilon_for_episode(session_episode, self.local_bbox_cfg)

    def _crop_rgb_depth(self, rgb: np.ndarray, depth: np.ndarray, robot_id: int):
        crop = self._r2_crop if robot_id == 2 else self._r1_crop
        h, w = rgb.shape[:2]
        y0 = int(crop['crop_y0'] * h)
        y1 = int(crop['crop_y1'] * h)
        x0 = int(crop['crop_x0'] * w)
        x1 = int(crop['crop_x1'] * w)
        rgb_c = rgb[y0:y1, x0:x1].copy()
        depth_c = depth[y0:y1, x0:x1].copy()
        try:
            debug_dir = Path(__file__).resolve().parent.parent / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            out = debug_dir / f"ai_vision_debug_rgb_sim_r{robot_id}.jpg"
            cv2.imwrite(str(out), cv2.cvtColor(rgb_c, cv2.COLOR_RGB2BGR))
        except Exception:
            pass
        return rgb_c, depth_c

    def _board_label_xz(self, payload: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Prefer live object XZ (matches camera) over logged spawn."""
        obj_pos = payload.get('object_pos')
        if isinstance(obj_pos, (list, tuple)) and len(obj_pos) >= 2:
            ox, oz = float(obj_pos[0]), float(obj_pos[1])
            if abs(ox) > 1e-6 or abs(oz) > 1e-6:
                return ox, oz
        ox = payload.get('object_x')
        oz = payload.get('object_z')
        if ox is not None and oz is not None:
            return float(ox), float(oz)
        sx = payload.get('spawn_x')
        sz = payload.get('spawn_z')
        if sx is not None and sz is not None:
            return float(sx), float(sz)
        return None, None

    def _board_cell_pixel(self, cell: int, board_h: int, board_w: int, robot_id: int = 1) -> Tuple[int, int]:
        wx, wz = board_cell_to_world(cell, robot_id, n=self._board_n, cfg=self.board_cfg)
        return world_xz_to_warp_pixel(
            wx, wz, robot_id, out_size=board_w, cfg=self.board_cfg, apply_post_flip=True,
        )

    def _warp_rgb_for_debug(self, camera_data: Dict, robot_id: int) -> np.ndarray:
        img = self.decode_b64_image(camera_data)
        rgb = cv2.cvtColor(img['rgb'], cv2.COLOR_BGR2RGB)
        depth = img['depth']
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        rgb_c, _ = self._crop_rgb_depth(rgb, depth, robot_id)
        rgb_w = warp_rgb_to_board(rgb_c, robot_id, out_size=224, cfg=self.board_cfg)
        return rgb_w

    def _save_board_debug_overlay(
        self,
        rgb_board: np.ndarray,
        robot_id: int,
        cnn_cell: Optional[int] = None,
        teacher_cell: Optional[int] = None,
        meta: Optional[Dict] = None,
        q_map: Optional[np.ndarray] = None,
        q_diag: Optional[Dict] = None,
        yolo_bbox: Optional[Tuple[float, float, float, float]] = None,
        window_xyxy: Optional[Tuple[int, int, int, int]] = None,
        local_pick_px: Optional[Tuple[int, int]] = None,
    ) -> None:
        try:
            debug_dir = Path(__file__).resolve().parent.parent / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            vis = rgb_board.copy()
            h, w = vis.shape[:2]
            # Draw in BGR so OpenCV colors match what is saved to disk.
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

            if teacher_cell is not None:
                tx, ty = self._board_cell_pixel(teacher_cell, h, w, robot_id)
                cv2.drawMarker(vis_bgr, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 16, 3)
                cv2.circle(vis_bgr, (tx, ty), 12, (0, 0, 255), 2)

            if yolo_bbox is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in yolo_bbox]
                cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 128, 255), 2)

            if window_xyxy is not None:
                wx1, wy1, wx2, wy2 = [int(v) for v in window_xyxy]
                cv2.rectangle(vis_bgr, (wx1, wy1), (wx2, wy2), (255, 0, 255), 2)

            if local_pick_px is not None:
                px, py = int(local_pick_px[0]), int(local_pick_px[1])
                cv2.drawMarker(vis_bgr, (px, py), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)

            if meta and meta.get('vlm_point_uv') is not None:
                try:
                    vu, vv = meta['vlm_point_uv']
                    cv2.drawMarker(
                        vis_bgr, (int(round(vu)), int(round(vv))),
                        (255, 255, 0), cv2.MARKER_STAR, 16, 2,
                    )
                except Exception:
                    pass
            if meta and meta.get('instruction'):
                try:
                    txt = str(meta['instruction'])[:80]
                    cv2.putText(
                        vis_bgr, txt, (4, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                except Exception:
                    pass

            if meta and meta.get('dqn_outside_bbox') and meta.get('dqn_cell') is not None:
                dx, dy = self._board_cell_pixel(int(meta['dqn_cell']), h, w, robot_id)
                cv2.drawMarker(vis_bgr, (dx, dy), (255, 255, 0), cv2.MARKER_DIAMOND, 12, 2)

            if cnn_cell is not None:
                gx, gy = self._board_cell_pixel(cnn_cell, h, w, robot_id)
                cv2.drawMarker(vis_bgr, (gx, gy), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)

            stem = f"board_warp_r{robot_id}_latest"
            out = debug_dir / f"{stem}.jpg"
            cv2.imwrite(str(out), vis_bgr)

            sidecar = {
                'robot_id': robot_id,
                'cnn_cell': cnn_cell,
                'teacher_cell': teacher_cell,
                'cnn_hit': (
                    int(teacher_cell) == int(cnn_cell)
                    if teacher_cell is not None and cnn_cell is not None else None
                ),
            }
            if meta:
                sidecar.update(meta)
            if q_diag:
                sidecar.update(q_diag)
            if teacher_cell is not None:
                lx = sidecar.get('label_x')
                lz = sidecar.get('label_z')
                if lx is not None and lz is not None:
                    tpx, tpy = world_xz_to_warp_pixel(
                        float(lx), float(lz), robot_id, out_size=h, cfg=self.board_cfg,
                    )
                    sidecar['teacher_warp_px'] = [tpx, tpy]
            if teacher_cell is not None and cnn_cell is not None:
                sidecar['cell_index_gap'] = abs(int(cnn_cell) - int(teacher_cell))
            with open(debug_dir / f"{stem}.json", 'w', encoding='utf-8') as f:
                json.dump(sidecar, f, indent=2)
        except Exception as e:
            print(f"[BOARD DEBUG R{robot_id}] overlay save failed: {e}")

    def _pose_from_board_cell(self, cell: int, robot_id: int) -> List[float]:
        wx, wz = board_cell_to_world(cell, robot_id, n=self._board_n, cfg=self.board_cfg)
        return compute_grasp_pose_from_object_world(
            wx, DEFAULT_OBJECT_Y_M, wz, robot_id, add_jitter=False,
        )

    def _predict_yolo_bbox_random_pose(self, full_message: Dict) -> Dict:
        """YOLO bbox on warp image → random valid cell inside box → grasp (no DQN)."""
        if self.yolo_client is None:
            return {'type': 'error', 'message': 'Local YOLO locator not initialized — use --yolo-locator-test'}

        robot_id = int(full_message.get('robot_id', 1))
        is_sim = full_message.get('source', 'real') == 'simulation'
        camera_data = full_message['data']
        rgb_w = self._warp_rgb_for_debug(camera_data, robot_id)
        h, w = rgb_w.shape[:2]

        try:
            bbox, raw_result = self.yolo_client.detect_bbox(rgb_w)
        except Exception as exc:
            return {'type': 'error', 'message': f'Local YOLO inference failed: {exc}'}

        if bbox is None:
            n_raw = 0
            if isinstance(raw_result, dict) and isinstance(raw_result.get("predictions"), list):
                n_raw = len(raw_result["predictions"])
            try:
                self._save_board_debug_overlay(
                    rgb_w, robot_id, cnn_cell=None, teacher_cell=None,
                    meta={'mode': 'yolo_bbox_random', 'yolo_detections_raw': n_raw},
                )
            except Exception:
                pass
            return {
                'type': 'error',
                'message': (
                    f'No valid YOLO bbox (raw_detections={n_raw}). '
                    f'Retrain on sim warps if block color/view differs from dataset.'
                ),
                'yolo_raw': raw_result,
            }

        inv_np = invalid_cell_mask(robot_id, self._board_n, self.board_cfg)
        cell, cells_count = random_cell_in_bbox(
            bbox, self._board_n, inv_np, h, w, robot_id, board_cfg=self.board_cfg,
        )
        if cell is None:
            return {
                'type': 'error',
                'message': 'YOLO bbox contains no valid board cells',
                'yolo_conf': bbox.confidence,
                'cells_in_bbox_count': 0,
            }

        row, col = divmod(cell, self._board_n)
        wx, wz = board_cell_to_world(cell, robot_id, n=self._board_n, cfg=self.board_cfg)
        grasp_pose = self._pose_from_board_cell(cell, robot_id)

        teacher_cell = None
        label_x, label_z = self._board_label_xz(full_message)
        if is_sim and label_x is not None and label_z is not None:
            teacher_cell = world_to_board_cell(
                label_x, label_z, robot_id, n=self._board_n, cfg=self.board_cfg,
            )

        yolo_bbox_px = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        self._save_board_debug_overlay(
            rgb_w, robot_id,
            cnn_cell=cell,
            teacher_cell=teacher_cell,
            yolo_bbox=yolo_bbox_px,
            meta={
                'mode': 'yolo_bbox_random',
                'label_x': label_x,
                'label_z': label_z,
                'yolo_bbox_px': list(yolo_bbox_px),
                'yolo_conf': bbox.confidence,
                'yolo_class': bbox.class_name,
                'random_cell': cell,
                'cells_in_bbox_count': cells_count,
            },
        )

        hit_line = ""
        if teacher_cell is not None:
            hit = int(cell) == int(teacher_cell)
            hit_line = f" teacher={teacher_cell} {'HIT' if hit else 'miss'}"
        print(
            f"[YOLO R{robot_id}] conf={bbox.confidence:.3f} "
            f"cells_in_bbox={cells_count} random_cell={cell}{hit_line} | "
            f"see debug/board_warp_r{robot_id}_latest.jpg",
            flush=True,
        )

        return {
            'type': 'grasp_prediction',
            'pose': grasp_pose,
            'mode': 'yolo_bbox_random',
            'board_cell': cell,
            'board_u': col,
            'board_v': row,
            'target_x_m': wx,
            'target_z_m': wz,
            'teacher_cell': teacher_cell,
            'yolo_conf': bbox.confidence,
            'yolo_bbox_px': list(yolo_bbox_px),
            'cells_in_bbox_count': cells_count,
            'confidence': bbox.confidence,
            'timestamp': time.time(),
        }

    def _warp_preview_ready(self, full_message: Dict) -> Dict:
        """Warp + save board_warp debug, then wait for client instruction."""
        robot_id = int(full_message.get('robot_id', 1))
        camera_data = full_message['data']
        rgb_w = self._warp_rgb_for_debug(camera_data, robot_id)
        instruction = str(full_message.get('instruction') or '').strip()
        try:
            self._save_board_debug_overlay(
                rgb_w, robot_id, cnn_cell=None, teacher_cell=None,
                meta={
                    'mode': 'warp_preview',
                    'instruction': instruction or None,
                    'awaiting_instruction': True,
                },
            )
        except Exception as exc:
            print(f"[WARP PREVIEW R{robot_id}] save failed: {exc}", flush=True)
        return {
            'type': 'warp_ready',
            'robot_id': robot_id,
            'timestamp': time.time(),
        }

    def _warp_rgb_depth_pair(self, camera_data: Dict, robot_id: int) -> Tuple[np.ndarray, np.ndarray]:
        img = self.decode_b64_image(camera_data)
        rgb = cv2.cvtColor(img['rgb'], cv2.COLOR_BGR2RGB)
        depth = img['depth']
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        rgb_c, depth_c = self._crop_rgb_depth(rgb, depth, robot_id)
        rgb_w = warp_rgb_to_board(rgb_c, robot_id, out_size=224, cfg=self.board_cfg)
        depth_w = warp_depth_to_board(depth_c, robot_id, out_size=224, cfg=self.board_cfg)
        return rgb_w, depth_w

    def _preprocess_local_bbox_crop(
        self, rgb_crop: np.ndarray, depth_crop: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        DEPTH_MIN, DEPTH_MAX = 0.50, 1.00
        depth_w = np.clip(depth_crop.astype(np.float32), DEPTH_MIN, DEPTH_MAX)
        depth_w = (depth_w - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
        rgb_t = torch.from_numpy(rgb_crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        depth_t = torch.from_numpy(depth_w).unsqueeze(0).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb_t = (rgb_t - mean) / std
        depth_t = (depth_t - 0.5) / 0.5
        return rgb_t.to(self.device), depth_t.to(self.device)

    def _local_bbox_crop_from_camera(
        self, camera_data: Dict, robot_id: int, center_x: float, center_z: float,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], np.ndarray]:
        rgb_w, depth_w = self._warp_rgb_depth_pair(camera_data, robot_id)
        rgb_c, depth_c, xyxy = crop_fixed_window(
            rgb_w, depth_w, center_x, center_z, robot_id,
            window_m=self._local_bbox_window_m,
            out_size=self._local_bbox_crop_size,
            board_cfg=self.board_cfg,
        )
        assert depth_c is not None
        return rgb_c, depth_c, xyxy, rgb_w

    def _predict_local_bbox_pose(self, full_message: Dict, explore: bool = False) -> Dict:
        """YOLO center → fixed window → local DQN cell → world grasp."""
        if self.yolo_client is None:
            return {'type': 'error', 'message': 'YOLO detector not initialized for local-bbox DQN'}
        robot_id = int(full_message.get('robot_id', 1))
        is_sim = full_message.get('source', 'real') == 'simulation'
        camera_data = full_message['data']
        mod = self.local_bbox_module2 if robot_id == 2 else self.local_bbox_module
        if mod is None:
            return {'type': 'error', 'message': 'Local-bbox DQN module not loaded'}

        use_vlm = self.use_vlm_select or bool(full_message.get('use_vlm_select', False))
        instruction = str(full_message.get('instruction') or '').strip()
        vlm_point_uv = None
        vlm_raw = None

        rgb_w, depth_w = self._warp_rgb_depth_pair(camera_data, robot_id)
        try:
            if use_vlm:
                if not instruction:
                    return {
                        'type': 'error',
                        'message': 'VLM select requires non-empty instruction',
                        'outcome_class': 'vlm_select_failed',
                    }
                boxes, raw_result = self.yolo_client.detect_bboxes(rgb_w)
                if not boxes:
                    try:
                        self._save_board_debug_overlay(
                            rgb_w, robot_id, cnn_cell=None, teacher_cell=None,
                            meta={
                                'mode': 'local_bbox_dqn', 'yolo_miss': True,
                                'instruction': instruction,
                            },
                        )
                    except Exception:
                        pass
                    return {
                        'type': 'error',
                        'message': 'No valid YOLO bbox for local-bbox DQN',
                        'yolo_raw': raw_result,
                    }
                try:
                    sel = select_bbox_by_vlm_point(
                        rgb_w, instruction, boxes,
                        ollama_url=self.ollama_url,
                        model=self.vlm_model,
                    )
                except VlmUnavailableError as exc:
                    return {
                        'type': 'error',
                        'message': f'vlm_unavailable: {exc}',
                        'outcome_class': 'vlm_unavailable',
                        'yolo_raw': raw_result,
                    }
                except VlmSelectFailedError as exc:
                    try:
                        self._save_board_debug_overlay(
                            rgb_w, robot_id, cnn_cell=None, teacher_cell=None,
                            meta={
                                'mode': 'local_bbox_dqn',
                                'instruction': instruction,
                                'vlm_fail': str(exc),
                            },
                        )
                    except Exception:
                        pass
                    return {
                        'type': 'error',
                        'message': f'vlm_select_failed: {exc}',
                        'outcome_class': 'vlm_select_failed',
                        'yolo_raw': raw_result,
                    }
                bbox = sel.bbox
                vlm_point_uv = list(sel.point_uv)
                vlm_raw = sel.raw_text
            else:
                bbox, raw_result = self.yolo_client.detect_bbox(rgb_w)
        except Exception as exc:
            return {'type': 'error', 'message': f'YOLO inference failed: {exc}'}

        if bbox is None:
            try:
                self._save_board_debug_overlay(
                    rgb_w, robot_id, cnn_cell=None, teacher_cell=None,
                    meta={'mode': 'local_bbox_dqn', 'yolo_miss': True},
                )
            except Exception:
                pass
            return {
                'type': 'error',
                'message': 'No valid YOLO bbox for local-bbox DQN',
                'yolo_raw': raw_result,
            }

        center_x, center_z = bbox_center_world(
            bbox, robot_id, warp_size=rgb_w.shape[1], board_cfg=self.board_cfg,
        )
        rgb_c, depth_c, win_xyxy = crop_fixed_window(
            rgb_w, depth_w, center_x, center_z, robot_id,
            window_m=self._local_bbox_window_m,
            out_size=self._local_bbox_crop_size,
            board_cfg=self.board_cfg,
        )
        assert depth_c is not None
        rgb_t, depth_t = self._preprocess_local_bbox_crop(rgb_c, depth_c)

        session_ep = max(1, int(full_message.get('session_episode', 1)))
        eps = self._local_bbox_epsilon(session_ep) if explore else 0.0
        with torch.no_grad():
            cell_t = mod.select_cell(rgb_t, depth_t, epsilon=eps)
        cell = int(cell_t.cpu().numpy()[0])
        wx, wz = local_cell_to_world(
            cell, center_x, center_z,
            n=self._local_bbox_n, window_m=self._local_bbox_window_m,
        )
        grasp_pose = compute_grasp_pose_from_object_world(
            wx, DEFAULT_OBJECT_Y_M, wz, robot_id, add_jitter=False,
        )

        teacher_cell = None
        label_x, label_z = self._board_label_xz(full_message)
        if is_sim and label_x is not None and label_z is not None:
            teacher_cell = world_to_local_cell(
                label_x, label_z, center_x, center_z,
                n=self._local_bbox_n, window_m=self._local_bbox_window_m,
            )

        yolo_bbox_px = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        pick_u, pick_v = world_xz_to_warp_pixel(
            wx, wz, robot_id, out_size=rgb_w.shape[1], cfg=self.board_cfg,
        )
        meta = {
            'mode': 'local_bbox_explore' if explore else 'local_bbox_dqn',
            'label_x': label_x,
            'label_z': label_z,
            'yolo_bbox_px': list(yolo_bbox_px),
            'yolo_conf': bbox.confidence,
            'window_xyxy': list(win_xyxy),
            'window_center_x': center_x,
            'window_center_z': center_z,
            'local_cell': cell,
            'teacher_local_cell': teacher_cell,
            'pick_warp_px': [pick_u, pick_v],
        }
        if use_vlm:
            meta['instruction'] = instruction
            meta['vlm_point_uv'] = vlm_point_uv
            meta['vlm_raw'] = vlm_raw
            meta['use_vlm_select'] = True
        self._save_board_debug_overlay(
            rgb_w, robot_id,
            cnn_cell=None,
            teacher_cell=None,
            yolo_bbox=yolo_bbox_px,
            meta=meta,
            local_pick_px=(pick_u, pick_v),
            window_xyxy=win_xyxy,
        )

        mode = 'local_bbox_explore' if explore else 'local_bbox_dqn'
        out = {
            'type': 'grasp_prediction',
            'pose': grasp_pose,
            'mode': mode,
            'local_cell': cell,
            'grid_cell': cell,
            'grid_center_x': center_x,
            'grid_center_z': center_z,
            'window_center_x': center_x,
            'window_center_z': center_z,
            'target_x_m': wx,
            'target_z_m': wz,
            'teacher_cell': teacher_cell,
            'yolo_conf': bbox.confidence,
            'yolo_bbox_px': list(yolo_bbox_px),
            'window_xyxy': list(win_xyxy),
            'epsilon': eps,
            'confidence': bbox.confidence,
            'timestamp': time.time(),
        }
        if use_vlm:
            out['instruction'] = instruction
            out['vlm_point_uv'] = vlm_point_uv
            out['vlm_raw'] = vlm_raw
            out['use_vlm_select'] = True
        return out

    def format_local_bbox_shaping_batch(
        self, batch: List[Dict], robot_id: int,
    ) -> Dict[str, torch.Tensor]:
        rgb_list, depth_list, labels = [], [], []
        for exp in batch:
            is_sim = exp.get('source', 'real') == 'simulation'
            cx = float(exp['window_center_x'])
            cz = float(exp['window_center_z'])
            rgb_c, depth_c, _, _ = self._local_bbox_crop_from_camera(
                exp['state'], robot_id, cx, cz,
            )
            rgb_t, depth_t = self._preprocess_local_bbox_crop(rgb_c, depth_c)
            rgb_list.append(rgb_t)
            depth_list.append(depth_t)
            labels.append(int(exp['cell_label']))
        return {
            'rgb': torch.cat(rgb_list, dim=0),
            'depth': torch.cat(depth_list, dim=0),
            'cell_labels': torch.tensor(labels, dtype=torch.long, device=self.device),
        }

    def format_local_bbox_rl_batch(
        self, batch: List[Dict], robot_id: int,
    ) -> Dict[str, torch.Tensor]:
        rgb_list, depth_list = [], []
        nrgb_list, ndepth_list = [], []
        actions, rewards, dones = [], [], []
        for exp in batch:
            cx = float(exp['window_center_x'])
            cz = float(exp['window_center_z'])
            rgb_c, depth_c, _, _ = self._local_bbox_crop_from_camera(
                exp['state'], robot_id, cx, cz,
            )
            rgb_t, depth_t = self._preprocess_local_bbox_crop(rgb_c, depth_c)
            next_state = exp.get('next_state', exp['state'])
            nrgb_c, ndepth_c, _, _ = self._local_bbox_crop_from_camera(
                next_state, robot_id, cx, cz,
            )
            nrgb_t, ndepth_t = self._preprocess_local_bbox_crop(nrgb_c, ndepth_c)
            rgb_list.append(rgb_t)
            depth_list.append(depth_t)
            nrgb_list.append(nrgb_t)
            ndepth_list.append(ndepth_t)
            actions.append(int(exp['cell_action']))
            rewards.append(float(exp.get('reward', 0.0)))
            dones.append(1.0 if exp.get('done', True) else 0.0)
        return {
            'rgb': torch.cat(rgb_list, dim=0),
            'depth': torch.cat(depth_list, dim=0),
            'next_rgb': torch.cat(nrgb_list, dim=0),
            'next_depth': torch.cat(ndepth_list, dim=0),
            'cell_actions': torch.tensor(actions, dtype=torch.long, device=self.device),
            'rewards': torch.tensor(rewards, dtype=torch.float32, device=self.device),
            'dones': torch.tensor(dones, dtype=torch.float32, device=self.device),
        }

    def _run_local_bbox_training_step(self, robot_id: int, rl: bool = False) -> None:
        lock = self.train_lock if robot_id == 1 else self.train_lock2
        if lock.locked():
            return
        with lock:
            try:
                mod = self.local_bbox_module2 if robot_id == 2 else self.local_bbox_module
                if mod is None:
                    return
                if rl:
                    replay = self.local_bbox_replay2 if robot_id == 2 else self.local_bbox_replay
                    if len(replay) < self.batch_size:
                        return
                    batch_raw = random.sample(list(replay), self.batch_size)
                    torch_batch = self.format_local_bbox_rl_batch(batch_raw, robot_id)
                    metrics = mod.update_dqn(torch_batch)
                    tgt_every = int(self.local_bbox_cfg.get('training', {}).get('target_update_steps', 100))
                    if mod._dqn_step % max(1, tgt_every) == 0:
                        mod.sync_target()
                    phase = 'rl'
                else:
                    weak = self.weak_buffer2 if robot_id == 2 else self.weak_buffer
                    normal = self.normal_buffer2 if robot_id == 2 else self.normal_buffer
                    pool = list(weak) + list(normal)
                    if len(pool) < self.batch_size:
                        return
                    batch_raw = random.sample(pool, self.batch_size)
                    torch_batch = self.format_local_bbox_shaping_batch(batch_raw, robot_id)
                    metrics = mod.update_shaping(torch_batch)
                    phase = 'shaping'

                step_attr = 'local_bbox_training_step_count2' if robot_id == 2 else 'local_bbox_training_step_count'
                step = getattr(self, step_attr) + 1
                setattr(self, step_attr, step)
                print(
                    f"[LOCAL-BBOX TRAIN R{robot_id}] {phase} step {step} "
                    f"loss={metrics.get('total', 0):.4f}",
                    flush=True,
                )
                if step % max(1, self._checkpoint_every) == 0:
                    name = self._save_path_r2_local_bbox if robot_id == 2 else self._save_path_r1_local_bbox
                    path = Path(__file__).resolve().parent.parent / "models" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    mod.save_model(str(path), training_step=step)
                    print(f"[LOCAL-BBOX] saved {path.name}", flush=True)
            except Exception as e:
                print(f"[LOCAL-BBOX TRAIN R{robot_id}] ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()

    def _handle_local_bbox_training_data(
        self, training_data: Dict, source: str, robot_id: int,
    ) -> Dict:
        try:
            collect_mode = training_data.get('mode', 'local_bbox_collect')
            label_x, label_z = self._board_label_xz(training_data)
            if label_x is None or label_z is None:
                obj_pos = training_data.get('object_pos', [0.0, 0.0])
                label_x = float(obj_pos[0]) if len(obj_pos) > 0 else 0.0
                label_z = float(obj_pos[1]) if len(obj_pos) > 1 else 0.0

            cx = training_data.get('window_center_x', training_data.get('grid_center_x'))
            cz = training_data.get('window_center_z', training_data.get('grid_center_z'))
            if cx is None or cz is None:
                if self.yolo_client is None or not training_data.get('state'):
                    return {'type': 'error', 'message': 'local-bbox training requires window_center_x/z or YOLO+state'}
                rgb_w = self._warp_rgb_for_debug(training_data['state'], robot_id)
                bbox, _ = self.yolo_client.detect_bbox(rgb_w)
                if bbox is None:
                    return {'type': 'error', 'message': 'YOLO missed during local-bbox collect'}
                center_x, center_z = bbox_center_world(
                    bbox, robot_id, warp_size=rgb_w.shape[1], board_cfg=self.board_cfg,
                )
            else:
                center_x, center_z = float(cx), float(cz)

            ack: Dict = {
                'type': 'training_ack',
                'local_bbox': True,
                'window_center_x': center_x,
                'window_center_z': center_z,
                'local_bbox_phase': self._local_bbox_training_phase(robot_id),
            }

            if collect_mode in ('local_bbox_collect', 'grid_collect'):
                cell = world_to_local_cell(
                    label_x, label_z, center_x, center_z,
                    n=self._local_bbox_n, window_m=self._local_bbox_window_m,
                )
                if cell is None:
                    ack['out_of_window'] = True
                    ack['skipped'] = True
                    return ack
                sample = {
                    'state': training_data['state'],
                    'cell_label': cell,
                    'window_center_x': center_x,
                    'window_center_z': center_z,
                    'source': source,
                    'robot_id': robot_id,
                }
                bucket = self._resolve_demo_bucket(training_data, robot_id)
                sample['demo_bucket'] = bucket
                target = (self.weak_buffer2 if bucket == 'weak' else self.normal_buffer2) if robot_id == 2 \
                    else (self.weak_buffer if bucket == 'weak' else self.normal_buffer)
                if self._local_bbox_training_phase(robot_id) == 'shaping':
                    target.append(sample)
                    if self._fine_tune_buffers_ready(robot_id):
                        threading.Thread(
                            target=self._run_local_bbox_training_step,
                            args=(robot_id, False), daemon=True,
                        ).start()
                ack.update({'teacher_cell': cell, 'local_cell': cell, 'out_of_window': False})
                return ack

            if collect_mode in ('local_bbox_rl', 'grid_rl'):
                cell_action = int(training_data.get('cell_action', training_data.get('local_cell', 0)))
                sample = {
                    'state': training_data['state'],
                    'next_state': training_data.get('next_state', training_data['state']),
                    'cell_action': cell_action,
                    'reward': float(training_data.get('reward', 0.0)),
                    'done': bool(training_data.get('done', True)),
                    'window_center_x': center_x,
                    'window_center_z': center_z,
                    'source': source,
                    'robot_id': robot_id,
                }
                replay = self.local_bbox_replay2 if robot_id == 2 else self.local_bbox_replay
                replay.append(sample)
                replay_min = int(self.local_bbox_cfg.get('training', {}).get('replay_min', 200))
                if self._local_bbox_training_phase(robot_id) == 'rl' and len(replay) >= replay_min:
                    threading.Thread(
                        target=self._run_local_bbox_training_step,
                        args=(robot_id, True), daemon=True,
                    ).start()
                ack.update({
                    'local_cell': cell_action,
                    'replay_len': len(replay),
                    'replay_used': True,
                    'reward': sample['reward'],
                })
                return ack

            return {'type': 'error', 'message': f'Unknown local-bbox collect mode: {collect_mode}'}
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    def _resolve_demo_bucket(self, training_data: Dict, robot_id: int) -> str:
        bucket = training_data.get('demo_bucket')
        if bucket in ('weak', 'normal'):
            return bucket

        spawn_x = float(training_data.get('spawn_x', 0.0))
        spawn_z = float(training_data.get('spawn_z', 0.0))
        spawn_phase = int(training_data.get('spawn_phase', 0))
        spawn_collection = training_data.get('spawn_collection')
        cx, cz = _PLATFORM_CENTER.get(robot_id, _PLATFORM_CENTER[1])
        return classify_demo_bucket(
            robot_id, spawn_phase, spawn_x, spawn_z, cx, cz,
            self._weak_regions, spawn_collection=spawn_collection,
        )

    def _fine_tune_buffers_ready(self, robot_id: int) -> bool:
        if robot_id == 2:
            return (len(self.weak_buffer2) + len(self.normal_buffer2)) >= self.batch_size
        return (len(self.weak_buffer) + len(self.normal_buffer)) >= self.batch_size

    # =========================================================================
    # DATA PROCESSING UTILITIES
    # =========================================================================

    def decode_b64_image(self, img_data: Dict) -> Dict:
        """Decodes base64 network payloads into OpenCV-compatible arrays."""
        rgb_bytes   = base64.b64decode(img_data['rgb'])
        depth_bytes = base64.b64decode(img_data['depth'])

        rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Depth is encoded as: [H uint32 LE][W uint32 LE][H*W uint16 LE raw pixels]
        shape_header = np.frombuffer(depth_bytes[:8], dtype=np.uint32)
        h, w = int(shape_header[0]), int(shape_header[1])
        depth = np.frombuffer(depth_bytes[8:], dtype=np.uint16).reshape(h, w)

        return {'rgb': rgb, 'depth': depth}

    # =========================================================================
    # MESSAGE HANDLERS & TRAINING LOGIC
    # =========================================================================

    def _handle_camera_data(self, full_message: Dict) -> Dict:
        """
        Processes incoming frames: local-bbox DQN (default inference/train),
        optional VLM select, or the YOLO-only debug path (--yolo-locator-test).
        """
        try:
            client_mode = full_message.get('mode', 'inference')

            if client_mode == 'local_bbox_dqn_train':
                return self._predict_local_bbox_pose(full_message, explore=True)

            if self.local_bbox_dqn or bool(full_message.get('use_local_bbox_dqn', False)):
                use_vlm = self.use_vlm_select or bool(full_message.get('use_vlm_select', False))
                instruction = str(full_message.get('instruction') or '').strip()
                if use_vlm and not instruction:
                    return self._warp_preview_ready(full_message)
                return self._predict_local_bbox_pose(full_message, explore=False)

            if self.yolo_locator_test or bool(full_message.get('use_yolo_locator_test', False)):
                return self._predict_yolo_bbox_random_pose(full_message)

            return {
                'type': 'error',
                'message': (
                    'No inference mode enabled — start gpu_server with '
                    '--local-bbox-dqn or --yolo-locator-test'
                ),
            }
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    def _handle_training_data(self, training_data: Dict, source: str = 'real',
                               robot_id: int = 1) -> Dict:
        """Stores teacher demonstrations and triggers async training updates."""
        try:
            if source == 'simulation':
                try:
                    img_data = self.decode_b64_image(training_data['state'])
                    rgb   = img_data['rgb']
                    depth = img_data['depth'].astype(np.float32) / 1000.0

                    crop = self._r2_crop if robot_id == 2 else self._r1_crop
                    h, w = rgb.shape[:2]
                    y0 = int(crop['crop_y0'] * h)
                    y1 = int(crop['crop_y1'] * h)
                    x0 = int(crop['crop_x0'] * w)
                    x1 = int(crop['crop_x1'] * w)

                    rgb_crop = rgb[y0:y1, x0:x1].copy()
                    debug_dir = Path(__file__).resolve().parent.parent / "debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(debug_dir / f"ai_vision_debug_rgb_sim_r{robot_id}.jpg"),
                        rgb_crop,
                    )
                    depth_crop = depth[y0:y1, x0:x1].copy()
                    depth_vis  = cv2.normalize(depth_crop, None, 0, 255,
                                               cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imwrite(
                        str(debug_dir / f"ai_vision_debug_depth_sim_r{robot_id}.png"),
                        depth_vis,
                    )
                except Exception as e:
                    print(f"[DEBUG R{robot_id}] Sim image save failed: {e}")

            if self.local_bbox_dqn_train:
                return self._handle_local_bbox_training_data(training_data, source, robot_id)

            return {
                'type': 'error',
                'message': 'No training mode enabled — start gpu_server with --local-bbox-dqn-train',
            }
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    def _handle_episode_end(self, robot_id: int) -> Dict:
        """
        Cross-client synchronization barrier.
        Blocks the responding thread until all registered robots report episode 
        completion, preventing desynchronized domain shifts.
        """
        with self._barrier_lock:
            self._barrier_ready_count += 1
            count = self._barrier_ready_count
            needed = self._barrier_num_robots

            if count >= needed:
                self._barrier_event.set()

        if not self._barrier_event.wait(timeout=self._barrier_wait_timeout_s):
            with self._barrier_lock:
                self._barrier_ready_count = max(0, self._barrier_ready_count - 1)
            return {
                'type': 'error',
                'message': (
                    f'timeout ({self._barrier_wait_timeout_s}s) waiting for '
                    f'{self._barrier_num_robots} robot(s) at episode_end '
                    f'(got {count}, R{robot_id} reported)'
                ),
            }

        with self._barrier_lock:
            self._barrier_ready_count -= 1
            if self._barrier_ready_count == 0:
                self._barrier_event.clear()
                self._setup_r1_event.clear()
                self._setup_all_event.clear()
                with self._setup_lock:
                    self._setup_ready_count = 0

        return {'type': 'proceed'}

    def _handle_episode_setup_done(self, robot_id: int) -> Dict:
        """Report local episode setup finished; release peers when all robots are ready."""
        with self._setup_lock:
            self._setup_ready_count += 1
            count = self._setup_ready_count
            needed = self._barrier_num_robots

        if robot_id == 1:
            print("[SETUP BARRIER] R1 world setup complete — R2 may proceed")
            self._setup_r1_event.set()

        if count >= needed:
            print(f"[SETUP BARRIER] All {needed} robot(s) setup complete — releasing to sim loop")
            self._setup_all_event.set()

        return {'type': 'proceed'}

    def _handle_episode_setup_wait(self, robot_id: int) -> Dict:
        """Robot 2 blocks until Robot 1 completes episode setup (dual-arm only)."""
        # Solo (barrier=1): whichever arm is running owns scene domain rand.
        # Dual (barrier=2): only R1 owns it (avoids dual-supervisor crash).
        owns_domain_rand = self._barrier_num_robots < 2 or robot_id == 1
        if self._barrier_num_robots < 2 or robot_id != 2:
            return {'type': 'proceed', 'owns_domain_rand': owns_domain_rand,
                    'num_robots': self._barrier_num_robots}
        print(f"[SETUP BARRIER] R{robot_id} waiting for R1 world setup...")
        if not self._setup_r1_event.wait(timeout=self._setup_wait_timeout_s):
            return {
                'type': 'error',
                'message': f'timeout ({self._setup_wait_timeout_s}s) waiting for R1 setup',
            }
        print(f"[SETUP BARRIER] R{robot_id} cleared — starting local setup")
        return {'type': 'proceed', 'owns_domain_rand': False,
                'num_robots': self._barrier_num_robots}

    def _handle_episode_setup_all_wait(self, robot_id: int) -> Dict:
        """Block until every robot has finished episode setup (dual-arm only)."""
        if self._barrier_num_robots < 2:
            return {'type': 'proceed'}
        print(f"[SETUP BARRIER] R{robot_id} waiting for all robots to finish setup...")
        if not self._setup_all_event.wait(timeout=self._setup_wait_timeout_s):
            return {
                'type': 'error',
                'message': f'timeout ({self._setup_wait_timeout_s}s) waiting for all setup',
            }
        print(f"[SETUP BARRIER] R{robot_id} setup sync complete — resuming simulation")
        return {'type': 'proceed'}

    # =========================================================================
    # NETWORKING ENGINE
    # =========================================================================

    def handle_client_request(self, client_socket, address):
        """Processes and routes inbound JSON payloads over TCP."""
        self.is_running = True
        try:
            while self.is_running:
                size_data = client_socket.recv(4)
                if not size_data:
                    break
                message_size = int.from_bytes(size_data, byteorder='big')

                message_data = b''
                while len(message_data) < message_size:
                    chunk = client_socket.recv(min(message_size - len(message_data), 4096))
                    if not chunk:
                        break
                    message_data += chunk

                message = json.loads(message_data.decode('utf-8'))

                if message['type'] == 'camera_data':
                    response = self._handle_camera_data(message)
                elif message['type'] == 'training_data':
                    response = self._handle_training_data(
                        message['data'],
                        source=message.get('source', 'real'),
                        robot_id=int(message.get('robot_id', 1))
                    )
                elif message['type'] == 'episode_end':
                    response = self._handle_episode_end(
                        robot_id=int(message.get('robot_id', 1))
                    )
                elif message['type'] == 'episode_setup_done':
                    response = self._handle_episode_setup_done(
                        robot_id=int(message.get('robot_id', 1))
                    )
                elif message['type'] == 'episode_setup_wait':
                    response = self._handle_episode_setup_wait(
                        robot_id=int(message.get('robot_id', 2))
                    )
                elif message['type'] == 'episode_setup_all_wait':
                    response = self._handle_episode_setup_all_wait(
                        robot_id=int(message.get('robot_id', 1))
                    )
                else:
                    response = {'type': 'ack'}

                response_data = json.dumps(response).encode('utf-8')
                client_socket.send(len(response_data).to_bytes(4, byteorder='big'))
                client_socket.send(response_data)
        except Exception as e:
            print(f"Socket error: {e}")
        finally:
            client_socket.close()

    def start_server(self):
        """Initializes the listener socket and spawns handler threads."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        host = self.config['network']['host_ip']
        port = self.config['network']['port']

        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        mode_label = "YOLO locator test" if self.yolo_locator_test else (
            "local-bbox DQN train" if self.local_bbox_dqn_train else (
                "local-bbox DQN + VLM" if (self.local_bbox_dqn and self.use_vlm_select) else (
                    "local-bbox DQN" if self.local_bbox_dqn else "idle"
                )
            )
        )
        print(f"GPU Server listening on {host}:{port} ({mode_label} mode)")
        if self.local_bbox_dqn and self.use_vlm_select:
            print(f"   VLM select: model={self.vlm_model} url={self.ollama_url}")
        if self.yolo_locator_test:
            yc = self.yolo_locator_cfg
            print(
                f"   YOLO locator test: local Ultralytics | "
                f"weights={yc.get('weights')} | conf={yc.get('conf')} | "
                f"min_conf={yc.get('min_confidence')}"
            )

        while True:
            conn, addr = self.server_socket.accept()
            threading.Thread(
                target=self.handle_client_request, args=(conn, addr), daemon=True
            ).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-locator-test', action='store_true',
                        help='Inference: local YOLO bbox → random cell grasp (no DQN); debug mode')
    parser.add_argument('--yolo-locator-config', type=str, default=None,
                        help='Path to yolo_locator_config.yaml')
    parser.add_argument('--local-bbox-dqn', action='store_true',
                        help='Inference: YOLO center → fixed window → local DQN grasp')
    parser.add_argument('--local-bbox-dqn-train', action='store_true',
                        help='Train local-bbox DQN (shaping + RL) with YOLO-centered window')
    parser.add_argument('--local-bbox-dqn-config', type=str, default=None,
                        help='Path to local_bbox_dqn_config.yaml')
    parser.add_argument('--local-bbox-model', type=str, default=None,
                        help='Local-bbox DQN checkpoint for R1')
    parser.add_argument('--local-bbox-model-r2', type=str, default=None,
                        help='Local-bbox DQN checkpoint for R2')
    parser.add_argument('--use-vlm-select', action='store_true',
                        help='With --local-bbox-dqn: Ollama VLM picks box via point on warp')
    parser.add_argument('--vlm-model', type=str, default=None,
                        help=f'Ollama model name (default {DEFAULT_VLM_MODEL})')
    parser.add_argument('--ollama-url', type=str, default=None,
                        help=f'Ollama base URL (default {DEFAULT_OLLAMA_URL})')
    args = parser.parse_args()

    # Default bare launch: local-bbox DQN inference.
    if not any((args.yolo_locator_test, args.local_bbox_dqn, args.local_bbox_dqn_train)):
        args.local_bbox_dqn = True

    if args.local_bbox_dqn_train and args.local_bbox_dqn:
        parser.error('Use --local-bbox-dqn-train for training; --local-bbox-dqn is inference only.')
    if args.use_vlm_select and not args.local_bbox_dqn:
        parser.error('--use-vlm-select requires --local-bbox-dqn.')
    if args.use_vlm_select and args.local_bbox_dqn_train:
        parser.error('--use-vlm-select is inference-only (not with --local-bbox-dqn-train).')
    if args.local_bbox_dqn and args.yolo_locator_test:
        parser.error('Use either --local-bbox-dqn or --yolo-locator-test, not both.')
    if args.local_bbox_dqn_train and args.yolo_locator_test:
        parser.error('Use either --local-bbox-dqn-train or --yolo-locator-test, not both.')

    server = GPUInferenceServer(
        yolo_locator_test=args.yolo_locator_test,
        yolo_locator_config_path=args.yolo_locator_config,
        local_bbox_dqn=args.local_bbox_dqn,
        local_bbox_dqn_train=args.local_bbox_dqn_train,
        local_bbox_dqn_config_path=args.local_bbox_dqn_config,
        local_bbox_model_path=args.local_bbox_model,
        local_bbox_model_path_r2=args.local_bbox_model_r2,
        use_vlm_select=args.use_vlm_select,
        vlm_model=args.vlm_model,
        ollama_url=args.ollama_url,
    )
    server.start_server()
