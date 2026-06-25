#!/usr/bin/env python3
"""
GPU Inference Server for UR3 Grasping System

Manages neural network inference, behavior cloning training buffers, and 
multi-robot episode synchronization. Operates exclusively in Behavior Cloning 
mode where training is driven by teacher demonstrations.
"""

import torch
import torch.nn as nn
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
import os
import math
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque

from enhanced_neural_network import (
    UR3GraspCNN_Enhanced, BehaviorCloningModule, LocalizationModule, TD3Module,
    create_model, create_td3_module, ImageProcessor,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "vm_simulation_system" / "src"))
from spawn_geometry import load_fine_tune_config, load_locator_train_config, classify_demo_bucket  # noqa: E402
from rl_reward import load_rl_train_config, get_robot_rl_config, exploration_noise_scale  # noqa: E402
from grasp_geometry import compute_grasp_pose_from_object_world, local_xz_to_world, DEFAULT_OBJECT_Y_M  # noqa: E402

_PLATFORM_CENTER = {1: (-0.646, 0.841), 2: (-1.365, 0.850)}

LOCATOR_STEP_CSV_FIELDS = [
    'timestamp_utc', 'robot_id', 'loc_step', 'aux_loss', 'grad_norm',
    'buffer_weak', 'buffer_normal', 'batch_weak', 'batch_normal', 'checkpoint_saved',
]


class GPUInferenceServer:
    """
    Centralized inference and training server for multi-robot reinforcement 
    and behavior cloning systems.
    """
    def __init__(self, config_path: str = "config/network_config.yaml", model_path: str = None,
                 model_path_r2: str = None, fine_tune: bool = False,
                 fine_tune_config_path: str = None,
                 locator_train: bool = False,
                 locator_config_path: str = None,
                 geo_grasp: bool = False,
                 rl_train: bool = False, rl_train_config_path: str = None,
                 rl_residual_r1: str = None, rl_residual_r2: str = None):
        self.config     = self._load_config(config_path)
        self.model_path = model_path
        self.model_path_r2 = model_path_r2
        self.fine_tune  = fine_tune
        self.locator_train = locator_train
        self.geo_grasp  = geo_grasp
        self.rl_train   = rl_train
        self.fine_tune_cfg: Dict = {}
        self.locator_cfg: Dict = {}
        self.rl_cfg: Dict = {}
        self._rl_robot_cfg: Dict[int, Dict] = {1: {}, 2: {}}
        self.td3_module: Optional[TD3Module] = None
        self.td3_module2: Optional[TD3Module] = None
        self._weak_regions: Dict = {}
        self._weak_ratio = 0.7
        self._checkpoint_every = 100
        self._save_path_r1 = "ur3_live_model_r1.pth"
        self._save_path_r2 = "ur3_live_model_r2.pth"
        self._save_path_r1_rl = "R1_RL_residual.pth"
        self._save_path_r2_rl = "R2_RL_residual.pth"
        self._rl_checkpoint_every = 100

        if rl_train:
            rl_path = rl_train_config_path or str(
                Path(__file__).resolve().parent.parent / "config" / "rl_train_config.yaml"
            )
            self.rl_cfg = load_rl_train_config(rl_path)
            self._rl_robot_cfg[1] = get_robot_rl_config(self.rl_cfg, 1)
            self._rl_robot_cfg[2] = get_robot_rl_config(self.rl_cfg, 2)
            self._rl_checkpoint_every = int(
                self.rl_cfg.get('training', {}).get('checkpoint_every_steps', 100)
            )
            ckpt = self.rl_cfg.get('checkpoints', {})
            self._save_path_r1_rl = Path(ckpt.get('save_r1', self._save_path_r1_rl)).name
            self._save_path_r2_rl = Path(ckpt.get('save_r2', self._save_path_r2_rl)).name

        if fine_tune:
            ft_path = fine_tune_config_path or str(
                Path(__file__).resolve().parent.parent / "config" / "fine_tune_config.yaml"
            )
            self.fine_tune_cfg = load_fine_tune_config(ft_path)
            self._weak_regions = self.fine_tune_cfg['_weak_regions_parsed']
            self._weak_ratio = float(self.fine_tune_cfg['sampling']['weak_ratio'])
            self._checkpoint_every = int(
                self.fine_tune_cfg.get('training', {}).get('checkpoint_every_steps', 100)
            )
            ckpt_cfg = self.fine_tune_cfg.get('checkpoints', {})
            ft_root = Path(__file__).resolve().parent.parent
            save_r1 = ckpt_cfg.get('save_r1', 'models/R1_BC_targeted_70weak_30normal.pth')
            save_r2 = ckpt_cfg.get('save_r2', 'models/R2_BC_targeted_70weak_30normal.pth')
            base_r1 = ckpt_cfg.get('base_r1', 'models/ur3_live_model_r1.pth')
            base_r2 = ckpt_cfg.get('base_r2', 'models/ur3_live_model_r2.pth')
            if not self.model_path:
                self.model_path = self._resolve_fine_tune_load_path(
                    ft_root, save_r1, base_r1, 'R1')
            if not self.model_path_r2:
                self.model_path_r2 = self._resolve_fine_tune_load_path(
                    ft_root, save_r2, base_r2, 'R2')
            self._save_path_r1 = Path(save_r1).name
            self._save_path_r2 = Path(save_r2).name

        if locator_train:
            loc_path = locator_config_path or str(
                Path(__file__).resolve().parent.parent / "config" / "locator_train_config.yaml"
            )
            self.locator_cfg = load_locator_train_config(loc_path)
            self._weak_regions = self.locator_cfg['_weak_regions_parsed']
            self._weak_ratio = float(self.locator_cfg['sampling']['weak_ratio'])
            self._checkpoint_every = int(
                self.locator_cfg.get('training', {}).get('checkpoint_every_steps', 100)
            )
            ckpt_cfg = self.locator_cfg.get('checkpoints', {})
            loc_root = Path(__file__).resolve().parent.parent
            save_r1 = ckpt_cfg.get('save_r1', 'models/R1_locator.pth')
            save_r2 = ckpt_cfg.get('save_r2', 'models/R2_locator.pth')
            base_r1 = ckpt_cfg.get('base_r1', 'models/ur3_live_model_r1.pth')
            base_r2 = ckpt_cfg.get('base_r2', 'models/ur3_live_model_r2.pth')
            if not self.model_path:
                self.model_path = self._resolve_fine_tune_load_path(
                    loc_root, save_r1, base_r1, 'R1 locator')
            if not self.model_path_r2:
                self.model_path_r2 = self._resolve_fine_tune_load_path(
                    loc_root, save_r2, base_r2, 'R2 locator')
            self._save_path_r1 = Path(save_r1).name
            self._save_path_r2 = Path(save_r2).name

        if geo_grasp and not locator_train and not fine_tune and not rl_train:
            if not self.model_path:
                self.model_path = "models/R1_locator.pth"
            if not self.model_path_r2:
                self.model_path_r2 = "models/R2_locator.pth"

        # =========================================================================
        # DEVICE & MODEL SETUP
        # =========================================================================
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_config = self._load_model_config()
        if fine_tune:
            model_config['learning_rate'] = float(
                self.fine_tune_cfg.get('training', {}).get('learning_rate', 1e-4)
            )
        if locator_train:
            model_config['learning_rate'] = float(
                self.locator_cfg.get('training', {}).get('learning_rate', 1e-4)
            )

        # Robot 1 — Intel D455 (Wider FOV)
        self.model,  self.bc_module,  self.image_processor  = create_model(model_config)
        self.model       = self.model.to(self.device)
        self.bc_module   = self.bc_module.to(self.device)

        # Robot 2 — Intel D415 (Narrower FOV)
        self.model2, self.bc_module2, self.image_processor2 = create_model(model_config)
        self.model2      = self.model2.to(self.device)
        self.bc_module2  = self.bc_module2.to(self.device)

        # =========================================================================
        # TRAINING BUFFER & SCHEDULING
        # =========================================================================
        if fine_tune:
            self.batch_size = int(self.fine_tune_cfg.get('training', {}).get('batch_size', 16))
        elif locator_train:
            self.batch_size = int(self.locator_cfg.get('training', {}).get('batch_size', 16))
        elif rl_train:
            self.batch_size = int(self.rl_cfg.get('training', {}).get('batch_size', 16))
        else:
            self.batch_size = 16

        buf_cap = 10000
        if rl_train:
            buf_cap = int(self.rl_cfg.get('training', {}).get('buffer_capacity', 10000))

        self.data_buffer          = deque(maxlen=buf_cap)   # Robot 1 Buffer
        self.data_buffer2         = deque(maxlen=buf_cap)   # Robot 2 Buffer
        self.rl_buffer            = deque(maxlen=buf_cap)
        self.rl_buffer2           = deque(maxlen=buf_cap)
        self.weak_buffer          = deque(maxlen=10000)
        self.normal_buffer        = deque(maxlen=10000)
        self.weak_buffer2         = deque(maxlen=10000)
        self.normal_buffer2       = deque(maxlen=10000)
        self.training_step_count  = 0
        self.training_step_count2 = 0

        self._load_model_weights()

        self.loc_module: Optional[LocalizationModule] = None
        self.loc_module2: Optional[LocalizationModule] = None
        if locator_train:
            loc_lr = float(self.locator_cfg.get('training', {}).get('learning_rate', 1e-4))
            loc_wd = float(model_config.get('weight_decay', 8e-4))
            self.loc_module = LocalizationModule(
                self.model, learning_rate=loc_lr, weight_decay=loc_wd,
            ).to(self.device)
            self.loc_module2 = LocalizationModule(
                self.model2, learning_rate=loc_lr, weight_decay=loc_wd,
            ).to(self.device)
            self._load_locator_optimizer_state()

        if rl_train or rl_residual_r1 or rl_residual_r2:
            if not self.rl_cfg:
                self._load_rl_config(rl_train_config_path)
            lr = float(self._rl_robot_cfg[1].get('td3', {}).get('learning_rate', 3e-4))
            self.td3_module = create_td3_module(
                self.model, self._rl_robot_cfg[1], learning_rate=lr,
            ).to(self.device)
            self.td3_module2 = create_td3_module(
                self.model2, self._rl_robot_cfg[2], learning_rate=lr,
            ).to(self.device)
            self.rl_training_step_count  = 0
            self.rl_training_step_count2 = 0
            self._load_rl_weights(rl_residual_r1, rl_residual_r2)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.bc_module.optimizer,
            mode='min', factor=0.5, patience=50, min_lr=1e-5
        )
        self.lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.bc_module2.optimizer,
            mode='min', factor=0.5, patience=50, min_lr=1e-5
        )
        self.loc_lr_scheduler = None
        self.loc_lr_scheduler2 = None
        if locator_train and self.loc_module is not None:
            self.loc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.loc_module.optimizer,
                mode='min', factor=0.5, patience=50, min_lr=1e-5
            )
            self.loc_lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.loc_module2.optimizer,
                mode='min', factor=0.5, patience=50, min_lr=1e-5
            )

        # =========================================================================
        # VISION PREPROCESSING CONFIGURATION
        # =========================================================================
        # Fractional crop boundaries (0.0-1.0) targeting the platform region.
        # R1 (Left Camera) and R2 (Right Camera) require asymmetric windows.
        
        # Robot 1: Left camera looking inward
        R1_CROP = dict(crop_y0=0.22, crop_y1=0.7, crop_x0=0.37,  crop_x1=0.60)

        # Robot 2: Right camera looking inward
        R2_CROP = dict(crop_y0=0.21, crop_y1=0.885, crop_x0=0.37,  crop_x1=0.71)

        self.image_processor.crop_y0  = R1_CROP['crop_y0']
        self.image_processor.crop_y1  = R1_CROP['crop_y1']
        self.image_processor.crop_x0  = R1_CROP['crop_x0']
        self.image_processor.crop_x1  = R1_CROP['crop_x1']
        self.image_processor.device   = self.device

        self.image_processor2.crop_y0 = R2_CROP['crop_y0']
        self.image_processor2.crop_y1 = R2_CROP['crop_y1']
        self.image_processor2.crop_x0 = R2_CROP['crop_x0']
        self.image_processor2.crop_x1 = R2_CROP['crop_x1']
        self.image_processor2.device  = self.device

        self._r1_crop = R1_CROP
        self._r2_crop = R2_CROP

        # =========================================================================
        # COORDINATE SYSTEM MAPPING
        # =========================================================================
        # Robot base positions in the Webots world frame (X only; Z is shared).
        # Used to map world-frame object_pos labels to robot-local coordinates.
        self._robot1_base_x = -0.685
        self._robot2_base_x = -1.226
        self._robot_base_z  =  0.47235   

        # =========================================================================
        # EPISODE SYNCHRONIZATION BARRIER
        # =========================================================================
        # Ensures all active robots complete their current episode before triggering
        # global domain randomizations (e.g., lighting, floor textures).
        self._barrier_num_robots  = 2          # Set to 1 for single-robot deployments
        self._barrier_ready_count = 0
        self._barrier_event       = threading.Event()
        self._barrier_lock        = threading.Lock()
        # R2 waits here until R1 finishes domain randomization + spawn (dual-arm only).
        self._setup_r1_event      = threading.Event()
        self._setup_all_event     = threading.Event()
        self._setup_ready_count   = 0
        self._setup_lock          = threading.Lock()
        self._setup_wait_timeout_s = 120.0

        # =========================================================================
        # NETWORKING & CONCURRENCY
        # =========================================================================
        self.is_running         = False
        self.server_socket      = None
        self.client_connections = []
        self.train_lock         = threading.Lock()
        self.train_lock2        = threading.Lock()
        self._locator_step_log_lock = threading.Lock()

    def _load_config(self, config_path: str) -> Dict:
        """Loads server network configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return {'network': {'host_ip': '0.0.0.0', 'port': 8888}}

    def _load_model_config(self) -> Dict:
        """Defines baseline hyperparameters for the CNN architecture."""
        return {
            'input_channels':    4,
            'input_size':        [224, 224],
            'num_grasp_classes': 4,
            'output_6dof':       True,
            'use_attention':     True,
            'learning_rate':     5e-4,
            'weight_decay':      8e-4
        }

    @staticmethod
    def _resolve_fine_tune_load_path(
        ft_root: Path, save_rel: str, base_rel: str, label: str
    ) -> str:
        """Prefer fine-tune checkpoint on restart; fall back to base BC weights on first run."""
        save_path = ft_root / save_rel
        base_path = ft_root / base_rel
        if save_path.exists():
            print(
                f"[FINE-TUNE {label}] Resuming from {save_path.name} "
                f"(training step restores from checkpoint)"
            )
            return save_rel
        print(
            f"[FINE-TUNE {label}] No fine-tune checkpoint yet — "
            f"starting from base weights ({base_path.name})"
        )
        return base_rel

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

    def _load_model_weights(self):
        """Locates and restores checkpoint weights, handling partial mismatches."""

        def _load(model, bc_module, path_override, default_name, step_attr):
            path = self._resolve_checkpoint_path(path_override, default_name)

            print(f"🔍 Looking for weights at: {path.resolve()}")
            if path.exists():
                try:
                    checkpoint  = torch.load(path, map_location=self.device)
                    model_state = checkpoint.get('model_state_dict', checkpoint)
                    
                    missing, unexpected = model.load_state_dict(model_state, strict=False)
                    if missing:
                        print(f"   ↳ New keys (random init): {missing}")
                    if unexpected:
                        print(f"   ↳ Ignored old keys: {unexpected}")
                        
                    if 'optimizer_state_dict' in checkpoint:
                        try:
                            bc_module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        except Exception as e:
                            print(f"   ↳ Optimizer state not restored: {e}")
                            
                    step = checkpoint.get('training_step', 0)
                    print(f"✅ Loaded weights: {path.resolve()}  (step {step})")
                    return step
                    
                except Exception as e:
                    print(f"⚠️  Failed to load checkpoint ({e}) — starting with RANDOM weights.")
            else:
                print(f"⚠️  No checkpoint found — starting with RANDOM weights.")
                print(f"   ↳ Expected: {path.resolve()}")
            return 0

        self.training_step_count  = _load(
            self.model,  self.bc_module,  self.model_path,  "ur3_live_model_r1.pth", None)
        self.training_step_count2 = _load(
            self.model2, self.bc_module2, self.model_path_r2, "ur3_live_model_r2.pth", None)

    def _load_locator_optimizer_state(self):
        """Restore locator optimizers from the same checkpoint paths as the model."""

        def _load_one(loc_mod, path_override, default_name):
            if not loc_mod:
                return
            path = self._resolve_checkpoint_path(path_override, default_name)
            if path.exists():
                try:
                    loc_mod.load_model(str(path))
                except Exception as e:
                    print(f"   ↳ Locator optimizer not restored from {path.name}: {e}")

        _load_one(self.loc_module, self.model_path, "R1_locator.pth")
        _load_one(self.loc_module2, self.model_path_r2, "R2_locator.pth")

    def _load_rl_config(self, rl_train_config_path: Optional[str] = None):
        rl_path = rl_train_config_path or str(
            Path(__file__).resolve().parent.parent / "config" / "rl_train_config.yaml"
        )
        self.rl_cfg = load_rl_train_config(rl_path)
        self._rl_robot_cfg[1] = get_robot_rl_config(self.rl_cfg, 1)
        self._rl_robot_cfg[2] = get_robot_rl_config(self.rl_cfg, 2)

    def _load_rl_weights(self, path_r1: Optional[str] = None, path_r2: Optional[str] = None):
        """Load TD3 residual checkpoints (separate from BC weights)."""
        models_dir = Path(__file__).resolve().parent.parent / "models"

        def _resolve(path_override: Optional[str], default_name: str) -> Path:
            if path_override:
                p = Path(path_override)
                if not p.is_absolute():
                    p = models_dir / p
                return p
            return models_dir / default_name

        def _load_one(td3: Optional[TD3Module], path: Path, step_attr: str):
            if td3 is None:
                return
            print(f"🔍 Looking for RL residual at: {path.resolve()}")
            if path.exists():
                step = td3.load_model(str(path))
                setattr(self, step_attr, step)
                print(f"✅ Loaded RL residual: {path.name}  (step {step})")
            elif self.rl_train:
                print(f"   ↳ No RL checkpoint yet — starting fresh residual (TD3)")

        _load_one(self.td3_module, _resolve(path_r1, self._save_path_r1_rl), 'rl_training_step_count')
        _load_one(self.td3_module2, _resolve(path_r2, self._save_path_r2_rl), 'rl_training_step_count2')

    def _predict_grasp_pose(self, full_message: Dict) -> Dict:
        """BC pose + optional residual delta (RL train or inference with use_residual)."""
        client_mode = full_message.get('mode', 'inference')
        robot_id    = int(full_message.get('robot_id', 1))
        use_residual = bool(full_message.get('use_residual', False))
        add_exploration = (client_mode == 'rl_train')

        is_sim      = full_message.get('source', 'real') == 'simulation'
        camera_data = full_message['data']
        rgbd_tensor = self.preprocess_rgbd_data(
            camera_data, is_simulation=is_sim, robot_id=robot_id
        )

        active_model = self.model2 if robot_id == 2 else self.model
        td3_mod      = self.td3_module2 if robot_id == 2 else self.td3_module
        active_model.eval()

        with torch.no_grad():
            prediction = active_model(rgbd_tensor)
            grasp_pose = prediction['pose_6dof'].cpu().numpy()[0].copy()

        base_x = self._robot2_base_x if robot_id == 2 else self._robot1_base_x
        bc_pose = grasp_pose.copy()
        bc_pose[0] += base_x
        bc_pose[2] += self._robot_base_z
        bc_pose[3] = 3.14
        bc_pose[4] = 0.0

        delta = [0.0, 0.0, 0.0]
        apply_residual = td3_mod is not None and (
            add_exploration or use_residual
        )
        if apply_residual:
            expl_noise = None
            if add_exploration:
                rcfg = self._rl_robot_cfg.get(robot_id, {})
                td3_cfg = rcfg.get('td3', {})
                session_ep = max(1, int(full_message.get('session_episode', 1)))
                noise_scale = exploration_noise_scale(td3_cfg, session_ep)
                limits = td3_mod.max_delta.to(self.device)
                expl_noise = torch.randn(1, 3, device=self.device) * noise_scale * limits
            with torch.no_grad():
                delta_t = td3_mod.select_delta(rgbd_tensor, exploration_noise=expl_noise)
            delta = delta_t.detach().cpu().numpy()[0].tolist()

        final_pose = bc_pose.copy()
        final_pose[0] += delta[0]
        final_pose[2] += delta[1]
        final_pose[5] += delta[2]

        return {
            'type':       'grasp_prediction',
            'pose':       final_pose.tolist(),
            'bc_pose':    bc_pose.tolist(),
            'delta':      delta,
            'mode':       'exploit',
            'confidence': 1.0,
            'timestamp':  time.time(),
        }

    def _predict_object_xz(self, state: Dict, robot_id: int, is_sim: bool,
                           label_xz: Optional[List[float]] = None) -> Dict:
        """CNN aux_position -> world X/Z (optional error vs label)."""
        rgbd_tensor = self.preprocess_rgbd_data(state, is_simulation=is_sim, robot_id=robot_id)
        active_model = self.model2 if robot_id == 2 else self.model
        was_training = active_model.training
        active_model.eval()
        try:
            with torch.no_grad():
                local_xz = active_model(rgbd_tensor)['aux_position'].cpu().numpy()[0]
        finally:
            if was_training:
                active_model.train()

        world_x, _, world_z = local_xz_to_world(
            float(local_xz[0]), float(local_xz[1]), robot_id,
            obj_y=DEFAULT_OBJECT_Y_M,
        )
        result = {
            'pred_obj_x':   float(world_x),
            'pred_obj_z':   float(world_z),
            'pred_local_x': float(local_xz[0]),
            'pred_local_z': float(local_xz[1]),
        }
        if label_xz is not None and len(label_xz) >= 2:
            result['locator_err_m'] = math.hypot(
                world_x - float(label_xz[0]), world_z - float(label_xz[1]),
            )
        return result

    def _append_locator_step_log(self, robot_id: int, row: Dict) -> None:
        path = _REPO_ROOT / "data" / f"locator_train_steps_r{robot_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._locator_step_log_lock:
            write_header = not path.exists()
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=LOCATOR_STEP_CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({k: row.get(k, '') for k in LOCATOR_STEP_CSV_FIELDS})

    def _predict_geo_grasp_pose(self, full_message: Dict) -> Dict:
        """CNN aux_position -> analytic grasp geometry (no supervisor)."""
        robot_id    = int(full_message.get('robot_id', 1))
        is_sim      = full_message.get('source', 'real') == 'simulation'
        camera_data = full_message['data']

        pred = self._predict_object_xz(camera_data, robot_id, is_sim)
        grasp_pose = compute_grasp_pose_from_object_world(
            pred['pred_obj_x'], DEFAULT_OBJECT_Y_M, pred['pred_obj_z'],
            robot_id, add_jitter=False,
        )

        return {
            'type':        'grasp_prediction',
            'pose':        grasp_pose,
            'mode':        'exploit_geo',
            'pred_obj_x':  pred['pred_obj_x'],
            'pred_obj_z':  pred['pred_obj_z'],
            'pred_local_x': pred['pred_local_x'],
            'pred_local_z': pred['pred_local_z'],
            'confidence':  1.0,
            'timestamp':   time.time(),
        }

    def format_rl_batch_for_torch(self, batch: List[Dict], robot_id: int) -> Dict[str, torch.Tensor]:
        states_list  = []
        actions_list = []
        rewards_list = []
        dones_list   = []

        for exp in batch:
            is_sim = exp.get('source', 'real') == 'simulation'
            states_list.append(
                self.preprocess_rgbd_data(exp['state'], is_simulation=is_sim, robot_id=robot_id)
            )
            d_raw = exp.get('delta', exp.get('action', [0.0, 0.0, 0.0]))
            actions_list.append(torch.tensor(d_raw[:3], dtype=torch.float32))
            rewards_list.append(torch.tensor(float(exp['reward']), dtype=torch.float32))
            dones_list.append(torch.tensor(1.0, dtype=torch.float32))

        return {
            'states':  torch.cat(states_list).to(self.device),
            'actions': torch.stack(actions_list).to(self.device),
            'rewards': torch.stack(rewards_list).to(self.device),
            'dones':   torch.stack(dones_list).to(self.device),
        }

    def _run_rl_training_step(self, robot_id: int = 1):
        lock = self.train_lock if robot_id == 1 else self.train_lock2
        if lock.locked():
            return

        with lock:
            try:
                if robot_id == 2:
                    td3_mod   = self.td3_module2
                    buf       = self.rl_buffer2
                    step_attr = 'rl_training_step_count2'
                    save_name = self._save_path_r2_rl
                else:
                    td3_mod   = self.td3_module
                    buf       = self.rl_buffer
                    step_attr = 'rl_training_step_count'
                    save_name = self._save_path_r1_rl

                if td3_mod is None or len(buf) < self.batch_size:
                    return

                batch_raw   = random.sample(list(buf), self.batch_size)
                torch_batch = self.format_rl_batch_for_torch(batch_raw, robot_id)
                losses      = td3_mod.update_networks(torch_batch)

                step = getattr(self, step_attr, 0) + 1
                setattr(self, step_attr, step)

                if step % 5 == 0:
                    print(
                        f"🎯 R{robot_id} RL Step {step:4d} | "
                        f"Loss: {losses['total']:.4f} "
                        f"(Critic:{losses['critic']:.4f} Actor:{losses['actor']:.4f}) | "
                        f"Buffer:{len(buf)}"
                    )

                if step % self._rl_checkpoint_every == 0:
                    base_dir = Path(__file__).resolve().parent.parent / "models"
                    base_dir.mkdir(parents=True, exist_ok=True)
                    full_path = base_dir / save_name
                    td3_mod.save_model(str(full_path), training_step=step)
                    print(f"💾 R{robot_id} SAVED RL RESIDUAL TO: {full_path}")

            except Exception as e:
                print(f"❌ CRITICAL RL TRAINING ERROR (R{robot_id}): {e}")
                import traceback
                traceback.print_exc()

    def _sample_mixed_batch(self, weak_buf: deque, normal_buf: deque,
                            batch_size: int, weak_ratio: float) -> Optional[List[Dict]]:
        """Sample ~weak_ratio from weak_buf and remainder from normal_buf."""
        if not weak_buf and not normal_buf:
            return None

        n_weak = int(round(batch_size * weak_ratio))
        n_weak = max(0, min(n_weak, batch_size))
        n_normal = batch_size - n_weak

        weak_list = list(weak_buf)
        normal_list = list(normal_buf)

        batch: List[Dict] = []
        if n_weak > 0 and weak_list:
            batch.extend(random.choices(weak_list, k=n_weak))
        elif n_weak > 0 and normal_list:
            batch.extend(random.choices(normal_list, k=n_weak))

        if n_normal > 0 and normal_list:
            batch.extend(random.choices(normal_list, k=n_normal))
        elif n_normal > 0 and weak_list:
            batch.extend(random.choices(weak_list, k=n_normal))

        if not batch:
            return None
        return batch

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

    def _world_to_robot_local_xz(self, world_x: float, world_z: float,
                                   robot_id: int) -> Tuple[float, float]:
        """
        Translates Webots absolute world coordinates to the specified robot's 
        local frame to ensure consistent aux_position_head learning.
        """
        base_x = self._robot2_base_x if robot_id == 2 else self._robot1_base_x
        base_z = self._robot_base_z
        return (world_x - base_x, world_z - base_z)

    def format_batch_for_torch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Converts raw buffer samples into batched device tensors for training."""
        states_list       = []
        pose_labels_list  = []
        grasp_labels_list = []
        rewards_list      = []
        aux_pos_list      = []

        for exp in batch:
            s_raw    = exp['state']
            a_raw    = exp['action']
            r_raw    = exp['reward']
            obj_pos  = exp.get('object_pos', [0.0, 0.0])
            is_sim   = exp.get('source', 'real') == 'simulation'
            robot_id = exp.get('robot_id', 1)

            states_list.append(
                self.preprocess_rgbd_data(s_raw, is_simulation=is_sim, robot_id=robot_id)
            )
       
            local_pose = list(a_raw)
            base_x = self._robot2_base_x if robot_id == 2 else self._robot1_base_x
            local_pose[0] -= base_x
            local_pose[2] -= self._robot_base_z
            pose_labels_list.append(torch.tensor(local_pose, dtype=torch.float32))

            grasp_class = 1 if float(r_raw) >= 0.5 else 0
            grasp_labels_list.append(torch.tensor(grasp_class, dtype=torch.long))

            rewards_list.append(torch.tensor(r_raw, dtype=torch.float32).unsqueeze(0))

            local_x, local_z = self._world_to_robot_local_xz(
                float(obj_pos[0]), float(obj_pos[1]), robot_id
            )
            aux_pos_list.append(torch.tensor([local_x, local_z], dtype=torch.float32))

        return {
            'states':              torch.cat(states_list).to(self.device),
            'pose_labels':         torch.stack(pose_labels_list).to(self.device),
            'grasp_labels':        torch.stack(grasp_labels_list).to(self.device),
            'rewards':             torch.stack(rewards_list).to(self.device),
            'aux_position_labels': torch.stack(aux_pos_list).to(self.device),
        }

    def preprocess_rgbd_data(self, rgbd_data: Dict, is_simulation: bool = False,
                              robot_id: int = 1) -> torch.Tensor:
        """Applies spatial cropping and normalization to raw RGB-D inputs."""
        img_data = self.decode_b64_image(rgbd_data)
        depth    = img_data['depth']
        rgb      = img_data['rgb']

        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0

        if not is_simulation:
            SIM_MEAN  = 0.700
            REAL_MEAN = 0.743
            depth_shift = 0  
            depth = np.where(depth > 0, depth - depth_shift, 0)

        if robot_id == 2:
            proc = self.image_processor2
            crop = self._r2_crop
        else:
            proc = self.image_processor
            crop = self._r1_crop
            
        proc.is_simulation = is_simulation

        h, w = rgb.shape[:2]
        y0 = int(crop['crop_y0'] * h)
        y1 = int(crop['crop_y1'] * h)
        x0 = int(crop['crop_x0'] * w)
        x1 = int(crop['crop_x1'] * w)

        processed = proc.process_rgbd_image(rgb, depth)
        
        return processed

    # =========================================================================
    # MESSAGE HANDLERS & TRAINING LOGIC
    # =========================================================================

    def _handle_camera_data(self, full_message: Dict) -> Dict:
        """
        Processes incoming frames. Routes to model inference if in 'inference' mode,
        or delegates to the client-side teacher algorithm in 'training' mode.
        """
        try:
            client_mode = full_message.get('mode', 'inference')
            robot_id    = int(full_message.get('robot_id', 1))

            if client_mode in ('training', 'fine_tune', 'locator_train'):
                return {
                    'type':      'grasp_prediction',
                    'mode':      'explore',
                    'pose':      [0.0] * 6,
                    'timestamp': time.time()
                }

            use_geo = self.geo_grasp or bool(full_message.get('use_geo_grasp', False))
            if use_geo:
                return self._predict_geo_grasp_pose(full_message)

            use_residual = bool(full_message.get('use_residual', False))
            if client_mode == 'rl_train' or use_residual:
                if self.td3_module is None and self.td3_module2 is None:
                    return {
                        'type': 'error',
                        'message': 'RL residual not loaded — start gpu_server with --rl-train or --rl-residual-r1/r2',
                    }
                return self._predict_grasp_pose(full_message)

            is_sim      = full_message.get('source', 'real') == 'simulation'
            camera_data = full_message['data']
            rgbd_tensor = self.preprocess_rgbd_data(
                camera_data, is_simulation=is_sim, robot_id=robot_id
            )

            active_model = self.model2 if robot_id == 2 else self.model
            active_model.eval()
            with torch.no_grad():
                prediction = active_model(rgbd_tensor)
                grasp_pose = prediction['pose_6dof'].cpu().numpy()[0]

            base_x = self._robot2_base_x if robot_id == 2 else self._robot1_base_x
            grasp_pose[0] += base_x
            grasp_pose[2] += self._robot_base_z
            grasp_pose[3] = 3.14
            grasp_pose[4] = 0.0

            return {
                'type':       'grasp_prediction',
                'pose':       grasp_pose.tolist(),
                'mode':       'exploit',
                'confidence': 1.0,
                'timestamp':  time.time()
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
                    cv2.imwrite(f"ai_vision_debug_rgb_sim_r{robot_id}.jpg", rgb_crop)
                    depth_crop = depth[y0:y1, x0:x1].copy()
                    depth_vis  = cv2.normalize(depth_crop, None, 0, 255,
                                               cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imwrite(f"ai_vision_debug_depth_sim_r{robot_id}.png", depth_vis)
                except Exception as e:
                    print(f"[DEBUG R{robot_id}] Sim image save failed: {e}")

            sample = {
                'state':      training_data['state'],
                'action':     training_data['action'],
                'reward':     training_data['reward'],
                'object_pos': training_data.get('object_pos', [0.0, 0.0]),
                'source':     source,
                'robot_id':   robot_id,
            }

            if self.rl_train:
                sample['delta'] = training_data.get(
                    'delta', training_data.get('residual_delta', [0.0, 0.0, 0.0])
                )
                skip_clamp = bool(
                    self.rl_cfg.get('training', {}).get('skip_buffer_when_clamp_limited', False)
                )
                clamp_limited = bool(training_data.get('clamp_limited', False))
                if skip_clamp and clamp_limited:
                    target_buf = self.rl_buffer2 if robot_id == 2 else self.rl_buffer
                    print(
                        f"[RL R{robot_id}] Skip replay (clamp_limited) | "
                        f"buffer={len(target_buf)} session_ep={training_data.get('session_episode', '?')}"
                    )
                    return {
                        'type': 'training_ack',
                        'buffer_len': len(target_buf),
                        'rl': True,
                        'skipped': True,
                    }
                target_buf = self.rl_buffer2 if robot_id == 2 else self.rl_buffer
                target_buf.append(sample)
                if len(target_buf) >= self.batch_size:
                    threading.Thread(
                        target=self._run_rl_training_step, args=(robot_id,), daemon=True
                    ).start()
                return {
                    'type': 'training_ack',
                    'buffer_len': len(target_buf),
                    'rl': True,
                }

            if self.fine_tune or self.locator_train:
                bucket = self._resolve_demo_bucket(training_data, robot_id)
                sample['demo_bucket'] = bucket
                if robot_id == 2:
                    target = self.weak_buffer2 if bucket == 'weak' else self.normal_buffer2
                else:
                    target = self.weak_buffer if bucket == 'weak' else self.normal_buffer
                target.append(sample)
                if self._fine_tune_buffers_ready(robot_id):
                    threading.Thread(
                        target=self._run_training_step, args=(robot_id,), daemon=True
                    ).start()
                weak_n = len(self.weak_buffer2 if robot_id == 2 else self.weak_buffer)
                normal_n = len(self.normal_buffer2 if robot_id == 2 else self.normal_buffer)
                ack = {
                    'type': 'training_ack',
                    'buffer_len': weak_n + normal_n,
                    'weak_len': weak_n,
                    'normal_len': normal_n,
                    'demo_bucket': bucket,
                    'locator': self.locator_train,
                }
                if self.locator_train:
                    obj_pos = training_data.get('object_pos', [0.0, 0.0])
                    loc_step = (
                        self.training_step_count2 if robot_id == 2 else self.training_step_count
                    )
                    pred_info = self._predict_object_xz(
                        training_data['state'], robot_id, source == 'simulation',
                        label_xz=obj_pos,
                    )
                    ack.update(pred_info)
                    ack['loc_step'] = loc_step
                return ack

            if robot_id == 2:
                self.data_buffer2.append(sample)
                if len(self.data_buffer2) >= self.batch_size:
                    threading.Thread(
                        target=self._run_training_step, args=(2,), daemon=True
                    ).start()
            else:
                self.data_buffer.append(sample)
                if len(self.data_buffer) >= self.batch_size:
                    threading.Thread(
                        target=self._run_training_step, args=(1,), daemon=True
                    ).start()

            return {'type': 'training_ack', 'buffer_len': len(self.data_buffer)}
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    def _run_training_step(self, robot_id: int = 1):
        """Executes a single mini-batch gradient descent update asynchronously."""
        lock = self.train_lock if robot_id == 1 else self.train_lock2
        if lock.locked():
            return

        with lock:
            try:
                if robot_id == 2:
                    bc_mod    = self.bc_module2
                    loc_mod   = self.loc_module2
                    model     = self.model2
                    buf       = self.data_buffer2
                    weak_buf  = self.weak_buffer2
                    normal_buf = self.normal_buffer2
                    sched     = self.loc_lr_scheduler2 if self.locator_train else self.lr_scheduler2
                    step_attr = 'training_step_count2'
                    if self.locator_train:
                        save_name = self._save_path_r2
                    elif self.fine_tune:
                        save_name = self._save_path_r2
                    else:
                        save_name = "ur3_live_model_r2.pth"
                else:
                    bc_mod    = self.bc_module
                    loc_mod   = self.loc_module
                    model     = self.model
                    buf       = self.data_buffer
                    weak_buf  = self.weak_buffer
                    normal_buf = self.normal_buffer
                    sched     = self.loc_lr_scheduler if self.locator_train else self.lr_scheduler
                    step_attr = 'training_step_count'
                    if self.locator_train:
                        save_name = self._save_path_r1
                    elif self.fine_tune:
                        save_name = self._save_path_r1
                    else:
                        save_name = "ur3_live_model_r1.pth"

                train_mod = loc_mod if self.locator_train else bc_mod

                if self.fine_tune or self.locator_train:
                    batch_raw = self._sample_mixed_batch(
                        weak_buf, normal_buf, self.batch_size, self._weak_ratio
                    )
                    if not batch_raw:
                        return
                else:
                    batch_raw = random.sample(list(buf), self.batch_size)

                torch_batch = self.format_batch_for_torch(batch_raw)
                losses      = train_mod.update_networks(torch_batch)
                sched_metric = losses['aux'] if self.locator_train else losses['pose']
                sched.step(sched_metric)
                
                step = getattr(self, step_attr) + 1
                setattr(self, step_attr, step)

                ckpt_every = self._checkpoint_every if (self.fine_tune or self.locator_train) else 100
                checkpoint_saved = int(step % ckpt_every == 0)

                if self.locator_train:
                    batch_weak = sum(1 for s in batch_raw if s.get('demo_bucket') == 'weak')
                    batch_normal = len(batch_raw) - batch_weak
                    self._append_locator_step_log(robot_id, {
                        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                        'robot_id': robot_id,
                        'loc_step': step,
                        'aux_loss': losses['aux'],
                        'grad_norm': losses['grad_norm'],
                        'buffer_weak': len(weak_buf),
                        'buffer_normal': len(normal_buf),
                        'batch_weak': batch_weak,
                        'batch_normal': batch_normal,
                        'checkpoint_saved': checkpoint_saved,
                    })

                if step % 5 == 0:
                    buf_info = ""
                    if self.fine_tune or self.locator_train:
                        buf_info = (
                            f" | Buffers weak:{len(weak_buf)} normal:{len(normal_buf)}"
                        )
                    tag = "LOC" if self.locator_train else "BC"
                    print(
                        f"🔥 R{robot_id} {tag} Step {step:4d} | "
                        f"Loss: {losses['total']:.4f} "
                        f"(Pose:{losses['pose']:.4f} "
                        f"Aux:{losses['aux']:.4f} "
                        f"Grasp:{losses['grasp']:.4f}[monitor only]) | "
                        f"GradNorm: {losses['grad_norm']:.3f}"
                        f"{buf_info}"
                    )

                if checkpoint_saved:
                    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    save_dir  = os.path.join(base_dir, "models")
                    os.makedirs(save_dir, exist_ok=True)
                    full_path = os.path.join(save_dir, save_name)
                    
                    ckpt = {
                        'model_state_dict':     model.state_dict(),
                        'training_step':        step,
                        'fine_tune':            self.fine_tune,
                        'locator_train':        self.locator_train,
                    }
                    if self.locator_train:
                        ckpt['optimizer_state_dict'] = loc_mod.optimizer.state_dict()
                    else:
                        ckpt['optimizer_state_dict'] = bc_mod.optimizer.state_dict()
                    torch.save(ckpt, full_path)
                    print(f"💾 R{robot_id} SAVED MODEL TO: {full_path}")

            except Exception as e:
                print(f"❌ CRITICAL TRAINING ERROR (R{robot_id}): {e}")
                import traceback
                traceback.print_exc()

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

        self._barrier_event.wait()

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
        if self._barrier_num_robots < 2 or robot_id != 2:
            return {'type': 'proceed'}
        print(f"[SETUP BARRIER] R{robot_id} waiting for R1 world setup...")
        if not self._setup_r1_event.wait(timeout=self._setup_wait_timeout_s):
            return {
                'type': 'error',
                'message': f'timeout ({self._setup_wait_timeout_s}s) waiting for R1 setup',
            }
        print(f"[SETUP BARRIER] R{robot_id} cleared — starting local setup")
        return {'type': 'proceed'}

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
        mode_label = "residual RL" if self.rl_train else (
            "locator train" if self.locator_train else (
                "targeted fine-tune" if self.fine_tune else "behavior cloning"
            )
        )
        print(f"🚀 BC Server listening on {host}:{port} ({mode_label} mode)")
        if self.geo_grasp and not self.locator_train:
            print(f"   Geo-grasp inference enabled (aux_position → grasp geometry)")
        if self.rl_train:
            lr = float(self._rl_robot_cfg[1].get('td3', {}).get('learning_rate', 1e-4))
            print(
                f"   RL TD3 minimal-dynamics | batch={self.batch_size} | LR={lr:g} | "
                f"noise anneal | align gate | skip clamp buffer | "
                f"checkpoints: {self._save_path_r1_rl}, {self._save_path_r2_rl}"
            )
        if self.fine_tune:
            print(
                f"   Fine-tune sampling: {self._weak_ratio:.0%} weak / "
                f"{1 - self._weak_ratio:.0%} normal | LR="
                f"{self.fine_tune_cfg.get('training', {}).get('learning_rate', 1e-4)}"
            )
        if self.locator_train:
            print(
                f"   Locator sampling: {self._weak_ratio:.0%} weak / "
                f"{1 - self._weak_ratio:.0%} normal | aux-only loss | LR="
                f"{self.locator_cfg.get('training', {}).get('learning_rate', 1e-4)} | "
                f"checkpoints: {self._save_path_r1}, {self._save_path_r2}"
            )
        
        while True:
            conn, addr = self.server_socket.accept()
            threading.Thread(
                target=self.handle_client_request, args=(conn, addr), daemon=True
            ).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        help='Checkpoint path for R1 (base weights in fine-tune mode)')
    parser.add_argument('--model-r2', type=str, default=None,
                        help='Checkpoint path for R2 (base weights in fine-tune mode)')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Targeted BC fine-tune with mixed weak/normal batch sampling')
    parser.add_argument('--fine-tune-config', type=str, default=None,
                        help='Path to fine_tune_config.yaml')
    parser.add_argument('--locator-train', action='store_true',
                        help='Supervised aux_position training for geo-grasp pipeline')
    parser.add_argument('--locator-config', type=str, default=None,
                        help='Path to locator_train_config.yaml')
    parser.add_argument('--geo-grasp', action='store_true',
                        help='Inference: aux_position → analytic grasp geometry (not pose_6dof)')
    parser.add_argument('--rl-train', action='store_true',
                        help='TD3 residual RL fine-tune (BC frozen; separate RL checkpoints)')
    parser.add_argument('--rl-train-config', type=str, default=None,
                        help='Path to rl_train_config.yaml')
    parser.add_argument('--rl-residual-r1', type=str, default=None,
                        help='RL residual checkpoint for R1 (inference with --use-residual on client)')
    parser.add_argument('--rl-residual-r2', type=str, default=None,
                        help='RL residual checkpoint for R2')
    args   = parser.parse_args()

    if args.rl_train and args.fine_tune:
        parser.error('Use either --rl-train or --fine-tune, not both.')
    if args.rl_train and args.locator_train:
        parser.error('Use either --rl-train or --locator-train, not both.')
    if args.fine_tune and args.locator_train:
        parser.error('Use either --fine-tune or --locator-train, not both.')
    if args.geo_grasp and args.locator_train:
        parser.error('--geo-grasp is for inference only; omit when --locator-train.')

    server = GPUInferenceServer(
        model_path=args.model,
        model_path_r2=args.model_r2,
        fine_tune=args.fine_tune,
        fine_tune_config_path=args.fine_tune_config,
        locator_train=args.locator_train,
        locator_config_path=args.locator_config,
        geo_grasp=args.geo_grasp,
        rl_train=args.rl_train,
        rl_train_config_path=args.rl_train_config,
        rl_residual_r1=args.rl_residual_r1,
        rl_residual_r2=args.rl_residual_r2,
    )
    server.start_server()