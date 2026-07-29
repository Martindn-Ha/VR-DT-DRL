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
from local_grid_module import LocalGridModule, create_local_grid_module
from full_board_dqn_module import FullBoardDQNModule, create_full_board_dqn_module
from board_locator_module import (
    BoardLocatorModule, create_board_locator_module, fuse_q_with_detector,
)
from yolo_locator import (
    LocalYoloLocator, load_yolo_locator_config, random_cell_in_bbox,
    best_q_cell_in_bbox, dist_cell_to_bbox_px, dist_cell_to_bbox_cells,
)
from local_bbox_dqn_module import LocalBBoxDQNModule, create_local_bbox_dqn_module
from local_bbox_window import (
    load_local_bbox_dqn_config, local_bbox_grid_params,
    bbox_center_world, crop_fixed_window, local_cell_to_world, world_to_local_cell,
    window_xyxy_warp,
)
from vlm_box_selector import (
    DEFAULT_OLLAMA_URL, DEFAULT_VLM_MODEL,
    VlmSelectFailedError, VlmUnavailableError, select_bbox_by_vlm_point,
)
from board_seg_labels import make_block_mask_warp, blend_mask_overlay, mask_grid_to_cell

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "vm_simulation_system" / "src"))
from board_q_diagnostics import compute_board_q_diagnostics, blend_q_heatmap_overlay  # noqa: E402
from spawn_geometry import (  # noqa: E402
    load_fine_tune_config, load_locator_train_config, load_board_locator_train_config,
    classify_demo_bucket,
)
from rl_reward import load_rl_train_config, get_robot_rl_config, exploration_noise_scale  # noqa: E402
from grasp_geometry import compute_grasp_pose_from_object_world, local_xz_to_world, DEFAULT_OBJECT_Y_M  # noqa: E402
from local_grid import (  # noqa: E402
    load_grid_train_config, grid_params, world_to_cell, cell_to_world, epsilon_for_episode,
    calculate_grid_reward,
)
from board_grid import (  # noqa: E402
    load_board_dqn_config, board_n, world_to_board_cell, board_cell_to_world,
    calculate_board_reward, epsilon_for_episode as board_epsilon_for_episode,
    invalid_cell_mask, board_shaping_q_targets,
)
from board_warp import (
    warp_rgb_to_board, warp_depth_to_board,
    world_xz_to_warp_pixel, warp_pixel_to_crop_pixel, warp_pixel_to_world_xz,
    get_warp_corners, _warp_image_corners,
)  # noqa: E402

_PLATFORM_CENTER = {1: (-0.646, 0.841), 2: (-1.365, 0.850)}

LOCATOR_STEP_CSV_FIELDS = [
    'timestamp_utc', 'robot_id', 'loc_step', 'aux_loss', 'grad_norm',
    'buffer_weak', 'buffer_normal', 'batch_weak', 'batch_normal', 'checkpoint_saved',
]

GRID_STEP_CSV_FIELDS = [
    'timestamp_utc', 'robot_id', 'grid_step', 'phase', 'loss', 'acc', 'grad_norm',
    'buffer_weak', 'buffer_normal', 'replay_len', 'checkpoint_saved',
]

BOARD_STEP_CSV_FIELDS = [
    'timestamp_utc', 'robot_id', 'board_step', 'phase', 'loss', 'shaping_mse', 'grad_norm',
    'buffer_weak', 'buffer_normal', 'replay_len', 'checkpoint_saved',
]

BOARD_LOCATOR_STEP_CSV_FIELDS = [
    'timestamp_utc', 'robot_id', 'locator_step', 'seg_bce', 'seg_dice', 'grad_norm',
    'buffer_weak', 'buffer_normal', 'checkpoint_saved',
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
                 grid_train: bool = False,
                 grid_config_path: str = None,
                 local_grid: bool = False,
                 grid_model_path: str = None,
                 grid_model_path_r2: str = None,
                 board_dqn_train: bool = False,
                 board_dqn: bool = False,
                 board_config_path: str = None,
                 board_model_path: str = None,
                 board_model_path_r2: str = None,
                 board_locator_train: bool = False,
                 board_locator: bool = False,
                 board_locator_config_path: str = None,
                 board_locator_model_path: str = None,
                 board_locator_model_path_r2: str = None,
                 yolo_locator_test: bool = False,
                 yolo_locator_config_path: str = None,
                 yolo_fuse: bool = False,
                 local_bbox_dqn: bool = False,
                 local_bbox_dqn_train: bool = False,
                 local_bbox_dqn_config_path: str = None,
                 local_bbox_model_path: str = None,
                 local_bbox_model_path_r2: str = None,
                 use_vlm_select: bool = False,
                 vlm_model: str = None,
                 ollama_url: str = None,
                 rl_train: bool = False, rl_train_config_path: str = None,
                 rl_residual_r1: str = None, rl_residual_r2: str = None):
        self.config     = self._load_config(config_path)
        self.model_path = model_path
        self.model_path_r2 = model_path_r2
        self.fine_tune  = fine_tune
        self.locator_train = locator_train
        self.geo_grasp  = geo_grasp
        self.grid_train = grid_train
        self.local_grid = local_grid
        self.grid_model_path = grid_model_path
        self.grid_model_path_r2 = grid_model_path_r2
        self.board_dqn_train = board_dqn_train
        self.board_dqn = board_dqn
        self.board_model_path = board_model_path
        self.board_model_path_r2 = board_model_path_r2
        self.board_locator_train = board_locator_train
        self.board_locator = board_locator
        self.board_locator_model_path = board_locator_model_path
        self.board_locator_model_path_r2 = board_locator_model_path_r2
        self.yolo_locator_test = yolo_locator_test
        self.yolo_fuse = yolo_fuse
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
        self.rl_train   = rl_train
        self.fine_tune_cfg: Dict = {}
        self.locator_cfg: Dict = {}
        self.grid_cfg: Dict = {}
        self.board_cfg: Dict = {}
        self.board_locator_cfg: Dict = {}
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

        if grid_train or local_grid:
            g_path = grid_config_path or str(
                Path(__file__).resolve().parent.parent / "config" / "grid_train_config.yaml"
            )
            self.grid_cfg = load_grid_train_config(g_path)
            self._grid_n, self._grid_window_m = grid_params(self.grid_cfg)
            self._grid_n_cells = self._grid_n * self._grid_n
            ckpt_cfg = self.grid_cfg.get('checkpoints', {})
            loc_root = Path(__file__).resolve().parent.parent
            loc_r1 = ckpt_cfg.get('locator_r1', 'models/R1_locator.pth')
            loc_r2 = ckpt_cfg.get('locator_r2', 'models/R2_locator.pth')
            save_g1 = ckpt_cfg.get('save_r1', 'models/R1_local_grid.pth')
            save_g2 = ckpt_cfg.get('save_r2', 'models/R2_local_grid.pth')
            if not self.model_path:
                self.model_path = str(self._resolve_checkpoint_path(loc_r1, 'R1_locator.pth'))
            if not self.model_path_r2:
                self.model_path_r2 = str(self._resolve_checkpoint_path(loc_r2, 'R2_locator.pth'))
            self._save_path_r1_grid = Path(save_g1).name
            self._save_path_r2_grid = Path(save_g2).name
            if grid_train:
                self._weak_regions = self.grid_cfg['_weak_regions_parsed']
                self._weak_ratio = float(self.grid_cfg['sampling']['weak_ratio'])
                self._checkpoint_every = int(
                    self.grid_cfg.get('training', {}).get('checkpoint_every_steps', 100)
                )
            if local_grid and not grid_train:
                if not self.grid_model_path:
                    self.grid_model_path = save_g1
                if not self.grid_model_path_r2:
                    self.grid_model_path_r2 = save_g2

        if (board_dqn_train or board_dqn or board_locator_train or board_locator
                or yolo_locator_test or local_bbox_dqn or local_bbox_dqn_train):
            b_path = board_config_path or str(
                Path(__file__).resolve().parent.parent / "config" / "board_dqn_config.yaml"
            )
            self.board_cfg = load_board_dqn_config(b_path)
            self._board_n = board_n(self.board_cfg)
            self._board_n_cells = self._board_n * self._board_n

        if yolo_locator_test or yolo_fuse:
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

        if board_dqn_train or board_dqn:
            ckpt_cfg = self.board_cfg.get('checkpoints', {})
            save_b1 = ckpt_cfg.get('save_r1', 'models/R1_board_dqn.pth')
            save_b2 = ckpt_cfg.get('save_r2', 'models/R2_board_dqn.pth')
            self._save_path_r1_board = Path(save_b1).name
            self._save_path_r2_board = Path(save_b2).name
            if board_dqn_train:
                self._weak_regions = self.board_cfg['_weak_regions_parsed']
                self._weak_ratio = float(self.board_cfg['sampling']['weak_ratio'])
                self._checkpoint_every = int(
                    self.board_cfg.get('training', {}).get('checkpoint_every_steps', 100)
                )
            if board_dqn and not board_dqn_train:
                if not self.board_model_path:
                    self.board_model_path = save_b1
                if not self.board_model_path_r2:
                    self.board_model_path_r2 = save_b2

        if board_locator_train or board_locator:
            bl_path = board_locator_config_path or str(
                Path(__file__).resolve().parent.parent / "config" / "board_locator_train_config.yaml"
            )
            self.board_locator_cfg = load_board_locator_train_config(bl_path)
            bl_ckpt = self.board_locator_cfg.get('checkpoints', {})
            save_l1 = bl_ckpt.get('save_r1', 'models/R1_board_locator.pth')
            save_l2 = bl_ckpt.get('save_r2', 'models/R2_board_locator.pth')
            self._save_path_r1_board_locator = Path(save_l1).name
            self._save_path_r2_board_locator = Path(save_l2).name
            if board_locator_train:
                self._weak_regions = self.board_locator_cfg['_weak_regions_parsed']
                self._weak_ratio = float(self.board_locator_cfg['sampling']['weak_ratio'])
                self._checkpoint_every = int(
                    self.board_locator_cfg.get('training', {}).get('checkpoint_every_steps', 100)
                )
            if board_locator and not board_locator_train:
                if not self.board_locator_model_path:
                    self.board_locator_model_path = save_l1
                if not self.board_locator_model_path_r2:
                    self.board_locator_model_path_r2 = save_l2

        if geo_grasp and not locator_train and not fine_tune and not rl_train and not grid_train and not local_grid and not board_dqn_train and not board_dqn and not board_locator_train and not board_locator and not yolo_locator_test and not local_bbox_dqn and not local_bbox_dqn_train:
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
        if grid_train:
            model_config['learning_rate'] = float(
                self.grid_cfg.get('training', {}).get('learning_rate', 1e-4)
            )
        if board_dqn_train:
            model_config['learning_rate'] = float(
                self.board_cfg.get('training', {}).get('learning_rate', 1e-3)
            )
        if board_locator_train:
            model_config['learning_rate'] = float(
                self.board_locator_cfg.get('training', {}).get('learning_rate', 1e-3)
            )
        if local_bbox_dqn_train:
            model_config['learning_rate'] = float(
                self.local_bbox_cfg.get('training', {}).get('learning_rate', 1e-3)
            )

        board_only = (
            board_dqn_train or board_dqn or board_locator_train or board_locator
            or yolo_locator_test or local_bbox_dqn or local_bbox_dqn_train
        )
        self._board_only = board_only
        self.model = None
        self.model2 = None
        self.bc_module = None
        self.bc_module2 = None
        self.image_processor = None
        self.image_processor2 = None

        if not board_only:
            # Robot 1 — Intel D455 (Wider FOV)
            self.model, self.bc_module, self.image_processor = create_model(model_config)
            self.model = self.model.to(self.device)
            self.bc_module = self.bc_module.to(self.device)

            # Robot 2 — Intel D415 (Narrower FOV)
            self.model2, self.bc_module2, self.image_processor2 = create_model(model_config)
            self.model2 = self.model2.to(self.device)
            self.bc_module2 = self.bc_module2.to(self.device)

        # =========================================================================
        # TRAINING BUFFER & SCHEDULING
        # =========================================================================
        if fine_tune:
            self.batch_size = int(self.fine_tune_cfg.get('training', {}).get('batch_size', 16))
        elif locator_train:
            self.batch_size = int(self.locator_cfg.get('training', {}).get('batch_size', 16))
        elif grid_train:
            self.batch_size = int(self.grid_cfg.get('training', {}).get('batch_size', 16))
        elif board_dqn_train:
            self.batch_size = int(self.board_cfg.get('training', {}).get('batch_size', 4))
        elif board_locator_train:
            self.batch_size = int(self.board_locator_cfg.get('training', {}).get('batch_size', 8))
        elif local_bbox_dqn_train:
            self.batch_size = int(self.local_bbox_cfg.get('training', {}).get('batch_size', 8))
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
        replay_cap = 10000
        if grid_train:
            replay_cap = int(self.grid_cfg.get('training', {}).get('replay_capacity', 10000))
        if local_bbox_dqn_train:
            replay_cap = int(self.local_bbox_cfg.get('training', {}).get('replay_capacity', 5000))
        self.grid_replay          = deque(maxlen=replay_cap)
        self.grid_replay2         = deque(maxlen=replay_cap)
        self.local_bbox_replay    = deque(maxlen=replay_cap)
        self.local_bbox_replay2   = deque(maxlen=replay_cap)
        self.training_step_count  = 0
        self.training_step_count2 = 0
        self.grid_training_step_count  = 0
        self.grid_training_step_count2 = 0
        self.local_bbox_training_step_count  = 0
        self.local_bbox_training_step_count2 = 0
        self.board_training_step_count  = 0
        self.board_training_step_count2 = 0
        self.board_locator_training_step_count  = 0
        self.board_locator_training_step_count2 = 0
        self.board_shaping_episode_count  = 0
        self.board_shaping_episode_count2 = 0
        self.board_cnn_correct_count  = 0
        self.board_cnn_correct_count2 = 0
        self.board_cnn_total_count  = 0
        self.board_cnn_total_count2 = 0

        if not board_only:
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

        self.grid_module: Optional[LocalGridModule] = None
        self.grid_module2: Optional[LocalGridModule] = None
        if grid_train or local_grid:
            self._init_grid_modules(model_config)
            if grid_train:
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model2.parameters():
                    param.requires_grad = False
                self.model.eval()
                self.model2.eval()

        self.board_module: Optional[FullBoardDQNModule] = None
        self.board_module2: Optional[FullBoardDQNModule] = None
        if board_dqn_train or board_dqn:
            self._init_board_modules()

        self.board_loc_module: Optional[BoardLocatorModule] = None
        self.board_loc_module2: Optional[BoardLocatorModule] = None
        if board_locator_train or board_locator:
            self._init_board_locator_modules()

        if local_bbox_dqn or local_bbox_dqn_train:
            self._init_local_bbox_modules()

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

        self.lr_scheduler = None
        self.lr_scheduler2 = None
        if not board_only:
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

        self._r1_crop = R1_CROP
        self._r2_crop = R2_CROP

        if not board_only:
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

    def _init_grid_modules(self, model_config: Dict) -> None:
        """Build local-grid Q modules (separate grasp_net per robot)."""
        tr = self.grid_cfg.get('training', {})
        freeze = bool(tr.get('freeze_backbone', True))
        lr = float(tr.get('learning_rate', 1e-4))
        wd = float(model_config.get('weight_decay', 8e-4))
        gamma = float(tr.get('gamma', 0.99))
        n_cells = self._grid_n_cells

        def _build_one(locator_model: UR3GraspCNN_Enhanced) -> LocalGridModule:
            grid_net, _, _ = create_model(model_config)
            grid_net.load_state_dict(locator_model.state_dict(), strict=False)
            return create_local_grid_module(
                grid_net,
                n_cells=n_cells,
                learning_rate=lr,
                weight_decay=wd,
                gamma=gamma,
                freeze_backbone=freeze,
            ).to(self.device)

        self.grid_module = _build_one(self.model)
        self.grid_module2 = _build_one(self.model2)
        self._load_grid_weights()

    def _load_grid_weights(self) -> None:
        host_root = Path(__file__).resolve().parent.parent

        def _load(mod: Optional[LocalGridModule], path_override: Optional[str], default_name: str, step_attr: str):
            if mod is None:
                return
            path = self._resolve_checkpoint_path(path_override, default_name)
            print(f"🔍 Looking for local grid at: {path.resolve()}")
            if path.exists():
                step = mod.load_model(str(path))
                setattr(self, step_attr, max(getattr(self, step_attr), step))
                print(f"✅ Loaded local grid: {path.name}  (step {step})")
            elif self.grid_train:
                print(f"   ↳ No grid checkpoint yet — starting from locator backbone + random Q-head")
            elif self.local_grid:
                print(f"⚠️  WARNING: No local grid checkpoint at {path.name} — Q-head is untrained (random cells)")

        _load(self.grid_module, self.grid_model_path, self._save_path_r1_grid, 'grid_training_step_count')
        _load(self.grid_module2, self.grid_model_path_r2, self._save_path_r2_grid, 'grid_training_step_count2')

    def _grid_training_phase(self, robot_id: int) -> str:
        phase_cfg = self.grid_cfg.get('training', {}).get('phase', 'both')
        if phase_cfg == 'shaping':
            return 'shaping'
        if phase_cfg == 'rl':
            return 'rl'
        step = self.grid_training_step_count2 if robot_id == 2 else self.grid_training_step_count
        shaping_steps = int(self.grid_cfg.get('training', {}).get('shaping_steps', 2000))
        return 'shaping' if step < shaping_steps else 'rl'

    def _grid_epsilon(self, session_episode: int) -> float:
        return epsilon_for_episode(session_episode, self.grid_cfg)

    def _append_grid_step_log(self, robot_id: int, row: Dict) -> None:
        path = _REPO_ROOT / "data" / f"grid_train_steps_r{robot_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._locator_step_log_lock:
            write_header = not path.exists()
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=GRID_STEP_CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({k: row.get(k, '') for k in GRID_STEP_CSV_FIELDS})

    def _cell_from_teacher(
        self, teacher_x: float, teacher_z: float, center_x: float, center_z: float,
    ) -> Optional[int]:
        return world_to_cell(
            teacher_x, teacher_z, center_x, center_z,
            n=self._grid_n, window_m=self._grid_window_m,
        )

    @staticmethod
    def _grid_label_spawn_xz(training_data: Dict, obj_pos: List) -> Tuple[float, float]:
        """Spawn-first block center for grid labels (ignore post-grasp drift)."""
        label_x = float(training_data.get('spawn_x', 0.0))
        label_z = float(training_data.get('spawn_z', 0.0))
        if label_x == 0.0 and label_z == 0.0:
            label_x = float(obj_pos[0]) if len(obj_pos) >= 1 else 0.0
            label_z = float(obj_pos[1]) if len(obj_pos) >= 2 else 0.0
        return label_x, label_z

    @staticmethod
    def _grid_center_xy(training_data: Dict, pred_info: Dict) -> Tuple[float, float]:
        """Prefer grasp-time grid center from sim; fall back to fresh locator pass."""
        gx = training_data.get('grid_center_x')
        gz = training_data.get('grid_center_z')
        if gx is not None and gz is not None:
            return float(gx), float(gz)
        return float(pred_info['pred_obj_x']), float(pred_info['pred_obj_z'])

    def _pose_from_cell(
        self, cell: int, center_x: float, center_z: float, robot_id: int,
    ) -> List[float]:
        wx, wz = cell_to_world(
            cell, center_x, center_z, n=self._grid_n, window_m=self._grid_window_m,
        )
        return compute_grasp_pose_from_object_world(
            wx, DEFAULT_OBJECT_Y_M, wz, robot_id, add_jitter=False,
        )

    def _predict_local_grid_pose(self, full_message: Dict) -> Dict:
        """Locator centers grid; Q-head picks cell; geometry builds grasp."""
        robot_id    = int(full_message.get('robot_id', 1))
        is_sim      = full_message.get('source', 'real') == 'simulation'
        camera_data = full_message['data']
        grid_mod    = self.grid_module2 if robot_id == 2 else self.grid_module
        if grid_mod is None:
            return {'type': 'error', 'message': 'Local grid module not loaded'}

        pred = self._predict_object_xz(camera_data, robot_id, is_sim)
        rgbd = self.preprocess_rgbd_data(camera_data, is_simulation=is_sim, robot_id=robot_id)
        with torch.no_grad():
            cell_t = grid_mod.select_cell(rgbd, epsilon=0.0)
        cell = int(cell_t.cpu().numpy()[0])
        grasp_pose = self._pose_from_cell(
            cell, pred['pred_obj_x'], pred['pred_obj_z'], robot_id,
        )
        return {
            'type':           'grasp_prediction',
            'pose':           grasp_pose,
            'mode':           'exploit_grid',
            'grid_cell':      cell,
            'grid_center_x':  pred['pred_obj_x'],
            'grid_center_z':  pred['pred_obj_z'],
            'pred_obj_x':     pred['pred_obj_x'],
            'pred_obj_z':     pred['pred_obj_z'],
            'confidence':     1.0,
            'timestamp':      time.time(),
        }

    def _predict_grid_explore_pose(self, full_message: Dict) -> Dict:
        """RL training: epsilon-greedy cell selection."""
        robot_id    = int(full_message.get('robot_id', 1))
        is_sim      = full_message.get('source', 'real') == 'simulation'
        camera_data = full_message['data']
        grid_mod    = self.grid_module2 if robot_id == 2 else self.grid_module
        session_ep  = max(1, int(full_message.get('session_episode', 1)))
        eps         = self._grid_epsilon(session_ep)

        pred = self._predict_object_xz(camera_data, robot_id, is_sim)
        rgbd = self.preprocess_rgbd_data(camera_data, is_simulation=is_sim, robot_id=robot_id)
        with torch.no_grad():
            cell_t = grid_mod.select_cell(rgbd, epsilon=eps)
        cell = int(cell_t.cpu().numpy()[0])
        grasp_pose = self._pose_from_cell(
            cell, pred['pred_obj_x'], pred['pred_obj_z'], robot_id,
        )
        return {
            'type':           'grasp_prediction',
            'pose':           grasp_pose,
            'mode':           'grid_explore',
            'grid_cell':      cell,
            'grid_center_x':  pred['pred_obj_x'],
            'grid_center_z':  pred['pred_obj_z'],
            'pred_obj_x':     pred['pred_obj_x'],
            'pred_obj_z':     pred['pred_obj_z'],
            'epsilon':        eps,
            'confidence':     1.0,
            'timestamp':      time.time(),
        }

    def format_grid_shaping_batch(self, batch: List[Dict], robot_id: int) -> Dict[str, torch.Tensor]:
        states_list = []
        labels_list = []
        for exp in batch:
            is_sim = exp.get('source', 'real') == 'simulation'
            states_list.append(
                self.preprocess_rgbd_data(exp['state'], is_simulation=is_sim, robot_id=robot_id)
            )
            labels_list.append(int(exp['cell_label']))
        return {
            'states': torch.cat(states_list).to(self.device),
            'cell_labels': torch.tensor(labels_list, dtype=torch.long, device=self.device),
        }

    def format_grid_dqn_batch(self, batch: List[Dict], robot_id: int) -> Dict[str, torch.Tensor]:
        states_list, next_states_list = [], []
        actions_list, rewards_list, dones_list = [], [], []
        for exp in batch:
            is_sim = exp.get('source', 'real') == 'simulation'
            states_list.append(
                self.preprocess_rgbd_data(exp['state'], is_simulation=is_sim, robot_id=robot_id)
            )
            ns = exp.get('next_state', exp['state'])
            next_states_list.append(
                self.preprocess_rgbd_data(ns, is_simulation=is_sim, robot_id=robot_id)
            )
            actions_list.append(int(exp['cell_action']))
            rewards_list.append(float(exp['reward']))
            dones_list.append(float(exp.get('done', True)))
        return {
            'states': torch.cat(states_list).to(self.device),
            'next_states': torch.cat(next_states_list).to(self.device),
            'cell_actions': torch.tensor(actions_list, dtype=torch.long, device=self.device),
            'rewards': torch.tensor(rewards_list, dtype=torch.float32, device=self.device),
            'dones': torch.tensor(dones_list, dtype=torch.float32, device=self.device),
        }

    def _run_grid_training_step(self, robot_id: int = 1, dqn: bool = False):
        lock = self.train_lock if robot_id == 1 else self.train_lock2
        if lock.locked():
            return

        with lock:
            try:
                if robot_id == 2:
                    grid_mod   = self.grid_module2
                    weak_buf   = self.weak_buffer2
                    normal_buf = self.normal_buffer2
                    replay_buf = self.grid_replay2
                    step_attr  = 'grid_training_step_count2'
                    save_name  = self._save_path_r2_grid
                else:
                    grid_mod   = self.grid_module
                    weak_buf   = self.weak_buffer
                    normal_buf = self.normal_buffer
                    replay_buf = self.grid_replay
                    step_attr  = 'grid_training_step_count'
                    save_name  = self._save_path_r1_grid

                phase = self._grid_training_phase(robot_id)
                if dqn:
                    replay_min = int(self.grid_cfg.get('training', {}).get('replay_min', 500))
                    if len(replay_buf) < replay_min:
                        return
                    batch_raw = random.sample(list(replay_buf), min(self.batch_size, len(replay_buf)))
                    torch_batch = self.format_grid_dqn_batch(batch_raw, robot_id)
                    losses = grid_mod.update_dqn(torch_batch)
                    tgt_every = int(self.grid_cfg.get('training', {}).get('target_update_steps', 100))
                    if grid_mod._dqn_step % tgt_every == 0:
                        grid_mod.sync_target()
                    loss_key = 'dqn'
                    acc = 0.0
                else:
                    batch_raw = self._sample_mixed_batch(
                        weak_buf, normal_buf, self.batch_size, self._weak_ratio,
                    )
                    if not batch_raw:
                        return
                    torch_batch = self.format_grid_shaping_batch(batch_raw, robot_id)
                    losses = grid_mod.update_shaping(torch_batch)
                    loss_key = 'ce'
                    acc = losses.get('acc', 0.0)

                step = getattr(self, step_attr) + 1
                setattr(self, step_attr, step)

                ckpt_every = self._checkpoint_every
                checkpoint_saved = int(step % ckpt_every == 0)

                batch_weak = sum(1 for s in (batch_raw if not dqn else []) if s.get('demo_bucket') == 'weak')
                self._append_grid_step_log(robot_id, {
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'robot_id': robot_id,
                    'grid_step': step,
                    'phase': 'rl' if dqn else phase,
                    'loss': losses.get(loss_key, losses['total']),
                    'acc': acc,
                    'grad_norm': losses.get('grad_norm', 0.0),
                    'buffer_weak': len(weak_buf),
                    'buffer_normal': len(normal_buf),
                    'replay_len': len(replay_buf),
                    'checkpoint_saved': checkpoint_saved,
                })

                if step % 5 == 0:
                    tag = 'GRID-DQN' if dqn else 'GRID'
                    print(
                        f"🔥 R{robot_id} {tag} Step {step:4d} | "
                        f"Loss: {losses['total']:.4f} | Acc: {acc:.3f} | "
                        f"GradNorm: {losses.get('grad_norm', 0):.3f} | "
                        f"weak:{len(weak_buf)} normal:{len(normal_buf)} replay:{len(replay_buf)}"
                    )

                if checkpoint_saved:
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    save_dir = os.path.join(base_dir, "models")
                    os.makedirs(save_dir, exist_ok=True)
                    full_path = os.path.join(save_dir, save_name)
                    grid_mod.save_model(full_path, training_step=step)
                    print(f"💾 R{robot_id} SAVED GRID MODEL TO: {full_path}")

            except Exception as e:
                print(f"❌ CRITICAL GRID TRAINING ERROR (R{robot_id}): {e}")
                import traceback
                traceback.print_exc()

    def _handle_grid_training_data(self, training_data: Dict, source: str, robot_id: int) -> Dict:
        """Teacher shaping labels and RL replay for local grid."""
        try:
            collect_mode = training_data.get('mode', 'grid_collect')
            is_sim = source == 'simulation'
            obj_pos = training_data.get('object_pos', [0.0, 0.0])
            label_x, label_z = self._grid_label_spawn_xz(training_data, obj_pos)
            label_xz_for_loc = [label_x, label_z] if (label_x != 0.0 or label_z != 0.0) else None
            pred_info = self._predict_object_xz(
                training_data['state'], robot_id, is_sim,
                label_xz=label_xz_for_loc,
            )
            center_x, center_z = self._grid_center_xy(training_data, pred_info)

            ack: Dict = {
                'type': 'training_ack',
                'grid': True,
                'pred_obj_x': center_x,
                'pred_obj_z': center_z,
                'grid_center_x': center_x,
                'grid_center_z': center_z,
                'grid_phase': self._grid_training_phase(robot_id),
            }
            if label_xz_for_loc is not None:
                ack['locator_err_m'] = math.hypot(label_x - center_x, label_z - center_z)
            elif 'locator_err_m' in pred_info:
                ack['locator_err_m'] = pred_info['locator_err_m']

            if collect_mode == 'grid_collect':
                # Label cell from spawn center, not grasp aim XY or post-grasp block drift.
                cell = self._cell_from_teacher(label_x, label_z, center_x, center_z)
                if cell is None:
                    ack['out_of_window'] = True
                    ack['skipped'] = True
                    return ack

                sample = {
                    'state': training_data['state'],
                    'cell_label': cell,
                    'source': source,
                    'robot_id': robot_id,
                }
                bucket = self._resolve_demo_bucket(training_data, robot_id)
                sample['demo_bucket'] = bucket
                if robot_id == 2:
                    target = self.weak_buffer2 if bucket == 'weak' else self.normal_buffer2
                else:
                    target = self.weak_buffer if bucket == 'weak' else self.normal_buffer

                phase = self._grid_training_phase(robot_id)
                if phase == 'shaping':
                    target.append(sample)
                    if self._fine_tune_buffers_ready(robot_id):
                        threading.Thread(
                            target=self._run_grid_training_step, args=(robot_id, False), daemon=True,
                        ).start()

                ack.update({
                    'teacher_cell': cell,
                    'grid_cell': cell,
                    'out_of_window': False,
                    'buffer_len': len(self.weak_buffer if robot_id == 1 else self.weak_buffer2)
                        + len(self.normal_buffer if robot_id == 1 else self.normal_buffer2),
                    'grid_step': self.grid_training_step_count2 if robot_id == 2 else self.grid_training_step_count,
                })
                return ack

            if collect_mode == 'grid_rl':
                cell_action = int(training_data.get('cell_action', training_data.get('grid_cell', 0)))

                label_cell = self._cell_from_teacher(label_x, label_z, center_x, center_z)
                replay_used = True
                replay_skip_reason = ''
                if label_cell is None:
                    replay_used = False
                    replay_skip_reason = 'out_of_window'

                sample = {
                    'state': training_data['state'],
                    'next_state': training_data.get('next_state', training_data['state']),
                    'cell_action': cell_action,
                    'reward': float(training_data.get('reward', 0.0)),
                    'done': bool(training_data.get('done', True)),
                    'source': source,
                    'robot_id': robot_id,
                }
                replay = self.grid_replay2 if robot_id == 2 else self.grid_replay
                if replay_used:
                    replay.append(sample)
                replay_min = int(self.grid_cfg.get('training', {}).get('replay_min', 500))
                phase = self._grid_training_phase(robot_id)
                if replay_used and phase == 'rl' and len(replay) >= replay_min:
                    threading.Thread(
                        target=self._run_grid_training_step, args=(robot_id, True), daemon=True,
                    ).start()
                ack.update({
                    'grid_cell': cell_action,
                    'teacher_cell': label_cell,
                    'replay_len': len(replay),
                    'replay_used': replay_used,
                    'replay_skip_reason': replay_skip_reason,
                    'reward': sample['reward'],
                })
                return ack

            return {'type': 'error', 'message': f'Unknown grid collect mode: {collect_mode}'}
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    def _init_board_modules(self) -> None:
        tr = self.board_cfg.get('training', {})
        lr = float(tr.get('learning_rate', 1e-3))
        wd = float(tr.get('weight_decay', 8e-5))
        gamma = float(tr.get('gamma', 0.99))
        enc_ph = float(tr.get('encoder_lr_factor', 0.1))
        n_cells = self._board_n_cells

        def _build_one(robot_id: int) -> FullBoardDQNModule:
            mask_np = invalid_cell_mask(robot_id, self._board_n, self.board_cfg)
            mask_t = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
            return create_full_board_dqn_module(
                n_cells=n_cells,
                learning_rate=lr,
                weight_decay=wd,
                encoder_lr_factor=enc_ph,
                gamma=gamma,
                invalid_mask=mask_t,
            ).to(self.device)

        self.board_module = _build_one(1)
        self.board_module2 = _build_one(2)
        self._load_board_weights()

    def _load_board_weights(self) -> None:
        def _load(mod: Optional[FullBoardDQNModule], path_override: Optional[str],
                  default_name: str, step_attr: str):
            if mod is None:
                return
            path = self._resolve_checkpoint_path(path_override, default_name)
            print(f"[BOARD] Looking for board DQN at: {path.resolve()}")
            if path.exists():
                step = mod.load_model(str(path))
                setattr(self, step_attr, max(getattr(self, step_attr), step))
                print(f"[BOARD] Loaded board DQN: {path.name}  (step {step})")
            elif self.board_dqn_train:
                print("   -> No board checkpoint yet — starting from pretrained MobileNet")
            elif self.board_dqn:
                print(f"[BOARD] WARNING: No board DQN checkpoint at {path.name}")

        _load(self.board_module, self.board_model_path, self._save_path_r1_board,
              'board_training_step_count')
        _load(self.board_module2, self.board_model_path_r2, self._save_path_r2_board,
              'board_training_step_count2')

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

    def _init_board_locator_modules(self) -> None:
        tr = self.board_locator_cfg.get('training', {})
        lr = float(tr.get('learning_rate', 1e-3))
        wd = float(tr.get('weight_decay', 8e-5))
        enc_ph = float(tr.get('encoder_lr_factor', 0.1))
        n_cells = self._board_n_cells
        grid_n = self._board_n

        def _build_one(robot_id: int) -> BoardLocatorModule:
            mask_np = invalid_cell_mask(robot_id, self._board_n, self.board_cfg)
            mask_t = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
            return create_board_locator_module(
                n_cells=n_cells,
                grid_n=grid_n,
                learning_rate=lr,
                weight_decay=wd,
                encoder_lr_factor=enc_ph,
                invalid_mask=mask_t,
            ).to(self.device)

        self.board_loc_module = _build_one(1)
        self.board_loc_module2 = _build_one(2)
        self._load_board_locator_weights()

    def _load_board_locator_weights(self) -> None:
        def _load(mod: Optional[BoardLocatorModule], path_override: Optional[str],
                  default_name: str, step_attr: str):
            if mod is None:
                return
            path = self._resolve_checkpoint_path(path_override, default_name)
            print(f"[BOARD-LOC] Looking for board locator at: {path.resolve()}")
            if path.exists():
                step = mod.load_model(str(path))
                setattr(self, step_attr, max(getattr(self, step_attr), step))
                print(f"[BOARD-LOC] Loaded board locator: {path.name}  (step {step})")
            elif self.board_locator_train:
                ckpt_cfg = self.board_locator_cfg.get('checkpoints', {})
                r_num = 1 if 'r1' in default_name.lower() else 2
                init_key = f'init_from_board_dqn_r{r_num}'
                init_name = Path(ckpt_cfg.get(init_key, '')).name
                if init_name:
                    dqn_path = self._resolve_checkpoint_path(None, init_name)
                    if dqn_path.exists():
                        tmp = create_full_board_dqn_module(
                            n_cells=self._board_n_cells,
                        ).to(self.device)
                        tmp.load_model(str(dqn_path))
                        mod.load_encoder_from_board_dqn(tmp)
                        print(f"[BOARD-LOC] Warm-started encoders from {dqn_path.name}")
                print("   -> No board locator checkpoint yet — starting from pretrained MobileNet")
            elif self.board_locator:
                print(f"[BOARD-LOC] WARNING: No board locator checkpoint at {path.name}")

        _load(self.board_loc_module, self.board_locator_model_path,
              self._save_path_r1_board_locator, 'board_locator_training_step_count')
        _load(self.board_loc_module2, self.board_locator_model_path_r2,
              self._save_path_r2_board_locator, 'board_locator_training_step_count2')

    def _append_board_locator_step_log(self, robot_id: int, row: Dict) -> None:
        path = _REPO_ROOT / "data" / f"board_locator_train_steps_r{robot_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._locator_step_log_lock:
            write_header = not path.exists()
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=BOARD_LOCATOR_STEP_CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({k: row.get(k, '') for k in BOARD_LOCATOR_STEP_CSV_FIELDS})

    def _board_training_phase(self, robot_id: int) -> str:
        """Paper session 2: sim uses oracle shaping only (no sparse-RL replay)."""
        return 'shaping'

    def _board_epsilon(self, session_episode: int) -> float:
        return board_epsilon_for_episode(session_episode, self.board_cfg)

    def _append_board_step_log(self, robot_id: int, row: Dict) -> None:
        path = _REPO_ROOT / "data" / f"board_dqn_train_steps_r{robot_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._locator_step_log_lock:
            write_header = not path.exists()
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=BOARD_STEP_CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({k: row.get(k, '') for k in BOARD_STEP_CSV_FIELDS})

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

    def _preprocess_board_pair(self, camera_data: Dict, is_simulation: bool,
                               robot_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

        DEPTH_MIN, DEPTH_MAX = 0.50, 1.00
        depth_w = np.clip(depth_w, DEPTH_MIN, DEPTH_MAX)
        depth_w = (depth_w - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)

        rgb_t = torch.from_numpy(rgb_w.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        depth_t = torch.from_numpy(depth_w.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb_t = (rgb_t - mean) / std
        depth_t = (depth_t - 0.5) / 0.5
        return rgb_t.to(self.device), depth_t.to(self.device)

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

    def _save_board_crop_debug(
        self,
        rgb_crop: np.ndarray,
        robot_id: int,
        label_x: Optional[float],
        label_z: Optional[float],
    ) -> None:
        try:
            debug_dir = Path(__file__).resolve().parent.parent / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            vis = cv2.cvtColor(rgb_crop.copy(), cv2.COLOR_RGB2BGR)
            h, w = vis.shape[:2]

            yaml_extra = {}
            warp_yaml = Path(__file__).resolve().parent.parent / "config" / f"board_warp_r{robot_id}.yaml"
            if warp_yaml.exists():
                with open(warp_yaml, encoding="utf-8") as f:
                    yaml_extra = yaml.safe_load(f) or {}
            corners = yaml_extra.get("image_corners") or []
            if len(corners) == 4:
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (255, 255, 0), 2)
                for i, (px, py) in enumerate(pts):
                    cv2.circle(vis, (int(px), int(py)), 4, (255, 255, 0), -1)
                    cv2.putText(
                        vis, str(i), (int(px) + 4, int(py) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA,
                    )

            if label_x is not None and label_z is not None:
                ux, vy = world_xz_to_warp_pixel(
                    label_x, label_z, robot_id, out_size=224, cfg=self.board_cfg,
                    apply_post_flip=True,
                )
                cx, cy = warp_pixel_to_crop_pixel(
                    ux, vy, robot_id, w, h, out_size=224, cfg=self.board_cfg,
                    apply_post_flip=True,
                )
                cv2.drawMarker(
                    vis, (int(round(cx)), int(round(cy))), (0, 0, 255),
                    cv2.MARKER_CROSS, 14, 2,
                )
                cv2.putText(
                    vis, "RED=crop point for sim XZ", (4, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA,
                )

            out = debug_dir / f"board_crop_r{robot_id}_latest.jpg"
            cv2.imwrite(str(out), vis)
        except Exception as e:
            print(f"[BOARD DEBUG R{robot_id}] crop debug save failed: {e}", flush=True)


    def _warp_rgb_for_debug(self, camera_data: Dict, robot_id: int) -> np.ndarray:
        img = self.decode_b64_image(camera_data)
        rgb = cv2.cvtColor(img['rgb'], cv2.COLOR_BGR2RGB)
        depth = img['depth']
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        rgb_c, _ = self._crop_rgb_depth(rgb, depth, robot_id)
        rgb_w = warp_rgb_to_board(rgb_c, robot_id, out_size=224, cfg=self.board_cfg)
        return rgb_w

    def _board_q_diag_from_tensors(
        self,
        rgb_t: torch.Tensor,
        depth_t: torch.Tensor,
        board_mod: FullBoardDQNModule,
        teacher_cell: int,
        argmax_cell: int,
        robot_id: int,
        label_x: Optional[float] = None,
        label_z: Optional[float] = None,
    ) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        try:
            with torch.no_grad():
                q_t = board_mod.q_map(rgb_t, depth_t)
            q_np = q_t.squeeze().cpu().numpy()
            diag = compute_board_q_diagnostics(
                q_np, teacher_cell, argmax_cell, robot_id, self.board_cfg,
                label_x=label_x, label_z=label_z,
            )
            return diag, q_np
        except Exception as exc:
            print(f"[BOARD Q-DIAG R{robot_id}] failed: {exc}", flush=True)
            return None, None

    def _save_board_debug_overlay(
        self,
        rgb_board: np.ndarray,
        robot_id: int,
        cnn_cell: Optional[int] = None,
        teacher_cell: Optional[int] = None,
        meta: Optional[Dict] = None,
        q_map: Optional[np.ndarray] = None,
        q_diag: Optional[Dict] = None,
        near_radius_cells: int = 2,
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

            if q_map is not None:
                heat_rgb = blend_q_heatmap_overlay(rgb_board, q_map, alpha=0.5)
                heat_bgr = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)
                if teacher_cell is not None:
                    tx, ty = self._board_cell_pixel(teacher_cell, h, w, robot_id)
                    cv2.drawMarker(heat_bgr, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 16, 3)
                    cv2.circle(heat_bgr, (tx, ty), 12, (0, 0, 255), 2)
                    cell_px = max(w, h) / float(self._board_n)
                    near_r = int(round(cell_px * near_radius_cells))
                    cv2.circle(heat_bgr, (tx, ty), near_r, (0, 0, 255), 1, cv2.LINE_AA)
                if cnn_cell is not None:
                    gx, gy = self._board_cell_pixel(cnn_cell, h, w, robot_id)
                    cv2.drawMarker(heat_bgr, (gx, gy), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.putText(
                    heat_bgr, "JET=Q heat  RED=teacher  GREEN=argmax", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.imwrite(str(debug_dir / f"board_q_heatmap_r{robot_id}_latest.jpg"), heat_bgr)

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

    def _predict_board_pose(self, full_message: Dict, explore: bool = False) -> Dict:
        robot_id = int(full_message.get('robot_id', 1))
        is_sim = full_message.get('source', 'real') == 'simulation'
        camera_data = full_message['data']
        board_mod = self.board_module2 if robot_id == 2 else self.board_module
        if board_mod is None:
            return {'type': 'error', 'message': 'Board DQN module not loaded'}

        session_ep = max(1, int(full_message.get('session_episode', 1)))
        eps = self._board_epsilon(session_ep) if explore else 0.0
        rgb_t, depth_t = self._preprocess_board_pair(camera_data, is_sim, robot_id)
        with torch.no_grad():
            q_map_t = board_mod.q_map(rgb_t, depth_t)
            cell_t = board_mod.select_cell(rgb_t, depth_t, epsilon=eps)
        dqn_cell = int(cell_t.cpu().numpy()[0])
        q_map_np = q_map_t.squeeze().cpu().numpy()

        use_loc = self.board_locator or bool(full_message.get('use_board_locator', False))
        use_yolo_fuse = self.yolo_fuse or bool(full_message.get('use_yolo_fuse', False))
        loc_mod = self.board_loc_module2 if robot_id == 2 else self.board_loc_module
        detector_cell = None
        detector_conf = None
        fusion_used = False
        cell = dqn_cell
        yolo_bbox_px = None
        yolo_conf = None
        cells_in_bbox_count = None
        dqn_outside_bbox = None
        dqn_dist_to_bbox_px = None
        dqn_dist_to_bbox_cells = None

        img = self.decode_b64_image(camera_data)
        rgb = cv2.cvtColor(img['rgb'], cv2.COLOR_BGR2RGB)
        rgb_c, _ = self._crop_rgb_depth(rgb, img['depth'].astype(np.float32), robot_id)
        rgb_w = warp_rgb_to_board(rgb_c, robot_id, out_size=224, cfg=self.board_cfg)
        h, w = rgb_w.shape[:2]
        inv_np = invalid_cell_mask(robot_id, self._board_n, self.board_cfg)

        if use_yolo_fuse and self.yolo_client is not None:
            try:
                bbox, _raw = self.yolo_client.detect_bbox(rgb_w)
            except Exception as exc:
                print(f"[YOLO FUSE R{robot_id}] detect failed: {exc} — falling back to DQN", flush=True)
                bbox = None
            if bbox is not None:
                yolo_bbox_px = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                yolo_conf = float(bbox.confidence)
                fused, cells_in_bbox_count = best_q_cell_in_bbox(
                    q_map_np, bbox, self._board_n, inv_np, h, w, robot_id,
                    board_cfg=self.board_cfg,
                )
                outside, dist_px = dist_cell_to_bbox_px(
                    dqn_cell, bbox, self._board_n, h, w, robot_id,
                )
                _, dist_cells = dist_cell_to_bbox_cells(
                    dqn_cell, bbox, self._board_n, inv_np, h, w, robot_id,
                    board_cfg=self.board_cfg,
                )
                dqn_outside_bbox = bool(outside)
                dqn_dist_to_bbox_px = float(dist_px)
                dqn_dist_to_bbox_cells = dist_cells
                if fused is not None:
                    cell = int(fused)
                    fusion_used = True
            # else: keep cell = dqn_cell (fallback)
        elif use_loc and loc_mod is not None:
            detector_cell, detector_conf, _ = loc_mod.predict_cell(rgb_t, depth_t)
            fusion_cfg = self.board_locator_cfg.get('fusion', {})
            if fusion_cfg.get('enabled', True):
                cell, fusion_used = fuse_q_with_detector(
                    q_map_np,
                    detector_cell,
                    dqn_cell,
                    inv_np,
                    grid_n=self._board_n,
                    radius_cells=int(fusion_cfg.get('radius_cells', 3)),
                    min_confidence=float(fusion_cfg.get('min_mask_confidence', 0.15)),
                    detector_confidence=float(detector_conf),
                )

        q_diag = None
        n = self._board_n
        row, col = divmod(cell, n)
        wx, wz = board_cell_to_world(cell, robot_id, n=n, cfg=self.board_cfg)
        grasp_pose = self._pose_from_board_cell(cell, robot_id)

        teacher_cell = None
        label_x, label_z = self._board_label_xz(full_message)
        if is_sim and label_x is not None and label_z is not None:
            self._save_board_crop_debug(rgb_c, robot_id, label_x, label_z)
            teacher_cell = world_to_board_cell(
                label_x, label_z, robot_id,
                n=self._board_n, cfg=self.board_cfg,
            )
            if teacher_cell is not None:
                q_diag, _ = self._board_q_diag_from_tensors(
                    rgb_t, depth_t, board_mod, teacher_cell, cell, robot_id,
                    label_x=label_x, label_z=label_z,
                )
        elif is_sim:
            print(
                f"[BOARD DEBUG R{robot_id}] no spawn in camera message — "
                f"red cross will appear after episode collect",
                flush=True,
            )
        self._save_board_debug_overlay(
            rgb_w, robot_id,
            cnn_cell=cell,
            teacher_cell=teacher_cell,
            yolo_bbox=yolo_bbox_px,
            meta={
                'label_x': label_x,
                'label_z': label_z,
                'spawn_x': full_message.get('spawn_x'),
                'spawn_z': full_message.get('spawn_z'),
                'object_x': full_message.get('object_x'),
                'object_z': full_message.get('object_z'),
                'session_episode': session_ep,
                'epsilon': eps,
                'dqn_cell': dqn_cell,
                'detector_cell': detector_cell,
                'detector_conf': detector_conf,
                'fused_cell': cell,
                'fusion_used': fusion_used,
                'yolo_fuse': use_yolo_fuse,
                'yolo_conf': yolo_conf,
                'yolo_bbox_px': list(yolo_bbox_px) if yolo_bbox_px is not None else None,
                'cells_in_bbox_count': cells_in_bbox_count,
                'dqn_outside_bbox': dqn_outside_bbox,
                'dqn_dist_to_bbox_px': dqn_dist_to_bbox_px,
                'dqn_dist_to_bbox_cells': dqn_dist_to_bbox_cells,
            },
            q_map=q_map_np,
            q_diag=q_diag,
        )
        if use_yolo_fuse:
            dist_line = ""
            if dqn_outside_bbox:
                dist_line = (
                    f" dqn_outside dist_px={dqn_dist_to_bbox_px:.1f}"
                    f" dist_cells={dqn_dist_to_bbox_cells}"
                )
            elif dqn_outside_bbox is False:
                dist_line = " dqn_in_bbox"
            conf_s = f"{yolo_conf:.3f}" if yolo_conf is not None else "none"
            print(
                f"[YOLO FUSE R{robot_id}] yolo_conf={conf_s} fused={cell} "
                f"dqn={dqn_cell} fusion={fusion_used}{dist_line} | "
                f"see debug/board_warp_r{robot_id}_latest.jpg",
                flush=True,
            )
        elif teacher_cell is not None:
            hit = int(cell) == int(teacher_cell)
            q_line = ""
            if q_diag:
                q_line = (
                    f" rank={int(q_diag['teacher_rank'])} "
                    f"mass_near={q_diag['mass_near_teacher']:.3f} "
                    f"topk_err={q_diag['topk_centroid_err_m']*100:.1f}cm"
                )
            loc_line = ""
            if detector_cell is not None:
                loc_hit = int(detector_cell) == int(teacher_cell)
                loc_line = f" det={detector_cell}({'HIT' if loc_hit else 'miss'})"
            print(
                f"[BOARD DEBUG R{robot_id}] teacher={teacher_cell} cnn={cell} "
                f"dqn={dqn_cell}{loc_line} fusion={fusion_used} "
                f"{'HIT' if hit else 'miss'}{q_line} | "
                f"see debug/board_warp_r{robot_id}_latest.jpg",
                flush=True,
            )
        elif detector_cell is not None:
            print(
                f"[BOARD DEBUG R{robot_id}] dqn={dqn_cell} det={detector_cell} "
                f"conf={detector_conf:.3f} fused={cell} fusion={fusion_used}",
                flush=True,
            )

        mode = 'board_explore' if explore else 'exploit_board'
        response = {
            'type': 'grasp_prediction',
            'pose': grasp_pose,
            'mode': mode,
            'board_cell': cell,
            'board_u': col,
            'board_v': row,
            'target_x_m': wx,
            'target_z_m': wz,
            'teacher_cell': teacher_cell,
            'dqn_cell': dqn_cell,
            'detector_cell': detector_cell,
            'detector_conf': detector_conf,
            'fusion_used': fusion_used,
            'yolo_fuse': use_yolo_fuse,
            'yolo_conf': yolo_conf,
            'yolo_bbox_px': list(yolo_bbox_px) if yolo_bbox_px is not None else None,
            'cells_in_bbox_count': cells_in_bbox_count,
            'dqn_outside_bbox': dqn_outside_bbox,
            'dqn_dist_to_bbox_px': dqn_dist_to_bbox_px,
            'dqn_dist_to_bbox_cells': dqn_dist_to_bbox_cells,
            'epsilon': eps,
            'confidence': 1.0,
            'timestamp': time.time(),
        }
        if q_diag:
            response.update(q_diag)
        return response

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

    def format_board_shaping_batch(self, batch: List[Dict], robot_id: int) -> Dict[str, torch.Tensor]:
        rgb_list, depth_list, q_target_list = [], [], []
        for exp in batch:
            is_sim = exp.get('source', 'real') == 'simulation'
            rgb_t, depth_t = self._preprocess_board_pair(exp['state'], is_sim, robot_id)
            rgb_list.append(rgb_t)
            depth_list.append(depth_t)
            sx = float(exp.get('spawn_x', 0.0))
            sz = float(exp.get('spawn_z', 0.0))
            q_tgt = board_shaping_q_targets(sx, sz, robot_id, n=self._board_n, cfg=self.board_cfg)
            q_target_list.append(torch.from_numpy(q_tgt))
        return {
            'rgb': torch.cat(rgb_list).to(self.device),
            'depth': torch.cat(depth_list).to(self.device),
            'q_targets': torch.stack(q_target_list).to(self.device),
        }

    def _run_board_training_step(self, robot_id: int = 1):
        lock = self.train_lock if robot_id == 1 else self.train_lock2
        if lock.locked():
            return
        with lock:
            try:
                if robot_id == 2:
                    board_mod = self.board_module2
                    weak_buf = self.weak_buffer2
                    normal_buf = self.normal_buffer2
                    step_attr = 'board_training_step_count2'
                    save_name = self._save_path_r2_board
                else:
                    board_mod = self.board_module
                    weak_buf = self.weak_buffer
                    normal_buf = self.normal_buffer
                    step_attr = 'board_training_step_count'
                    save_name = self._save_path_r1_board

                batch_raw = self._sample_mixed_batch(
                    weak_buf, normal_buf, self.batch_size, self._weak_ratio,
                )
                if not batch_raw:
                    return
                torch_batch = self.format_board_shaping_batch(batch_raw, robot_id)
                losses = board_mod.update_shaping(torch_batch)
                shaping_mse = losses.get('shaping_mse', 0.0)

                step = getattr(self, step_attr) + 1
                setattr(self, step_attr, step)
                ckpt_every = self._checkpoint_every
                checkpoint_saved = int(step % ckpt_every == 0)
                self._append_board_step_log(robot_id, {
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'robot_id': robot_id,
                    'board_step': step,
                    'phase': 'shaping',
                    'loss': losses.get('shaping_mse', losses['total']),
                    'shaping_mse': shaping_mse,
                    'grad_norm': losses.get('grad_norm', 0.0),
                    'buffer_weak': len(weak_buf),
                    'buffer_normal': len(normal_buf),
                    'replay_len': 0,
                    'checkpoint_saved': checkpoint_saved,
                })
                print(
                    f"[BOARD TRAIN R{robot_id}] step {step:4d} | "
                    f"MSE: {shaping_mse:.4f} | "
                    f"buf weak:{len(weak_buf)} normal:{len(normal_buf)}",
                    flush=True,
                )
                if checkpoint_saved:
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    save_dir = os.path.join(base_dir, "models")
                    os.makedirs(save_dir, exist_ok=True)
                    full_path = os.path.join(save_dir, save_name)
                    board_mod.save_model(full_path, training_step=step)
                    print(f"[BOARD TRAIN R{robot_id}] SAVED checkpoint: {full_path}", flush=True)
            except Exception as e:
                print(f"[BOARD TRAIN R{robot_id}] CRITICAL TRAINING ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()

    def format_board_locator_batch(self, batch: List[Dict], robot_id: int) -> Dict[str, torch.Tensor]:
        rgb_list, depth_list, mask_list = [], [], []
        for exp in batch:
            is_sim = exp.get('source', 'real') == 'simulation'
            rgb_t, depth_t = self._preprocess_board_pair(exp['state'], is_sim, robot_id)
            rgb_list.append(rgb_t)
            depth_list.append(depth_t)
            sx = float(exp.get('spawn_x', 0.0))
            sz = float(exp.get('spawn_z', 0.0))
            _, mask_grid = make_block_mask_warp(
                sx, sz, robot_id,
                grid_n=self._board_n,
                board_cfg=self.board_cfg,
                locator_cfg=self.board_locator_cfg,
            )
            mask_list.append(torch.from_numpy(mask_grid).unsqueeze(0).unsqueeze(0))
        return {
            'rgb': torch.cat(rgb_list).to(self.device),
            'depth': torch.cat(depth_list).to(self.device),
            'masks': torch.cat(mask_list).to(self.device),
        }

    def _run_board_locator_training_step(self, robot_id: int = 1):
        lock = self.train_lock if robot_id == 1 else self.train_lock2
        if lock.locked():
            return
        with lock:
            try:
                if robot_id == 2:
                    loc_mod = self.board_loc_module2
                    weak_buf = self.weak_buffer2
                    normal_buf = self.normal_buffer2
                    step_attr = 'board_locator_training_step_count2'
                    save_name = self._save_path_r2_board_locator
                else:
                    loc_mod = self.board_loc_module
                    weak_buf = self.weak_buffer
                    normal_buf = self.normal_buffer
                    step_attr = 'board_locator_training_step_count'
                    save_name = self._save_path_r1_board_locator

                batch_raw = self._sample_mixed_batch(
                    weak_buf, normal_buf, self.batch_size, self._weak_ratio,
                )
                if not batch_raw:
                    return
                torch_batch = self.format_board_locator_batch(batch_raw, robot_id)
                losses = loc_mod.update(torch_batch)
                step = getattr(self, step_attr) + 1
                setattr(self, step_attr, step)
                ckpt_every = self._checkpoint_every
                checkpoint_saved = int(step % ckpt_every == 0)
                self._append_board_locator_step_log(robot_id, {
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'robot_id': robot_id,
                    'locator_step': step,
                    'seg_bce': losses.get('seg_bce', losses['total']),
                    'seg_dice': losses.get('seg_dice', 0.0),
                    'grad_norm': losses.get('grad_norm', 0.0),
                    'buffer_weak': len(weak_buf),
                    'buffer_normal': len(normal_buf),
                    'checkpoint_saved': checkpoint_saved,
                })
                print(
                    f"[BOARD-LOC TRAIN R{robot_id}] step {step:4d} | "
                    f"BCE: {losses.get('seg_bce', losses['total']):.4f} | "
                    f"Dice: {losses.get('seg_dice', 0.0):.3f} | "
                    f"buf weak:{len(weak_buf)} normal:{len(normal_buf)}",
                    flush=True,
                )
                debug_every = int(self.board_locator_cfg.get('debug', {}).get('save_every_steps', 50))
                if debug_every > 0 and step % debug_every == 0 and batch_raw:
                    try:
                        exp0 = batch_raw[0]
                        self._save_board_seg_train_debug(
                            exp0['state'], robot_id,
                            float(exp0.get('spawn_x', 0.0)),
                            float(exp0.get('spawn_z', 0.0)),
                            loc_mod=loc_mod,
                            step=step,
                        )
                    except Exception as exc:
                        print(f"[BOARD-LOC DEBUG R{robot_id}] save failed: {exc}", flush=True)
                if checkpoint_saved:
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    save_dir = os.path.join(base_dir, "models")
                    os.makedirs(save_dir, exist_ok=True)
                    full_path = os.path.join(save_dir, save_name)
                    loc_mod.save_model(full_path, training_step=step)
                    print(f"[BOARD-LOC TRAIN R{robot_id}] SAVED checkpoint: {full_path}", flush=True)
            except Exception as e:
                print(f"[BOARD-LOC TRAIN R{robot_id}] CRITICAL TRAINING ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()

    def _save_board_seg_train_debug(
        self,
        camera_data: Dict,
        robot_id: int,
        label_x: float,
        label_z: float,
        loc_mod: Optional[BoardLocatorModule] = None,
        step: Optional[int] = None,
    ) -> None:
        debug_dir = Path(__file__).resolve().parent.parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        rgb_w = self._warp_rgb_for_debug(camera_data, robot_id)
        mask_warp, mask_grid = make_block_mask_warp(
            label_x, label_z, robot_id,
            grid_n=self._board_n,
            board_cfg=self.board_cfg,
            locator_cfg=self.board_locator_cfg,
        )
        _, teacher_conf = mask_grid_to_cell(
            mask_grid, invalid_cell_mask(robot_id, self._board_n, self.board_cfg),
        )
        # Canonical teacher cell (same convention as board_warp overlay).
        teacher_cell = world_to_board_cell(
            label_x, label_z, robot_id, n=self._board_n, cfg=self.board_cfg,
        )
        vis = blend_mask_overlay(rgb_w, mask_warp, color_rgb=(0, 255, 0), alpha=0.45)
        if teacher_cell is not None:
            # Red cross at the exact block center (matches green mask center).
            tx, ty = world_xz_to_warp_pixel(
                label_x, label_z, robot_id,
                out_size=vis.shape[1], cfg=self.board_cfg, apply_post_flip=True,
            )
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.drawMarker(vis_bgr, (int(tx), int(ty)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            pred_cell = None
            pred_conf = None
            pred_mask = None
            if loc_mod is not None:
                rgb_t, depth_t = self._preprocess_board_pair(camera_data, True, robot_id)
                pred_cell, pred_conf, pred_mask = loc_mod.predict_cell(rgb_t, depth_t)
                pred_x, pred_y = self._board_cell_pixel(pred_cell, vis.shape[0], vis.shape[1], robot_id)
                cv2.drawMarker(vis_bgr, (pred_x, pred_y), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
                if pred_mask is not None:
                    pm = cv2.resize(pred_mask, (vis.shape[1], vis.shape[0]))
                    contours, _ = cv2.findContours(
                        (pm > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(vis_bgr, contours, -1, (255, 255, 0), 1)
            stem = f"board_seg_train_r{robot_id}_latest"
            if step is not None:
                cv2.imwrite(str(debug_dir / f"board_seg_train_r{robot_id}_step{step}.jpg"), vis_bgr)
            cv2.imwrite(str(debug_dir / f"{stem}.jpg"), vis_bgr)
            sidecar = {
                'robot_id': robot_id,
                'label_x': label_x,
                'label_z': label_z,
                'teacher_cell': teacher_cell,
                'teacher_mask_conf': teacher_conf,
                'pred_cell': pred_cell,
                'pred_conf': pred_conf,
                'step': step,
            }
            with open(debug_dir / f"{stem}.json", 'w', encoding='utf-8') as f:
                json.dump(sidecar, f, indent=2)

    def _save_board_locator_raw_sample(
        self, camera_data: Dict, robot_id: int, label_x: float, label_z: float,
    ) -> Dict:
        """Save clean warped RGB for manual segmentation (no auto-label)."""
        ds_dir = _REPO_ROOT / "data" / "board_locator_dataset" / f"r{robot_id}"
        img_dir = ds_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        count_attr = f'_board_locator_raw_count_r{robot_id}'
        idx = getattr(self, count_attr, 0)
        setattr(self, count_attr, idx + 1)
        stem = f"r{robot_id}_{idx:05d}"

        rgb_w = self._warp_rgb_for_debug(camera_data, robot_id)
        warp_path = img_dir / f"{stem}_warp.png"
        cv2.imwrite(str(warp_path), cv2.cvtColor(rgb_w, cv2.COLOR_RGB2BGR))

        meta = {
            'stem': stem,
            'robot_id': robot_id,
            'warp_image': warp_path.name,
            'warp_size': 224,
            'spawn_x': label_x,
            'spawn_z': label_z,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'note': 'segment the block manually; spawn_x/z is reference only, not a label',
        }
        with open(ds_dir / f"{stem}.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        return {'stem': stem, 'dir': str(ds_dir), 'index': idx}

    def _handle_board_locator_training_data(self, training_data: Dict, source: str, robot_id: int) -> Dict:
        try:
            label_x, label_z = self._board_label_xz(training_data)
            if label_x is None or label_z is None:
                obj_pos = training_data.get('object_pos', [0.0, 0.0])
                label_x = float(obj_pos[0]) if len(obj_pos) > 0 else 0.0
                label_z = float(obj_pos[1]) if len(obj_pos) > 1 else 0.0

            ack: Dict = {'type': 'training_ack', 'board_locator': True}
            if not training_data.get('state'):
                ack['skipped'] = True
                return ack

            info = self._save_board_locator_raw_sample(
                training_data['state'], robot_id, label_x, label_z,
            )
            ack['saved_stem'] = info['stem']
            ack['saved_index'] = info['index']
            print(
                f"[BOARD-LOC COLLECT R{robot_id}] saved {info['stem']} "
                f"(warp) → {info['dir']}",
                flush=True,
            )
            return ack
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    def _handle_board_training_data(self, training_data: Dict, source: str, robot_id: int) -> Dict:
        try:
            collect_mode = training_data.get('mode', 'board_collect')
            label_x, label_z = self._board_label_xz(training_data)
            if label_x is None or label_z is None:
                obj_pos = training_data.get('object_pos', [0.0, 0.0])
                label_x = float(obj_pos[0]) if len(obj_pos) > 0 else 0.0
                label_z = float(obj_pos[1]) if len(obj_pos) > 1 else 0.0

            ack: Dict = {
                'type': 'training_ack',
                'board': True,
            }

            if collect_mode == 'board_collect':
                cnn_cell_raw = None
                cell = world_to_board_cell(
                    label_x, label_z, robot_id, n=self._board_n, cfg=self.board_cfg,
                )
                if cell is None:
                    ack['skipped'] = True
                    ack['invalid_cell'] = True
                else:
                    sample = {
                        'state': training_data['state'],
                        'spawn_x': label_x,
                        'spawn_z': label_z,
                        'source': source,
                        'robot_id': robot_id,
                    }
                    bucket = self._resolve_demo_bucket(training_data, robot_id)
                    sample['demo_bucket'] = bucket
                    target = (self.weak_buffer2 if bucket == 'weak' else self.normal_buffer2) if robot_id == 2 \
                        else (self.weak_buffer if bucket == 'weak' else self.normal_buffer)
                    target.append(sample)
                    if self._fine_tune_buffers_ready(robot_id):
                        threading.Thread(
                            target=self._run_board_training_step, args=(robot_id,), daemon=True,
                        ).start()
                    row, col = divmod(cell, self._board_n)
                    ack.update({
                        'teacher_cell': cell,
                        'board_cell': cell,
                        'board_u': col,
                        'board_v': row,
                    })
                    cnn_cell_raw = training_data.get('cnn_cell', training_data.get('board_cell'))
                    if cnn_cell_raw is not None:
                        cnn_cell = int(cnn_cell_raw)
                        correct = cnn_cell == cell
                        correct_attr = 'board_cnn_correct_count2' if robot_id == 2 else 'board_cnn_correct_count'
                        total_attr = 'board_cnn_total_count2' if robot_id == 2 else 'board_cnn_total_count'
                        setattr(self, total_attr, getattr(self, total_attr) + 1)
                        if correct:
                            setattr(self, correct_attr, getattr(self, correct_attr) + 1)
                        ack['cnn_cell'] = cnn_cell
                        ack['cnn_correct'] = correct
                    if cell is not None and training_data.get('state'):
                        try:
                            img = self.decode_b64_image(training_data['state'])
                            rgb = cv2.cvtColor(img['rgb'], cv2.COLOR_BGR2RGB)
                            rgb_c, _ = self._crop_rgb_depth(
                                rgb, img['depth'].astype(np.float32), robot_id,
                            )
                            self._save_board_crop_debug(rgb_c, robot_id, label_x, label_z)
                            rgb_w = self._warp_rgb_for_debug(training_data['state'], robot_id)
                            cnn_for_overlay = int(cnn_cell_raw) if cnn_cell_raw is not None else None
                            q_diag = None
                            q_map_np = None
                            board_mod = self.board_module2 if robot_id == 2 else self.board_module
                            if (
                                board_mod is not None
                                and cnn_for_overlay is not None
                                and cell is not None
                            ):
                                rgb_t, depth_t = self._preprocess_board_pair(
                                    training_data['state'], source == 'simulation', robot_id,
                                )
                                q_diag, q_map_np = self._board_q_diag_from_tensors(
                                    rgb_t, depth_t, board_mod, cell, cnn_for_overlay, robot_id,
                                    label_x=label_x, label_z=label_z,
                                )
                            self._save_board_debug_overlay(
                                rgb_w, robot_id,
                                cnn_cell=cnn_for_overlay,
                                teacher_cell=cell,
                                meta={
                                    'label_x': label_x,
                                    'label_z': label_z,
                                    'spawn_x': training_data.get('spawn_x'),
                                    'spawn_z': training_data.get('spawn_z'),
                                    'object_pos': training_data.get('object_pos'),
                                    'source': 'board_collect',
                                },
                                q_map=q_map_np,
                                q_diag=q_diag,
                            )
                            print(
                                f"[BOARD DEBUG R{robot_id}] saved overlay teacher={cell} "
                                f"cnn={cnn_for_overlay} | debug/board_warp_r{robot_id}_latest.jpg",
                                flush=True,
                            )
                        except Exception as overlay_exc:
                            print(
                                f"[BOARD DEBUG R{robot_id}] overlay at collect failed: {overlay_exc}",
                                flush=True,
                            )
                ep_attr = 'board_shaping_episode_count2' if robot_id == 2 else 'board_shaping_episode_count'
                ep = getattr(self, ep_attr) + 1
                setattr(self, ep_attr, ep)
                ack['board_phase'] = 'shaping'
                ack['board_step'] = self.board_training_step_count2 if robot_id == 2 else self.board_training_step_count
                weak_buf = self.weak_buffer2 if robot_id == 2 else self.weak_buffer
                normal_buf = self.normal_buffer2 if robot_id == 2 else self.normal_buffer
                buf_total = len(weak_buf) + len(normal_buf)
                batch_size = self.batch_size
                if cnn_cell_raw is not None:
                    hit = 'HIT' if ack.get('cnn_correct') else 'miss'
                    print(
                        f"[BOARD TRAIN R{robot_id}] shaping ep {ep} | "
                        f"cnn={int(cnn_cell_raw)} teacher={ack.get('teacher_cell', '?')} {hit} | "
                        f"buffer {buf_total}/{batch_size}",
                        flush=True,
                    )
                elif cell is None:
                    print(
                        f"[BOARD TRAIN R{robot_id}] shaping ep {ep} | "
                        f"invalid spawn cell | buffer {buf_total}/{batch_size}",
                        flush=True,
                    )
                else:
                    print(
                        f"[BOARD TRAIN R{robot_id}] shaping ep {ep} | "
                        f"teacher={cell} (no cnn_cell) | buffer {buf_total}/{batch_size}",
                        flush=True,
                    )
                shaping_episodes = int(self.board_cfg.get('training', {}).get('shaping_episodes', 250))
                log_every = int(self.board_cfg.get('training', {}).get('accuracy_log_every', 10))
                if cnn_cell_raw is not None and ep % log_every == 0:
                    correct_attr = 'board_cnn_correct_count2' if robot_id == 2 else 'board_cnn_correct_count'
                    total_attr = 'board_cnn_total_count2' if robot_id == 2 else 'board_cnn_total_count'
                    total = getattr(self, total_attr)
                    correct_n = getattr(self, correct_attr)
                    acc = 100.0 * correct_n / total if total else 0.0
                    print(
                        f"[BOARD TRAIN R{robot_id}] CNN accuracy @ ep {ep}: "
                        f"{acc:.1f}% ({correct_n}/{total})",
                        flush=True,
                    )
                if ep >= shaping_episodes and ep == shaping_episodes:
                    print(
                        f"[BOARD TRAIN R{robot_id}] Shaping complete ({ep} episodes). "
                        f"Stop sim — continue on real arm (paper session 3).",
                        flush=True,
                    )
                return ack

            return {'type': 'error', 'message': f'Unknown board collect mode: {collect_mode}'}
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

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

            if client_mode in ('training', 'fine_tune', 'locator_train', 'board_locator_train'):
                return {
                    'type':      'grasp_prediction',
                    'mode':      'explore',
                    'pose':      [0.0] * 6,
                    'timestamp': time.time()
                }

            if client_mode == 'grid_train':
                if self._grid_training_phase(robot_id) == 'rl':
                    return self._predict_grid_explore_pose(full_message)
                return {
                    'type':      'grasp_prediction',
                    'mode':      'explore',
                    'pose':      [0.0] * 6,
                    'timestamp': time.time(),
                    'grid_phase': 'shaping',
                }

            if client_mode == 'board_dqn_train':
                return self._predict_board_pose(full_message, explore=True)

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

            use_board = self.board_dqn or bool(full_message.get('use_board_dqn', False))
            if use_board:
                return self._predict_board_pose(full_message, explore=False)

            use_local = self.local_grid or bool(full_message.get('use_local_grid', False))
            if use_local:
                return self._predict_local_grid_pose(full_message)

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

            if self.grid_train:
                return self._handle_grid_training_data(training_data, source, robot_id)

            if self.local_bbox_dqn_train:
                return self._handle_local_bbox_training_data(training_data, source, robot_id)

            if self.board_dqn_train:
                return self._handle_board_training_data(training_data, source, robot_id)

            if self.board_locator_train:
                return self._handle_board_locator_training_data(training_data, source, robot_id)

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
                    "local-bbox DQN" if self.local_bbox_dqn else (
                        "board locator train" if self.board_locator_train else (
                            "board DQN train" if self.board_dqn_train else (
                                "board DQN" if self.board_dqn else (
                                    "residual RL" if self.rl_train else (
                                        "grid train" if self.grid_train else (
                                            "locator train" if self.locator_train else (
                                                "targeted fine-tune" if self.fine_tune else "behavior cloning"
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        print(f"GPU Server listening on {host}:{port} ({mode_label} mode)")
        if self.local_bbox_dqn and self.use_vlm_select:
            print(f"   VLM select: model={self.vlm_model} url={self.ollama_url}")
        if self.local_grid and not self.grid_train:
            print(f"   Local grid inference enabled (locator + Q-cell → geometry)")
        if self.geo_grasp and not self.locator_train and not self.local_grid:
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
        if self.grid_train:
            tr = self.grid_cfg.get('training', {})
            print(
                f"   Grid train: {self._grid_n}x{self._grid_n} window={self._grid_window_m}m | "
                f"phase={tr.get('phase', 'both')} shaping_steps={tr.get('shaping_steps', 2000)} | "
                f"checkpoints: {self._save_path_r1_grid}, {self._save_path_r2_grid}"
            )
        if self.board_locator_train:
            tr = self.board_locator_cfg.get('training', {})
            print(
                f"   Board locator: {self._board_n}x{self._board_n} seg on warp RGB-D | "
                f"LR={tr.get('learning_rate', 1e-3)} | "
                f"checkpoints: {self._save_path_r1_board_locator}, {self._save_path_r2_board_locator}"
            )
        if self.yolo_locator_test:
            yc = self.yolo_locator_cfg
            print(
                f"   YOLO locator test: local Ultralytics | "
                f"weights={yc.get('weights')} | conf={yc.get('conf')} | "
                f"min_conf={yc.get('min_confidence')}"
            )
        if self.yolo_fuse:
            yc = self.yolo_locator_cfg
            print(
                f"   YOLO+DQN fuse: bbox → best Q in box | "
                f"weights={yc.get('weights')} | conf={yc.get('conf')} | "
                f"fallback=plain DQN on miss"
            )
        if self.board_dqn or self.board_locator:
            fusion_on = self.board_locator and self.board_locator_cfg.get('fusion', {}).get('enabled', True)
            if self.board_locator:
                print(f"   Board locator inference enabled (fusion={'on' if fusion_on and self.board_dqn else 'n/a'})")
        if self.board_dqn_train or self.board_dqn:
            tr = self.board_cfg.get('training', {})
            print(
                f"   Board DQN: {self._board_n}x{self._board_n} full platform | "
                f"shaping_episodes={tr.get('shaping_episodes', 250)} | "
                f"checkpoints: {self._save_path_r1_board}, {self._save_path_r2_board}"
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
    parser.add_argument('--grid-train', action='store_true',
                        help='Local grid DQN train (teacher shaping + RL)')
    parser.add_argument('--grid-config', type=str, default=None,
                        help='Path to grid_train_config.yaml')
    parser.add_argument('--local-grid', action='store_true',
                        help='Inference: locator-centered grid Q-head → geometry')
    parser.add_argument('--grid-model', type=str, default=None,
                        help='Local grid checkpoint for R1')
    parser.add_argument('--grid-model-r2', type=str, default=None,
                        help='Local grid checkpoint for R2')
    parser.add_argument('--board-dqn-train', action='store_true',
                        help='Full-board DQN train (Gomes paper: 112x112, shaping + RL)')
    parser.add_argument('--board-dqn', action='store_true',
                        help='Inference: full-board DQN cell picker')
    parser.add_argument('--board-config', type=str, default=None,
                        help='Path to board_dqn_config.yaml')
    parser.add_argument('--board-model', type=str, default=None,
                        help='Board DQN checkpoint for R1')
    parser.add_argument('--board-model-r2', type=str, default=None,
                        help='Board DQN checkpoint for R2')
    parser.add_argument('--board-locator-train', action='store_true',
                        help='Warp-space block segmentation train (auxiliary board locator)')
    parser.add_argument('--board-locator', action='store_true',
                        help='Inference: fuse board locator with board DQN cell picker')
    parser.add_argument('--board-locator-config', type=str, default=None,
                        help='Path to board_locator_train_config.yaml')
    parser.add_argument('--board-locator-model', type=str, default=None,
                        help='Board locator checkpoint for R1')
    parser.add_argument('--board-locator-model-r2', type=str, default=None,
                        help='Board locator checkpoint for R2')
    parser.add_argument('--yolo-locator-test', action='store_true',
                        help='Inference: local YOLO bbox → random cell grasp (no DQN)')
    parser.add_argument('--yolo-locator-config', type=str, default=None,
                        help='Path to yolo_locator_config.yaml')
    parser.add_argument('--yolo-fuse', action='store_true',
                        help='With --board-dqn: YOLO bbox → best Q cell inside (fallback plain DQN)')
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
    parser.add_argument('--rl-train', action='store_true',
                        help='TD3 residual RL fine-tune (BC frozen; separate RL checkpoints)')
    parser.add_argument('--rl-train-config', type=str, default=None,
                        help='Path to rl_train_config.yaml')
    parser.add_argument('--rl-residual-r1', type=str, default=None,
                        help='RL residual checkpoint for R1 (inference with --use-residual on client)')
    parser.add_argument('--rl-residual-r2', type=str, default=None,
                        help='RL residual checkpoint for R2')
    args   = parser.parse_args()

    # Default bare launch: local-bbox DQN inference (THIS ONE checkpoint via config).
    if not any((
        args.fine_tune, args.locator_train, args.geo_grasp, args.grid_train,
        args.local_grid, args.board_dqn_train, args.board_dqn,
        args.board_locator_train, args.board_locator, args.yolo_locator_test,
        args.local_bbox_dqn, args.local_bbox_dqn_train, args.rl_train,
    )):
        args.local_bbox_dqn = True

    if args.rl_train and args.fine_tune:
        parser.error('Use either --rl-train or --fine-tune, not both.')
    if args.rl_train and args.locator_train:
        parser.error('Use either --rl-train or --locator-train, not both.')
    if args.fine_tune and args.locator_train:
        parser.error('Use either --fine-tune or --locator-train, not both.')
    if args.geo_grasp and args.locator_train:
        parser.error('--geo-grasp is for inference only; omit when --locator-train.')
    if args.grid_train and args.locator_train:
        parser.error('Use either --grid-train or --locator-train, not both.')
    if args.grid_train and args.rl_train:
        parser.error('Use either --grid-train or --rl-train, not both.')
    if args.grid_train and args.fine_tune:
        parser.error('Use either --grid-train or --fine-tune, not both.')
    if args.local_grid and args.geo_grasp:
        parser.error('Use either --local-grid or --geo-grasp, not both.')
    if args.grid_train and args.local_grid:
        parser.error('Use --grid-train for training; --local-grid is inference only.')
    if args.board_dqn_train and args.locator_train:
        parser.error('Use either --board-dqn-train or --locator-train, not both.')
    if args.board_dqn_train and args.grid_train:
        parser.error('Use either --board-dqn-train or --grid-train, not both.')
    if args.board_dqn_train and args.rl_train:
        parser.error('Use either --board-dqn-train or --rl-train, not both.')
    if args.board_dqn_train and args.fine_tune:
        parser.error('Use either --board-dqn-train or --fine-tune, not both.')
    if args.board_dqn and args.geo_grasp:
        parser.error('Use either --board-dqn or --geo-grasp, not both.')
    if args.board_dqn and args.local_grid:
        parser.error('Use either --board-dqn or --local-grid, not both.')
    if args.board_dqn_train and args.board_dqn:
        parser.error('Use --board-dqn-train for training; --board-dqn is inference only.')
    if args.board_locator_train and args.board_locator:
        parser.error('Use --board-locator-train for training; --board-locator is inference only.')
    if args.board_locator_train and args.locator_train:
        parser.error('Use either --board-locator-train or --locator-train, not both.')
    if args.board_locator_train and args.board_dqn_train:
        parser.error('Use either --board-locator-train or --board-dqn-train, not both.')
    if args.board_locator_train and args.fine_tune:
        parser.error('Use either --board-locator-train or --fine-tune, not both.')
    if args.board_locator_train and args.grid_train:
        parser.error('Use either --board-locator-train or --grid-train, not both.')
    if args.board_locator_train and args.rl_train:
        parser.error('Use either --board-locator-train or --rl-train, not both.')
    if args.board_locator and args.geo_grasp:
        parser.error('Use either --board-locator with --board-dqn or --geo-grasp, not both.')
    if args.board_locator and not args.board_dqn:
        parser.error('--board-locator requires --board-dqn for fused inference.')
    if args.yolo_locator_test and args.board_dqn:
        parser.error('Use either --yolo-locator-test or --board-dqn, not both.')
    if args.yolo_fuse and not args.board_dqn:
        parser.error('--yolo-fuse requires --board-dqn.')
    if args.yolo_fuse and args.yolo_locator_test:
        parser.error('Use either --yolo-fuse or --yolo-locator-test, not both.')
    if args.yolo_locator_test and args.board_locator:
        parser.error('Use either --yolo-locator-test or --board-locator, not both.')
    if args.yolo_locator_test and args.board_dqn_train:
        parser.error('Use either --yolo-locator-test or --board-dqn-train, not both.')
    if args.yolo_locator_test and args.board_locator_train:
        parser.error('Use either --yolo-locator-test or --board-locator-train, not both.')
    if args.yolo_locator_test and args.geo_grasp:
        parser.error('Use either --yolo-locator-test or --geo-grasp, not both.')
    if args.yolo_locator_test and args.local_grid:
        parser.error('Use either --yolo-locator-test or --local-grid, not both.')
    if args.local_bbox_dqn_train and args.local_bbox_dqn:
        parser.error('Use --local-bbox-dqn-train for training; --local-bbox-dqn is inference only.')
    if args.use_vlm_select and not args.local_bbox_dqn:
        parser.error('--use-vlm-select requires --local-bbox-dqn.')
    if args.use_vlm_select and args.local_bbox_dqn_train:
        parser.error('--use-vlm-select is inference-only (not with --local-bbox-dqn-train).')
    if args.local_bbox_dqn and args.board_dqn:
        parser.error('Use either --local-bbox-dqn or --board-dqn, not both.')
    if args.local_bbox_dqn and args.yolo_locator_test:
        parser.error('Use either --local-bbox-dqn or --yolo-locator-test, not both.')
    if args.local_bbox_dqn and args.local_grid:
        parser.error('Use either --local-bbox-dqn or --local-grid, not both.')
    if args.local_bbox_dqn_train and args.board_dqn_train:
        parser.error('Use either --local-bbox-dqn-train or --board-dqn-train, not both.')
    if args.local_bbox_dqn_train and args.grid_train:
        parser.error('Use either --local-bbox-dqn-train or --grid-train, not both.')

    server = GPUInferenceServer(
        model_path=args.model,
        model_path_r2=args.model_r2,
        fine_tune=args.fine_tune,
        fine_tune_config_path=args.fine_tune_config,
        locator_train=args.locator_train,
        locator_config_path=args.locator_config,
        geo_grasp=args.geo_grasp,
        grid_train=args.grid_train,
        grid_config_path=args.grid_config,
        local_grid=args.local_grid,
        grid_model_path=args.grid_model,
        grid_model_path_r2=args.grid_model_r2,
        board_dqn_train=args.board_dqn_train,
        board_dqn=args.board_dqn,
        board_config_path=args.board_config,
        board_model_path=args.board_model,
        board_model_path_r2=args.board_model_r2,
        board_locator_train=args.board_locator_train,
        board_locator=args.board_locator,
        board_locator_config_path=args.board_locator_config,
        board_locator_model_path=args.board_locator_model,
        board_locator_model_path_r2=args.board_locator_model_r2,
        yolo_locator_test=args.yolo_locator_test,
        yolo_locator_config_path=args.yolo_locator_config,
        yolo_fuse=args.yolo_fuse,
        local_bbox_dqn=args.local_bbox_dqn,
        local_bbox_dqn_train=args.local_bbox_dqn_train,
        local_bbox_dqn_config_path=args.local_bbox_dqn_config,
        local_bbox_model_path=args.local_bbox_model,
        local_bbox_model_path_r2=args.local_bbox_model_r2,
        use_vlm_select=args.use_vlm_select,
        vlm_model=args.vlm_model,
        ollama_url=args.ollama_url,
        rl_train=args.rl_train,
        rl_train_config_path=args.rl_train_config,
        rl_residual_r1=args.rl_residual_r1,
        rl_residual_r2=args.rl_residual_r2,
    )
    server.start_server()