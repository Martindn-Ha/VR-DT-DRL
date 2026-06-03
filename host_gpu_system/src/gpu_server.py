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
import threading
import time
import yaml
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque

from enhanced_neural_network import UR3GraspCNN_Enhanced, BehaviorCloningModule, create_model, ImageProcessor


class GPUInferenceServer:
    """
    Centralized inference and training server for multi-robot reinforcement 
    and behavior cloning systems.
    """
    def __init__(self, config_path: str = "config/network_config.yaml", model_path: str = None):
        self.config     = self._load_config(config_path)
        self.model_path = model_path

        # =========================================================================
        # DEVICE & MODEL SETUP
        # =========================================================================
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_config = self._load_model_config()

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
        self.batch_size           = 16
        self.data_buffer          = deque(maxlen=10000)   # Robot 1 Buffer
        self.data_buffer2         = deque(maxlen=10000)   # Robot 2 Buffer
        self.training_step_count  = 0
        self.training_step_count2 = 0

        self._load_model_weights()

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.bc_module.optimizer,
            mode='min', factor=0.5, patience=50, min_lr=1e-5
        )
        self.lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.bc_module2.optimizer,
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

    def _load_model_weights(self):
        """Locates and restores checkpoint weights, handling partial mismatches."""
        script_dir = Path(__file__).resolve().parent

        def _load(model, bc_module, path_override, default_name, step_attr):
            if path_override:
                path = Path(path_override)
                if not path.is_absolute():
                    path = script_dir / path
            else:
                path = script_dir.parent / "models" / default_name
                if not path.exists():
                    path = script_dir.parent / "models" / "ur3_model.pth"

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
            self.model2, self.bc_module2, None,             "ur3_live_model_r2.pth", None)

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

            if client_mode == 'training':
                return {
                    'type':      'grasp_prediction',
                    'mode':      'explore',
                    'pose':      [0.0] * 6,   
                    'timestamp': time.time()
                }

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

            # Convert local-frame prediction back to world coordinates
            base_x = self._robot2_base_x if robot_id == 2 else self._robot1_base_x
            grasp_pose[0] += base_x
            grasp_pose[2] += self._robot_base_z

            # Force tool-down orientation
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
                import random, os
                if robot_id == 2:
                    bc_mod    = self.bc_module2
                    model     = self.model2
                    buf       = self.data_buffer2
                    sched     = self.lr_scheduler2
                    step_attr = 'training_step_count2'
                    save_name = "ur3_live_model_r2.pth"
                else:
                    bc_mod    = self.bc_module
                    model     = self.model
                    buf       = self.data_buffer
                    sched     = self.lr_scheduler
                    step_attr = 'training_step_count'
                    save_name = "ur3_live_model_r1.pth"

                batch_raw   = random.sample(list(buf), self.batch_size)
                torch_batch = self.format_batch_for_torch(batch_raw)
                losses      = bc_mod.update_networks(torch_batch)
                sched.step(losses['pose'])
                
                step = getattr(self, step_attr) + 1
                setattr(self, step_attr, step)

                if step % 5 == 0:
                    print(
                        f"🔥 R{robot_id} Step {step:4d} | "
                        f"Loss: {losses['total']:.4f} "
                        f"(Pose:{losses['pose']:.4f} "
                        f"Aux:{losses['aux']:.4f} "
                        f"Grasp:{losses['grasp']:.4f}[monitor only]) | "
                        f"GradNorm: {losses['grad_norm']:.3f}"
                    )

                if step % 100 == 0:
                    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    save_dir  = os.path.join(base_dir, "models")
                    os.makedirs(save_dir, exist_ok=True)
                    full_path = os.path.join(save_dir, save_name)
                    
                    torch.save({
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': bc_mod.optimizer.state_dict(),
                        'training_step':        step,
                    }, full_path)
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
        print(f"🚀 BC Server listening on {host}:{port} (behavior cloning mode)")
        
        while True:
            conn, addr = self.server_socket.accept()
            threading.Thread(
                target=self.handle_client_request, args=(conn, addr), daemon=True
            ).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    args   = parser.parse_args()
    
    server = GPUInferenceServer(model_path=args.model)
    server.start_server()