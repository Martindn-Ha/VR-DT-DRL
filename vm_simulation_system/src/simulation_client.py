#!/usr/bin/env python3
"""
VM Simulation Client for UR3e System

This module handles the simulation and real-world execution of a UR3e robotic arm.
It manages network communications, state tracking, domain randomization, and an
automated curriculum for reinforcement learning.
"""

import socket
import json
import os
import sys
import numpy as np
import math
import cv2
import time
import threading
import yaml
import argparse
import base64
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from failure_taxonomy import classify_outcome, is_clamp_limited
from rl_reward import load_rl_train_config, get_robot_rl_config, calculate_rl_reward
from spawn_geometry import (
    load_fine_tune_config,
    load_locator_train_config,
    pick_random_weak_cell,
    quadrant_from_spawn,
    sample_spawn_in_phase_quadrant,
)
from grasp_geometry import (
    compute_grasp_pose_from_object_world,
    fallback_grasp_pose,
)
from collections import deque

# Repo-root data/ (VR-DT-DRL/data), not vm_simulation_system/data
VR_DRL_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_AGENT_DEBUG_LOG = VR_DRL_DATA_DIR.parent / "debug-4ce223.log"


def _agent_debug_log(location: str, message: str, data: Optional[Dict] = None,
                     hypothesis_id: str = "", run_id: str = "pre-fix") -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "4ce223",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with _AGENT_DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
    # #endregion


def running_inside_webots() -> bool:
    """True when Webots launched this process as an extern controller."""
    return bool(os.environ.get("WEBOTS_CONTROLLER_URL"))


EPISODE_LOG_COLUMNS = [
    'timestamp', 'robot_id', 'episode', 'session_episode', 'run_mode', 'inference_mode',
    'curriculum_phase', 'spawn_phase',
    'spawn_x', 'spawn_y', 'spawn_z', 'spawn_radius_cm',
    'cam_delta_x_cm', 'cam_delta_y_cm', 'cam_delta_z_cm',
    'cam_delta_pitch_deg', 'cam_delta_yaw_deg', 'cam_delta_roll_deg',
    'grasp_mode',
    'ai_pose_0', 'ai_pose_1', 'ai_pose_2', 'ai_pose_3', 'ai_pose_4', 'ai_pose_5',
    'clamp_pose_0', 'clamp_pose_1', 'clamp_pose_2',
    'success', 'lifted_m', 'closest_dist_m',
    'lateral_aim_err_m', 'lateral_aim_bc_err_m', 'lateral_improve_m',
    'reward', 'object_found',
    'outcome_class', 'clamp_limited',
    'residual_dx', 'residual_dz', 'residual_dyaw',
    'pred_obj_x_m', 'pred_obj_z_m', 'locator_err_m',
    'support_shade_1', 'support_shade_2',
    'spawn_quadrant', 'demo_bucket', 'spawn_collection',
]

LOCATOR_EPISODE_LOG_COLUMNS = [
    'timestamp', 'robot_id', 'episode', 'session_episode', 'run_mode',
    'spawn_phase',
    'spawn_x', 'spawn_y', 'spawn_z', 'spawn_radius_cm',
    'cam_delta_x_cm', 'cam_delta_y_cm', 'cam_delta_z_cm',
    'cam_delta_pitch_deg', 'cam_delta_yaw_deg', 'cam_delta_roll_deg',
    'grasp_mode',
    'label_x_m', 'label_z_m', 'label_source',
    'demo_sent', 'demo_ok', 'object_found',
    'pred_obj_x_m', 'pred_obj_z_m', 'locator_err_m', 'loc_step_at_pred',
    'spawn_quadrant', 'demo_bucket', 'spawn_collection',
]

# Excel-only: column B is local wall time derived from UTC ISO in column A.
EPISODE_LOG_TIMESTAMP_LOCAL_COL = 'timestamp_local'
EPISODE_LOG_XLSX_LOCAL_TIME_FORMAT = 'yyyy-mm-dd h:mm:ss AM/PM'
EPISODE_LOG_XLSX_HEADERS = (
    ['timestamp', EPISODE_LOG_TIMESTAMP_LOCAL_COL]
    + [c for c in EPISODE_LOG_COLUMNS if c != 'timestamp']
)

EPISODE_LOG_XLSX_INT_COLS = frozenset({
    'robot_id', 'episode', 'session_episode', 'curriculum_phase', 'success', 'object_found', 'clamp_limited',
})
EPISODE_LOG_XLSX_FLOAT_COLS = frozenset({
    'spawn_x', 'spawn_y', 'spawn_z', 'spawn_radius_cm',
    'cam_delta_x_cm', 'cam_delta_y_cm', 'cam_delta_z_cm',
    'cam_delta_pitch_deg', 'cam_delta_yaw_deg', 'cam_delta_roll_deg',
    'ai_pose_0', 'ai_pose_1', 'ai_pose_2', 'ai_pose_3', 'ai_pose_4', 'ai_pose_5',
    'clamp_pose_0', 'clamp_pose_1', 'clamp_pose_2',
    'lifted_m', 'closest_dist_m',
    'lateral_aim_err_m', 'lateral_aim_bc_err_m', 'lateral_improve_m',
    'reward',
    'residual_dx', 'residual_dz', 'residual_dyaw',
    'pred_obj_x_m', 'pred_obj_z_m', 'locator_err_m',
    'support_shade_1', 'support_shade_2',
})
LOCATOR_EPISODE_LOG_XLSX_INT_COLS = frozenset({
    'robot_id', 'episode', 'session_episode', 'object_found', 'demo_sent', 'demo_ok',
    'loc_step_at_pred',
})
LOCATOR_EPISODE_LOG_XLSX_FLOAT_COLS = frozenset({
    'spawn_x', 'spawn_y', 'spawn_z', 'spawn_radius_cm',
    'cam_delta_x_cm', 'cam_delta_y_cm', 'cam_delta_z_cm',
    'cam_delta_pitch_deg', 'cam_delta_yaw_deg', 'cam_delta_roll_deg',
    'label_x_m', 'label_z_m',
    'pred_obj_x_m', 'pred_obj_z_m', 'locator_err_m',
})


def _iso_utc_to_local_naive(iso_str: str) -> Optional[datetime]:
    """Parse UTC ISO timestamp from column A into local naive datetime for Excel."""
    if not iso_str:
        return None
    try:
        text = str(iso_str).replace('Z', '+00:00')
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def _xlsx_cell_value(col_name: str, value: Any) -> Any:
    """Coerce numeric log fields to int/float so Excel does not store them as text."""
    if value == '' or value is None:
        return ''
    if col_name in EPISODE_LOG_XLSX_INT_COLS or col_name in LOCATOR_EPISODE_LOG_XLSX_INT_COLS:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    if col_name in EPISODE_LOG_XLSX_FLOAT_COLS or col_name in LOCATOR_EPISODE_LOG_XLSX_FLOAT_COLS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value

#  RealSense dependencies (--real mode only) 
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

#  Robotiq gripper dependencies (--real mode only) 
try:
    import roslib; roslib.load_manifest('robotiq_2f_gripper_control')
    from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as RobotiqOutput
    from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input  as RobotiqInput
    ROBOTIQ_AVAILABLE = True
except Exception:
    ROBOTIQ_AVAILABLE = False

#  Actionlib for real robot joint trajectory (--real mode only) 
try:
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectoryPoint
    ACTIONLIB_AVAILABLE = True
except ImportError:
    ACTIONLIB_AVAILABLE = False

#  ROS imports 
try:
    import rospy
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32MultiArray, Bool, Empty
    from geometry_msgs.msg import Pose, PoseStamped
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from cv_bridge import CvBridge, CvBridgeError
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS not available, using simulation mode")
    class CvBridge: pass
    class Image: pass
    class JointState: pass
    class Float32MultiArray: pass
    class Bool: pass
    class Empty: pass
    class Pose: pass
    class PoseStamped: pass
    class MockRospy:
        def init_node(self, *args, **kwargs): pass
        def loginfo(self, msg): print(f"[INFO] {msg}")
        def logwarn(self, msg): print(f"[WARN] {msg}")
        def logerr(self, msg): print(f"[ERROR] {msg}")
        def logwarn_throttle(self, period, msg): print(f"[WARN] {msg}")
        def loginfo_throttle(self, period, msg): print(f"[INFO] {msg}")
        def is_shutdown(self): return False
        def Rate(self, hz):
            class MockRate:
                def sleep(self): time.sleep(0.1)
            return MockRate()
        def sleep(self, secs=0.1): time.sleep(secs)
        def Subscriber(self, *args, **kwargs): pass
        def Publisher(self, *args, **kwargs):
            class MockPub:
                def publish(self, *args): pass
            return MockPub()
        class Duration:
            def __init__(self, secs=0): self.secs = secs
        class ROSInterruptException(Exception): pass
    if 'rospy' not in locals():
        rospy = MockRospy()

from enhanced_robot_controller import create_robot_system, create_dual_robot_system
from enhanced_camera_handler import EnhancedCameraHandler
from webots_bridge import WebotsBridge


class CurriculumManager:
    """
    Performance-gated curriculum manager.
    
    Each phase advancement requires:
      1. A minimum number of episodes completed in the current phase.
      2. A minimum AI-only success rate over the last N AI episodes.
    Teacher (explore) results do not count toward phase advancement.
    """

    PLATFORM_CENTER_X  = -0.646
    PLATFORM_CENTER_Z  = 0.841
    PLATFORM_HALF_SIZE_X = 0.143675 
    PLATFORM_HALF_SIZE_Z = 0.083675  

    # =========================================================================
    # CURRICULUM PHASE CONFIGURATION
    # =========================================================================
    # Tuple format: 
    # (r_min, r_max, min_episodes_in_phase, mastery_threshold, ai_window)
    #
    # Dimensions: 0.30m (X) x 0.23m (Z) -> Half-sizes: 0.15m x 0.115m
    # Max usable radius before Z edge = 0.115m
    # =========================================================================
    FULL_BOARD_PHASE = 5  # Uniform spawn anywhere on usable platform (inference --phase 5)

    PHASE_CONFIG = [
        (0.000, 0.000,  60,  0.85, 20),  # Phase 0: Static center
        (0.005, 0.015, 200,  0.75, 40),  # Phase 1: ±0.5cm - 1.5cm
        (0.015, 0.035, 250,  0.65, 50),  # Phase 2: ±1.5cm - 3.5cm
        (0.035, 0.070, 300,  0.55, 50),  # Phase 3: ±3.5cm - 7.0cm
        (0.070, 0.115, 9999, 0.00, 50),  # Phase 4: Radius ring 7–11.5cm (inference --phase 4)
        (0.000, 0.000, 9999, 0.00, 50),  # Phase 5: Full board (uniform; see get_spawn / inference)
    ]

    def __init__(self, state_file="config/curriculum_state.json"):
        self.state_file = Path(state_file)
        self.phase             = 4  # Training permanently set to random spawn
        self.episodes_in_phase = 0
        self.episode           = 0
        self.ai_recent_results = deque(maxlen=50)
        
        self._load_state()

    def _load_state(self):
        """Restores curriculum state from disk."""
        ai_window = self.PHASE_CONFIG[self.phase][4]
        self.ai_recent_results = deque(maxlen=ai_window)
        if self.state_file.exists():
            try:
                import json
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.phase = state.get('phase', 0)
                self.episodes_in_phase = state.get('episodes_in_phase', 0)
                self.episode = state.get('episode', 0)
                
                ai_window = self.PHASE_CONFIG[self.phase][4]
                self.ai_recent_results = deque(maxlen=ai_window)
                print(
                    f"[CURRICULUM] Resumed Phase {self.phase}, Episode {self.episode} "
                    f"| AI window reset for this session (0/{ai_window})"
                )
            except Exception as e:
                print(f"[CURRICULUM] WARNING: Could not load state: {e}")

    def _save_state(self):
        """Persists current curriculum state to disk."""
        try:
            import json
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({
                    'phase': self.phase,
                    'episodes_in_phase': self.episodes_in_phase,
                    'episode': self.episode,
                    'ai_recent_results': list(self.ai_recent_results)
                }, f)
        except Exception:
            pass

    def record_result(self, success: bool, mode: str):
        """
        Records the outcome of a completed episode.
        Only 'exploit' (AI) mode results trigger potential phase advancements.
        """
        self.episodes_in_phase += 1
        if mode == 'exploit':
            self.ai_recent_results.append(success)
            
        self._save_state() 

    def update(self, episode: int):
        self.episode = episode
        self._save_state()

    def get_ai_success_rate(self) -> float:
        if not self.ai_recent_results:
            return 0.0
        return sum(self.ai_recent_results) / len(self.ai_recent_results)

    def check_phase_advance(self) -> bool:
        """
        Evaluates conditions for phase progression.
        Requires a minimum sample size (20 AI attempts) to prevent 
        premature advancement from statistical anomalies.
        """
        if self.phase >= len(self.PHASE_CONFIG) - 1:
            return False

        r_min, r_max, min_eps, threshold, ai_window = self.PHASE_CONFIG[self.phase]

        # Resize the tracking window to match the current phase constraints
        self.ai_recent_results = deque(self.ai_recent_results, maxlen=ai_window)

        enough_episodes = self.episodes_in_phase >= min_eps
        ai_attempts     = len(self.ai_recent_results)
        ai_rate         = self.get_ai_success_rate()
        ai_mastered     = ai_attempts >= 20 and ai_rate >= threshold

        if enough_episodes and ai_mastered:
            old_phase   = self.phase
            self.phase += 1
            self.episodes_in_phase = 0
            self.ai_recent_results.clear()

            r_new_min, r_new_max, _, new_threshold, new_window = self.PHASE_CONFIG[self.phase]
            print(f"[CURRICULUM] Phase {old_phase} -> {self.phase} | "
                  f"AI mastery: {ai_rate*100:.1f}% over {ai_attempts} attempts")
            print(f"[CURRICULUM] New radius: {r_new_min*100:.1f}–{r_new_max*100:.1f}cm | "
                  f"Next target: {new_threshold*100:.0f}% over {new_window} AI attempts")
            return True
        return False

    def get_spawn_radius(self) -> tuple:
        r_min, r_max, _, _, _ = self.PHASE_CONFIG[self.phase]
        return r_min, r_max

    def get_spawn_position(self) -> tuple:
        """Determines the next object spawn position based on the current phase."""
        r_min, r_max = self.get_spawn_radius()

        cx = self.PLATFORM_CENTER_X
        cz = self.PLATFORM_CENTER_Z
        
        half_x = 0.143675   
        half_z = 0.083675  

        if self.phase in (4, self.FULL_BOARD_PHASE):
            # Full platform random distribution (training phase 4/5, or normal inference at phase 4+)
            spawn_x = np.random.uniform(cx - half_x, cx + half_x)
            spawn_z = np.random.uniform(cz - half_z, cz + half_z)
            
        else:
            # Controlled radius expansion for early phases
            if r_max < 0.001:
                spawn_x, spawn_z = cx, cz
            else:
                while True:
                    sample_x = np.random.uniform(cx - half_x, cx + half_x)
                    sample_z = np.random.uniform(cz - half_z, cz + half_z)
                    dist = np.sqrt((sample_x - cx)**2 + (sample_z - cz)**2)
                    
                    if r_min <= dist <= r_max:
                        spawn_x, spawn_z = sample_x, sample_z
                        break

        ai_rate    = self.get_ai_success_rate()
        ai_window  = self.PHASE_CONFIG[self.phase][4]
        print(f"[CURRICULUM] Episode {self.episode} | Phase {self.phase} | "
              f"Spawn: ({spawn_x:.3f}, {spawn_z:.3f}) | "
              f"AI rate: {ai_rate*100:.1f}% ({len(self.ai_recent_results)}/{ai_window} AI attempts)")

        return (spawn_x, None, spawn_z)

    def _get_phase_number(self) -> int:
        return self.phase


class CurriculumManagerRobot2(CurriculumManager):
    """
    Curriculum manager for Robot 2 (ur3_robot2).

    Handles the unique geometric constraints of Platform 2, which features a 
    curved rear edge. Ensures objects do not spawn in unreachable areas.
    """

    # World-frame origin of the crossSection (0,0) corner
    PLATFORM_ORIGIN_X = -1.215
    PLATFORM_ORIGIN_Z =  0.755
    SPAWN_Y = 0.461

    # Arc boundary definitions
    _ARC_CX      =  0.0241   
    _ARC_CZ      = -0.2965   
    _ARC_R       =  0.4871   
    _ARC_R_INSET = 0.4771   

    # Operational boundaries with safety buffer
    PLATFORM_WORLD_X_MIN = PLATFORM_ORIGIN_X - 0.290  
    PLATFORM_WORLD_X_MAX = PLATFORM_ORIGIN_X - 0.010 
    PLATFORM_WORLD_Z_MIN = PLATFORM_ORIGIN_Z + 0.010
    PLATFORM_WORLD_Z_MAX = PLATFORM_ORIGIN_Z + 0.180  

    PLATFORM_CENTER_X = (PLATFORM_WORLD_X_MIN + PLATFORM_WORLD_X_MAX) / 2
    PLATFORM_CENTER_Z = (PLATFORM_WORLD_Z_MIN + PLATFORM_WORLD_Z_MAX) / 2


    def __init__(self, state_file="config/curriculum_state_robot2.json"):
        super().__init__(state_file=state_file)

    def _world_to_local(self, wx, wz):
        """Converts world coordinates to local cross-section coordinates."""
        local_x = self.PLATFORM_ORIGIN_X - wx
        local_z = wz - self.PLATFORM_ORIGIN_Z
        return local_x, local_z

    def _in_spawn_area(self, wx, wz):
        """Validates if a world point is inside the boundary."""
        lx, lz = self._world_to_local(wx, wz)
        dist = np.sqrt((lx - self._ARC_CX)**2 + (lz - self._ARC_CZ)**2)
        return dist <= self._ARC_R_INSET

    def get_spawn_position(self) -> tuple:
        """Samples a safe spawn position inside the curved platform area."""
        r_min, r_max, _, _, _ = self.PHASE_CONFIG[self.phase]
        cx = self.PLATFORM_CENTER_X
        cz = self.PLATFORM_CENTER_Z

        MAX_ATTEMPTS = 200

        if self.phase in (4, self.FULL_BOARD_PHASE):
            wx_min, wx_max = self.PLATFORM_WORLD_X_MIN, self.PLATFORM_WORLD_X_MAX
            wz_min, wz_max = self.PLATFORM_WORLD_Z_MIN, self.PLATFORM_WORLD_Z_MAX
            
            for _ in range(MAX_ATTEMPTS):
                spawn_x = np.random.uniform(wx_min, wx_max)
                spawn_z = np.random.uniform(wz_min, wz_max)
                if self._in_spawn_area(spawn_x, spawn_z):
                    break
            else:
                spawn_x, spawn_z = cx, cz  

        elif r_max < 0.001:
            spawn_x, spawn_z = cx, cz

        else:
            for _ in range(MAX_ATTEMPTS):
                sample_x = np.random.uniform(cx - r_max, cx + r_max)
                sample_z = np.random.uniform(cz - r_max, cz + r_max)
                dist = np.sqrt((sample_x - cx)**2 + (sample_z - cz)**2)
                if r_min <= dist <= r_max and self._in_spawn_area(sample_x, sample_z):
                    spawn_x, spawn_z = sample_x, sample_z
                    break
            else:
                spawn_x, spawn_z = cx, cz 

        ai_rate   = self.get_ai_success_rate()
        ai_window = self.PHASE_CONFIG[self.phase][4]
        print(f"[CURRICULUM R2] Episode {self.episode} | Phase {self.phase} | "
              f"Spawn: ({spawn_x:.3f}, {spawn_z:.3f}) | "
              f"AI rate: {ai_rate*100:.1f}% ({len(self.ai_recent_results)}/{ai_window} AI attempts)")

        return (spawn_x, None, spawn_z)


class SimulationClient:
    """Main execution client coordinating the physical/simulated robot and neural network."""

    def __init__(self, config_path: str = "config/network_config.yaml",
                 mode: str = 'inference', real_robot: bool = False,
                 robot_id: int = 1, ros_camera: bool = False,
                 use_residual: bool = False, use_geo_grasp: bool = False,
                 no_workspace_clamp: bool = False,
                 rl_train_config_path: Optional[str] = None):
        
        self.mode       = mode
        self.real_robot = real_robot
        self.robot_id   = robot_id   
        self.ros_camera = ros_camera 
        self.use_residual = use_residual
        self.use_geo_grasp = use_geo_grasp
        self.no_workspace_clamp = no_workspace_clamp
        self._rl_reward_cfg: Dict[str, Any] = {}
        self.config     = self._load_config(config_path)

        if ROS_AVAILABLE:
            rospy.init_node('ur3_simulation_client', anonymous=True)
            rospy.loginfo(f"UR3 Client started (Mode: {mode}, Real robot: {real_robot}, "
                          f"Robot ID: {robot_id})")

        self.host_socket        = None
        self.connected          = False
        self.connection_lock    = threading.Lock()
        self.bridge             = CvBridge() if ROS_AVAILABLE else None
        
        self.latest_rgb_image   = None
        self.latest_depth_image = None
        self.latest_rgb_b64     = None
        self.latest_depth_b64   = None
        self.latest_joint_states = {'names': [], 'positions': [0]*6}

        self.curriculum = CurriculumManagerRobot2() if robot_id == 2 else CurriculumManager()

        self.episode_count        = self.curriculum.episode
        self.session_episode_count = 0
        self.episode_active       = False
        self.last_grasp_mode      = 'explore'
        self._nan_reset_pending   = False

        # Inference sub-mode parameters
        self.inference_mode           = 'normal'
        self.cycle_episodes_per_phase = 10
        self.fixed_phase              = 0
        self._cycle_phases            = list(range(len(CurriculumManager.PHASE_CONFIG)))
        self._cycle_phase             = self._cycle_phases[0]
        self._cycle_count_in_phase    = 0

        self._fine_tune_cfg           = None
        self._fine_tune_weak_regions  = {}
        self._fine_tune_weak_spawn_p  = 0.7
        self._fine_tune_demo_bucket   = ''
        self._fine_tune_spawn_phase   = 0
        self._fine_tune_spawn_collection = ''
        self._fine_tune_spawn_quadrant = 0
        if mode == 'fine_tune':
            self._load_fine_tune_settings()
        if mode == 'locator_train':
            self._load_locator_train_settings()
        if mode == 'rl_train':
            self._load_rl_train_settings(rl_train_config_path)

        self._episode_log_dir = VR_DRL_DATA_DIR
        self._episode_xlsx_path = None
        self._episode_log_lock = threading.Lock()
        self._reset_episode_log_fields()

        if real_robot:
            self.webots_bridge = None
            if ros_camera:
                self._init_ros_camera()
            else:
                self._init_realsense()
            self._init_real_robot_motion()
            self._init_robotiq_gripper()
        else:
            self.webots_bridge = WebotsBridge(simulation=False, robot_id=robot_id)
            if self.webots_bridge.shared_robot is None:
                raise RuntimeError(
                    f"Webots not connected for robot {robot_id}. "
                    "Open updated_world/worlds/Environmentnewww.wbt, press Play, then restart this client."
                )
            print(f"[STARTUP] Webots connected (robot {robot_id})")

            self.robot_controller, self.gripper_controller, self.motion_planner = \
                create_robot_system(
                    config_path="config/robot_config.yaml",
                    simulation=True,
                    webots_bridge=self.webots_bridge,
                    robot_id=robot_id
                )

            cam_name = 'camera2' if robot_id == 2 else 'robot1'
            self.camera_handler = EnhancedCameraHandler(
                config_path="config/camera_config.yaml",
                simulation=True,
                camera_type="simulation",
                webots_bridge=self.webots_bridge,
                camera_name=cam_name
            )

            if ROS_AVAILABLE:
                self._setup_ros_interface()

            self._cam_base = {}
            cam = getattr(self.webots_bridge, 'camera2' if robot_id == 2 else 'camera', None)
            if cam is not None:
                status = "ready" if cam.devices_ready else "NOT bound"
                print(f"[WebotsBridge] R{robot_id} RGB-D devices: {status}")
            if robot_id == 2:
                print(
                    "[STARTUP] Dual-robot world: start a Robot 1 client too "
                    "(--robot-id 1) or Webots may not step the simulation."
                )

    def _load_fine_tune_settings(self, config_path: Optional[str] = None):
        path = config_path or str(
            VR_DRL_DATA_DIR.parent / "host_gpu_system" / "config" / "fine_tune_config.yaml"
        )
        self._fine_tune_cfg = load_fine_tune_config(path)
        self._fine_tune_weak_regions = self._fine_tune_cfg['_weak_regions_parsed']
        self._fine_tune_weak_spawn_p = float(
            self._fine_tune_cfg.get('collection', {}).get('weak_spawn_probability', 0.7)
        )
        print(
            f"[FINE-TUNE R{self.robot_id}] weak_spawn_p={self._fine_tune_weak_spawn_p:.0%} | "
            f"weak cells={self._fine_tune_weak_regions.get(self.robot_id, [])}"
        )

    def _load_locator_train_settings(self, config_path: Optional[str] = None):
        path = config_path or str(
            VR_DRL_DATA_DIR.parent / "host_gpu_system" / "config" / "locator_train_config.yaml"
        )
        cfg = load_locator_train_config(path)
        self._fine_tune_cfg = cfg
        self._fine_tune_weak_regions = cfg['_weak_regions_parsed']
        self._fine_tune_weak_spawn_p = float(
            cfg.get('collection', {}).get('weak_spawn_probability', 0.7)
        )
        print(
            f"[LOCATOR-TRAIN R{self.robot_id}] weak_spawn_p={self._fine_tune_weak_spawn_p:.0%} | "
            f"weak cells={self._fine_tune_weak_regions.get(self.robot_id, [])}"
        )

    def _load_rl_train_settings(self, config_path: Optional[str] = None):
        path = config_path or str(
            VR_DRL_DATA_DIR.parent / "host_gpu_system" / "config" / "rl_train_config.yaml"
        )
        rl_cfg = load_rl_train_config(path)
        self._rl_reward_cfg = get_robot_rl_config(rl_cfg, self.robot_id)
        rw = self._rl_reward_cfg.get('reward', {})
        print(
            f"[RL-TRAIN R{self.robot_id}] reward weights loaded from {path} | "
            f"w_dist={rw.get('w_dist')} w_lift={rw.get('w_lift')} "
            f"w_miss={rw.get('w_miss')} w_corr={rw.get('w_corr')}"
        )

    def _episode_log_stem(self) -> str:
        if self.mode == 'rl_train':
            return f"episode_log_r{self.robot_id}_rl_train"
        if self.mode == 'fine_tune':
            return f"episode_log_r{self.robot_id}_fine_tune"
        if self.mode == 'locator_train':
            return f"episode_log_r{self.robot_id}_locator_train"
        if self.mode == 'inference' and self.inference_mode == 'phase':
            return f"episode_log_r{self.robot_id}_phase{self.fixed_phase}"
        return f"episode_log_r{self.robot_id}"

    def refresh_episode_log_paths(self):
        """Set Excel log path under VR-DT-DRL/data (phase-specific name for --phase inference)."""
        stem = self._episode_log_stem()
        self._episode_xlsx_path = self._episode_log_dir / f"{stem}.xlsx"
        print(f"[LOG R{self.robot_id}] XLSX → {self._episode_xlsx_path.resolve()}")

    def _episode_log_columns(self) -> List[str]:
        if self.mode == 'locator_train':
            return LOCATOR_EPISODE_LOG_COLUMNS
        return EPISODE_LOG_COLUMNS

    def _episode_log_xlsx_headers(self) -> List[str]:
        cols = self._episode_log_columns()
        return (
            ['timestamp', EPISODE_LOG_TIMESTAMP_LOCAL_COL]
            + [c for c in cols if c != 'timestamp']
        )

    def _reset_episode_log_fields(self):
        """Clears per-episode fields before a new grasp attempt."""
        self._episode_log_fields = {col: '' for col in self._episode_log_columns()}

    def _refresh_xlsx_local_time_column(self, ws) -> None:
        """Rewrite column B from UTC ISO in column A (12-hour local display)."""
        for row_num in range(2, ws.max_row + 1):
            utc_val = ws.cell(row_num, 1).value
            local_dt = _iso_utc_to_local_naive(str(utc_val) if utc_val is not None else '')
            cell = ws.cell(row_num, 2, local_dt if local_dt is not None else '')
            cell.number_format = EPISODE_LOG_XLSX_LOCAL_TIME_FORMAT

    def _ensure_xlsx_local_time_column(self, ws) -> None:
        """Insert column B with timestamp_local if upgrading an older log file."""
        if ws.max_row == 0:
            return
        if ws.cell(1, 2).value != EPISODE_LOG_TIMESTAMP_LOCAL_COL:
            ws.insert_cols(2)
            ws.cell(1, 2, EPISODE_LOG_TIMESTAMP_LOCAL_COL)
            self._refresh_xlsx_local_time_column(ws)
            return
        sample = ws.cell(2, 2).value if ws.max_row >= 2 else None
        if isinstance(sample, str) and sample.startswith('='):
            self._refresh_xlsx_local_time_column(ws)

    def _write_episode_xlsx_row(self, ws, row: dict) -> None:
        """Write one data row: A=timestamp, B=local-time, C+=episode fields."""
        row_num = ws.max_row + 1
        rest_cols = [c for c in self._episode_log_columns() if c != 'timestamp']

        utc_iso = row.get('timestamp', '')
        ws.cell(row_num, 1, utc_iso)
        local_dt = _iso_utc_to_local_naive(utc_iso)
        local_cell = ws.cell(row_num, 2, local_dt if local_dt is not None else '')
        local_cell.number_format = EPISODE_LOG_XLSX_LOCAL_TIME_FORMAT
        for col_idx, col_name in enumerate(rest_cols, start=3):
            ws.cell(row_num, col_idx, _xlsx_cell_value(col_name, row.get(col_name, '')))

    def _append_episode_log_row(self, success: bool):
        """Append one episode row to the Excel workbook in VR-DT-DRL/data."""
        row = dict(self._episode_log_fields)
        row['timestamp']        = datetime.now(timezone.utc).isoformat()
        row['robot_id']         = self.robot_id
        row['episode']          = self.episode_count
        row['session_episode']  = self.session_episode_count
        row['run_mode']         = self.mode
        if self.mode == 'locator_train':
            if row.get('demo_ok') == '':
                row['demo_ok'] = int(bool(success))
        else:
            row['inference_mode']   = self.inference_mode if self.mode == 'inference' else ''
            row['curriculum_phase'] = self.curriculum.phase
            row['success']          = int(bool(success))
        if row.get('spawn_phase') == '':
            row['spawn_phase'] = self._spawn_phase_label()

        xlsx_path = self._episode_xlsx_path
        headers = self._episode_log_xlsx_headers()
        try:
            import openpyxl
        except ImportError as e:
            raise ImportError(
                "Episode logging requires openpyxl. Install with: pip install openpyxl"
            ) from e

        try:
            with self._episode_log_lock:
                xlsx_path.parent.mkdir(parents=True, exist_ok=True)
                if xlsx_path.exists():
                    wb = openpyxl.load_workbook(xlsx_path)
                    ws = wb.active
                    self._ensure_xlsx_local_time_column(ws)
                    for col_idx, header in enumerate(headers, start=1):
                        ws.cell(1, col_idx, header)
                else:
                    wb = openpyxl.Workbook()
                    ws = wb.active
                    ws.title = "episodes"
                    for col_idx, header in enumerate(headers, start=1):
                        ws.cell(1, col_idx, header)
                self._write_episode_xlsx_row(ws, row)
                wb.save(xlsx_path)
            print(f"[LOG R{self.robot_id}] Appended episode {self.episode_count} → {xlsx_path.resolve()}")
        except Exception as e:
            rospy.logwarn(f"[LOG R{self.robot_id}] Failed to write episode row: {e}")

    def _spawn_phase_label(self) -> str:
        if self.mode == 'fine_tune':
            return str(self._fine_tune_spawn_phase)
        if self.mode == 'locator_train':
            return str(self._fine_tune_spawn_phase)
        if self.mode != 'inference':
            return str(self.curriculum.phase)
        if self.inference_mode == 'free':
            return 'free'
        if self.inference_mode == 'phase':
            return str(self.fixed_phase)
        if self.inference_mode == 'cycle':
            return str(self._cycle_phase)
        return str(self.curriculum.phase)

    def _record_spawn_for_log(self, spawn_x: float, spawn_y: float, spawn_z: float,
                              spawn_radius_cm: Optional[float] = None):
        cx = self.curriculum.PLATFORM_CENTER_X
        cz = self.curriculum.PLATFORM_CENTER_Z
        if spawn_radius_cm is None:
            spawn_radius_cm = math.hypot(spawn_x - cx, spawn_z - cz) * 100.0
        self._episode_log_fields.update({
            'spawn_phase':       self._spawn_phase_label(),
            'spawn_x':           f'{spawn_x:.6f}',
            'spawn_y':           f'{spawn_y:.6f}',
            'spawn_z':           f'{spawn_z:.6f}',
            'spawn_radius_cm':   f'{spawn_radius_cm:.4f}',
        })

    def _record_cam_rand_for_log(self, dx: float, dy: float, dz: float,
                                 d_pitch: float, d_yaw: float, d_roll: float):
        self._episode_log_fields.update({
            'cam_delta_x_cm':        f'{dx * 100:.4f}',
            'cam_delta_y_cm':        f'{dy * 100:.4f}',
            'cam_delta_z_cm':        f'{dz * 100:.4f}',
            'cam_delta_pitch_deg':   f'{math.degrees(d_pitch):.4f}',
            'cam_delta_yaw_deg':     f'{math.degrees(d_yaw):.4f}',
            'cam_delta_roll_deg':    f'{math.degrees(d_roll):.4f}',
        })

    @staticmethod
    def _fmt_log_float(value: Optional[float], decimals: int = 6) -> str:
        if value is None:
            return ''
        return f'{float(value):.{decimals}f}'

    @staticmethod
    def _empty_platform_log() -> Dict[str, Any]:
        return {'texture': '', 'color_r': None, 'color_g': None, 'color_b': None}

    def _record_domain_rand_for_log(self, domain: Dict[str, Any]) -> None:
        """Persist bed-support shading for this episode (R1 only; shared scene)."""
        fields: Dict[str, str] = {
            'support_shade_1': self._fmt_log_float(domain.get('support_shade_1')),
            'support_shade_2': self._fmt_log_float(domain.get('support_shade_2')),
        }
        self._episode_log_fields.update(fields)

    def _spawn_block_xz(self) -> Optional[Tuple[float, float]]:
        """Spawn-time block center (world X-Z) for aim metrics — avoids post-grasp drift."""
        sx = self._episode_log_fields.get('spawn_x', '')
        sz = self._episode_log_fields.get('spawn_z', '')
        if sx == '' or sz == '':
            return None
        try:
            return float(sx), float(sz)
        except (TypeError, ValueError):
            return None

    def _object_pos_for_locator_label(self, duck_node) -> List[float]:
        """World X,Z for aux training — spawn-first, pre-grasp fallback."""
        xz = self._spawn_block_xz()
        if xz is not None:
            return [xz[0], xz[1]]
        if duck_node:
            p = duck_node.getPosition()
            return [float(p[0]), float(p[2])]
        return [0.0, 0.0]

    def _collect_locator_training_demo(self, current_state: Dict, duck_node, node_found: bool):
        """Spawn-aligned RGB-D demo for locator_train (no teacher grasp)."""
        robot_id = self.robot_id
        self.last_grasp_mode = 'locator_collect'
        spawn_xz = self._spawn_block_xz()
        label_source = 'spawn' if spawn_xz is not None else 'pre_grasp'
        obj_pos = self._object_pos_for_locator_label(duck_node)
        print(
            f"[LOCATOR-COLLECT R{robot_id}] label=({obj_pos[0]:.3f}, {obj_pos[1]:.3f}) "
            f"| source={label_source} | bucket={self._fine_tune_demo_bucket}"
        )

        demo_sent = 0
        demo_ok = 0
        response = None

        if current_state.get('rgb') and self.mode != 'inference':
            spawn_x_log = self._episode_log_fields.get('spawn_x', '')
            spawn_z_log = self._episode_log_fields.get('spawn_z', '')
            train_payload = {
                'state':      current_state,
                'action':     [0.0] * 6,
                'reward':     0.0,
                'next_state': current_state,
                'done':       True,
                'mode':       'locator_collect',
                'object_pos': obj_pos,
                'spawn_phase':      self._fine_tune_spawn_phase,
                'spawn_x':          float(spawn_x_log) if spawn_x_log != '' else 0.0,
                'spawn_z':          float(spawn_z_log) if spawn_z_log != '' else 0.0,
                'quadrant':         self._fine_tune_spawn_quadrant,
                'demo_bucket':      self._fine_tune_demo_bucket,
                'spawn_collection': self._fine_tune_spawn_collection,
            }
            response = self._send_message_to_host({
                'type':     'training_data',
                'source':   'simulation',
                'robot_id': robot_id,
                'data':     train_payload,
            })
            if response and response.get('type') == 'training_ack':
                demo_sent = 1
                demo_ok = 1

        locator_fields: Dict[str, Any] = {
            'grasp_mode':   'locator_collect',
            'label_x_m':    self._fmt_log_float(obj_pos[0]),
            'label_z_m':    self._fmt_log_float(obj_pos[1]),
            'label_source': label_source,
            'object_found': int(bool(node_found)),
            'demo_sent':    demo_sent,
            'demo_ok':      demo_ok,
        }
        if response and response.get('type') == 'training_ack':
            pred_x = response.get('pred_obj_x')
            pred_z = response.get('pred_obj_z')
            if pred_x is not None and pred_z is not None:
                locator_fields['pred_obj_x_m'] = self._fmt_log_float(pred_x)
                locator_fields['pred_obj_z_m'] = self._fmt_log_float(pred_z)
            err = response.get('locator_err_m')
            if err is not None:
                locator_fields['locator_err_m'] = self._fmt_log_float(err)
            loc_step = response.get('loc_step')
            if loc_step is not None:
                locator_fields['loc_step_at_pred'] = int(loc_step)

        self._episode_log_fields.update(locator_fields)
        self._end_episode_and_restart(success=bool(demo_ok))

    def _compute_lateral_aim_metrics(
        self,
        block_node,
        clamp_pose: Optional[List[float]],
        bc_pose: Optional[List[float]] = None,
        block_xz: Optional[Tuple[float, float]] = None,
    ):
        """Oracle horizontal aim error (world X-Z) vs block center — sim only."""
        if clamp_pose is None:
            return None, None, None
        if block_xz is not None:
            bx, bz = block_xz
        elif block_node is not None:
            bp = block_node.getPosition()
            bx, bz = float(bp[0]), float(bp[2])
        else:
            return None, None, None
        aim_err = math.hypot(float(clamp_pose[0]) - bx, float(clamp_pose[2]) - bz)
        bc_err = None
        improve = None
        if bc_pose is not None and len(bc_pose) >= 3:
            bc_err = math.hypot(float(bc_pose[0]) - bx, float(bc_pose[2]) - bz)
            improve = bc_err - aim_err
        return aim_err, bc_err, improve

    def _record_grasp_for_log(self, *, grasp_mode: str, raw_pose: Optional[List[float]],
                              clamp_pose: Optional[List[float]], success: bool,
                              lift_delta: Optional[float], closest_dist: float,
                              reward: float, object_found: bool,
                              lateral_aim_err: Optional[float] = None,
                              lateral_aim_bc_err: Optional[float] = None,
                              lateral_improve: Optional[float] = None):
        outcome_class = classify_outcome(
            success=success,
            grasp_mode=grasp_mode,
            object_found=object_found,
            closest_dist_m=closest_dist,
            lifted_m=lift_delta,
        )
        clamp_limited = int(is_clamp_limited(raw_pose, clamp_pose))
        fields = {
            'grasp_mode':      grasp_mode,
            'success':         int(bool(success)),
            'lifted_m':        '' if lift_delta is None else f'{lift_delta:.6f}',
            'closest_dist_m':  f'{closest_dist:.6f}',
            'reward':          f'{reward:.6f}',
            'object_found':    int(bool(object_found)),
            'outcome_class':   outcome_class,
            'clamp_limited':   clamp_limited,
        }
        if lateral_aim_err is not None:
            fields['lateral_aim_err_m'] = f'{float(lateral_aim_err):.6f}'
        if lateral_aim_bc_err is not None:
            fields['lateral_aim_bc_err_m'] = f'{float(lateral_aim_bc_err):.6f}'
        if lateral_improve is not None:
            fields['lateral_improve_m'] = f'{float(lateral_improve):.6f}'
        if raw_pose is not None:
            for i, val in enumerate(raw_pose[:6]):
                fields[f'ai_pose_{i}'] = f'{float(val):.6f}'
        if clamp_pose is not None:
            for i, val in enumerate(clamp_pose[:3]):
                fields[f'clamp_pose_{i}'] = f'{float(val):.6f}'
        self._episode_log_fields.update(fields)

    def _record_residual_for_log(self, delta: List[float]):
        if len(delta) >= 3:
            self._episode_log_fields.update({
                'residual_dx':   f'{float(delta[0]):.6f}',
                'residual_dz':   f'{float(delta[1]):.6f}',
                'residual_dyaw': f'{float(delta[2]):.6f}',
            })

    def _cache_camera_base_poses(self):
        """
        Caches the initial camera poses. 
        This ensures perturbations are relative to the original setup, avoiding drift.
        """
        robot = self.webots_bridge.shared_robot
        if robot is None:
            self._cam_base = {}
            return

        groups = {
            1: ["realsense_color",  "realsense_range"],
            2: ["realsense_color2", "realsense_range2"],
        }
        self._cam_base = {}
        for rid, defs in groups.items():
            entries = []
            ok = True
            for def_name in defs:
                node = robot.getFromDef(def_name)
                if node is None:
                    print(f"[CAM CACHE] DEF '{def_name}' not found. Disabling camera noise for robot {rid}")
                    ok = False
                    break
                base_rot   = list(node.getField('rotation').getSFRotation())
                base_trans = list(node.getField('translation').getSFVec3f())
                entries.append((def_name, node, base_trans, base_rot))
            if ok:
                self._cam_base[rid] = entries
                print(f"[CAM CACHE] Robot {rid} poses cached: {defs}")

    def _randomize_camera_poses(self):
        """
        Applies a randomized spatial offset (translation and rotation) to the robot's cameras 
        each episode to improve model robustness. Applies identically across RGB/Depth pairs.
        """
        from scipy.spatial.transform import Rotation as Rot

        entries = getattr(self, '_cam_base', {}).get(self.robot_id)
        if not entries:
            return

        def signed(lo, hi):
            return float(np.random.uniform(lo, hi) * np.random.choice([-1.0, 1.0]))

        # =========================================================================
        # CAMERA PERTURBATION SETTINGS
        # Edit these variables to adjust the domain randomization properties.
        # =========================================================================
        
        # --- Translation Offsets (Meters) ---
        dx = signed(0.0, 0.015)
        dy = signed(0.0, 0.015)
        dz = signed(0.0, 0.008)

        # --- Rotation Offsets (Degrees) ---
        pitch_max_deg = 1.0   # Up/down tilt variance
        yaw_max_deg   = 0.5   # Left/right pan variance
        roll_max_deg  = 0.35  # In-plane roll variance (set to 0.0 to disable)

        # =========================================================================

        d_pitch = signed(0.0, np.deg2rad(pitch_max_deg)) if pitch_max_deg > 0 else 0.0
        d_yaw   = signed(0.0, np.deg2rad(yaw_max_deg))   if yaw_max_deg   > 0 else 0.0
        d_roll  = signed(0.0, np.deg2rad(roll_max_deg))  if roll_max_deg  > 0 else 0.0

        delta_rot = Rot.from_euler('xyz', [d_pitch, d_yaw, d_roll])

        for def_name, node, base_trans, base_rot_aa in entries:
            # Apply translation
            node.getField('translation').setSFVec3f([
                base_trans[0] + dx,
                base_trans[1] + dy,
                base_trans[2] + dz,
            ])

            # Apply local rotation matrix
            ax, ay, az, angle = base_rot_aa
            nominal  = Rot.from_rotvec(np.array([ax, ay, az]) * angle)
            combined = nominal * delta_rot          
            rotvec   = combined.as_rotvec()
            new_angle = float(np.linalg.norm(rotvec))
            
            if new_angle < 1e-9:
                new_axis  = [0.0, 1.0, 0.0]
                new_angle = 0.0
            else:
                new_axis = (rotvec / new_angle).tolist()
            node.getField('rotation').setSFRotation(new_axis + [new_angle])

        print(f"[CAM RAND R{self.robot_id}] "
              f"Δxyz=({dx*100:.2f},{dy*100:.2f},{dz*100:.2f}) cm  "
              f"Δpitch={np.rad2deg(d_pitch):.2f}°  "
              f"Δyaw={np.rad2deg(d_yaw):.2f}°  "
              f"Δroll={np.rad2deg(d_roll):.2f}°")
        self._record_cam_rand_for_log(dx, dy, dz, d_pitch, d_yaw, d_roll)

    # =========================================================================
    # HARDWARE INITIALIZATION & CONTROL (--real mode)
    # =========================================================================

    def _init_realsense(self):
        """Initializes direct PyRealSense2 hardware stream."""
        if not REALSENSE_AVAILABLE:
            rospy.logerr("[REAL] pyrealsense2 not installed - cannot run in real mode")
            raise RuntimeError("pyrealsense2 required for --real mode")

        self._rs_pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 360, rs.format.z16,  30)
        profile = self._rs_pipeline.start(cfg)

        depth_sensor      = profile.get_device().first_depth_sensor()
        self._rs_depth_scale = depth_sensor.get_depth_scale()
        self._rs_align    = rs.align(rs.stream.color)

        self._rs_spatial  = rs.spatial_filter()
        self._rs_temporal = rs.temporal_filter()
        self._rs_holefill = rs.hole_filling_filter()

        # Warm up buffer
        for _ in range(30):
            self._rs_pipeline.wait_for_frames()
        rospy.loginfo("[REAL] RealSense D455 ready")

    def _capture_realsense(self):
        """Polls and processes the next RealSense frame."""
        frames   = self._rs_pipeline.wait_for_frames(timeout_ms=5000)
        aligned  = self._rs_align.process(frames)
        c_frame  = aligned.get_color_frame()
        d_frame  = aligned.get_depth_frame()
        if not c_frame or not d_frame:
            return None, None
        d_frame = self._rs_spatial.process(d_frame)
        d_frame = self._rs_temporal.process(d_frame)
        d_frame = self._rs_holefill.process(d_frame)
        rgb   = np.asanyarray(c_frame.get_data())
        depth = np.asanyarray(d_frame.get_data()).astype(np.float32) * self._rs_depth_scale
        return rgb, depth

    def _init_ros_camera(self):
        """
        Subscribes to external ROS camera nodes instead of direct SDK access.
        Used primarily for headless deployments (e.g. Raspberry Pi).
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("[ROS CAM] ROS is not available — cannot use --ros-camera mode")

        ns = '/camera2' if self.robot_id == 2 else '/camera'

        self._ros_cam_lock  = threading.Lock()
        self._ros_rgb_frame  = None
        self._ros_depth_frame = None
        self._ros_cam_ready  = False

        rospy.Subscriber(f'{ns}/color/image_raw', Image, self._ros_rgb_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(f'{ns}/aligned_depth_to_color/image_raw', Image, self._ros_depth_cb, queue_size=1, buff_size=2**24)
        
        rospy.loginfo(f"[ROS CAM R{self.robot_id}] Waiting for first frames on {ns}...")
        deadline = time.time() + 15.0
        rate = rospy.Rate(10)
        while not self._ros_cam_ready and not rospy.is_shutdown():
            if time.time() > deadline:
                raise RuntimeError(
                    f"[ROS CAM R{self.robot_id}] Timed out waiting for camera on {ns}. "
                    "Is realsense2_camera running?"
                )
            rate.sleep()
        rospy.loginfo(f"[ROS CAM R{self.robot_id}] Camera ready on {ns}")

    @staticmethod
    def _imgmsg_to_numpy(msg, encoding):
        """Decodes ROS sensor_msgs/Image to numpy arrays without cv_bridge dependency."""
        dtype_map = {
            'rgb8':   (np.uint8,  3),
            'bgr8':   (np.uint8,  3),
            'mono8':  (np.uint8,  1),
            '8UC1':   (np.uint8,  1),
            '8UC3':   (np.uint8,  3),
            '16UC1':  (np.uint16, 1),
            '32FC1':  (np.float32, 1),
        }
        if encoding not in dtype_map:
            raise ValueError(f"Unsupported encoding: {encoding}")
        dtype, channels = dtype_map[encoding]
        frame = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
        if channels == 1:
            frame = frame[:, :, 0]
        if encoding == 'rgb8':
            frame = frame[:, :, ::-1].copy()
        return frame

    def _ros_rgb_cb(self, msg):
        """Store the latest RGB frame from the ROS topic."""
        try:
            enc = msg.encoding if msg.encoding else 'rgb8'
            frame = self._imgmsg_to_numpy(msg, enc)
            
            with self._ros_cam_lock:
                self._ros_rgb_frame = frame
                if self._ros_depth_frame is not None:
                    self._ros_cam_ready = True
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[ROS CAM R{self.robot_id}] RGB decode error: {e}")

    def _ros_depth_cb(self, msg):
        """Store the latest depth frame from the ROS topic."""
        try:
            enc = msg.encoding if msg.encoding else '16UC1'
            raw = self._imgmsg_to_numpy(msg, enc)
            depth_m = raw.astype(np.float32) / 1000.0
            with self._ros_cam_lock:
                self._ros_depth_frame = depth_m
                if self._ros_rgb_frame is not None:
                    self._ros_cam_ready = True
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[ROS CAM R{self.robot_id}] Depth decode error: {e}")

    def _capture_ros_camera(self):
        """Fetches the latest async cached frame from ROS subscribers."""
        with self._ros_cam_lock:
            rgb   = self._ros_rgb_frame
            depth = self._ros_depth_frame
        if rgb is None or depth is None:
            return None, None
        return rgb.copy(), depth.copy()

    def _init_real_robot_motion(self):
        """Establishes actionlib client connections to UR3e trajectory hardware."""
        self.robot_controller, self.gripper_controller, self.motion_planner = \
            create_robot_system(
                config_path="config/robot_config.yaml",
                simulation=False,    
                webots_bridge=None,
                robot_id=self.robot_id
            )

        self._traj_client = None
        if not ACTIONLIB_AVAILABLE:
            rospy.logwarn("[REAL] actionlib not available - motion will be stubbed")
            return

        ns = '/ur3_robot2' if self.robot_id == 2 else ''
        UR3_ACTION = f'{ns}/scaled_pos_joint_traj_controller/follow_joint_trajectory'
        self._traj_client = actionlib.SimpleActionClient(UR3_ACTION, FollowJointTrajectoryAction)
        
        if self._traj_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.loginfo("[REAL] UR3e trajectory action server connected")
        else:
            rospy.logwarn("[REAL] Could not connect to trajectory server.")
            self._traj_client = None

        js_topic = '/ur3_robot2/joint_states' if self.robot_id == 2 else '/joint_states'
        rospy.Subscriber(js_topic, JointState, self._real_joint_state_cb)

    def _real_joint_state_cb(self, msg):
        """Synchronizes controller state with physical joint positions."""
        NAMES = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                 'wrist_1_joint','wrist_2_joint','wrist_3_joint']
        positions = dict(zip(msg.name, msg.position))
        joints = [positions.get(n, 0.0) for n in NAMES]
        self.robot_controller.joints_state = joints

    def _send_real_joints(self, joints: List[float], duration: float):
        """
        Translates waypoints to joint trajectory payloads and dispatches them 
        to the real UR hardware. Blocks execution until hardware arrival.
        """
        NAMES = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                 'wrist_1_joint','wrist_2_joint','wrist_3_joint']

        wait_timeout = duration * 25.0 + 15.0

        if self._traj_client is None:
            rospy.logwarn(f"[REAL STUB] Move joints {[round(j,3) for j in joints]} "
                          f"(duration {duration:.1f}s)")
            rospy.sleep(duration)
            return True

        goal = FollowJointTrajectoryGoal()
        from trajectory_msgs.msg import JointTrajectory
        import actionlib
        traj = JointTrajectory()
        traj.joint_names = NAMES
        pt = JointTrajectoryPoint()
        pt.positions       = joints
        pt.velocities      = [0.0] * 6
        pt.time_from_start = rospy.Duration(duration)
        traj.points = [pt]
        goal.trajectory = traj

        self._traj_client.send_goal(goal)
        finished = self._traj_client.wait_for_result(timeout=rospy.Duration(wait_timeout))

        if not finished:
            rospy.logerr("[REAL] Trajectory timed out. Cancelling goal.")
            self._traj_client.cancel_goal()
            return False

        if self._traj_client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.logwarn("[REAL] Trajectory Action failed. Check robot emergency status.")
            return False

        return True

    GRIPPER_CTRL_REV = 2  # bump when gripper logic changes (check startup log on VM)

    def _gripper_ns(self) -> str:
        return '/ur3e_robot2' if self.robot_id == 2 else '/ur3e_robot1'

    def _make_gripper_cmd(self, rACT=1, rGTO=1, rATR=0, rPR=0, rSP=255, rFR=150):
        cmd = RobotiqOutput.Robotiq2FGripper_robot_output()
        cmd.rACT = rACT
        cmd.rGTO = rGTO
        cmd.rATR = rATR
        cmd.rPR  = rPR
        cmd.rSP  = rSP
        cmd.rFR  = rFR
        return cmd

    def _wait_gripper_subscriber(self, timeout: float = 5.0) -> bool:
        """RTU node must be subscribed before Output messages take effect."""
        if self._gripper_pub is None:
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._gripper_pub.get_num_connections() > 0:
                return True
            rospy.sleep(0.1)
        return False

    def _publish_gripper_cmd(self, rACT=1, rGTO=1, rATR=0, rPR=0, rSP=255, rFR=150,
                             repeats: int = 15, period: float = 0.1):
        """Stream gripper commands (Robotiq RTU needs repeated publishes, not a single msg)."""
        if self._gripper_pub is None:
            return
        if not self._wait_gripper_subscriber(2.0):
            rospy.logwarn(
                f"[REAL] No subscriber on {self._gripper_ns()}/Robotiq2FGripperRobotOutput "
                f"— is gripper_node running?"
            )
        for _ in range(repeats):
            self._gripper_pub.publish(
                self._make_gripper_cmd(rACT=rACT, rGTO=rGTO, rATR=rATR, rPR=rPR, rSP=rSP, rFR=rFR))
            rospy.sleep(period)

    def _log_gripper_status(self, label: str):
        if self._gripper_status is None:
            rospy.logwarn(f"[REAL] {label}: no gripper status (wrong namespace or node down?)")
            return
        s = self._gripper_status
        rospy.loginfo(
            f"[REAL] {label}: gSTA={s.gSTA} gFLT={s.gFLT} gOBJ={s.gOBJ} gPR={s.gPR} gPO={s.gPO}"
        )

    def _init_robotiq_gripper(self):
        """Bootstraps Robotiq gripper communications over ROS."""
        self._gripper_ready  = False
        self._gripper_status = None
        self._gripper_pub    = None

        if not ROBOTIQ_AVAILABLE or not ROS_AVAILABLE:
            rospy.logwarn("[REAL] Robotiq package not found — gripper stubbed.")
            return

        ns = self._gripper_ns()
        out_topic = f'{ns}/Robotiq2FGripperRobotOutput'

        self._gripper_pub = rospy.Publisher(
            out_topic,
            RobotiqOutput.Robotiq2FGripper_robot_output,
            queue_size=10,
            latch=False)

        rospy.Subscriber(
            f'{ns}/Robotiq2FGripperRobotInput',
            RobotiqInput.Robotiq2FGripper_robot_input,
            self._gripper_status_cb)

        rospy.loginfo(
            f"[REAL] Gripper ctrl rev {self.GRIPPER_CTRL_REV} | pub {out_topic}"
        )

        rospy.sleep(2.0)
        if not self._wait_gripper_subscriber(5.0):
            rospy.logwarn("[REAL] gripper_node not subscribed yet — start it before grasping")

        if self._gripper_status and self._gripper_status.gSTA == 3 and self._gripper_status.gFLT == 0:
            self._gripper_ready = True
            rospy.loginfo("[REAL] Gripper already activated (gSTA=3)")
            return

        # Do not hardware-reset here — external gripper_node may already be active.
        self._ensure_gripper_activated()
    
    def _gripper_status_cb(self, msg):
        """Updates internal status based on physical gripper feedback."""
        self._gripper_status = msg
        if msg.gACT == 1 and msg.gSTA == 3 and msg.gFLT == 0:
            self._gripper_ready = True

    def _ensure_gripper_activated(self):
        """Re-activate if gSTA dropped below ready (common after e-stop or client restart)."""
        if self._gripper_pub is None:
            rospy.logwarn("[REAL] Gripper publisher missing — cannot activate")
            return False
        if self._gripper_status and self._gripper_status.gSTA == 3 and self._gripper_status.gFLT == 0:
            self._gripper_ready = True
            return True
        rospy.loginfo("[REAL] Gripper not ready — streaming activate...")
        self._publish_gripper_cmd(rACT=1, rGTO=1, rATR=0, rPR=0, rSP=255, rFR=150, repeats=30)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._gripper_status and self._gripper_status.gSTA == 3 and self._gripper_status.gFLT == 0:
                self._gripper_ready = True
                rospy.loginfo("[REAL] Gripper activated ✓")
                return True
            rospy.sleep(0.1)
        rospy.logwarn("[REAL] Gripper activate failed before grasp")
        return False

    def _gripper_open(self):
        """Sends open command to the physical Robotiq gripper."""
        if self._gripper_pub is None:
            rospy.logwarn("[REAL] Gripper open skipped — no publisher")
            return
        if not self._ensure_gripper_activated():
            return
        self._publish_gripper_cmd(rACT=1, rGTO=1, rATR=0, rPR=0, rSP=255, rFR=150, repeats=15)
        rospy.sleep(0.5)

    def _gripper_close(self):
        """Sends close command to the physical Robotiq gripper."""
        if self._gripper_pub is None:
            rospy.logwarn("[REAL] Gripper close skipped — no publisher")
            return
        if not self._ensure_gripper_activated():
            self._log_gripper_status("close aborted")
            return
        self._log_gripper_status("before close")
        rospy.loginfo(
            f"[REAL] Streaming close on {self._gripper_ns()}/Robotiq2FGripperRobotOutput "
            f"(subs={self._gripper_pub.get_num_connections()})"
        )
        self._publish_gripper_cmd(rACT=1, rGTO=1, rATR=0, rPR=255, rSP=255, rFR=150, repeats=25)
        rospy.sleep(1.5)
        self._log_gripper_status("after close")

    def _gripper_reactivate(self):
        """Ensures gripper logic is synced to physical state per episode."""
        if self._gripper_pub is None:
            return
        self._ensure_gripper_activated()

    def _gripper_grasped(self) -> bool:
        """Determines grasp success from physical gripper feedback."""
        if self._gripper_status is None:
            return False
        return self._gripper_status.gOBJ in (1, 2)

    def _execute_real_grasp(self, prediction: Dict):
        """
        Executes a grasp sequence on the physical hardware based on network output.
        Safeguards execution via IK filtering and static height geometry limits.
        """
        import math as _m

        raw_pose = list(prediction['pose'])
        rospy.loginfo(f"[REAL] Network output: {raw_pose}")

        # Webots coordinate bounds filtering
        pose = raw_pose.copy()
        pose[0] = float(np.clip(pose[0], -0.862, -0.578))   
        pose[1] = float(np.clip(pose[1],  0.420,  0.460))   
        pose[2] = float(np.clip(pose[2],  0.65,  0.972))    
        
        if any(abs(raw_pose[i] - pose[i]) > 0.001 for i in range(3)):
            rospy.loginfo(f"[REAL CLAMP] {raw_pose[:3]} → {pose[:3]}")

        x, y, z = pose[0], pose[1], pose[2]
        yaw     = pose[5]

        # Convert to UR3 Base Frame
        ik_x, ik_y, _ = self.robot_controller.transform_real_to_ur3(x, y, z)

        # =========================================================================
        # REAL HARDWARE SAFETY LIMITS
        # =========================================================================
        PLATFORM_Z   = 0.068   
        FLOOR_MARGIN = 0.050   
        GRIPPER_OFF  = 0.115 # Gripper target height above platform
        HOVER_OFF    = 0.08    

        target_z = PLATFORM_Z + FLOOR_MARGIN        
        grasp_z  = target_z   + GRIPPER_OFF         
        hover_z  = grasp_z    + HOVER_OFF           
        safe_z   = hover_z    + 0.05                
        
        WRIST_ANGLE = _m.pi   
        # =========================================================================

        def _wrap(a):
            return min(abs(a), 2 * _m.pi - abs(a))

        def solve_wp(tx, ty, tz):
            """Analytically computes IK with heuristic filtering for safe paths."""
            R_down = np.array([[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]])
            cy, sy = _m.cos(yaw), _m.sin(yaw)
            R_yaw  = np.array([[cy, -sy, 0], [sy,  cy, 0], [ 0,   0, 1]])
            T           = np.eye(4)
            T[:3, 3]    = [tx, ty, tz]
            T[:3, :3]   = R_yaw @ R_down

            sols = self.robot_controller._solve_ik_analytical(T)
            if not sols:
                return None

            J0_SAFE_MIN, J0_SAFE_MAX = math.radians(35), math.radians(95)
            J0_FIND_MIN, J0_FIND_MAX = math.radians(30), math.radians(130)
            
            valid = [s for s in sols
                     if J0_FIND_MIN < s[0] < J0_FIND_MAX
                     and s[1] < 0.0
                     and s[2] > 0.0]
                     
            if not valid:
                rospy.logerr("[REAL] No viable elbow-down IK solution in range.")
                return None

            cur = np.array(self.robot_controller.joints_state)
            best, best_s = None, float('inf')
            for s in valid:
                sc = (np.linalg.norm(np.array(s) - cur)
                      + 20.0 * _wrap(s[1] - cur[1])   
                      + 20.0 * _wrap(s[2] - cur[2])   
                      + 50.0 * _wrap(s[3] - cur[3])   
                      + 50.0 * _wrap(s[4] - cur[4]))  
                if sc < best_s:
                    best_s, best = sc, s

            best = list(best)
            best[0] = float(np.clip(best[0], J0_SAFE_MIN, J0_SAFE_MAX))
            best[5] = WRIST_ANGLE   
            return best

        j_safe  = solve_wp(ik_x, ik_y, safe_z)
        j_hover = solve_wp(ik_x, ik_y, hover_z)
        j_grasp = solve_wp(ik_x, ik_y, grasp_z)

        if not all([j_safe, j_hover, j_grasp]):
            rospy.logerr("[REAL] IK generation failed — skipping grasp")
            return

        HOME = list(self.robot_controller.get_home_joints(simulation=False))
        HOME[5] = WRIST_ANGLE  

        JOINT_SPEED  = 0.5   
        MIN_DURATION = 1.5   

        def duration_for(j_from, j_to):
            dist = np.linalg.norm(np.array(j_to) - np.array(j_from))
            return float(max(MIN_DURATION, dist / JOINT_SPEED))

        rospy.loginfo("[REAL] Re-activating gripper...")
        self._gripper_reactivate()
        self._gripper_open()

        # Step 1: Hover Phase
        rospy.loginfo("[REAL] → Hover")
        self._send_real_joints(j_hover, duration_for(j_safe, j_hover))

        # Step 2: Wrist Compensation Matrix Alignment
        HOME_BASE_ANGLE  = _m.pi / 2   
        HOME_WRIST_ANGLE = _m.pi       
        base_delta = j_hover[0] - HOME_BASE_ANGLE
        wrist_compensated = HOME_WRIST_ANGLE + base_delta

        j_hover_comp = list(j_hover)
        j_hover_comp[5] = wrist_compensated
        self._send_real_joints(j_hover_comp, duration_for(j_hover, j_hover_comp))

        # Step 3: Descend Phase
        j_grasp_comp = list(j_grasp)
        j_grasp_comp[5] = wrist_compensated
        rospy.loginfo("[REAL] → Descend")
        self._send_real_joints(j_grasp_comp, duration_for(j_hover_comp, j_grasp_comp))
        rospy.sleep(0.5)

        # Step 4: Actuate Gripper (re-activate after long arm moves)
        self._ensure_gripper_activated()
        rospy.loginfo("[REAL] → Closing gripper")
        self._gripper_close()
        rospy.sleep(1.0)

        success = self._gripper_grasped()
        rospy.loginfo(f"[REAL] Grasp {'SUCCESS ✓' if success else 'FAIL ✗'}")

        # Step 5: Retreat & Reset Phase
        rospy.loginfo("[REAL] → Lift straight up")
        self._send_real_joints(j_safe, duration_for(j_grasp_comp, j_safe))
        
        rospy.loginfo("[REAL] → Return home")
        self._send_real_joints(HOME, duration_for(j_safe, HOME))

        rospy.sleep(1.0)
        self._gripper_open()
        return success

    # =========================================================================
    # GENERAL SIMULATION PIPELINES
    # =========================================================================

    def _load_config(self, config_path: str) -> Dict:
        """Loads client configuration or returns default parameters."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(
                f"[CONFIG] Failed to read {config_path}: {e}\n"
                f"  Fix YAML indentation. Using fallback host 127.0.0.1:8888."
            )
            return {'network': {'host_ip': '127.0.0.1', 'host_port': 8888}}

    def _setup_ros_interface(self):
        """Binds ROS image and joint state topic subscribers."""
        self.rgb_sub         = rospy.Subscriber('/camera/image_raw', Image, self._rgb_callback)
        self.depth_sub       = rospy.Subscriber('/camera/depth/image_raw', Image, self._depth_callback)
        self.joint_state_sub = rospy.Subscriber('/ur3/joint_states', JointState, self._joint_state_callback)

    def _rgb_callback(self, msg):
        self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _depth_callback(self, msg):
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def _joint_state_callback(self, msg):
        self.latest_joint_states = {'positions': list(msg.position)}

    def connect_to_host(self) -> bool:
        """Establishes connection to GPU server network component."""
        host_ip   = self.config['network']['host_ip']
        host_port = self.config['network']['host_port']
        try:
            with self.connection_lock:
                self.host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.host_socket.settimeout(30)
                self.host_socket.connect((host_ip, host_port))
                self.connected = True
            rospy.loginfo(f"Connected to GPU server at {host_ip}:{host_port}")
            return True
        except Exception as e:
            self.connected = False
            print(
                f"[NETWORK R{self.robot_id}] Cannot reach GPU server at "
                f"{host_ip}:{host_port} — {e}\n"
                f"  Start gpu_server.py first. Check config/network_config.yaml "
                f"(host_ip should be 127.0.0.1 for Windows Part 5)."
            )
            return False

    def _send_camera_data_to_host(self):
        """Serializes and dispatches local camera buffer to inference server."""
        try:
            if self.latest_rgb_image is None:
                return

            data_dir = Path("~/catkin_ws/src/vm_simulation_system/data").expanduser() #Saves image it took to your desktop
            data_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(data_dir / f"latest_camera_view_r{self.robot_id}.jpg"),
                        self.latest_rgb_image)

            _, rgb_enc = cv2.imencode('.jpg', self.latest_rgb_image)
            self.latest_rgb_b64 = base64.b64encode(rgb_enc).decode('utf-8')

            if self.latest_depth_image is not None:
                depth_mm = (self.latest_depth_image * 1000).astype(np.uint16)
                h, w = depth_mm.shape
                header = np.array([h, w], dtype=np.uint32).tobytes()
                self.latest_depth_b64 = base64.b64encode(header + depth_mm.tobytes()).decode('utf-8')
            else:
                self.latest_depth_b64 = ""

            payload = {
                'type':     'camera_data',
                'data':     {'rgb': self.latest_rgb_b64, 'depth': self.latest_depth_b64},
                'mode':     self.mode,
                'source':   'real' if self.real_robot else 'simulation',
                'robot_id': self.robot_id,
                'use_residual': self.use_residual,
                'use_geo_grasp': self.use_geo_grasp,
                'session_episode': self.session_episode_count,
            }
            
            response = self._send_message_to_host(payload)
            if response and response.get('type') == 'grasp_prediction':
                if self.real_robot:
                    success = self._execute_real_grasp(response)
                    if success is None:
                        success = False
                    
                    self.episode_count += 1
                    self.last_grasp_mode = response.get('mode', 'exploit')
                    status = chr(10003) if success else chr(10007)
                    rospy.loginfo(f"[REAL R{self.robot_id}] {status} Episode {self.episode_count} complete — waiting at barrier")
                    
                    barrier_resp = self._send_message_to_host({
                        'type':     'episode_end',
                        'success':  success,
                        'robot_id': self.robot_id,
                    })
                    
                    if barrier_resp and barrier_resp.get('type') == 'proceed':
                        rospy.loginfo(f"[REAL R{self.robot_id}] Barrier cleared — starting next capture cycle")
                    else:
                        rospy.logwarn(f"[REAL R{self.robot_id}] Unexpected barrier response: {barrier_resp}")
                else:
                    self._execute_grasp_prediction(response)
            elif response and response.get('type') == 'error':
                print(f"[GPU R{self.robot_id}] Inference error: {response.get('message')}")
            elif self.episode_active:
                print(f"[GPU R{self.robot_id}] Unexpected response: {response}")
        except Exception as e:
            rospy.logerr(f"Camera Send Error (R{self.robot_id}): {e}")

    def _send_message_to_host(self, message: Dict) -> Optional[Dict]:
        """Handles low level socket transmissions to remote servers."""
        if not self.connected:
            return None
        try:
            with self.connection_lock:
                data = json.dumps(message).encode('utf-8')
                self.host_socket.sendall(len(data).to_bytes(4, byteorder='big'))
                self.host_socket.sendall(data)

                header = self.host_socket.recv(4)
                if not header:
                    return None
                resp_size = int.from_bytes(header, byteorder='big')

                resp_data = b''
                while len(resp_data) < resp_size:
                    chunk = self.host_socket.recv(min(resp_size - len(resp_data), 4096))
                    if not chunk:
                        break
                    resp_data += chunk
                return json.loads(resp_data.decode('utf-8'))
        except:
            self.connected = False
            return None

    def _calculate_shaped_reward(self, success: bool, closest_dist: float) -> float:
        """Returns binary BC training signal: 1.0 on success, 0.0 otherwise."""
        reward = 1.0 if success else 0.0
        print(f"[REWARD] ClosestDist: {closest_dist:.4f}m | Success: {success} | Reward: {reward:.4f}")
        return float(reward)

    def _calculate_episode_reward(
        self,
        success: bool,
        closest_dist: float,
        lift_delta: Optional[float],
        delta: List[float],
        clamp_limited: bool,
        object_found: bool,
        lateral_improve: Optional[float] = None,
    ) -> float:
        if self.mode == 'rl_train':
            reward_cfg = self._rl_reward_cfg.get('reward', self._rl_reward_cfg)
            reward = calculate_rl_reward(
                robot_id=self.robot_id,
                success=success,
                closest_dist=closest_dist,
                lifted_m=lift_delta,
                delta_x=float(delta[0]),
                delta_z=float(delta[1]),
                delta_yaw=float(delta[2]),
                clamp_limited=bool(clamp_limited),
                object_found=object_found,
                reward_cfg=reward_cfg,
                max_delta=(
                    float(self._rl_reward_cfg.get('max_delta_x_m', 0.03)),
                    float(self._rl_reward_cfg.get('max_delta_z_m', 0.03)),
                    float(self._rl_reward_cfg.get('max_delta_yaw_rad', 0.17)),
                ),
                lateral_improve_m=lateral_improve,
            )
            align_note = (
                f" align={lateral_improve:+.4f}m"
                if lateral_improve is not None else ""
            )
            print(
                f"[RL REWARD R{self.robot_id}] dist={closest_dist:.4f} lift={lift_delta} "
                f"delta={delta} clamp={clamp_limited}{align_note} → {reward:.4f}"
            )
            return reward
        return self._calculate_shaped_reward(success, closest_dist)

    def _generate_guided_random_grasp(self) -> List[float]:
        """Provides an algorithmic teacher path for the 'explore' policy block."""
        robot_id = self.robot_id
        object_def = "TARGET_OBJECT2" if robot_id == 2 else "TARGET_OBJECT"
        fallback = fallback_grasp_pose(robot_id)

        try:
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor

            duck_node = supervisor.getFromDef(object_def)
            if not duck_node:
                rospy.logwarn(f"[GUIDED] {object_def} not found!")
                return fallback

            d_pos = np.array(duck_node.getPosition())

            if np.any(np.isnan(d_pos)):
                if not self._nan_reset_pending:
                    rospy.logwarn(f"[NaN GUARD R{robot_id}] Object NaN — resetting once.")
                    self._nan_reset_pending = True
                    self._reset_simulation_for_nan()
                return fallback

            pose = compute_grasp_pose_from_object_world(
                float(d_pos[0]), float(d_pos[1]), float(d_pos[2]),
                robot_id, add_jitter=True,
            )
            print(
                f"[CLIENT R{robot_id}] Target: {pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f} | "
                f"geo teacher"
            )
            return pose

        except Exception as e:
            rospy.logerr(f"Guided Random Error (R{robot_id}): {e}")
            return fallback

    def _execute_grasp_prediction(self, prediction: Dict):
        """Processes the neural network outputs inside Webots Simulation."""
        if not self.episode_active:
            return
        self.episode_active = False

        robot_id = self.robot_id
        try:
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor

            object_def = "TARGET_OBJECT2" if robot_id == 2 else "TARGET_OBJECT"
            if robot_id == 2:
                CLAMP_X = (-1.457, -1.093)
            else:
                CLAMP_X = (-0.90, -0.50)
            CLAMP_Z = (0.70, 1.05)

            duck_node  = supervisor.getFromDef(object_def)
            initial_y  = 0.0
            node_found = False
            self.last_grasp_mode = prediction.get('mode', 'exploit')

            if duck_node:
                initial_y  = duck_node.getPosition()[1]
                node_found = True

                if math.isnan(initial_y) or any(math.isnan(v) for v in duck_node.getPosition()):
                    if not self._nan_reset_pending:
                        rospy.logwarn(f"[NaN GUARD R{robot_id}] NaN at grasp start — resetting once.")
                        self._nan_reset_pending = True
                        self._reset_simulation_for_nan()
                    else:
                        rospy.logwarn(f"[NaN GUARD R{robot_id}] NaN persists — skipping episode.")
                    self._record_grasp_for_log(
                        grasp_mode='nan_abort', raw_pose=None, clamp_pose=None,
                        success=False, lift_delta=None, closest_dist=9999.0,
                        reward=0.0, object_found=node_found)
                    self._end_episode_and_restart(False)
                    return

            current_state = {'rgb': self.latest_rgb_b64, 'depth': self.latest_depth_b64}
            mode = prediction.get('mode', 'unknown')

            if self.mode == 'locator_train':
                self._collect_locator_training_demo(current_state, duck_node, node_found)
                return

            residual_delta = [0.0, 0.0, 0.0]
            if mode == 'explore':
                pose = self._generate_guided_random_grasp()
                raw_pose = None
            else:
                raw_pose = list(prediction['pose'])
                residual_delta = list(prediction.get('delta', [0.0, 0.0, 0.0]))
                if mode == 'exploit_geo':
                    pred_x = prediction.get('pred_obj_x')
                    pred_z = prediction.get('pred_obj_z')
                    print(
                        f"[GEO GRASP R{robot_id}] pred_obj=({pred_x:.3f}, {pred_z:.3f}) "
                        f"→ grasp={raw_pose[:3]}…"
                    )
                elif prediction.get('bc_pose'):
                    print(
                        f"[AI R{robot_id}] BC={prediction['bc_pose'][:3]}… "
                        f"Δ={residual_delta} → final={raw_pose[:3]}…"
                    )
                else:
                    print(f"[AI PREDICTION R{robot_id}] Network output: {raw_pose}")
                pose = raw_pose.copy()
                if self.no_workspace_clamp:
                    print(
                        f"[NO CLAMP R{robot_id}] Using raw pose (diagnostic): "
                        f"[{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}]"
                    )
                else:
                    pose[0] = float(np.clip(pose[0], CLAMP_X[0], CLAMP_X[1]))
                    pose[1] = float(np.clip(pose[1], 0.483, 0.490))
                    pose[2] = float(np.clip(pose[2], CLAMP_Z[0], CLAMP_Z[1]))

                    if raw_pose[0] != pose[0] or raw_pose[1] != pose[1] or raw_pose[2] != pose[2]:
                        print(f"[AI CLAMP R{robot_id}] [{raw_pose[0]:.3f},{raw_pose[1]:.3f},{raw_pose[2]:.3f}]"
                              f" → [{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}]")

            clamp_xyz = [pose[0], pose[1], pose[2]]
            clamp_limited_flag = bool(
                raw_pose is not None and is_clamp_limited(raw_pose, clamp_xyz)
            )
            self._record_residual_for_log(residual_delta)

            self.robot_controller._closest_approach_dist = 9999.0
            self.robot_controller.execute_grasp(pose)

            closest_dist = getattr(self.robot_controller, '_closest_approach_dist', 9999.0)

            success = False
            lift_delta = None
            if node_found:
                final_y    = duck_node.getPosition()[1]
                lift_delta = final_y - initial_y
                REQUIRED_LIFT = 0.023
                success = lift_delta > REQUIRED_LIFT
                status  = "SUCCESS" if success else "FAIL"
                print(f"[RESULT R{robot_id}] {status}. Lifted {lift_delta:.4f}m")
                if not math.isnan(lift_delta):
                    self._nan_reset_pending = False
            else:
                print(f"[RESULT R{robot_id}] FAIL. Object not found.")

            block_xz = self._spawn_block_xz()
            lateral_aim_err, lateral_bc_err, lateral_improve = self._compute_lateral_aim_metrics(
                duck_node, clamp_xyz, prediction.get('bc_pose'), block_xz=block_xz,
            )

            reward = self._calculate_episode_reward(
                success, closest_dist, lift_delta, residual_delta,
                clamp_limited_flag, node_found,
                lateral_improve=lateral_improve,
            )

            self._record_grasp_for_log(
                grasp_mode=mode, raw_pose=raw_pose, clamp_pose=clamp_xyz,
                success=success, lift_delta=lift_delta, closest_dist=closest_dist,
                reward=reward, object_found=node_found,
                lateral_aim_err=lateral_aim_err,
                lateral_aim_bc_err=lateral_bc_err,
                lateral_improve=lateral_improve,
            )
            if mode == 'exploit_geo':
                pred_x = prediction.get('pred_obj_x')
                pred_z = prediction.get('pred_obj_z')
                if pred_x is not None and pred_z is not None:
                    self._episode_log_fields['pred_obj_x_m'] = f'{float(pred_x):.6f}'
                    self._episode_log_fields['pred_obj_z_m'] = f'{float(pred_z):.6f}'
                    spawn_x_log = self._episode_log_fields.get('spawn_x', '')
                    spawn_z_log = self._episode_log_fields.get('spawn_z', '')
                    if spawn_x_log != '' and spawn_z_log != '':
                        locator_err = math.hypot(
                            float(pred_x) - float(spawn_x_log),
                            float(pred_z) - float(spawn_z_log),
                        )
                        self._episode_log_fields['locator_err_m'] = f'{locator_err:.6f}'
                        print(f"[LOCATOR R{robot_id}] err vs spawn center: {locator_err:.4f}m")
            if lateral_aim_err is not None:
                print(
                    f"[AIM R{robot_id}] lateral_err={lateral_aim_err:.4f}m"
                    + (f" bc_err={lateral_bc_err:.4f}m improve={lateral_improve:+.4f}m"
                       if lateral_bc_err is not None else "")
                )

            self.webots_bridge.step()
            self.camera_handler.update_from_webots()

            if self.camera_handler.current_rgb_frame is not None and self.mode != 'inference':
                _, r_enc = cv2.imencode('.jpg', self.camera_handler.current_rgb_frame)
                depth_mm = (self.camera_handler.current_depth_frame * 1000).astype(np.uint16)
                h, w = depth_mm.shape
                header = np.array([h, w], dtype=np.uint32).tobytes()
                d_b64 = base64.b64encode(header + depth_mm.tobytes()).decode('utf-8')

                next_state = {
                    'rgb':   base64.b64encode(r_enc).decode('utf-8'),
                    'depth': d_b64
                }

                network_action = [
                    pose[0], pose[1], pose[2],
                    3.14, 0.0, pose[5],
                ]

                obj_pos = [0.0, 0.0]
                if duck_node:
                    dp = duck_node.getPosition()
                    obj_pos = [float(dp[0]), float(dp[2])]

                train_payload = {
                    'state':      current_state,
                    'action':     network_action,
                    'reward':     reward,
                    'next_state': next_state,
                    'done':       True,
                    'mode':       mode,
                    'object_pos': obj_pos,
                }
                if self.mode == 'rl_train':
                    train_payload.update({
                        'delta':           residual_delta,
                        'residual_delta':  residual_delta,
                        'bc_pose':         prediction.get('bc_pose'),
                        'session_episode': self.session_episode_count,
                        'clamp_limited':   int(clamp_limited_flag),
                    })
                if self.mode == 'fine_tune' or self.mode == 'locator_train':
                    spawn_x_log = self._episode_log_fields.get('spawn_x', '')
                    spawn_z_log = self._episode_log_fields.get('spawn_z', '')
                    train_payload.update({
                        'spawn_phase':      self._fine_tune_spawn_phase,
                        'spawn_x':          float(spawn_x_log) if spawn_x_log != '' else 0.0,
                        'spawn_z':          float(spawn_z_log) if spawn_z_log != '' else 0.0,
                        'quadrant':         self._fine_tune_spawn_quadrant,
                        'demo_bucket':      self._fine_tune_demo_bucket,
                        'spawn_collection': self._fine_tune_spawn_collection,
                    })

                self._send_message_to_host({
                    'type':     'training_data',
                    'source':   'simulation',
                    'robot_id': robot_id,
                    'data':     train_payload,
                })

            self._end_episode_and_restart(success)

        except Exception as e:
            rospy.logerr(f"Grasp Execution Error (R{robot_id}): {e}")

    def _end_episode_and_restart(self, success: bool):
        """Records episode results and issues environment resets."""
        _agent_debug_log(
            "simulation_client.py:_end_episode_and_restart:entry",
            "episode_end_restart_begin",
            {"robot_id": self.robot_id, "episode": self.episode_count, "success": success},
            hypothesis_id="C",
        )
        self.end_current_episode(success)
        if self.mode != 'locator_train':
            self.robot_controller.home_position()
        flush_steps = 20 if self.mode == 'locator_train' else 40
        self._flush_camera_buffers(steps=flush_steps)

        print(f"[BARRIER R{self.robot_id}] Waiting for other robot to finish episode...")
        _agent_debug_log(
            "simulation_client.py:_end_episode_and_restart:pre_barrier",
            "gpu_barrier_request",
            {"robot_id": self.robot_id, "episode": self.episode_count},
            hypothesis_id="D",
        )
        response = self._send_message_to_host({
            'type':     'episode_end',
            'success':  success,
            'robot_id': self.robot_id
        })
        if response and response.get('type') == 'proceed':
            print(f"[BARRIER R{self.robot_id}] Barrier cleared — starting next episode")
        else:
            print(f"[BARRIER R{self.robot_id}] Unexpected barrier response: {response}")
        _agent_debug_log(
            "simulation_client.py:_end_episode_and_restart:post_barrier",
            "gpu_barrier_response",
            {"robot_id": self.robot_id, "response_type": (response or {}).get("type"), "connected": self.connected},
            hypothesis_id="D",
        )

        time.sleep(2.0)
        if self._episode_limit_reached(getattr(self, '_max_episodes', None)):
            label = 'session' if self.mode == 'inference' else 'total'
            print(
                f"[CLIENT R{self.robot_id}] Reached {self._max_episodes} {label} episodes. Stopping."
            )
            return
        try:
            self._begin_next_episode_serialized()
        except Exception as exc:
            _agent_debug_log(
                "simulation_client.py:_end_episode_and_restart:start_fail",
                "start_new_episode_exception",
                {"robot_id": self.robot_id, "error": repr(exc)},
                hypothesis_id="C",
            )
            raise
        self._flush_camera_buffers(steps=20)

    def _reset_simulation_for_nan(self):
        """Teleports physics entities to reset physics engines resolving NaN anomalies."""
        robot_id   = self.robot_id
        object_def = "TARGET_OBJECT2" if robot_id == 2 else "TARGET_OBJECT"
        cx = CurriculumManagerRobot2.PLATFORM_CENTER_X if robot_id == 2 else CurriculumManager.PLATFORM_CENTER_X
        cz = CurriculumManagerRobot2.PLATFORM_CENTER_Z if robot_id == 2 else CurriculumManager.PLATFORM_CENTER_Z
        try:
            print(f"[NaN GUARD R{robot_id}] Re-spawning {object_def} at platform centre...")
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor
            obj_node = supervisor.getFromDef(object_def)
            if obj_node:
                position_field = obj_node.getField("translation")
                if position_field:
                    position_field.setSFVec3f([cx, 0.461, cz])
                rotation_field = obj_node.getField("rotation")
                if rotation_field:
                    rotation_field.setSFRotation([0.0, 1.0, 0.0, 0.0])
                obj_node.resetPhysics()
                print(f"[NaN GUARD R{robot_id}] OK {object_def} re-spawned at ({cx:.3f}, 0.461, {cz:.3f})")
            else:
                print(f"[NaN GUARD R{robot_id}] WARNING: {object_def} not found during NaN recovery.")
        except Exception as e:
            rospy.logerr(f"[NaN GUARD R{robot_id}] Reset failed: {e}")

    @staticmethod
    def _discover_textures(tex_dir: str) -> list:
        """Globally fetches available texture assets for platform domain randomization."""
        import os
        EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tga'}
        tex_dir = os.path.expanduser(tex_dir)
        if not os.path.isdir(tex_dir):
            return []
        return sorted(
            os.path.join(tex_dir, f)
            for f in os.listdir(tex_dir)
            if os.path.splitext(f)[1].lower() in EXTENSIONS
        )
 
    def _apply_platform_texture(self, supervisor, tex_node_def, transform_def,
                                mat_node_def, tex_images, tex_chance=0.75) -> Dict[str, Any]:
        """Apply platform texture or colour; return applied state for domain dict."""
        import random, math, os

        log = self._empty_platform_log()
        floor_tex     = supervisor.getFromDef(tex_node_def)
        tex_transform = supervisor.getFromDef(transform_def)
        platform_mat  = supervisor.getFromDef(mat_node_def)

        if not floor_tex or not platform_mat:
            return log

        url_field = floor_tex.getField("url")

        def _set_mat_color(node, r, g, b):
            t = node.getTypeName()
            if t == "Material":
                node.getField("diffuseColor").setSFColor([r, g, b])
            elif t == "PBRAppearance":
                node.getField("baseColor").setSFColor([r, g, b])
                node.getField("roughness").setSFFloat(random.uniform(0.1, 1.0))
                node.getField("metalness").setSFFloat(random.uniform(0.0, 1.0))

        if tex_images and random.random() < tex_chance:
            img_path = random.choice(tex_images)
            if url_field.getCount() == 0:
                url_field.insertMFString(0, img_path)
            else:
                url_field.setMFString(0, img_path)

            if tex_transform:
                rotations = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
                tex_transform.getField("rotation").setSFFloat(random.choice(rotations))

            _set_mat_color(platform_mat, 1.0, 1.0, 1.0)
            log['texture'] = os.path.basename(img_path)
        else:
            if url_field.getCount() > 0:
                url_field.removeMF(0)
            if tex_transform:
                tex_transform.getField("rotation").setSFFloat(0.0)
            pr = random.uniform(0.05, 0.95)
            pg = random.uniform(0.05, 0.95)
            pb = random.uniform(0.05, 0.95)
            _set_mat_color(platform_mat, pr, pg, pb)
            log['texture'] = 'color_only'
            log['color_r'], log['color_g'], log['color_b'] = pr, pg, pb
        return log

    @staticmethod
    def _read_platform_texture_state(supervisor, tex_node_def,
                                     mat_node_def) -> Dict[str, Any]:
        """Read current platform texture/colour without changing the scene."""
        import os

        log = SimulationClient._empty_platform_log()
        floor_tex = supervisor.getFromDef(tex_node_def)
        platform_mat = supervisor.getFromDef(mat_node_def)
        if not floor_tex or not platform_mat:
            return log

        url_field = floor_tex.getField("url")
        if url_field.getCount() > 0:
            url = url_field.getMFString(0)
            log['texture'] = os.path.basename(url) if url else ''
            return log

        t = platform_mat.getTypeName()
        try:
            if t == "Material":
                c = platform_mat.getField("diffuseColor").getSFColor()
            elif t == "PBRAppearance":
                c = platform_mat.getField("baseColor").getSFColor()
            else:
                return log
            log['texture'] = 'color_only'
            log['color_r'], log['color_g'], log['color_b'] = float(c[0]), float(c[1]), float(c[2])
        except Exception:
            pass
        return log

    @staticmethod
    def _read_block_color(supervisor, obj_def: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        node = supervisor.getFromDef(obj_def)
        if not node:
            return None, None, None
        try:
            c = node.getField("baseColor").getSFColor()
            return float(c[0]), float(c[1]), float(c[2])
        except Exception:
            return None, None, None

    def _capture_domain_state_from_scene(self) -> Dict[str, Any]:
        """Snapshot domain-randomization state already present in Webots."""
        domain = self._empty_domain_log()
        try:
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor
            if supervisor is None:
                return domain

            for obj_def in ("TARGET_OBJECT", "TARGET_OBJECT2"):
                domain['block_colors'][obj_def] = self._read_block_color(supervisor, obj_def)

            for support_def, key in (("BedSupports_1", "support_shade_1"),
                                     ("BedSupports_2", "support_shade_2")):
                support_node = supervisor.getFromDef(support_def)
                if support_node:
                    c = support_node.getField("baseColor").getSFColor()
                    domain[key] = float(c[0])

            domain['floor'] = self._read_platform_texture_state(
                supervisor, "FLOOR_TEXTURE", "FLOOR_MATERIAL")
            domain['platform'] = self._read_platform_texture_state(
                supervisor, "PLATFORM_TEXTURE", "PLATFORM_MATERIAL")
            domain['platform2'] = self._read_platform_texture_state(
                supervisor, "PLATFORM_TEXTURE2", "PLATFORM_MATERIAL2")

            light_node = supervisor.getFromDef("MAIN_LIGHT")
            if light_node:
                domain['light_intensity'] = float(
                    light_node.getField("intensity").getSFFloat())
            fill_light = supervisor.getFromDef("FILL_LIGHT")
            if fill_light:
                domain['fill_light_intensity'] = float(
                    fill_light.getField("intensity").getSFFloat())
        except Exception as e:
            print(f"[DOMAIN RAND] Could not read scene state: {e}")
        return domain

    @staticmethod
    def _empty_domain_log() -> Dict[str, Any]:
        return {
            'block_colors': {
                'TARGET_OBJECT': (None, None, None),
                'TARGET_OBJECT2': (None, None, None),
            },
            'support_shade_1': None,
            'support_shade_2': None,
            'light_intensity': None,
            'fill_light_intensity': None,
            'floor': SimulationClient._empty_platform_log(),
            'platform': SimulationClient._empty_platform_log(),
            'platform2': SimulationClient._empty_platform_log(),
        }

    def _randomize_domain(self) -> Dict[str, Any]:
        """Randomize simulator visuals; return applied domain state."""
        import random, math, os

        domain = self._empty_domain_log()
        _agent_debug_log(
            "simulation_client.py:_randomize_domain:entry",
            "domain_rand_begin",
            {"robot_id": self.robot_id, "episode": self.episode_count},
            hypothesis_id="B",
        )
        try:
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor

            _pkg_root = Path(__file__).resolve().parent.parent
            _local_dataset = _pkg_root / "Webots" / "protos" / "textures" / "Dataset"
            if _local_dataset.is_dir():
                tex_dir = str(_local_dataset)
            else:
                tex_dir = os.path.expanduser(
                    "~/catkin_ws/src/vm_simulation_system/Webots/protos/textures/Dataset"
                )
            tex_images = self._discover_textures(tex_dir)
            if not tex_images:
                print("[DOMAIN RAND] No texture images found — using colour-only randomisation")

            for obj_def in ("TARGET_OBJECT", "TARGET_OBJECT2"):
                node = supervisor.getFromDef(obj_def)
                if node:
                    r, g, b = random.random(), random.random(), random.random()
                    node.getField("baseColor").setSFColor([r, g, b])
                    domain['block_colors'][obj_def] = (r, g, b)

            for support_def, key in (("BedSupports_1", "support_shade_1"),
                                     ("BedSupports_2", "support_shade_2")):
                support_node = supervisor.getFromDef(support_def)
                if support_node:
                    shade = random.uniform(0.02, 0.25)
                    support_node.getField("baseColor").setSFColor([shade, shade, shade])
                    domain[key] = shade
                    support_node.getField("roughness").setSFFloat(random.uniform(0.4, 0.85))
                    support_node.getField("metalness").setSFFloat(random.uniform(0.7, 1.0))

            domain['floor'] = self._apply_platform_texture(
                supervisor,
                tex_node_def="FLOOR_TEXTURE",
                transform_def="FLOOR_TEX_TRANSFORM",
                mat_node_def="FLOOR_MATERIAL",
                tex_images=tex_images,
                tex_chance=0.90,
            )
            domain['platform'] = self._apply_platform_texture(
                supervisor,
                tex_node_def="PLATFORM_TEXTURE",
                transform_def="PLATFORM_TEX_TRANSFORM",
                mat_node_def="PLATFORM_MATERIAL",
                tex_images=tex_images,
                tex_chance=0.90,
            )
            domain['platform2'] = self._apply_platform_texture(
                supervisor,
                tex_node_def="PLATFORM_TEXTURE2",
                transform_def="PLATFORM_TEX_TRANSFORM2",
                mat_node_def="PLATFORM_MATERIAL2",
                tex_images=tex_images,
                tex_chance=0.90,
            )

            light_node = supervisor.getFromDef("MAIN_LIGHT")
            if light_node:
                intensity = random.uniform(0.2, 4.0)
                light_node.getField("intensity").setSFFloat(intensity)
                domain['light_intensity'] = intensity
                light_node.getField("ambientIntensity").setSFFloat(random.uniform(0.05, 1.0))
                light_node.getField("color").setSFColor([
                    random.uniform(0.7, 1.0),
                    random.uniform(0.7, 1.0),
                    random.uniform(0.7, 1.0),
                ])
                if light_node.getTypeName() == "DirectionalLight":
                    light_node.getField("direction").setSFVec3f([
                        random.uniform(-1.0, 1.0),
                        random.uniform(-1.0, -0.3),
                        random.uniform(-1.0, 1.0),
                    ])

            fill_light = supervisor.getFromDef("FILL_LIGHT")
            if fill_light:
                fill_intensity = random.uniform(0.0, 2.0)
                fill_light.getField("intensity").setSFFloat(fill_intensity)
                domain['fill_light_intensity'] = fill_intensity
                fill_light.getField("ambientIntensity").setSFFloat(random.uniform(0.0, 0.5))
                fill_light.getField("color").setSFColor([
                    random.uniform(0.6, 1.0),
                    random.uniform(0.6, 1.0),
                    random.uniform(0.6, 1.0),
                ])
                if fill_light.getTypeName() == "DirectionalLight":
                    fill_light.getField("direction").setSFVec3f([
                        random.uniform(-1.0, 1.0),
                        random.uniform(-1.0, -0.1),
                        random.uniform(-1.0, 1.0),
                    ])

            camera_defs = [
                "realsense_color1",
                "realsense_color2",
                "realsense_range1",
                "realsense_range2",
            ]
            for cam_def in camera_defs:
                cam_node = supervisor.getFromDef(cam_def)
                if cam_node:
                    cam_node.getField("noise").setSFFloat(random.uniform(0.0, 0.03))

        except Exception as e:
            print(f"[DOMAIN RAND] Skipping randomization (nodes not found or error): {e}")
            _agent_debug_log(
                "simulation_client.py:_randomize_domain:error",
                "domain_rand_exception",
                {"robot_id": self.robot_id, "error": repr(e)},
                hypothesis_id="B",
            )
        _agent_debug_log(
            "simulation_client.py:_randomize_domain:exit",
            "domain_rand_done",
            {"robot_id": self.robot_id},
            hypothesis_id="B",
        )
        return domain

    def start_new_episode(self):
        _agent_debug_log(
            "simulation_client.py:start_new_episode:entry",
            "start_new_episode_begin",
            {"robot_id": self.robot_id, "prev_episode": self.episode_count},
            hypothesis_id="C",
        )
        self._reset_episode_log_fields()
        self.episode_count += 1
        self.session_episode_count += 1
        self.episode_active = True
        self.curriculum.update(self.episode_count)

        if not self.real_robot:
            if self.robot_id == 1:
                domain_log = self._randomize_domain()
                self._record_domain_rand_for_log(domain_log)
            # R2: support-shade columns stay empty — R1 logs scene rand (avoids dual-supervisor read crash)

        if not self.real_robot:
            self._randomize_camera_poses()

        self._spawn_object_at_curriculum_position()
        _agent_debug_log(
            "simulation_client.py:start_new_episode:exit",
            "start_new_episode_done",
            {"robot_id": self.robot_id, "episode": self.episode_count},
            hypothesis_id="C",
        )
        self._send_message_to_host({'type': 'episode_start',
                                    'episode':  self.episode_count,
                                    'robot_id': self.robot_id})

    def _begin_next_episode_serialized(self) -> None:
        """
        Dual-arm: R1 setup → R2 setup → both wait until all setup done before sim loop.
        Single-arm (R1 only): no extra wait.
        """
        if self.real_robot:
            self.start_new_episode()
            return

        if self.robot_id == 2:
            print(f"[SETUP BARRIER R{self.robot_id}] Waiting for R1 world setup...")
            _agent_debug_log(
                "simulation_client.py:_begin_next_episode_serialized",
                "setup_wait_request",
                {"robot_id": self.robot_id},
                hypothesis_id="D",
            )
            response = self._send_message_to_host({
                'type':     'episode_setup_wait',
                'robot_id': self.robot_id,
            })
            if not response or response.get('type') != 'proceed':
                msg = (response or {}).get('message', response)
                raise RuntimeError(f"Setup barrier failed for R2: {msg}")
            print(f"[SETUP BARRIER R{self.robot_id}] R1 setup done — starting R2 episode setup")

        self.start_new_episode()

        if self.robot_id == 1:
            print(f"[SETUP BARRIER R{self.robot_id}] World setup done — releasing R2")
        _agent_debug_log(
            "simulation_client.py:_begin_next_episode_serialized",
            "setup_done_signal",
            {"robot_id": self.robot_id, "episode": self.episode_count},
            hypothesis_id="D",
        )
        done_resp = self._send_message_to_host({
            'type':     'episode_setup_done',
            'robot_id': self.robot_id,
        })
        if not done_resp or done_resp.get('type') != 'proceed':
            msg = (done_resp or {}).get('message', done_resp)
            raise RuntimeError(f"Setup done signal failed for R{self.robot_id}: {msg}")

        _agent_debug_log(
            "simulation_client.py:_begin_next_episode_serialized",
            "setup_all_wait_request",
            {"robot_id": self.robot_id, "episode": self.episode_count},
            hypothesis_id="D",
        )
        all_resp = self._send_message_to_host({
            'type':     'episode_setup_all_wait',
            'robot_id': self.robot_id,
        })
        if not all_resp or all_resp.get('type') != 'proceed':
            msg = (all_resp or {}).get('message', all_resp)
            raise RuntimeError(f"Setup all-wait failed for R{self.robot_id}: {msg}")

        _agent_debug_log(
            "simulation_client.py:_end_episode_and_restart:post_start",
            "start_new_episode_ok",
            {"robot_id": self.robot_id, "episode": self.episode_count, "episode_active": self.episode_active},
            hypothesis_id="C",
        )

    def _spawn_fine_tune_position(self) -> Tuple[float, float, float, Optional[float]]:
        """Spawn for targeted fine-tune: weak cell or full-grid normal demo."""
        robot_id = self.robot_id
        cx = self.curriculum.PLATFORM_CENTER_X
        cz = self.curriculum.PLATFORM_CENTER_Z
        half_x = getattr(self.curriculum, 'PLATFORM_HALF_SIZE_X', 0.143675)
        half_z = getattr(self.curriculum, 'PLATFORM_HALF_SIZE_Z', 0.083675)
        in_spawn = getattr(self.curriculum, '_in_spawn_area', None)

        if np.random.random() < self._fine_tune_weak_spawn_p:
            cell = pick_random_weak_cell(robot_id, self._fine_tune_weak_regions)
            sx, sz, radius_m = sample_spawn_in_phase_quadrant(
                cell['phase'], cell['quadrant'], cx, cz, half_x, half_z, in_spawn,
            )
            self._fine_tune_demo_bucket = 'weak'
            self._fine_tune_spawn_phase = int(cell['phase'])
            self._fine_tune_spawn_collection = 'weak_cell'
            self._fine_tune_spawn_quadrant = quadrant_from_spawn(sx, sz, cx, cz)
            print(
                f"[FINE-TUNE R{robot_id}] WEAK spawn band={cell['phase']} "
                f"Q{self._fine_tune_spawn_quadrant} ({sx:.3f}, {sz:.3f}) "
                f"r={radius_m * 100:.1f}cm"
            )
            return sx, sz, radius_m * 100.0

        sx, sz, radius_cm = self._sample_full_board_spawn()
        self._fine_tune_demo_bucket = 'normal'
        self._fine_tune_spawn_phase = CurriculumManager.FULL_BOARD_PHASE
        self._fine_tune_spawn_collection = 'full_grid'
        self._fine_tune_spawn_quadrant = quadrant_from_spawn(sx, sz, cx, cz)
        print(
            f"[FINE-TUNE R{robot_id}] NORMAL full-grid ({sx:.3f}, {sz:.3f}) "
            f"Q{self._fine_tune_spawn_quadrant}"
        )
        return sx, sz, radius_cm

    def _spawn_object_at_curriculum_position(self):
        robot_id   = self.robot_id
        object_def = "TARGET_OBJECT2" if robot_id == 2 else "TARGET_OBJECT"
        try:
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor

            obj_node = supervisor.getFromDef(object_def)
            if obj_node is None:
                rospy.logwarn(f"[CURRICULUM R{robot_id}] {object_def} not found, skipping spawn.")
                return

            if self.mode == 'inference' and self.inference_mode == 'free':
                pos = obj_node.getPosition()
                print(f"[INFERENCE R{robot_id}/free] Object left at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                self._record_spawn_for_log(float(pos[0]), float(pos[1]), float(pos[2]))
                return

            spawn_radius_cm = None
            if self.mode in ('fine_tune', 'locator_train'):
                spawn_x, spawn_z, spawn_radius_cm = self._spawn_fine_tune_position()
                self._episode_log_fields['spawn_quadrant'] = str(self._fine_tune_spawn_quadrant)
                self._episode_log_fields['demo_bucket'] = self._fine_tune_demo_bucket
                self._episode_log_fields['spawn_collection'] = self._fine_tune_spawn_collection
            elif self.mode == 'inference' and self.inference_mode == 'cycle':
                spawn_x, spawn_z, spawn_radius_cm = self._get_spawn_for_phase(self._cycle_phase)
            elif self.mode == 'inference' and self.inference_mode == 'phase':
                spawn_x, spawn_z, spawn_radius_cm = self._get_spawn_for_phase(self.fixed_phase)
            else:
                spawn_x, _, spawn_z = self.curriculum.get_spawn_position()

            spawn_y = 0.461
            if self.mode in ('fine_tune', 'locator_train'):
                self._episode_log_fields['spawn_phase'] = str(self._fine_tune_spawn_phase)

            position_field = obj_node.getField("translation")
            if position_field:
                position_field.setSFVec3f([spawn_x, spawn_y, spawn_z])
            rotation_field = obj_node.getField("rotation")
            if rotation_field:
                rotation_field.setSFRotation([0.0, 1.0, 0.0, 0.0])
            obj_node.resetPhysics()
            self._record_spawn_for_log(spawn_x, spawn_y, spawn_z, spawn_radius_cm)

        except Exception as e:
            rospy.logerr(f"[CURRICULUM R{robot_id}] Spawn error: {e}")

    def _get_spawn_for_phase(self, phase_index: int) -> tuple:
        """Inference spawn for a locked curriculum phase (--phase N)."""
        cfg = CurriculumManager.PHASE_CONFIG
        phase_index = max(0, min(phase_index, len(cfg) - 1))

        if phase_index == CurriculumManager.FULL_BOARD_PHASE:
            return self._sample_full_board_spawn()

        r_min, r_max, _, _, _ = cfg[phase_index]

        cx     = self.curriculum.PLATFORM_CENTER_X
        cz     = self.curriculum.PLATFORM_CENTER_Z
        half_x = self.curriculum.PLATFORM_HALF_SIZE_X
        half_z = self.curriculum.PLATFORM_HALF_SIZE_Z

        if r_max < 0.001:
            print(f"[INFERENCE R{self.robot_id}] Phase {phase_index}: static centre ({cx:.3f}, {cz:.3f})")
            return (cx, cz, 0.0)

        angle  = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(r_min, r_max)
        sx = np.clip(cx + radius * np.cos(angle), cx - half_x, cx + half_x)
        sz = np.clip(cz + radius * np.sin(angle), cz - half_z, cz + half_z)
        print(f"[INFERENCE R{self.robot_id}] Phase {phase_index} spawn: ({sx:.3f}, {sz:.3f}) | "
              f"radius {radius*100:.1f}cm (band {r_min*100:.1f}–{r_max*100:.1f}cm)")
        return (sx, sz, radius * 100.0)

    def _sample_full_board_spawn(self) -> tuple:
        """Uniform random spawn anywhere on the usable platform (phase 5 / full board)."""
        cx = self.curriculum.PLATFORM_CENTER_X
        cz = self.curriculum.PLATFORM_CENTER_Z
        max_attempts = 200

        if isinstance(self.curriculum, CurriculumManagerRobot2):
            wx_min = self.curriculum.PLATFORM_WORLD_X_MIN
            wx_max = self.curriculum.PLATFORM_WORLD_X_MAX
            wz_min = self.curriculum.PLATFORM_WORLD_Z_MIN
            wz_max = self.curriculum.PLATFORM_WORLD_Z_MAX
            sx, sz = cx, cz
            for _ in range(max_attempts):
                candidate_x = np.random.uniform(wx_min, wx_max)
                candidate_z = np.random.uniform(wz_min, wz_max)
                if self.curriculum._in_spawn_area(candidate_x, candidate_z):
                    sx, sz = candidate_x, candidate_z
                    break
        else:
            half_x = self.curriculum.PLATFORM_HALF_SIZE_X
            half_z = self.curriculum.PLATFORM_HALF_SIZE_Z
            sx = np.random.uniform(cx - half_x, cx + half_x)
            sz = np.random.uniform(cz - half_z, cz + half_z)

        radius_cm = math.hypot(sx - cx, sz - cz) * 100.0
        print(f"[INFERENCE R{self.robot_id}] Phase {CurriculumManager.FULL_BOARD_PHASE} "
              f"(full board) spawn: ({sx:.3f}, {sz:.3f}) | radius {radius_cm:.1f}cm")
        return (sx, sz, radius_cm)

    def end_current_episode(self, success: bool):
        self.episode_active = False
        mode     = getattr(self, 'last_grasp_mode', 'explore')
        robot_id = self.robot_id

        if self.mode in ('training', 'rl_train'):
            self.curriculum.record_result(success, mode)
            advanced = self.curriculum.check_phase_advance()
            if advanced:
                print(f"[CURRICULUM R{robot_id}] Phase advanced to {self.curriculum.phase}!")
                self._send_message_to_host({
                    'type':     'reset_epsilon',
                    'value':    0.4,
                    'robot_id': robot_id
                })

        if self.mode == 'inference' and self.inference_mode == 'cycle':
            self._cycle_count_in_phase += 1
            if self._cycle_count_in_phase >= self.cycle_episodes_per_phase:
                self._cycle_count_in_phase = 0
                idx = self._cycle_phases.index(self._cycle_phase)
                idx = (idx + 1) % len(self._cycle_phases)
                self._cycle_phase = self._cycle_phases[idx]
                print(f"[INFERENCE R{robot_id}/cycle] Moving to Phase {self._cycle_phase}")

        status = "OK" if success else "FAIL"
        if self.mode == 'inference':
            print(f"[INFERENCE R{robot_id}] {status} Episode {self.episode_count} (session {self.session_episode_count}) complete")
        elif self.mode == 'fine_tune':
            print(
                f"[FINE-TUNE R{robot_id}] {status} Ep {self.episode_count} (session {self.session_episode_count}) | "
                f"bucket={self._fine_tune_demo_bucket} | "
                f"band={self._fine_tune_spawn_phase} Q{self._fine_tune_spawn_quadrant} | "
                f"grasp={mode}"
            )
        elif self.mode == 'locator_train':
            print(
                f"[LOCATOR-TRAIN R{robot_id}] {status} Ep {self.episode_count} (session {self.session_episode_count}) | "
                f"bucket={self._fine_tune_demo_bucket} | "
                f"band={self._fine_tune_spawn_phase} Q{self._fine_tune_spawn_quadrant} | "
                f"grasp={mode}"
            )
        elif self.mode == 'rl_train':
            print(
                f"[RL-TRAIN R{robot_id}] {status} Ep {self.episode_count} (session {self.session_episode_count}) | "
                f"Phase {self.curriculum.phase} | reward logged | grasp={mode}"
            )
        else:
            ai_rate     = self.curriculum.get_ai_success_rate()
            ai_attempts = len(self.curriculum.ai_recent_results)
            ai_window   = self.curriculum.PHASE_CONFIG[self.curriculum.phase][4]
            print(f"[EPISODE R{robot_id}] {status} Ep {self.episode_count} | "
                  f"Phase {self.curriculum.phase} | "
                  f"AI: {ai_rate*100:.1f}% ({ai_attempts}/{ai_window}) | Mode: {mode}")

        self._append_episode_log_row(success)

    def _flush_camera_buffers(self, steps: int = 40):
        """
        Step the sim and discard frames until the camera catches up.
        Required after the arm moves (e.g. home) — otherwise RGB/depth can show
        a stale frame from before the move.
        """
        if not self.webots_bridge:
            return
        for i in range(steps):
            if not self.webots_bridge.step():
                _agent_debug_log(
                    "simulation_client.py:_flush_camera_buffers",
                    "flush_step_failed",
                    {"robot_id": self.robot_id, "step_index": i, "total_steps": steps},
                    hypothesis_id="A",
                )
                break
            self.camera_handler.update_from_webots()
            time.sleep(0.016)
        self.latest_rgb_image   = self.camera_handler.current_rgb_frame
        self.latest_depth_image = self.camera_handler.current_depth_frame
        if self.latest_rgb_image is not None:
            print(f"[CAM R{self.robot_id}] Fresh camera frame after {steps}-step flush")
        else:
            print(
                f"[CAM R{self.robot_id}] WARNING: No frame after flush. "
                "Webots playing? Both extern controllers running?"
            )

    def _episode_limit_reached(self, max_episodes: Optional[int]) -> bool:
        """Inference: session episode count. Training: global curriculum episode."""
        if max_episodes is None:
            return False
        if self.mode == 'inference':
            return self.session_episode_count >= max_episodes
        return self.episode_count > max_episodes

    def run_simulation_loop(self, max_episodes: int = None):
        self._max_episodes = max_episodes
        if not self.connect_to_host():
            print(f"[CLIENT R{self.robot_id}] Exiting — no GPU server connection.")
            return

        if not self.real_robot:
            self.robot_controller.home_position()
            time.sleep(1.0)
            self._flush_camera_buffers()
            self._cache_camera_base_poses()
            print(f"[BARRIER R{self.robot_id}] Waiting at startup barrier...")
            response = self._send_message_to_host({
                'type':     'episode_end',
                'success':  False,
                'robot_id': self.robot_id
            })
            print(f"[BARRIER R{self.robot_id}] Startup barrier cleared")
            self._begin_next_episode_serialized()
            self._flush_camera_buffers(steps=20)
        else:
            rospy.loginfo(f'[REAL R{self.robot_id}] Moving to home position before starting...')
            home = self.robot_controller.get_home_joints(simulation=False)
            self._send_real_joints(home, duration=4.0)
            rospy.sleep(1.0)
            self._gripper_open()
            rospy.loginfo(f'[REAL R{self.robot_id}] At home — waiting at startup barrier...')
            
            startup_resp = self._send_message_to_host({
                'type':     'episode_end',
                'success':  False,
                'robot_id': self.robot_id,
            })
            rospy.loginfo(f'[REAL R{self.robot_id}] Startup barrier cleared — ready for first cycle')

        if self.real_robot:
            FLUSH_FRAMES   = 10   
            SETTLE_SLEEP   = 1.0  

            while not rospy.is_shutdown():
                if self._episode_limit_reached(max_episodes):
                    label = 'session' if self.mode == 'inference' else 'total'
                    print(f"[REAL R{self.robot_id}] Reached {max_episodes} {label} episodes. Stopping.")
                    break

                if not self.connected:
                    break

                if self.ros_camera:
                    rospy.loginfo(f"[REAL R{self.robot_id}] Settling after home move...")
                    rospy.sleep(SETTLE_SLEEP)
                    rgb, depth = self._capture_ros_camera()
                else:
                    rospy.loginfo(f"[REAL R{self.robot_id}] Flushing camera buffer before capture...")
                    for _ in range(FLUSH_FRAMES):
                        self._capture_realsense()
                    rospy.sleep(SETTLE_SLEEP)
                    rgb, depth = self._capture_realsense()

                if rgb is None:
                    rospy.logwarn(f"[REAL R{self.robot_id}] Camera returned None — retrying")
                    continue
                self.latest_rgb_image   = rgb
                self.latest_depth_image = depth

                rospy.loginfo(f"[REAL R{self.robot_id}] Captured settled frame — sending to GPU server")
                self._send_camera_data_to_host()

        else:
            rate = rospy.Rate(10)
            _cam_warn_at = 0.0
            _loop_iter = 0
            while not rospy.is_shutdown():
                if self._episode_limit_reached(max_episodes):
                    label = 'session' if self.mode == 'inference' else 'total'
                    print(f"[CLIENT R{self.robot_id}] Reached {max_episodes} {label} episodes. Stopping.")
                    _agent_debug_log(
                        "simulation_client.py:run_simulation_loop:exit",
                        "loop_exit_max_episodes",
                        {"robot_id": self.robot_id, "max_episodes": max_episodes},
                        hypothesis_id="E",
                    )
                    break

                step_ok = self.webots_bridge.step()
                _loop_iter += 1
                if not step_ok:
                    _agent_debug_log(
                        "simulation_client.py:run_simulation_loop:step_fail",
                        "webots_step_returned_false",
                        {"robot_id": self.robot_id, "episode": self.episode_count, "iter": _loop_iter},
                        hypothesis_id="A",
                    )
                    print(f"[CLIENT R{self.robot_id}] Webots step failed — simulation disconnected?")
                    break
                if _loop_iter % 500 == 0:
                    _agent_debug_log(
                        "simulation_client.py:run_simulation_loop:heartbeat",
                        "loop_alive",
                        {
                            "robot_id": self.robot_id,
                            "episode": self.episode_count,
                            "episode_active": self.episode_active,
                            "connected": self.connected,
                            "iter": _loop_iter,
                        },
                        hypothesis_id="E",
                    )
                self.camera_handler.update_from_webots()
                self.latest_rgb_image   = self.camera_handler.current_rgb_frame
                self.latest_depth_image = self.camera_handler.current_depth_frame

                if self.episode_active and self.latest_rgb_image is not None and self.connected:
                    self._send_camera_data_to_host()
                elif self.episode_active and self.latest_rgb_image is None:
                    now = time.time()
                    if now - _cam_warn_at > 5.0:
                        _cam_warn_at = now
                        print(
                            f"[CAM R{self.robot_id}] Waiting for camera frames "
                            "(Webots playing? both extern controllers running?)"
                        )
                rate.sleep()
            _agent_debug_log(
                "simulation_client.py:run_simulation_loop:exit",
                "loop_exit_shutdown",
                {"robot_id": self.robot_id, "episode": self.episode_count, "connected": self.connected},
                hypothesis_id="E",
            )


def main():
    parser = argparse.ArgumentParser(
        description='UR3 Simulation Client',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--mode', type=str, default='training',
                        help='training | fine_tune | locator_train | rl_train | inference')
    parser.add_argument('--fine-tune-config', type=str, default=None,
                        help='Path to fine_tune_config.yaml (fine_tune mode only)')
    parser.add_argument('--locator-config', type=str, default=None,
                        help='Path to locator_train_config.yaml (locator_train mode only)')
    parser.add_argument('--rl-train-config', type=str, default=None,
                        help='Path to rl_train_config.yaml (rl_train mode only)')
    parser.add_argument('--use-residual', action='store_true',
                        help='Inference: apply trained RL residual (requires gpu_server --rl-residual-r1/r2)')
    parser.add_argument('--use-geo-grasp', action='store_true',
                        help='Inference: CNN aux_position → analytic grasp geometry (requires gpu_server --geo-grasp)')
    parser.add_argument('--no-workspace-clamp', action='store_true',
                        help='Inference: skip X/Y/Z workspace clip on exploit/exploit_geo poses (diagnostic; teacher explore unchanged)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Episode cap for this process. Inference: session episodes (resets each launch).\n'
                             'Training: global curriculum episode total (resumes from curriculum_state.json).\n'
                             'Omit for infinite.')
    parser.add_argument('--real', action='store_true',
                        help='Run on real UR3e. Forces inference mode.')
    parser.add_argument('--robot-id', type=int, default=1, choices=[1, 2],
                        help='Which robot this process controls (1 or 2).\n'
                             'Must match the WEBOTS_ROBOT_NAME environment variable:\n'
                             '  Robot 1: export WEBOTS_ROBOT_NAME="ur3e_robot"\n'
                             '  Robot 2: export WEBOTS_ROBOT_NAME="ur3e_robot2"\n'
                             'Run two separate terminals, one per robot.')
    parser.add_argument('--ros-camera', action='store_true',
                        help='Use ROS image subscribers for the camera instead of\n'
                             'opening the RealSense SDK directly.  Required on a\n'
                             'Raspberry Pi where pyrealsense2 is not available, or\n'
                             'when the camera is driven by a separate\n'
                             '  roslaunch realsense2_camera rs_camera.launch\n'
                             'process.  Implies --real.')

    inf_group = parser.add_argument_group(
        'Inference sub-modes',
        'These flags only take effect when --mode inference is set.\n'
        'Only one may be used at a time.'
    )
    inf_group.add_argument('--cycle', type=int, default=None, metavar='N',
                           help='Cycle through curriculum phases, N episodes per phase.\n'
                                'Use --cycle-from / --cycle-to to limit the range.\n'
                                'Example: --mode inference --cycle 20 --cycle-from 1 --cycle-to 4')
    inf_group.add_argument('--cycle-from', type=int, default=0, metavar='N',
                           help='First phase in --cycle rotation (default 0). Requires --cycle.')
    inf_group.add_argument('--cycle-to', type=int, default=5, metavar='N',
                           help='Last phase in --cycle rotation (default 5). Requires --cycle.')
    inf_group.add_argument('--free', action='store_true',
                           help='No automatic spawning. Place the object manually.\n'
                                'The AI will attempt a grasp wherever you put it.\n'
                                'Example: --mode inference --free')
    inf_group.add_argument('--phase', type=int, default=None, metavar='N',
                           help='Lock to a specific curriculum phase (0–5).\n'
                                'Phases 0–4: spawn on a radius band; phase 5: full board.\n'
                                'Example: --mode inference --phase 5')

    args = parser.parse_args()

    is_real = args.real or args.ros_camera
    mode = 'inference' if is_real else args.mode

    if mode not in ('training', 'fine_tune', 'locator_train', 'rl_train', 'inference'):
        parser.error("--mode must be training, fine_tune, locator_train, rl_train, or inference.")

    if mode == 'rl_train' and is_real:
        parser.error("rl_train mode is simulation-only (omit --real).")

    if args.use_residual and mode != 'inference':
        parser.error("--use-residual requires --mode inference.")
    if args.use_geo_grasp and mode != 'inference':
        parser.error("--use-geo-grasp requires --mode inference.")
    if args.use_residual and args.use_geo_grasp:
        parser.error("Use either --use-residual or --use-geo-grasp, not both.")
    if args.no_workspace_clamp and mode != 'inference':
        parser.error("--no-workspace-clamp requires --mode inference (or --real).")

    inf_flags = [args.cycle is not None, args.free, args.phase is not None]
    if sum(inf_flags) > 1:
        parser.error("Only one of --cycle, --free, --phase may be used at a time.")
    if any(inf_flags) and mode != 'inference':
        parser.error("--cycle / --free / --phase require --mode inference (or --real / --ros-camera).")
    if mode == 'fine_tune' and any(inf_flags):
        parser.error("fine_tune mode does not support --cycle / --free / --phase.")
    if mode == 'locator_train' and any(inf_flags):
        parser.error("locator_train mode does not support --cycle / --free / --phase.")
    if mode == 'rl_train' and any(inf_flags):
        parser.error("rl_train mode does not support --cycle / --free / --phase.")
    if args.cycle is None and (args.cycle_from != 0 or args.cycle_to != 5):
        parser.error("--cycle-from / --cycle-to require --cycle.")

    robot_id = args.robot_id
    print(f"[STARTUP] Launching as Robot {robot_id}")
    if is_real:
        print(f"[STARTUP] Gripper ctrl rev {SimulationClient.GRIPPER_CTRL_REV} "
              f"(must match updated simulation_client.py on this machine)")

    if not is_real:
        webots_robot = "ur3e_robot2" if robot_id == 2 else "ur3e_robot"
        os.environ["WEBOTS_ROBOT_NAME"] = webots_robot
        print(f"[STARTUP] WEBOTS_ROBOT_NAME={webots_robot}")

    if not is_real and not running_inside_webots():
        print(
            "[STARTUP] Note: launched outside Webots (e.g. PowerShell). "
            "Phase 4/5 both work the same way; real cameras/motors need Webots Play."
        )

    try:
        client = SimulationClient(
            mode=mode, real_robot=is_real, robot_id=robot_id,
            ros_camera=args.ros_camera,
            use_residual=args.use_residual,
            use_geo_grasp=args.use_geo_grasp,
            no_workspace_clamp=args.no_workspace_clamp,
            rl_train_config_path=args.rl_train_config,
        )
    except Exception:
        import traceback
        print("[STARTUP] SimulationClient failed during initialization:")
        traceback.print_exc()
        raise

    print(f"[STARTUP] SimulationClient ready (mode={mode})")

    if mode == 'fine_tune' and args.fine_tune_config:
        client._load_fine_tune_settings(args.fine_tune_config)

    if mode == 'locator_train' and args.locator_config:
        client._load_locator_train_settings(args.locator_config)

    if mode == 'fine_tune':
        cfg = client._fine_tune_cfg or {}
        wr = cfg.get('sampling', {}).get('weak_ratio', 0.7)
        nr = cfg.get('sampling', {}).get('normal_ratio', 0.3)
        print(
            f"[FINE-TUNE R{robot_id}] Targeted BC collection | "
            f"batch mix {wr:.0%} weak / {nr:.0%} normal"
        )

    if mode == 'locator_train':
        cfg = client._fine_tune_cfg or {}
        wr = cfg.get('sampling', {}).get('weak_ratio', 0.7)
        nr = cfg.get('sampling', {}).get('normal_ratio', 0.3)
        print(
            f"[LOCATOR-TRAIN R{robot_id}] Supervised (X,Z) collection | "
            f"batch mix {wr:.0%} weak / {nr:.0%} normal | "
            f"start gpu_server.py with --locator-train"
        )

    if mode == 'rl_train':
        print(
            f"[RL-TRAIN R{robot_id}] TD3 residual collection | "
            f"start gpu_server.py with --rl-train and BC checkpoints"
        )
        if args.rl_train_config:
            client._load_rl_train_settings(args.rl_train_config)

    if mode == 'inference' and args.use_geo_grasp:
        print(f"[INFERENCE R{robot_id}] Geo grasp (--use-geo-grasp) | aux_position → geometry")

    if mode == 'inference' and args.no_workspace_clamp:
        print(f"[INFERENCE R{robot_id}] Workspace clamp DISABLED (--no-workspace-clamp)")

    if mode == 'inference' and args.use_residual:
        print(f"[INFERENCE R{robot_id}] BC + RL residual (--use-residual)")

    if mode == 'inference' and args.episodes is not None:
        print(f"[INFERENCE R{robot_id}] Session episode cap: {args.episodes}")

    if mode == 'inference':
        if args.cycle is not None:
            max_phase = len(CurriculumManager.PHASE_CONFIG) - 1
            if not 0 <= args.cycle_from <= max_phase:
                parser.error(f"--cycle-from must be between 0 and {max_phase}.")
            if not 0 <= args.cycle_to <= max_phase:
                parser.error(f"--cycle-to must be between 0 and {max_phase}.")
            if args.cycle_from > args.cycle_to:
                parser.error("--cycle-from must be <= --cycle-to.")
            client.inference_mode           = 'cycle'
            client.cycle_episodes_per_phase = args.cycle
            client._cycle_phases            = list(range(args.cycle_from, args.cycle_to + 1))
            client._cycle_phase             = client._cycle_phases[0]
            print(f"[INFERENCE R{robot_id}] Mode: CYCLE | {args.cycle} episodes × "
                  f"phases {args.cycle_from}–{args.cycle_to}")
        elif args.free:
            client.inference_mode = 'free'
            print(f"[INFERENCE R{robot_id}] Mode: FREE | Place the object manually each episode")
        elif args.phase is not None:
            max_phase = len(CurriculumManager.PHASE_CONFIG) - 1
            if not 0 <= args.phase <= max_phase:
                parser.error(f"--phase must be between 0 and {max_phase}.")
            client.inference_mode = 'phase'
            client.fixed_phase    = args.phase
            cfg = CurriculumManager.PHASE_CONFIG[args.phase]
            if args.phase == CurriculumManager.FULL_BOARD_PHASE:
                print(f"[INFERENCE R{robot_id}] Mode: PHASE {args.phase} | "
                      f"uniform spawn on full usable platform")
            else:
                print(f"[INFERENCE R{robot_id}] Mode: PHASE {args.phase} | "
                      f"radius {cfg[0]*100:.1f}–{cfg[1]*100:.1f}cm")
        else:
            client.inference_mode = 'normal'
            print(f"[INFERENCE R{robot_id}] Mode: NORMAL | Following curriculum as usual")

    client.refresh_episode_log_paths()
    _agent_debug_log(
        "simulation_client.py:main",
        "client_starting_loop",
        {"robot_id": robot_id, "mode": mode},
        hypothesis_id="E",
    )
    try:
        client.run_simulation_loop(max_episodes=None if args.real else args.episodes)
    except Exception as exc:
        _agent_debug_log(
            "simulation_client.py:main",
            "client_loop_exception",
            {"robot_id": robot_id, "error": repr(exc)},
            hypothesis_id="C",
        )
        raise
    finally:
        _agent_debug_log(
            "simulation_client.py:main",
            "client_loop_finished",
            {"robot_id": robot_id},
            hypothesis_id="E",
        )


if __name__ == "__main__":
    main()
