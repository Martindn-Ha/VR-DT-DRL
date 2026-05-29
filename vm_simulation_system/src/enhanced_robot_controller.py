#!/usr/bin/env python3
"""
Enhanced Robot Controller for UR3e System

Provides kinematics, hardware bridging, and motion planning for the UR3e robotic arm.
Supports execution in both Webots simulation and on physical hardware via ROS.
"""

import numpy as np
import time
import yaml
import math
import logging
import sys
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from math import pi, sin, cos, acos, atan2, sqrt

# --- Optional ROS Imports ---
try:
    import rospy
    import actionlib
    from geometry_msgs.msg import Pose, JointState
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from std_msgs.msg import Bool
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS not available, operating in standalone Simulation mode.")

# --- Optional Scipy Imports ---
try:
    from scipy.spatial.transform import Rotation as Rot
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    class MockRotation:
        @staticmethod
        def from_euler(seq, angles): return MockRotation()
        @staticmethod
        def from_matrix(matrix): return MockRotation()
        def as_matrix(self): return np.eye(3)
        def as_euler(self, seq): return [0, 0, 0]
    Rot = MockRotation


ROBOT1_DEF_CANDIDATES = ("ur3e_robot", "UR3", "ur3_robot")
ROBOT2_DEF_CANDIDATES = ("ur3e_robot2", "ur3_robot2")


def resolve_webots_robot_node(supervisor, robot_id: int):
    """Return the Webots Robot node for robot_id using world-file DEF fallbacks."""
    if supervisor is None:
        return None
    candidates = ROBOT2_DEF_CANDIDATES if robot_id == 2 else ROBOT1_DEF_CANDIDATES
    for def_name in candidates:
        try:
            node = supervisor.getFromDef(def_name)
            if node is not None:
                return node
        except Exception:
            continue
    return None

class UR3KinematicsController:
    """
    Kinematics and motion controller for the UR3 arm.
    Manages coordinate transformations, inverse kinematics (IK), and joint-space interpolation.
    """

    # =========================================================================
    # HOME POSTURE CONFIGURATION
    # =========================================================================
    # Shared "elbow-up" home poses.
    # SIM:  joint 0 = 0.0      (Base faces the platform in Webots)
    # REAL: joint 0 = +pi/2    (Real base is physically rotated +90 deg relative to Webots)
    # =========================================================================
    _HOME_JOINTS_SIM = [
        0.0,
        math.radians(-105.18),
        math.radians( 102.93),
        math.radians( -87.75),
        math.radians( -90.05),
        3.0,
    ]
    
    _HOME_JOINTS_REAL = [
        math.pi / 2,              
        math.radians(-105.18),
        math.radians( 102.93),
        math.radians( -87.75),
        math.radians( -90.05),
        math.pi,                  
    ]

    @classmethod
    def get_home_joints(cls, simulation: bool = True) -> list:
        """Returns the correct home joint angles for the active environment."""
        return list(cls._HOME_JOINTS_SIM if simulation else cls._HOME_JOINTS_REAL)

    HOME_JOINTS = _HOME_JOINTS_SIM

    def __init__(self, config_path: str = "config/robot_config.yaml", 
                 simulation: bool = True, 
                 robot_instance: Any = None,
                 webots_bridge: Any = None,
                 robot_id: int = 1):
                 
        self.logger = logging.getLogger('UR3Controller')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.is_sim = simulation
        self.webots_bridge = webots_bridge
        self.gripper: Optional['GripperController'] = None
        self.robot_id = robot_id  
        
        # UR3e DH parameters
        self.d = [0.15185, 0, 0, 0.13105, 0.08535, 0.0921]
        self.a = [0, -0.24355, -0.2132, 0, 0, 0]
        self.alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]

        self.joints_state = [0.0] * 6
        self.motors = []

        if self.is_sim and robot_instance:
            self._setup_webots_hardware(robot_instance)
        
        if ROS_AVAILABLE:
            self._setup_ros_interface()

    def _setup_webots_hardware(self, robot_instance):
        """Binds Webots motor devices to the controller."""
        motor_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        
        for name in motor_names:
            motor = robot_instance.getDevice(name)
            if motor:
                motor.setPosition(float('inf')) 
                motor.setVelocity(1.0)
                self.motors.append(motor)
            else:
                self.logger.error(f"Motor {name} NOT FOUND!")
        print(f"[UR3 R{self.robot_id}] Webots motors bound: {len(self.motors)}/6")

    def _setup_ros_interface(self):
        """Initializes ROS subscribers for real-world state tracking."""
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self._joint_state_callback)

    def _joint_state_callback(self, msg):
        if len(msg.position) >= 6:
            self.joints_state = list(msg.position[:6])

    def _wait_step(self, duration: float):
        """Deterministically steps the simulation for a specific duration."""
        if self.webots_bridge:
            n = max(1, int(duration * 80))
            for _ in range(n):
                self.webots_bridge.step()
        else:
            time.sleep(duration)

    def _axis_angle_to_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Converts an axis-angle representation into a 3x3 rotation matrix."""
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


    # =========================================================================
    # COORDINATE TRANSFORMS
    # =========================================================================

    def transform_webots_to_ur3(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Converts Webots world coordinates to UR3 base frame (Robot 1 Simulation)."""
        base_pos_world = np.array([-0.685, 0.372, 0.47235])

        axis = np.array([-0.57735, -0.57735, -0.577351])
        angle = 2.0944
        R_wb = self._axis_angle_to_rotation(axis, angle)

        P_object_world = np.array([x, y, z])
        P_diff_world = P_object_world - base_pos_world

        P_local = R_wb.T @ P_diff_world
        return P_local[0], P_local[1], P_local[2]

    def transform_real_to_ur3(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Converts world coordinates to UR3 base frame for the physical hardware."""
        base_pos_world = np.array([-0.685, 0.372, 0.47235])

        axis_wb = np.array([-0.57735, -0.57735, -0.577351])
        R_wb = self._axis_angle_to_rotation(axis_wb, 2.0944)

        # Apply -90 degree offset for physical base orientation
        R_Ym90 = self._axis_angle_to_rotation(np.array([0.0, 1.0, 0.0]), -math.pi / 2)
        R_real = R_Ym90 @ R_wb

        P_diff = np.array([x, y, z]) - base_pos_world
        P_local = R_real.T @ P_diff
        return P_local[0], P_local[1], P_local[2]

    def transform_webots_to_ur3_robot2(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Converts Webots world coordinates to UR3 base frame (Robot 2 Simulation)."""
        base_pos_world = np.array([-1.226, 0.372, 0.47235])

        axis  = np.array([-0.57735, -0.57735, -0.577351])
        angle = 2.0944
        R_wb  = self._axis_angle_to_rotation(axis, angle)

        P_object_world = np.array([x, y, z])
        P_diff_world   = P_object_world - base_pos_world
        P_local        = R_wb.T @ P_diff_world
        return P_local[0], P_local[1], P_local[2]


    # =========================================================================
    # INVERSE KINEMATICS (IK) SOLVER
    # =========================================================================

    def _solve_ik_analytical(self, T_target: np.ndarray) -> List[List[float]]:
        """Generates possible joint configurations for a target Cartesian pose."""
        solutions = []
        try:
            # 1. Calculate Wrist Center (P_wc)
            P_tcp = T_target[0:3, 3]
            Z_tool = T_target[0:3, 2]  
            
            d6 = self.d[5]
            P_wc = P_tcp - d6 * Z_tool 

            # 2. Solve for Shoulder Pan (Theta 1)
            x_wc, y_wc = P_wc[0], P_wc[1]
            theta1 = atan2(y_wc, x_wc)
            
            # 3. Solve 2D Planar Arm (Shoulder Lift & Elbow)
            d1 = self.d[0]
            d4 = self.d[3]
            r_wc = sqrt(x_wc**2 + y_wc**2) 
            
            if abs(r_wc) < d4: 
                return [] 
                
            r_arm = sqrt(r_wc**2 - d4**2) 
            h_arm = P_wc[2] - d1
            dist_s_w = sqrt(r_arm**2 + h_arm**2)
            
            a2, a3 = abs(self.a[1]), abs(self.a[2])
            
            max_dist = (a2 + a3) - 0.001 
            if dist_s_w > max_dist:
                dist_s_w = max_dist

            cos_theta3 = (dist_s_w**2 - a2**2 - a3**2) / (2 * a2 * a3)
            cos_theta3 = max(-1.0, min(1.0, cos_theta3))
            
            # Candidate 1: Elbow Down / Candidate 2: Elbow Up
            elbow_candidates = [-acos(cos_theta3), acos(cos_theta3)]

            for theta3 in elbow_candidates:
                alpha = atan2(h_arm, r_arm)
                beta = atan2(a3 * sin(theta3), a2 + a3 * cos(theta3))
                
                theta2 = -(alpha + beta) 

                yaw = atan2(T_target[1, 0], T_target[0, 0])
                
                theta4 = -(pi/2) - theta2 - theta3
                theta5 = -pi/2 
                theta6_raw = yaw - theta1
                theta6 = ((theta6_raw + pi) % (2 * pi)) - pi

                solutions.append([theta1, theta2, theta3, theta4, theta5, theta6])

            return solutions
        except Exception as e:
            return []


    # =========================================================================
    # MOTION EXECUTION
    # =========================================================================

    def move_linear_path(self, start_pose: List[float], end_pose: List[float], 
                     steps: int = 50, step_duration: float = 0.12) -> bool:
        """Executes a linear Cartesian path by interpolating waypoints."""
        p_start = np.array(start_pose[:3])
        p_end   = np.array(end_pose[:3])
        current_joints = np.array(self.joints_state)
        
        R_down = np.array([[ 0, -1,  0], 
                           [-1,  0,  0], 
                           [ 0,  0, -1]])
        
        yaw = end_pose[3] if len(end_pose) > 3 else 0.0
        cy, sy = cos(yaw), sin(yaw)
        R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        
        R_target = R_yaw @ R_down

        for i in range(1, steps + 1):
            t = i / float(steps)
            p_current = p_start + (p_end - p_start) * t
            
            T = np.eye(4)
            T[0:3, 3] = p_current
            T[0:3, 0:3] = R_target
            
            solutions = self._solve_ik_analytical(T)
            
            if not solutions: return False

            # Heuristic selection for optimal joint configuration
            best_sol = None
            min_score = float('inf')
            
            for sol in solutions:
                sol_arr = np.array(sol)
                dist = np.linalg.norm(sol_arr - current_joints)
    
                shoulder_val = sol[1]
                penalty = 1000.0 if shoulder_val > -0.5 else 0.0
    
                wrist3_delta = abs(sol[5] - current_joints[5])
                if wrist3_delta > math.pi:
                    wrist3_delta = 2 * math.pi - wrist3_delta
                wrist3_penalty = 50.0 * wrist3_delta
    
                score = dist + penalty + wrist3_penalty
                if score < min_score:
                    min_score = score
                    best_sol = sol
            
            if best_sol:
                current_joints = np.array(best_sol)
                self.move_to_joint_positions(best_sol, duration=step_duration, wait=True)
            else:
                return False
                
        return True

    def move_to_joint_positions(self, target_joints, duration=3.0, wait=True):
        """Commands synchronized joint-space movement."""
        if not self.motors: return False
    
        normalized = list(target_joints)
        for i in range(len(normalized)):
            if i < len(self.joints_state):
                delta = normalized[i] - self.joints_state[i]
                while delta > math.pi:
                    delta -= 2 * math.pi
                    normalized[i] -= 2 * math.pi
                while delta < -math.pi:
                    delta += 2 * math.pi
                    normalized[i] += 2 * math.pi

        safe_duration = max(0.001, float(duration))
        
        for i, motor in enumerate(self.motors):
            if i < len(normalized):
                distance = abs(normalized[i] - self.joints_state[i])
                speed = distance / safe_duration
                speed = max(0.01, min(speed, 3.14))
                
                motor.setVelocity(speed)
                motor.setPosition(normalized[i])
                
        self.joints_state = normalized
        if wait: 
            self._wait_step(duration)
        return True

    def move_to_pose(self, target_pose: List[float], duration: float = 3.0, wait: bool = True) -> bool:
        """Solves IK for a specific Cartesian pose and executes the movement."""
        ik_x, ik_y, ik_z = target_pose[0], target_pose[1], target_pose[2]
        yaw = target_pose[3] if len(target_pose) > 3 else 0.0

        R_down = np.array([[ 0, -1,  0], 
                           [-1,  0,  0], 
                           [ 0,  0, -1]])
        cy, sy = math.cos(yaw), math.sin(yaw)
        R_yaw = np.array([[cy, -sy, 0], 
                          [sy,  cy, 0], 
                          [ 0,   0, 1]])
        R_target = R_yaw @ R_down

        T = np.eye(4)
        T[0:3, 3] = [ik_x, ik_y, ik_z]
        T[0:3, 0:3] = R_target

        solutions = self._solve_ik_analytical(T)
    
        if solutions:
            current_joints = np.array(self.joints_state)
            best_sol = None
            min_score = float('inf')
        
            for sol in solutions:
                sol_arr = np.array(sol)
                joint_dist = np.linalg.norm(sol_arr - current_joints)
            
                shoulder_val = sol[1]
                penalty = 1000.0 if shoulder_val > -0.5 else 0.0
            
                wrist3_delta = abs(sol[5] - self.joints_state[5])
                if wrist3_delta > math.pi:
                    wrist3_delta = 2 * math.pi - wrist3_delta
                wrist3_penalty = 50.0 * wrist3_delta  
            
                score = joint_dist + penalty + wrist3_penalty
                if score < min_score:
                    min_score = score
                    best_sol = sol
        
            return self.move_to_joint_positions(best_sol, duration, wait)
    
        print(f"[MoveToPose] No IK solution for {target_pose}")
        return False

    def move_joints_linear(self, target_joints: List[float], 
                        steps: int = 30, 
                        step_duration: float = 0.15) -> bool:
        """Interpolates directly in joint space using cosine easing."""
        if not self.motors:
            return False

        start_joints = np.array(self.joints_state)
        end_joints   = np.array(target_joints)

        for i in range(len(end_joints)):
            delta = end_joints[i] - start_joints[i]
            while delta > math.pi:
                delta -= 2 * math.pi
                end_joints[i] -= 2 * math.pi
            while delta < -math.pi:
                delta += 2 * math.pi
                end_joints[i] += 2 * math.pi

        for step in range(1, steps + 1):
            t = step / float(steps)
            t_smooth = (1 - math.cos(t * math.pi)) / 2.0
            interp = start_joints + t_smooth * (end_joints - start_joints)
            self.move_to_joint_positions(interp.tolist(), duration=step_duration, wait=True)

        return True

    def home_position(self):
        """Returns the arm to the designated home configuration."""
        return self.move_to_joint_positions(
            self.get_home_joints(simulation=self.is_sim), duration=2.0)

    # =========================================================================
    # GRASP SEQUENCE EXECUTION
    # =========================================================================

    def execute_grasp(self, pose: List[float]) -> bool:
        """
        Executes a complete pick sequence based on a 6-DOF Cartesian pose prediction.
        Pose format: [x, y, z, rx, ry, rz]
        """
        w_x, w_y, w_z = pose[0], pose[1], pose[2]
        yaw = pose[5] if len(pose) > 5 else (pose[3] if len(pose) > 3 else 0.0)

        if self.robot_id == 2:
            ik_x, ik_y, ik_z = self.transform_webots_to_ur3_robot2(w_x, w_y, w_z)
        else:
            ik_x, ik_y, ik_z = self.transform_webots_to_ur3(w_x, w_y, w_z)

        # Apply kinematic reach limits
        r = math.sqrt(ik_x**2 + ik_y**2)
        MAX_SAFE_RADIUS = 0.43   
        if r > MAX_SAFE_RADIUS:
            scale = MAX_SAFE_RADIUS / r
            ik_x *= scale
            ik_y *= scale

        target_z = ik_z

        # ---------------------------------------------------------------------
        # HEIGHT SETTINGS (Meters relative to robot base)
        # ---------------------------------------------------------------------
        PLATFORM_Z    = 0.068
        FLOOR_MARGIN  = 0.035
        target_z = max(target_z, PLATFORM_Z + FLOOR_MARGIN)

        grasp_z  = target_z + 0.129
        hover_z  = grasp_z  + 0.11
        safe_z   = 0.40
        # ---------------------------------------------------------------------

        # Dynamic Compass Compensation (Maintains absolute world orientation)
        base_yaw_offset = 0.0
        try:
            sup = self.webots_bridge.supervisor
            if hasattr(sup, 'supervisor'):
                sup = sup.supervisor

            robot_node = resolve_webots_robot_node(sup, self.robot_id)
            if robot_node:
                rot_field = robot_node.getField("rotation").getSFRotation()
                axis_y = rot_field[1] if len(rot_field) == 4 else 1.0
                base_yaw_offset = rot_field[3] * np.sign(axis_y)
        except Exception as e:
            print(f"Could not read dynamic base rotation: {e}")

        GRIPPER_MOUNT_OFFSET = math.pi / 4 - (math.pi / 16) + math.radians(11.2)
        ik_yaw = -(yaw - base_yaw_offset + GRIPPER_MOUNT_OFFSET)

        # Initialize sequence
        if self.gripper:
            self.gripper.open_gripper()
            self._wait_step(0.8)

        R_down = np.array([[ 0, -1,  0],
                       [-1,  0,  0],
                       [ 0,  0, -1]])
        cy, sy = math.cos(ik_yaw), math.sin(ik_yaw)
        R_yaw  = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]])
        R_target = R_yaw @ R_down

        def solve_waypoint(x, y, z):
            T = np.eye(4)
            T[0:3, 3]   = [x, y, z]
            T[0:3, 0:3] = R_target
            solutions = self._solve_ik_analytical(T)
            if not solutions:
                return None
            
            current = np.array(self.joints_state)
            best, best_score = None, float('inf')
            for sol in solutions:
                sol_arr = np.array(sol)
                dist = np.linalg.norm(sol_arr - current)
                shoulder_penalty = 1000.0 if sol[1] > -0.5 else 0.0
                wrist_delta = abs(sol[5] - current[5])
                if wrist_delta > math.pi:
                    wrist_delta = 2 * math.pi - wrist_delta
                wrist_penalty = 50.0 * wrist_delta
                score = dist + shoulder_penalty + wrist_penalty
                if score < best_score:
                    best_score = score
                    best = sol
            return best

        joints_safe  = solve_waypoint(ik_x, ik_y, safe_z)
        joints_hover = solve_waypoint(ik_x, ik_y, hover_z)
        joints_grasp = solve_waypoint(ik_x, ik_y, grasp_z)

        if not all([joints_safe, joints_hover, joints_grasp]):
            print("[GRASP] Could not solve IK for one or more waypoints")
            return False

        # Phase 1: Hover
        self.move_to_joint_positions(joints_hover, duration=1.0)

        # Phase 2: Apply Compass Compensation to Wrist
        HOME_BASE_ANGLE  = 0.0
        HOME_WRIST_ANGLE = 3.0
        base_delta = joints_hover[0] - HOME_BASE_ANGLE
        wrist_compensated = HOME_WRIST_ANGLE + base_delta

        joints_hover_compensated = list(joints_hover)
        joints_hover_compensated[5] = wrist_compensated
        self.move_to_joint_positions(joints_hover_compensated, duration=0.8)

        # Phase 3: Vertical Descent
        N_STEPS = 15
        for i in range(1, N_STEPS + 1):
            t = i / float(N_STEPS)
            interp_z = hover_z + t * (grasp_z - hover_z)
            j = solve_waypoint(ik_x, ik_y, interp_z)
            if j is None:
                print("[GRASP] IK failed during descent")
                return False
            j = list(j)
            j[5] = wrist_compensated
            self.move_to_joint_positions(j, duration=0.05, wait=True)
        self._wait_step(0.2)

        # Phase 4: Proximity Check
        try:
            closest = 9999.0
            if self.webots_bridge and hasattr(self.webots_bridge, 'supervisor'):
                sup = self.webots_bridge.supervisor
                if hasattr(sup, 'supervisor'):
                    sup = sup.supervisor
                target_def = "TARGET_OBJECT" if self.robot_id == 1 else "TARGET_OBJECT2"
                o_node = sup.getFromDef(target_def)
                g_node = sup.getFromDef("GRIPPER_MAIN")
                if g_node is None: g_node = sup.getFromDef("UR3e")
                if g_node is None: g_node = sup.getFromDef("UR3")
                if o_node and g_node:
                    import numpy as _np
                    o_pos = _np.array(o_node.getPosition())
                    g_pos = _np.array(g_node.getPosition())
                    closest = float(_np.linalg.norm(o_pos - g_pos))
            self._closest_approach_dist = closest
        except Exception:
            self._closest_approach_dist = 9999.0

        # Phase 5: Actuate Gripper
        if self.gripper:
            self.gripper.close_gripper()
            self._wait_step(1.0)

        # Phase 6: Retreat
        self.move_to_joint_positions(joints_hover, duration=1.0)

        return True


class GripperController:
    """
    Hardware interface for the end-effector.
    Synchronizes state between Webots devices and ROS topics.
    """
    
    def __init__(self, robot_instance=None):
        self.finger1 = None
        self.finger2 = None
        self.is_closed = False
        
        if robot_instance:
            self.finger1 = robot_instance.getDevice("finger1")
            self.finger2 = robot_instance.getDevice("finger2")
            
            # Safety Check: Verify devices initialized
            if self.finger1 and self.finger2:
                # Velocity Control (Gentle Closing)
                self.finger1.setVelocity(0.15) 
                self.finger2.setVelocity(0.15)
                
                # Force Limit (Prevents object ejection)
                self.finger1.setAvailableForce(10.0)
                self.finger2.setAvailableForce(10.0)
            else:
                print("Warning: Gripper fingers not found! Check device names.")

        if ROS_AVAILABLE:
            self.gripper_pub = rospy.Publisher('/gripper/command', Bool, queue_size=1)

    def open_gripper(self, force: float = 50.0):
        """Opens the gripper fingers fully."""
        self.is_closed = False
        
        if self.finger1: self.finger1.setPosition(0.0)
        if self.finger2: self.finger2.setPosition(0.0)
        
        if ROS_AVAILABLE:
            self.gripper_pub.publish(Bool(data=False))

    def close_gripper(self):
        """Closes the gripper fingers to the designated threshold."""
        self.is_closed = True
        
        # 0.024 represents closed state for the Hand-e gripper geometry
        if self.finger1: self.finger1.setPosition(0.024) 
        if self.finger2: self.finger2.setPosition(0.024)
        
        if ROS_AVAILABLE:
            self.gripper_pub.publish(Bool(data=True))


class MotionPlanner:
    """
    High-level motion planning and coordination module.
    """
    def __init__(self, robot_controller: UR3KinematicsController):
        self.robot = robot_controller
        self.logger = logging.getLogger('MotionPlanner')

    def plan_and_execute_grasp(self, pose: List[float]) -> bool:
        """Wrapper for triggering the robot's standardized grasp execution."""
        return self.robot.execute_grasp(pose)


def create_robot_system(config_path: str = "config.yaml", 
                        simulation: bool = True, 
                        webots_bridge: Any = None,
                        robot_id: int = 1) -> Tuple[UR3KinematicsController, GripperController, MotionPlanner]:
    """
    Initializes a complete robot architecture instance.
    robot_id=1 -> Base UR3
    robot_id=2 -> Offset UR3
    """
    robot_instance = None
    if webots_bridge:
        if hasattr(webots_bridge, 'supervisor'):
            if hasattr(webots_bridge.supervisor, 'supervisor'):
                robot_instance = webots_bridge.supervisor.supervisor
            else:
                robot_instance = webots_bridge.supervisor

    # Devices (motors, cameras) must use the live Supervisor/Robot handle from
    # WEBOTS_ROBOT_NAME — getFromDef() returns a Node without getDevice().

    robot_controller = UR3KinematicsController(
        config_path=config_path, 
        simulation=simulation, 
        robot_instance=robot_instance,
        webots_bridge=webots_bridge,
        robot_id=robot_id
    )
    
    gripper_controller = GripperController(robot_instance=robot_instance)
    robot_controller.gripper = gripper_controller
    motion_planner = MotionPlanner(robot_controller)
    
    return robot_controller, gripper_controller, motion_planner


def create_dual_robot_system(config_path: str = "config.yaml",
                              webots_bridge: Any = None):
    """Initializes both standard and offset robot systems sharing a common bridge."""
    ur3_r1, grip_r1, plan_r1 = create_robot_system(
        config_path, simulation=True, webots_bridge=webots_bridge, robot_id=1
    )
    ur3_r2, grip_r2, plan_r2 = create_robot_system(
        config_path, simulation=True, webots_bridge=webots_bridge, robot_id=2
    )
    return ur3_r1, grip_r1, plan_r1, ur3_r2, grip_r2, plan_r2


if __name__ == "__main__":
    try:
        if ROS_AVAILABLE:
            rospy.init_node('ur3_controller_node', anonymous=True)
        
        from webots_bridge import WebotsBridge
        bridge = WebotsBridge(simulation=True)
        
        ur3_r1, gripper_r1, planner_r1, \
        ur3_r2, gripper_r2, planner_r2 = create_dual_robot_system("config.yaml", bridge)

        print("--> Dual Robot Controller Ready. Press Ctrl+C to exit.")
        print("    Robot 1 (UR3):       base at x=-0.685, TARGET_OBJECT")
        print("    Robot 2 (ur3e_robot2): base at x=-1.226, TARGET_OBJECT2")
        
        rate = rospy.Rate(30) if ROS_AVAILABLE else None
        
        while not (rospy.is_shutdown() if ROS_AVAILABLE else False):
            bridge.step()
            if rate: rate.sleep()
            else: time.sleep(0.033)

    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Critical Error: {e}")