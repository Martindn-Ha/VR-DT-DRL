#!/usr/bin/env python3
"""
Enhanced Webots Bridge for UR3e Hybrid System

Integrates the robotic control systems with the Webots simulation environment.
Handles dynamic library path resolution, supervisor node management, and 
simulated camera data extraction.
"""

import json
import os
import sys
import subprocess
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

_AGENT_DEBUG_LOG = Path(__file__).resolve().parent.parent.parent / "debug-4ce223.log"


def _agent_debug_log(location: str, message: str, data: Optional[Dict] = None,
                     hypothesis_id: str = "") -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "4ce223",
            "runId": "pre-fix",
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

# =========================================================================
# WEBOTS LIBRARY PATH DETECTION
# =========================================================================
# Locates Webots install (WEBOTS_HOME env or common paths), prepends the
# matching controller Python API, and exports WEBOTS_HOME for controller.py.
# =========================================================================

def _resolve_webots_home() -> Optional[str]:
  """Return Webots root if installed, else None."""
  env_home = os.environ.get("WEBOTS_HOME")
  if env_home and os.path.isdir(env_home):
    return os.path.normpath(env_home)

  candidates = []
  if sys.platform == "win32":
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
      candidates.append(os.path.join(local, "Programs", "Webots"))
    candidates.extend([
      r"C:\Program Files\Webots",
      r"C:\Program Files (x86)\Webots",
    ])
  else:
    candidates.append("/opt/webots")

  for path in candidates:
    if path and os.path.isdir(path):
      return os.path.normpath(path)
  return None


def _controller_python_lib(webots_home: Optional[str] = None) -> Optional[str]:
    """Return Webots controller API folder matching THIS Python version only."""
    home = webots_home or _resolve_webots_home()
    if not home:
        return None
    folder = f"python{sys.version_info.major}{sys.version_info.minor}"
    lib = os.path.join(home, "lib", "controller", folder)
    return lib if os.path.isdir(lib) else None


def _add_webots_dll_directories(webots_home: Optional[str] = None) -> None:
    """Windows: load Controller.dll / mingw64 deps (matches controller.py)."""
    if os.name != "nt" or sys.version_info < (3, 8):
        return
    home = webots_home or _resolve_webots_home()
    if not home:
        return
    for sub in (
        os.path.join("lib", "controller"),
        os.path.join("msys64", "mingw64", "bin", "cpp"),
        os.path.join("msys64", "mingw64", "bin"),
    ):
        path = os.path.join(home, sub)
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
            except (AttributeError, OSError):
                pass


def _configure_webots_env() -> None:
    """Apply R2021a Windows extern-controller environment (PATH, PYTHONPATH, PID)."""
    if not WEBOTS_HOME:
        return

    os.environ.setdefault("WEBOTS_HOME", WEBOTS_HOME)
    os.environ.setdefault("PYTHONIOENCODING", "UTF-8")

    path_add = [
        os.path.join(WEBOTS_HOME, "lib", "controller"),
        os.path.join(WEBOTS_HOME, "msys64", "mingw64", "bin"),
        os.path.join(WEBOTS_HOME, "msys64", "mingw64", "bin", "cpp"),
    ]
    existing = os.environ.get("PATH", "")
    prefix = os.pathsep.join(p for p in path_add if os.path.isdir(p))
    if prefix and prefix not in existing:
        os.environ["PATH"] = prefix + os.pathsep + existing

    py_lib = _controller_python_lib(WEBOTS_HOME) or ""
    if py_lib:
        os.environ["PYTHONPATH"] = py_lib

    if not os.environ.get("WEBOTS_PID"):
        try:
            import psutil
            for proc in psutil.process_iter(["pid", "name", "exe"]):
                try:
                    name = (proc.info.get("name") or "").lower()
                    exe = (proc.info.get("exe") or "").lower()
                    if "webots" in name or "webots" in exe:
                        if "unins" not in exe:
                            os.environ["WEBOTS_PID"] = str(proc.info["pid"])
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            pass


def _webots_is_running() -> bool:
    """True if a Webots process appears to be running."""
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                name = (proc.info.get("name") or "").lower()
                exe = (proc.info.get("exe") or "").lower()
                if "webots" in name or "webots" in exe:
                    if "unins" not in exe:
                        if not os.environ.get("WEBOTS_PID"):
                            os.environ["WEBOTS_PID"] = str(proc.info["pid"])
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        return True
    return False


def running_inside_webots() -> bool:
    """True when connected as a Webots extern controller (URL set by Webots or after Supervisor())."""
    return bool(os.environ.get("WEBOTS_CONTROLLER_URL"))


def _probe_webots_connection() -> bool:
    """
    Check whether Webots is playing and accepting an extern controller.

    Calling Supervisor() in-process when Webots is not ready aborts the Python
    interpreter (native crash). The probe runs in a subprocess so the client
    survives and can print a useful error.
    """
    script = (
        Path(__file__).resolve().parent.parent
        / "Webots" / "scripts" / "probe_webots_connection.py"
    )
    if not script.is_file():
        return False

    env = os.environ.copy()
    _configure_webots_env()
    env = os.environ.copy()
    if WEBOTS_HOME:
        env.setdefault("WEBOTS_HOME", WEBOTS_HOME)
    robot = env.get("WEBOTS_ROBOT_NAME", "?")

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[WebotsBridge] Timed out waiting for Webots (robot={robot}). "
            "Is the simulation playing?"
        )
        return False

    if result.returncode == 0 and "OK" in (result.stdout or ""):
        return True

    print(
        "[WebotsBridge] Could not connect extern controller to Webots.\n"
        f"  WEBOTS_ROBOT_NAME={robot}\n"
        "  Checklist:\n"
        "  1. Webots: updated_world/worlds/Environmentnewww.wbt open -> Reset Simulation -> Play.\n"
        "     Console should say 'Waiting for external controller' for ur3e_robot.\n"
        "  2. Start this client within ~30s of pressing Play.\n"
        "  3. Task Manager: kill stale python.exe from old simulation_client runs.\n"
        "  4. Run probe: python Webots/scripts/probe_webots_connection.py\n"
        "     Log: %TEMP%\\webots_probe.log"
    )
    if result.stdout:
        print(f"  probe stdout: {result.stdout.strip()}")
    if result.stderr:
        print(f"  probe stderr: {result.stderr.strip()}")
    return False


WEBOTS_HOME = _resolve_webots_home()
CONTROLLER_BASE = (
  os.path.join(WEBOTS_HOME, "lib", "controller") if WEBOTS_HOME else ""
)

if WEBOTS_HOME:
  os.environ.setdefault("WEBOTS_HOME", WEBOTS_HOME)

LIB_PATH = _controller_python_lib(WEBOTS_HOME) or ""

if LIB_PATH:
  if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)
  _add_webots_dll_directories(WEBOTS_HOME)
elif WEBOTS_HOME:
  ver = f"{sys.version_info.major}.{sys.version_info.minor}"
  print(
    f"--> [FAIL] Webots has no python{sys.version_info.major}"
    f"{sys.version_info.minor} API folder. This venv is Python {ver}. "
    f"Use Python 3.9 with Webots 2021a (see UR3e_Setup_Guide Part 5)."
  )

_configure_webots_env()

try:
    from controller import Supervisor, Robot
    from scipy.spatial.transform import Rotation as Rot
    WEBOTS_AVAILABLE = True
except ImportError as e:
    WEBOTS_AVAILABLE = False
    print(f"--> [FAIL] Webots controller not available: {e}. Using mock mode.")

# =========================================================================
# ROS & OPENCV DEPENDENCIES
# =========================================================================

try:
    import rospy
    from std_msgs.msg import Int8
    from integrator.msg import BlockPose
    from integrator.srv import SupervisorGrabService, SupervisorPositionService
    from integrator.srv import SimImageCameraService, SimDepthCameraService
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    class MockROS: pass
    BlockPose = MockROS
    Image = MockROS

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


# =========================================================================
# WEBOTS SUPERVISOR
# =========================================================================

class WebotsSupervisor:
    """
    Manages the global state of the Webots simulation.
    Tracks blocks, the end-effector GPS, and handles simulation stepping and resets.
    """
    def __init__(self, simulation: bool = True, world_file: str = "Environmentnewww.wbt", robot_instance=None):
        self.simulation = simulation
        self.logger = logging.getLogger('WebotsSupervisor')
        
        self.supervisor = robot_instance
        
        self.number_of_blocks = 5
        self.timestep = 16 
        self.ur3e_position = [0.69, 0.74, 0]
        self.ur3e_rotation = None
        
        if not self.supervisor and not simulation and WEBOTS_AVAILABLE:
            self._init_webots_supervisor()
        elif simulation or not WEBOTS_AVAILABLE:
            self._init_mock_supervisor()
        else:
            self.timestep = int(self.supervisor.getBasicTimeStep())
            self._setup_nodes() 

    def _setup_nodes(self):
        """Binds block targets and GPS nodes to the supervisor instance."""
        self.ur3e_rotation = Rot.from_rotvec(-(np.pi / 2) * np.array([1.0, 0.0, 0.0]))
        self.blocks = []
        for i in range(self.number_of_blocks):
            block = self.supervisor.getFromDef(f"block{i}")
            if block: self.blocks.append(block)
        self.end_effector = self.supervisor.getFromDef("gps")

    def _init_webots_supervisor(self):
        try:
            self.supervisor = Supervisor()
            self.timestep = int(self.supervisor.getBasicTimeStep())
            self._setup_nodes()
            self.logger.info("Webots supervisor initialized")
        except Exception as e:
            self.logger.error(f"Failed to init supervisor: {e}")
            self._init_mock_supervisor()
            
    def _init_mock_supervisor(self):
        self.supervisor = None
        self.blocks = []
        self.end_effector = None
        
        for i in range(self.number_of_blocks):
            mock_block = {
                'id': i,
                'position': [np.random.uniform(-0.5, 0.5), 
                           np.random.uniform(-0.5, 0.5),
                           np.random.uniform(0.7, 0.9)],
                'rotation': [0, 0, np.random.uniform(0, 2*np.pi)]
            }
            self.blocks.append(mock_block)
            
        self.logger.info(f"Mock supervisor initialized with {len(self.blocks)} blocks")
        
    def _init_ros_services(self):
        """Initializes ROS integration services for external control."""
        try:
            if not rospy.get_node_uri():
                rospy.init_node('webots_supervisor', anonymous=True)
                
            self.grab_service = rospy.Service(
                'supervisor_grab_service', 
                SupervisorGrabService, 
                self._handle_grab_request
            )
            
            self.position_service = rospy.Service(
                'supervisor_position_service',
                SupervisorPositionService,
                self._handle_position_request  
            )
            
            self.logger.info("ROS services initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS services: {e}")
            
    def step(self) -> bool:
        if self.supervisor:
            return self.supervisor.step(self.timestep) != -1
        else:
            time.sleep(self.timestep / 1000.0) 
            return True
            
    def get_block_poses(self) -> List[Dict[str, Any]]:
        """Extracts world coordinate poses for all tracked scene blocks."""
        block_poses = []
        
        if self.supervisor and hasattr(self.supervisor, 'getFromDef'):
            for i, block in enumerate(self.blocks):
                if block:
                    try:
                        position = block.getPosition()
                        rotation = block.getOrientation()
                        
                        block_poses.append({
                            'id': i,
                            'position': list(position) if position else [0, 0, 0],
                            'rotation': list(rotation) if rotation else [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            'timestamp': time.time()
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to get pose for block {i}: {e}")
        else:
            for i, block in enumerate(self.blocks):
                if isinstance(block, dict):
                    block_poses.append({
                        'id': i,
                        'position': block['position'],
                        'rotation': block['rotation'] + [1, 0, 0, 0, 1, 0], 
                        'timestamp': time.time()
                    })
                    
        return block_poses
        
    def set_block_pose(self, block_id: int, position: List[float], 
                      rotation: Optional[List[float]] = None) -> bool:
        """Teleports a specified block to a new position and orientation."""
        if block_id >= len(self.blocks):
            self.logger.error(f"Block ID {block_id} out of range")
            return False
            
        if self.supervisor and hasattr(self.supervisor, 'getFromDef'):
            block = self.blocks[block_id]
            if block:
                try:
                    block.getField('translation').setSFVec3f(position)
                    if rotation:
                        block.getField('rotation').setSFRotation(rotation + [1.0]) 
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to set block {block_id} pose: {e}")
                    return False
        else:
            if isinstance(self.blocks[block_id], dict):
                self.blocks[block_id]['position'] = position
                if rotation:
                    self.blocks[block_id]['rotation'] = rotation
                return True
                
        return False
        
    def get_robot_state(self) -> Dict[str, Any]:
        """Returns the current spatial state of the robot and end-effector."""
        robot_state = {
            'position': self.ur3e_position.copy(),
            'rotation': [0, 0, 0],
            'joint_angles': [0.0] * 6,
            'end_effector_pose': [0, 0, 0, 0, 0, 0],
            'timestamp': time.time()
        }
        
        if self.supervisor and self.end_effector:
            try:
                ee_pos = self.end_effector.getPosition()
                if ee_pos:
                    robot_state['end_effector_pose'][:3] = list(ee_pos)
                    
                ee_rot = self.end_effector.getOrientation()
                if ee_rot:
                    rot_matrix = np.array(ee_rot).reshape(3, 3)
                    if WEBOTS_AVAILABLE:
                        euler = Rot.from_matrix(rot_matrix).as_euler('xyz')
                        robot_state['end_effector_pose'][3:] = list(euler)
                        
            except Exception as e:
                self.logger.warning(f"Failed to get robot state: {e}")
                
        return robot_state
        
    def reset_simulation(self) -> bool:
        if self.supervisor:
            try:
                self.supervisor.simulationReset()
                return True
            except Exception as e:
                self.logger.error(f"Failed to reset simulation: {e}")
                return False
        else:
            for block in self.blocks:
                if isinstance(block, dict):
                    block['position'] = [
                        np.random.uniform(-0.5, 0.5),
                        np.random.uniform(-0.5, 0.5), 
                        np.random.uniform(0.7, 0.9)
                    ]
                    block['rotation'] = [0, 0, np.random.uniform(0, 2*np.pi)]
            return True
            
    def _handle_grab_request(self, request):
        return True
        
    def _handle_position_request(self, request):
        return self.get_robot_state()


# =========================================================================
# WEBOTS CAMERA INTERFACE
# =========================================================================

class WebotsCamera:
    """
    Interfaces with Webots Camera nodes to extract and format RGB-D buffers.
    Applies resolution downscaling and necessary rotational corrections depending
    on the physical mounting orientation of the sensor in the simulation world.
    """

    OUTPUT_WIDTH  = 640
    OUTPUT_HEIGHT = 360

    def __init__(self, simulation: bool = True, robot_instance=None,
                 robot_hosts: Optional[List[Any]] = None,
                 supervisor=None,
                 color_device_name: str = 'realsense_color',
                 range_device_name: str = 'realsense_range',
                 color_def: Optional[str] = None,
                 range_def: Optional[str] = None,
                 alt_color_names: Optional[List[str]] = None,
                 alt_range_names: Optional[List[str]] = None,
                 discover_tag: Optional[str] = None,
                 rot90_k: int = 3,
                 flip_lr: bool = True,
                 defer_setup: bool = False):
        """
        Args:
            simulation: True = mock/offline mode; False = live Webots devices.
            robot_instance: Legacy single host (appended to robot_hosts if set).
            robot_hosts: Webots Robot/Supervisor handles to search for devices.
            supervisor: Supervisor used to read Camera/RangeFinder DEF -> name fields.
            color_device_name: Primary RGB Camera device name.
            range_device_name: Primary RangeFinder device name.
            color_def / range_def: Scene DEF names (device 'name' may differ from DEF).
            alt_color_names / alt_range_names: Kinect-style fallbacks (paired in order).
            discover_tag: If set, auto-pick the sole color/range device whose name contains this.
            rot90_k: Orientation correction integer (0=none, 1=90° CCW, 2=180°, 3=270° CCW).
            flip_lr: Boolean flag to apply a left/right mirror correction.
            defer_setup: If True, call setup_devices() later (lazy init).
        """
        self.simulation = simulation
        self.logger = logging.getLogger('WebotsCamera')
        self.robot = robot_instance
        self.supervisor = supervisor

        self.color_device_name = color_device_name
        self.range_device_name = range_device_name
        self.color_def = color_def
        self.range_def = range_def
        self._alt_color_names = list(alt_color_names or [])
        self._alt_range_names = list(alt_range_names or [])
        self.discover_tag = discover_tag

        self.rot90_k = rot90_k
        self.flip_lr = flip_lr

        self.timestep = 4
        self.image_width  = 1280
        self.image_height = 720
        self.devices_ready = False

        hosts: List[Any] = list(robot_hosts or [])
        if robot_instance is not None and robot_instance not in hosts:
            hosts.insert(0, robot_instance)
        self.robot_hosts = hosts

        if defer_setup:
            self._init_mock_camera()
        elif not simulation and self.robot_hosts:
            self.setup_devices()
        else:
            self._init_mock_camera()

    def _resolve_def_device_name(self, def_name: str) -> Optional[str]:
        """Read the Webots device 'name' field from a DEF (often differs from DEF id)."""
        if not self.supervisor or not def_name:
            return None
        try:
            node = self.supervisor.getFromDef(def_name)
            if node is None:
                return None
            name_field = node.getField('name')
            if name_field:
                return name_field.getSFString()
        except Exception as e:
            self.logger.debug(f"Could not read name from DEF '{def_name}': {e}")
        return None

    def _ordered_device_pairs(self) -> List[Tuple[str, str]]:
        """Build (color, range) pairs to try — avoids cartesian-product getDevice spam."""
        pairs: List[Tuple[str, str]] = []

        if self.color_def and self.range_def:
            def_cn = self._resolve_def_device_name(self.color_def)
            def_rn = self._resolve_def_device_name(self.range_def)
            if def_cn and def_rn:
                pairs.append((def_cn, def_rn))
            pairs.append((self.color_def, self.range_def))

        pairs.append((self.color_device_name, self.range_device_name))

        for cn, rn in zip(self._alt_color_names, self._alt_range_names):
            pairs.append((cn, rn))

        seen = set()
        unique: List[Tuple[str, str]] = []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                unique.append(pair)
        return unique

    @staticmethod
    def _discover_singleton_rgbd_pair(host, tag: Optional[str] = None
                                      ) -> Optional[Tuple[str, str]]:
        """If a host exposes exactly one color + one range (optionally matching tag), use them."""
        if host is None or not hasattr(host, 'getNumberOfDevices'):
            return None
        try:
            names = [host.getDeviceName(i) for i in range(host.getNumberOfDevices())]
        except Exception:
            return None

        colors = [n for n in names if 'color' in n.lower()]
        ranges = [n for n in names if 'range' in n.lower()]
        if tag:
            colors = [n for n in colors if tag in n]
            ranges = [n for n in ranges if tag in n]
        if len(colors) == 1 and len(ranges) == 1:
            return colors[0], ranges[0]
        return None

    def setup_devices(self):
        """Bind RGB-D devices. Safe to call again after lazy defer_setup."""
        if self.simulation or not self.robot_hosts:
            return

        self.camera = None
        self.depth_camera = None
        pairs = self._ordered_device_pairs()

        for host_idx, host in enumerate(self.robot_hosts):
            if host is None or not hasattr(host, 'getDevice'):
                continue

            discovered = self._discover_singleton_rgbd_pair(host, self.discover_tag)
            if discovered:
                pairs = [discovered] + [p for p in pairs if p != discovered]

            for cn, rn in pairs:
                try:
                    cam = host.getDevice(cn)
                    dep = host.getDevice(rn)
                except Exception:
                    cam, dep = None, None
                if cam and dep:
                    self.camera = cam
                    self.depth_camera = dep
                    self.color_device_name = cn
                    self.range_device_name = rn
                    self.camera.enable(self.timestep)
                    self.depth_camera.enable(self.timestep)
                    self.image_width = self.camera.getWidth()
                    self.image_height = self.camera.getHeight()
                    self.devices_ready = True
                    self.logger.info(
                        f"RGB-D bound on host[{host_idx}]: "
                        f"color='{cn}', range='{rn}' "
                        f"({self.image_width}x{self.image_height})"
                    )
                    return

        self.logger.error(
            f"Camera devices not found. Tried pairs={pairs} "
            f"on {len(self.robot_hosts)} host(s)."
        )

    def _init_mock_camera(self):
        self.camera = None
        self.depth_camera = None
        self.devices_ready = False
        
    def _init_ros_services(self):
        try:
            if ROS_AVAILABLE:
                if not rospy.get_node_uri():
                    rospy.init_node('webots_camera', anonymous=True)
                    
                self.bridge = CvBridge()
                
                self.image_service = rospy.Service(
                    'image_camera_service',
                    SimImageCameraService,
                    self._handle_image_request
                )
                
                self.depth_service = rospy.Service(
                    'depth_camera_service', 
                    SimDepthCameraService,
                    self._handle_depth_request
                )
                
                self.logger.info("Camera ROS services initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize camera ROS services: {e}")
            
    def capture_rgb_image(self) -> Optional[np.ndarray]:
        if self.camera and WEBOTS_AVAILABLE:
            try:
                image_data = self.camera.getImageArray()
                if image_data:
                    image = np.array(image_data, dtype=np.uint8)
                    if self.rot90_k:
                        image = np.rot90(image, k=self.rot90_k)
                    if self.flip_lr:
                        image = np.fliplr(image)
                    if OPENCV_AVAILABLE:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if image.shape[1] != self.OUTPUT_WIDTH or image.shape[0] != self.OUTPUT_HEIGHT:
                        image = cv2.resize(image, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT),
                                           interpolation=cv2.INTER_AREA)
                    return image
            except Exception as e:
                self.logger.error(f"Failed to capture RGB image: {e}")
        return None

    def capture_depth_image(self) -> Optional[np.ndarray]:
        if self.depth_camera and WEBOTS_AVAILABLE:
            try:
                depth_data = self.depth_camera.getRangeImageArray()
                if depth_data:
                    depth = np.array(depth_data, dtype=np.float32)
                    if self.rot90_k:
                        depth = np.rot90(depth, k=self.rot90_k)
                    if self.flip_lr:
                        depth = np.fliplr(depth)
                    if depth.shape[1] != self.OUTPUT_WIDTH or depth.shape[0] != self.OUTPUT_HEIGHT:
                        depth = cv2.resize(depth, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT),
                                           interpolation=cv2.INTER_LINEAR)
                    return depth
            except Exception as e:
                self.logger.error(f"Failed to capture depth image: {e}")
        return None
        
    def capture_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        rgb_image = self.capture_rgb_image()
        depth_image = self.capture_depth_image()
        return rgb_image, depth_image
        
    def _handle_image_request(self, request):
        rgb_image = self.capture_rgb_image()
        if rgb_image is not None and ROS_AVAILABLE:
            try:
                ros_image = self.bridge.cv2_to_imgmsg(rgb_image, "rgb8")
                return ros_image
            except Exception as e:
                self.logger.error(f"Failed to convert image to ROS message: {e}")
        return None
        
    def _handle_depth_request(self, request):
        depth_image = self.capture_depth_image()
        if depth_image is not None and ROS_AVAILABLE:
            try:
                ros_depth = self.bridge.cv2_to_imgmsg(depth_image, "32FC1")
                return ros_depth
            except Exception as e:
                self.logger.error(f"Failed to convert depth to ROS message: {e}")
        return None


# =========================================================================
# CENTRAL INTEGRATION BRIDGE
# =========================================================================

# Webots Robot DEF names in Environmentnewww.wbt (try canonical first, then legacy).
ROBOT1_DEF_CANDIDATES = ("ur3e_robot", "UR3", "ur3_robot")
ROBOT2_DEF_CANDIDATES = ("ur3e_robot2", "ur3_robot2")


class WebotsBridge:
    """
    Main orchestration class that ties together the Supervisor and individual Camera
    nodes. Supports dual-robot setups by mapping independent camera streams to a 
    single shared supervisor backend.
    """
    @staticmethod
    def _resolve_robot_device_host(supervisor, robot_defs):
        """
        Return the Webots node that owns a robot's devices (motors, cameras).
        getDevice() only works on the robot node that contains the device, not on
        a top-level Supervisor when devices live under a nested Robot DEF.

        Args:
            robot_defs: A single DEF string or ordered tuple of fallbacks.
        Returns:
            (node, matched_def) or (None, None).
        """
        if supervisor is None:
            return None, None
        if isinstance(robot_defs, str):
            robot_defs = (robot_defs,)
        logger = logging.getLogger('WebotsBridge')
        for robot_def in robot_defs:
            try:
                node = supervisor.getFromDef(robot_def)
                if node is not None:
                    return node, robot_def
            except Exception as e:
                logger.debug(f"Could not resolve robot DEF '{robot_def}': {e}")
        return None, None

    @staticmethod
    def _list_device_names(host, label: str) -> List[str]:
        """Logs available Webots device names on a Robot/Supervisor (for debugging)."""
        names: List[str] = []
        if host is None or not hasattr(host, 'getNumberOfDevices'):
            return names
        try:
            for i in range(host.getNumberOfDevices()):
                names.append(host.getDeviceName(i))
        except Exception as e:
            logging.getLogger('WebotsBridge').debug(
                f"Could not list devices on {label}: {e}"
            )
        return names

    def __init__(self, simulation: bool = True, world_file: str = "Environmentnewww.wbt",
                 robot_id: int = 1):
        self.simulation = simulation
        self.logger = logging.getLogger('WebotsBridge')
        self.robot_id = robot_id
        
        self.shared_robot = None
        if not simulation and WEBOTS_AVAILABLE:
            _configure_webots_env()
            if not _webots_is_running():
                print(
                    "[WebotsBridge] No Webots process found.\n"
                    "  Open updated_world/worlds/Environmentnewww.wbt and press Play, then restart this client."
                )
                simulation = True
            else:
                try:
                    self.shared_robot = Supervisor()
                    self.timestep = int(self.shared_robot.getBasicTimeStep())
                    name = self.shared_robot.getName()
                    self.logger.info("Connected to Webots (extern controller).")
                    print(f"[WebotsBridge] Connected to robot '{name}'")
                except Exception as e:
                    self.logger.error(f"Could not connect to Webots: {e}")
                    print(
                        f"[WebotsBridge] Supervisor() failed: {e}\n"
                        "  Reset Simulation in Webots, press Play, then restart this client."
                    )
                    simulation = True

        self.supervisor = WebotsSupervisor(
            simulation, world_file, robot_instance=self.shared_robot
        )

        # One extern controller process attaches to one Webots robot (WEBOTS_ROBOT_NAME).
        # getDevice() works on that Supervisor/Robot handle — not on getFromDef() nodes.
        host = [self.shared_robot] if self.shared_robot else []

        if self.shared_robot:
            devices = self._list_device_names(self.shared_robot, "controller")
            self.logger.info(f"Webots controller devices ({len(devices)}): {devices}")

        self.camera: Optional[WebotsCamera] = None
        self.camera2: Optional[WebotsCamera] = None

        if robot_id == 1:
            self.camera = WebotsCamera(
                simulation,
                robot_hosts=host,
                supervisor=self.shared_robot,
                color_device_name='realsense_color',
                range_device_name='realsense_range',
                color_def='realsense_color',
                range_def='realsense_range',
                rot90_k=3,
                flip_lr=True
            )
            print(
                "[WebotsBridge] Robot 2 cameras not opened (running as Robot 1). "
                "Use --robot-id 2 if you need camera 2."
            )
        else:
            self.camera2 = WebotsCamera(
                simulation,
                robot_hosts=host,
                supervisor=self.shared_robot,
                color_device_name='realsense_color2',
                range_device_name='realsense_range2',
                color_def='realsense_color2',
                range_def='realsense_range2',
                rot90_k=3,
                flip_lr=True
            )
            if not self.camera2.devices_ready:
                self.logger.error(
                    "Robot 2 cameras NOT bound. Ensure WEBOTS_ROBOT_NAME=ur3e_robot2 "
                    "and Webots is playing."
                )
            print(
                "[WebotsBridge] Robot 1 cameras not opened (running as Robot 2)."
            )

        self.logger.info(f"Webots bridge initialized (simulation={simulation}, robot_id={robot_id})")

    def _init_camera2(self, log_devices: bool = False) -> 'WebotsCamera':
        """Return Robot 2 camera (created at init when robot_id==2)."""
        if self.camera2 is None:
            raise RuntimeError(
                "Robot 2 camera not initialized — launch with --robot-id 2 "
                "and WEBOTS_ROBOT_NAME=ur3e_robot2"
            )
        return self.camera2

    def step(self) -> bool:
        """Step the simulation forward"""
        try:
            ok = bool(self.supervisor.step())
            if not ok:
                _agent_debug_log(
                    "webots_bridge.py:WebotsBridge.step",
                    "supervisor_step_false",
                    {"robot_id": getattr(self, "robot_id", None)},
                    hypothesis_id="A",
                )
            return ok
        except Exception as exc:
            _agent_debug_log(
                "webots_bridge.py:WebotsBridge.step",
                "supervisor_step_exception",
                {"robot_id": getattr(self, "robot_id", None), "error": repr(exc)},
                hypothesis_id="A",
            )
            return False
 
    def get_block_poses(self) -> List[Dict[str, Any]]:
        return self.supervisor.get_block_poses()
        
    def get_robot_state(self) -> Dict[str, Any]:
        return self.supervisor.get_robot_state()
        
    def capture_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture RGB and depth images from Robot 1's cameras."""
        if self.camera is None:
            return None, None
        return self.camera.capture_rgbd()

    def capture_images2(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture RGB and depth images from Robot 2's cameras."""
        return self._init_camera2().capture_rgbd()

    def get_camera(self, robot_id: int = 1) -> 'WebotsCamera':
        """Return the WebotsCamera instance for the given robot_id (1 or 2)."""
        return self.camera if robot_id == 1 else self._init_camera2()

    @staticmethod
    def _signed_uniform(lo: float, hi: float) -> float:
        """
        Returns a value with magnitude [lo, hi] and a randomized sign.
        Ensures perturbations never strictly center on the nominal zero pose.
        """
        return np.random.uniform(lo, hi) * np.random.choice([-1.0, 1.0])

    def randomize_camera_pose(self,
                              camera_defs: List[str],
                              base_translations: List[List[float]],
                              base_rotation_matrices: List[np.ndarray],
                              xy_min: float = 0.005,
                              xy_max: float = 0.020,
                              z_min:  float = 0.005,
                              z_max:  float = 0.010,
                              angle_min_deg: float = 0.0,
                              angle_max_deg: float = 0.5) -> bool:
        """
        Applies identically calculated spatial noise to a coupled set of cameras.
        Crucial for maintaining RGB and Depth hardware alignment during domain randomization.

        Args:
            camera_defs: List of DEF names (e.g., ["realsense_color", "realsense_range"]).
            base_translations: Nominal [x, y, z] anchors for the nodes.
            base_rotation_matrices: Nominal 3x3 rotation anchors.
            xy_min / xy_max: Bounds for X/Y planar noise (Meters).
            z_min  / z_max: Bounds for Z elevation noise (Meters).
            angle_min_deg / angle_max_deg: Bounds for Euler angular noise (Degrees).

        Returns:
            True if nodes were located and successfully translated.
        """
        if not self.shared_robot:
            return False

        nodes = []
        for def_name in camera_defs:
            node = self.shared_robot.getFromDef(def_name)
            if node is None:
                self.logger.warning(
                    f"randomize_camera_pose: DEF '{def_name}' not found — "
                    f"skipping entire group {camera_defs}"
                )
                return False
            nodes.append(node)

        try:
            dx = self._signed_uniform(xy_min, xy_max)
            dy = self._signed_uniform(xy_min, xy_max)
            dz = self._signed_uniform(z_min,  z_max)

            d_roll = d_pitch = d_yaw = 0.0
            if WEBOTS_AVAILABLE:
                angle_min_rad = np.deg2rad(angle_min_deg)
                angle_max_rad = np.deg2rad(angle_max_deg)
                d_roll  = self._signed_uniform(angle_min_rad, angle_max_rad)
                d_pitch = self._signed_uniform(angle_min_rad, angle_max_rad)
                d_yaw   = self._signed_uniform(angle_min_rad, angle_max_rad)
                noise_rot = Rot.from_euler('xyz', [d_roll, d_pitch, d_yaw])

            for node, base_t, base_r in zip(nodes, base_translations, base_rotation_matrices):
                node.getField('translation').setSFVec3f([
                    base_t[0] + dx,
                    base_t[1] + dy,
                    base_t[2] + dz,
                ])

                if WEBOTS_AVAILABLE:
                    combined   = Rot.from_matrix(base_r) * noise_rot
                    axis_angle = combined.as_rotvec()
                    angle      = np.linalg.norm(axis_angle)
                    if angle < 1e-9:
                        axis  = [0.0, 1.0, 0.0]
                        angle = 0.0
                    else:
                        axis = (axis_angle / angle).tolist()
                    node.getField('rotation').setSFRotation(axis + [float(angle)])

            self.logger.debug(
                f"Camera group {camera_defs} nudged "
                f"Δxyz=({dx*100:.2f},{dy*100:.2f},{dz*100:.2f}) cm  "
                f"Δrpy=({np.rad2deg(d_roll):.2f},{np.rad2deg(d_pitch):.2f},"
                f"{np.rad2deg(d_yaw):.2f})°"
            )
            return True

        except Exception as e:
            self.logger.error(f"randomize_camera_pose failed for group {camera_defs}: {e}")
            return False

    def reset_simulation(self) -> bool:
        return self.supervisor.reset_simulation()
        
    def set_block_pose(self, block_id: int, position: List[float], 
                      rotation: Optional[List[float]] = None) -> bool:
        return self.supervisor.set_block_pose(block_id, position, rotation)


def create_webots_bridge(config: Optional[Dict[str, Any]] = None,
                        simulation: bool = True) -> WebotsBridge:
    """
    Factory builder for deploying the Webots interconnect bridge.
    """
    world_file = "Environmentnewww.wbt"
    if config and 'world_file' in config:
        world_file = config['world_file']
        
    return WebotsBridge(simulation=simulation, world_file=world_file)