#!/usr/bin/env python3
"""
Enhanced Camera Handler for UR3e System

Integrates RGB-D camera processing with ROS and Webots simulation.
Supports hardware streams (RealSense) and synthetic simulation rendering,
providing automated depth filtering, alignment, and basic object segmentation.
"""

import numpy as np
import cv2
import time
import yaml
import os
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from collections import deque
import logging

# =========================================================================
# DEPENDENCY FALLBACKS & MOCKS
# =========================================================================

try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import Point, PointStamped
    from std_msgs.msg import Header
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS not installed — camera handler will use webots_bridge when provided.")
    class Image: pass
    class CameraInfo: pass
    class Header: pass

# Compatibility fallback for cv_bridge under Python 3 (ROS Melodic environments)
CV_BRIDGE_AVAILABLE = False
CvBridgeError = Exception  
if ROS_AVAILABLE:
    try:
        from cv_bridge import CvBridge, CvBridgeError
        CV_BRIDGE_AVAILABLE = True
    except (ImportError, Exception):
        pass

class CvBridge: pass

try:
    import pyrealsense2 as rs2
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

try:
    from integrator.srv import SimImageCameraService, SimDepthCameraService
    INTEGRATOR_SERVICES_AVAILABLE = True
except ImportError:
    INTEGRATOR_SERVICES_AVAILABLE = False
    # Optional ROS catkin package — not used when webots_bridge is passed in.
    print(
        "integrator ROS services not available "
        "(OK on Windows Part 5 — cameras use webots_bridge directly)."
    )
    class SimImageCameraService: pass
    class SimDepthCameraService: pass


class EnhancedCameraHandler:
    """
    Unified interface for RGB-D camera data acquisition and processing.
    Handles hardware pipelines, ROS topics, and Webots simulation buffers seamlessly.
    """
    
    def __init__(self, config_path: str = "config/camera_config.yaml", 
                 simulation: bool = True, camera_type: str = "realsense",
                 webots_bridge=None, camera_name: str = 'robot1'):
        """
        Args:
            config_path: Path to the YAML configuration file.
            simulation: Flag to toggle between Webots and physical hardware.
            camera_type: Specifies hardware backend (default: 'realsense').
            webots_bridge: Active bridge instance to the simulation supervisor.
            camera_name: Target camera identifier ('robot1' or 'camera2'). 
                         Maps to the respective WebotsCamera instance on the bridge.
        """
        self.webots_bridge = webots_bridge
        self._camera_robot_id = 2 if camera_name == 'camera2' else 1

        self.logger = logging.getLogger('CameraHandler')
        self.config = self._load_config(config_path)
        
        self.is_sim = simulation
        self.camera_type = camera_type
        self.image_size = tuple(self.config.get('image_size', [360, 640]))
        self.h, self.w = self.image_size
        
        # Image processing parameters
        self.depth_scale = self.config.get('depth_scale', 0.001)  
        self.depth_max = self.config.get('depth_max', 2.0)  
        self.depth_min = self.config.get('depth_min', 0.1)   
        
        # Camera intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None
        self.depth_camera_matrix = None
        
        # State tracking
        self.current_rgb_frame = None
        self.current_depth_frame = None
        self.current_aligned_depth = None
        self.frame_timestamp = None
        
        # Buffer histories for temporal stability
        self.distances = deque(maxlen=10)
        self.rgb_history = deque(maxlen=3)
        self.depth_history = deque(maxlen=3)
        
        if ROS_AVAILABLE:
            self.bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
            self._setup_ros_interface()
        
        self.camera_pipeline = None
        if not simulation:
            self._initialize_hardware_camera()
        
        self.logger.info(f"Camera handler initialized - Type: {camera_type}, Simulation: {simulation}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Loads and validates camera configuration parameters."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if not config or not isinstance(config, dict):
                    config = {}
                
                default_config = self._get_default_config()
                for key, value in default_config.items():
                    if key not in config or not isinstance(config.get(key), type(value)):
                        config[key] = value
                return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Provides fallback calibration and processing defaults."""
        return {
            'image_size': [360, 640],
            'fps': 30,
            'depth_scale': 0.001,
            'depth_max': 2.0,
            'depth_min': 0.1,
            'camera_intrinsics': {
                'fx': 525.0, 'fy': 525.0,
                'cx': 320.0, 'cy': 240.0
            },
            'processing': {
                'bilateral_filter': True,
                'temporal_filter': True,
                'spatial_filter': True,
                'hole_filling': True
            }
        }

    # =========================================================================
    # ROS & HARDWARE INITIALIZATION
    # =========================================================================

    def _setup_ros_interface(self):
        """Initializes ROS publishers and external data subscribers."""
        self.rgb_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=1)
        self.aligned_depth_pub = rospy.Publisher('/camera/aligned_depth/image_raw', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=1)
        
        self.external_rgb_sub = rospy.Subscriber('/external_camera/rgb', Image, self._external_rgb_callback)
        self.external_depth_sub = rospy.Subscriber('/external_camera/depth', Image, self._external_depth_callback)
        
        if INTEGRATOR_SERVICES_AVAILABLE:
            try:
                rospy.wait_for_service('/sim_image_camera_service', timeout=5.0)
                rospy.wait_for_service('/sim_depth_camera_service', timeout=5.0)
                
                self.sim_rgb_service = rospy.ServiceProxy('/sim_image_camera_service', SimImageCameraService)
                self.sim_depth_service = rospy.ServiceProxy('/sim_depth_camera_service', SimDepthCameraService)
                
                self.logger.info("Connected to Webots camera services")
            except rospy.ROSException:
                self.logger.warning("Could not connect to Webots camera services")
    
    def _initialize_hardware_camera(self):
        """Bootstraps physical RealSense pipeline and aligns intrinsics."""
        if self.camera_type == "realsense":
            if not REALSENSE_AVAILABLE:
                self.logger.error("pyrealsense2 not available — cannot initialise hardware camera")
                return
            try:
                self.camera_pipeline = rs2.pipeline()
                config = rs2.config()
                
                config.enable_stream(rs2.stream.color, self.w, self.h, rs2.format.bgr8, 30)
                config.enable_stream(rs2.stream.depth, self.w, self.h, rs2.format.z16, 30)
                
                profile = self.camera_pipeline.start(config)
                
                color_profile = profile.get_stream(rs2.stream.color)
                color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
                
                self.camera_matrix = np.array([
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1]
                ])
                
                self.dist_coeffs = np.array(color_intrinsics.coeffs)
                self.align = rs2.align(rs2.stream.color)
                self._setup_depth_filters()
                
                self.logger.info("RealSense camera initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize RealSense camera: {e}")
                self.camera_pipeline = None
    
    def _setup_depth_filters(self):
        """Configures SDK-level depth processing filters."""
        if not hasattr(self, 'camera_pipeline') or self.camera_pipeline is None:
            return
            
        self.spatial_filter = rs2.spatial_filter()
        self.spatial_filter.set_option(rs2.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs2.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs2.option.filter_smooth_delta, 20)
        
        self.temporal_filter = rs2.temporal_filter()
        self.temporal_filter.set_option(rs2.option.filter_smooth_alpha, 0.4)
        self.temporal_filter.set_option(rs2.option.filter_smooth_delta, 20)
        
        self.hole_filling_filter = rs2.hole_filling_filter()
        
        self.logger.info("Depth filters configured")

    # =========================================================================
    # FRAME ACQUISITION PIPELINES
    # =========================================================================

    def _external_rgb_callback(self, msg: Image):
        try:
            if CV_BRIDGE_AVAILABLE and self.bridge is not None:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            else:
                data = np.frombuffer(msg.data, dtype=np.uint8)
                cv_image = data.reshape((msg.height, msg.width, 3))
            self.current_rgb_frame = cv_image
            self.frame_timestamp = msg.header.stamp
            self.rgb_history.append(cv_image.copy())
        except Exception as e:
            self.logger.error(f"Error converting RGB image: {e}")
    
    def _external_depth_callback(self, msg: Image):
        try:
            if CV_BRIDGE_AVAILABLE and self.bridge is not None:
                if msg.encoding == "16UC1":
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                    depth_image = cv_image.astype(np.float32) * self.depth_scale
                elif msg.encoding == "32FC1":
                    depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                else:
                    self.logger.warning(f"Unsupported depth encoding: {msg.encoding}")
                    return
            else:
                if msg.encoding == "16UC1":
                    data = np.frombuffer(msg.data, dtype=np.uint16)
                    cv_image = data.reshape((msg.height, msg.width))
                    depth_image = cv_image.astype(np.float32) * self.depth_scale
                elif msg.encoding == "32FC1":
                    data = np.frombuffer(msg.data, dtype=np.float32)
                    depth_image = data.reshape((msg.height, msg.width))
                else:
                    self.logger.warning(f"Unsupported depth encoding: {msg.encoding}")
                    return

            self.current_depth_frame = depth_image
            self.depth_history.append(depth_image.copy())
        except Exception as e:
            self.logger.error(f"Error converting depth image: {e}")
    
    def capture_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Captures synchronized RGB and depth frames from the active source."""
        if self.is_sim:
            return self._capture_simulation_frames()
        elif self.camera_pipeline is not None:
            return self._capture_hardware_frames()
        else:
            return self.current_rgb_frame, self.current_depth_frame
    
    def _capture_simulation_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieves frames via ROS services from Webots.
        Native frames are captured at 1280x720 and optimally downscaled to pipeline resolution.
        """
        WEBOTS_H, WEBOTS_W = 720, 1280

        if not INTEGRATOR_SERVICES_AVAILABLE or not hasattr(self, 'sim_rgb_service'):
            # Fallback synthetic frames
            rgb_frame = np.random.randint(0, 255, (self.h, self.w, 3), dtype=np.uint8)
            depth_frame = np.random.uniform(0.5, 2.0, (self.h, self.w)).astype(np.float32)
            return rgb_frame, depth_frame
        
        try:
            rgb_response = self.sim_rgb_service()
            if rgb_response.success:
                rgb_array = np.array(rgb_response.image_data, dtype=np.uint8)
                rgb_frame = rgb_array.reshape((WEBOTS_H, WEBOTS_W, 3))
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                # INTER_AREA optimal for color downsampling
                rgb_frame = cv2.resize(rgb_frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
            else:
                rgb_frame = None
            
            depth_response = self.sim_depth_service()
            if depth_response.success:
                depth_array = np.array(depth_response.depth_data, dtype=np.float32)
                depth_frame = depth_array.reshape((WEBOTS_H, WEBOTS_W))
                # INTER_LINEAR prevents phantom edge artifacts in depth maps
                depth_frame = cv2.resize(depth_frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            else:
                depth_frame = None
            
            return rgb_frame, depth_frame
            
        except rospy.ServiceException as e:
            self.logger.error(f"Service call failed: {e}")
            return None, None
    
    def _capture_hardware_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Polls physical RealSense devices and applies post-processing filters."""
        try:
            frames = self.camera_pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            if hasattr(self, 'spatial_filter'):
                depth_frame = self.spatial_filter.process(depth_frame)
                depth_frame = self.temporal_filter.process(depth_frame)
                depth_frame = self.hole_filling_filter.process(depth_frame)
            
            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = depth_image.astype(np.float32) * self.depth_scale

            # Diagnostic logging for depth bounds validation
            in_range = ((depth_image >= 0.50) & (depth_image <= 0.90)).mean() * 100
            print(f"[DEBUG DEPTH] min={depth_image.min():.3f}  max={depth_image.max():.3f}  "
                  f"in_range(0.50-0.90m)={in_range:.1f}%")

            depth_image = np.where(
                (depth_image < self.depth_min) | (depth_image > self.depth_max),
                0, depth_image
            )
            
            return rgb_image, depth_image
            
        except Exception as e:
            self.logger.error(f"Error capturing hardware frames: {e}")
            return None, None

    # =========================================================================
    # COMPUTER VISION PIPELINE
    # =========================================================================

    def process_frames(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> Dict[str, Any]:
        """Main CV entry point for enhancement, segmentation, and feature extraction."""
        processed_data = {
            'rgb_frame': rgb_frame,
            'depth_frame': depth_frame,
            'timestamp': time.time()
        }
        
        if rgb_frame is None or depth_frame is None:
            return processed_data
        
        processed_data.update({
            'rgb_enhanced': self._enhance_rgb_image(rgb_frame),
            'depth_filtered': self._filter_depth_image(depth_frame),
            'depth_colormap': self._create_depth_colormap(depth_frame),
            'segmentation_mask': self._simple_segmentation(rgb_frame, depth_frame)
        })
        
        objects = self._detect_objects(rgb_frame, depth_frame)
        processed_data['detected_objects'] = objects
        
        if objects:
            grasp_candidates = self._calculate_grasp_candidates(objects, depth_frame)
            processed_data['grasp_candidates'] = grasp_candidates
        
        return processed_data
    
    def _enhance_rgb_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """Applies CLAHE and bilateral filtering for noise reduction."""
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _filter_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """Inpaints structural holes and applies median smoothing to depth arrays."""
        filtered_depth = depth_image.copy()
        filtered_depth[filtered_depth == 0] = np.nan
        
        filtered_depth = cv2.medianBlur(filtered_depth.astype(np.float32), 5)
        
        mask = np.isnan(filtered_depth).astype(np.uint8)
        if np.any(mask):
            depth_for_inpaint = filtered_depth.copy()
            depth_for_inpaint[np.isnan(depth_for_inpaint)] = 0
            filtered_depth = cv2.inpaint(
                depth_for_inpaint, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
            )
            filtered_depth[mask == 1] = np.nan
        
        return filtered_depth
    
    def _create_depth_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """Generates visual representation of the depth map."""
        depth_normalized = (depth_image - self.depth_min) / (self.depth_max - self.depth_min)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        
        return depth_colormap
    
    def _simple_segmentation(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """Isolates potential target objects using depth and morphological masking."""
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        depth_mask = ((depth_image > 0.3) & (depth_image < 1.5)).astype(np.uint8) * 255
        
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        combined_mask = cv2.bitwise_and(depth_mask, binary_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def _detect_objects(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> List[Dict]:
        """Extracts geometric contours and localizes object coordinates."""
        objects = []
        mask = self._simple_segmentation(rgb_image, depth_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 500:  
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            if (0 <= center_y < depth_image.shape[0] and 
                0 <= center_x < depth_image.shape[1]):
                object_depth = depth_image[center_y, center_x]
            else:
                continue
            
            if self.camera_matrix is not None:
                world_pos = self._pixel_to_world(center_x, center_y, object_depth)
            else:
                world_pos = [0, 0, object_depth]
            
            obj = {
                'id': i,
                'bbox': [x, y, w, h],
                'center_2d': [center_x, center_y],
                'center_3d': world_pos,
                'area': area,
                'depth': object_depth,
                'contour': contour
            }
            objects.append(obj)
        
        return objects
    
    def _calculate_grasp_candidates(self, objects: List[Dict], depth_image: np.ndarray) -> List[Dict]:
        """Proposes planar top-down grasp orientations based on object geometry."""
        grasp_candidates = []
        
        for obj in objects:
            contour = obj['contour']
            
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
            else:
                angle = 0
            
            center_3d = obj['center_3d']
            
            for grasp_angle in [0, 45, 90, 135]:
                grasp_orientation = angle + grasp_angle
                approach_vector = [0, 0, -1]  
                
                grasp_candidate = {
                    'object_id': obj['id'],
                    'position': center_3d,
                    'orientation': grasp_orientation,
                    'approach_vector': approach_vector,
                    'quality_score': 0.5,  
                    'grasp_type': 'top_down'
                }
                
                grasp_candidates.append(grasp_candidate)
        
        grasp_candidates.sort(key=lambda x: x['quality_score'], reverse=True)
        return grasp_candidates
    
    def _pixel_to_world(self, u: int, v: int, depth: float) -> List[float]:
        """Projects 2D image coordinates into 3D camera space."""
        if self.camera_matrix is None:
            return [0, 0, depth]
        
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return [x, y, z]

    # =========================================================================
    # ROS PUBLISHING & BRIDGE UPDATES
    # =========================================================================

    def _make_image_msg(self, header, data: np.ndarray, encoding: str) -> Image:
        """
        Manually constructs sensor_msgs/Image payloads avoiding cv_bridge constraints.
        Supports standard BGR and float/uint depths.
        """
        if CV_BRIDGE_AVAILABLE and self.bridge is not None:
            return self.bridge.cv2_to_imgmsg(data, encoding)

        msg = Image()
        msg.header = header
        msg.height, msg.width = data.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = 0
        if encoding == "bgr8":
            msg.step = msg.width * 3
            msg.data = data.astype(np.uint8).tobytes()
        elif encoding == "16UC1":
            msg.step = msg.width * 2
            msg.data = data.astype(np.uint16).tobytes()
        elif encoding == "32FC1":
            msg.step = msg.width * 4
            msg.data = data.astype(np.float32).tobytes()
        else:
            msg.step = msg.width * data.itemsize
            msg.data = data.tobytes()
        return msg

    def publish_frames(self, rgb_frame: np.ndarray, depth_frame: np.ndarray):
        """Dispatches active frames to the ROS topic network."""
        if not ROS_AVAILABLE:
            return

        try:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_link"

            if rgb_frame is not None:
                rgb_msg = self._make_image_msg(header, rgb_frame, "bgr8")
                self.rgb_pub.publish(rgb_msg)

            if depth_frame is not None:
                depth_mm = (depth_frame * 1000).astype(np.uint16)
                depth_msg = self._make_image_msg(header, depth_mm, "16UC1")
                self.depth_pub.publish(depth_msg)

            self._publish_camera_info(header)

        except Exception as e:
            self.logger.error(f"Error publishing frames: {e}")
    
    def _publish_camera_info(self, header: Header):
        """Publishes active matrix intrinsics and distortion mappings."""
        if self.camera_matrix is None:
            return
        
        camera_info = CameraInfo()
        camera_info.header = header
        camera_info.width = self.w
        camera_info.height = self.h
        
        camera_info.K = self.camera_matrix.flatten().tolist()
        
        if self.dist_coeffs is not None:
            camera_info.D = self.dist_coeffs.tolist()
        
        camera_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        camera_info.P = self.camera_matrix.flatten().tolist() + [0, 0, 0, 0]
        
        self.camera_info_pub.publish(camera_info)
    
    def get_average_distance(self, center_x: int, center_y: int, radius: int = 10) -> float:
        """Samples the local neighborhood depth avoiding nan/zero artifacts."""
        if self.current_depth_frame is None:
            return 0.0
        
        y1, y2 = max(0, center_y - radius), min(self.h, center_y + radius)
        x1, x2 = max(0, center_x - radius), min(self.w, center_x + radius)
        
        region = self.current_depth_frame[y1:y2, x1:x2]
        
        valid_depths = region[region > 0]
        if len(valid_depths) > 0:
            return np.mean(valid_depths)
        else:
            return 0.0
    
    def cleanup(self):
        """Safely terminates hardware streams."""
        if hasattr(self, 'camera_pipeline') and self.camera_pipeline is not None:
            self.camera_pipeline.stop()
            self.logger.info("Camera pipeline stopped")
    
    def __del__(self):
        self.cleanup()

    def update_from_webots(self):
        """
        Polls fresh frames from the associated WebotsBridge.
        Automatically routes arrays via bridge.capture_images() or capture_images2() 
        depending on the active _camera_robot_id.
        """
        if self.webots_bridge:
            if self._camera_robot_id == 2:
                rgb, depth = self.webots_bridge.capture_images2()
            else:
                rgb, depth = self.webots_bridge.capture_images()

            if rgb is not None and depth is not None:
                if rgb.shape[0] != self.h or rgb.shape[1] != self.w:
                    rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_AREA)
                if depth.shape[0] != self.h or depth.shape[1] != self.w:
                    depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
                
                self.current_rgb_frame   = rgb
                self.current_depth_frame = depth
                self.publish_frames(rgb, depth)
                return True
        return False


def create_camera_system(config_path: str = "config/camera_config.yaml",
                        simulation: bool = True,
                        camera_type: str = "realsense") -> 'EnhancedCameraHandler':
    """Factory builder for deploying the vision handling architecture."""
    return EnhancedCameraHandler(config_path, simulation, camera_type)