from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
# Fix for newer NumPy type hints
np.float = float
np.int = int
np.bool = bool
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from image_geometry import PinholeCameraModel
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_pose_stamped
import tf2_ros
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import gymnasium as gym
from gymnasium import spaces
import open_clip
import torch
import torch.nn.functional as F

from etrga_wrapper import ETRGA

# --------------------------------------------------------------------------- #
# Frames, topics, and constants                                               #
# --------------------------------------------------------------------------- #
BASE_FRAME      = "world"
EE_FRAME        = "rx150/ee_gripper_link"
CAM_OPT_FRAME   = "camera_link_optical"
RGB_TOPIC       = "/camera1/camera1/image_raw"
DEPTH_TOPIC     = "/camera1/depth_camera1/depth/image_raw"
CAM_INFO_TOPIC  = "/camera1/camera1/camera_info"

# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #

def robust_percentile(img: np.ndarray, lo: float = 5, hi: float = 95) -> Tuple[float, float]:
    """Return *lo*/*hi* percentiles while ignoring zero-pixels.

    Args:
        img: Input array (typically a depth map).
        lo: Low percentile.
        hi: High percentile.

    Returns:
        Tuple ``(vmin, vmax)`` with the requested percentiles. If *img* is
        entirely zeros the pair ``(0.0,1.0)`` is returned instead.
    """
    valid = img[img > 0]
    return np.percentile(valid, [lo, hi]) if valid.size else (0.0, 1.0)


class MetricReward:
    """Cosine-similarity reward between an RGB frame and a language prompt.

    The class embeds both the current observation and the goal text in the
    CLIP latent space and outputs a similarity scaled to the ``[0, 1]`` range.
    """

    def __init__(self, prompt: str, device: str = "cpu") -> None:
        """Pre-load CLIP and cache the text embedding.

        Args:
            prompt: Natural-language description of the goal (e.g. *"pick up
                the red cube"*).
            device: Execution device for CLIP (``"cpu"`` or ``"cuda"``).
        """
        self.device = device
        self.clip, self.pre, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai", device=device
        )
        self.clip.eval()
        with torch.no_grad():
            tok = open_clip.tokenize([prompt]).to(device)
            self.goal_vec = self.clip.encode_text(tok).float() / 100.0

    @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> float:
        """Compute similarity reward for *frame*.

        Args:
            frame: RGB image (``HxWx3`` uint8).

        Returns:
            Reward in ``[0,1]`` -1means perfect visual alignment with the
            prompt, 0means opposite.
        """
        img = self.pre(PILImage.fromarray(frame)).unsqueeze(0).to(self.device)
        sim = F.cosine_similarity(self.clip.encode_image(img).float(), self.goal_vec).item()
        return (sim + 1) / 2


class GazeboRX150Env(gym.Env):
    """Vision-based grasping task with a RobotisRX-150 arm in Gazebo."""

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------- #
    # Construction / lifecycle                                            #
    # ------------------------------------------------------------------- #

    def __init__(
        self,
        prompt: str = "pick up the red cube",
        img_size: int = 224,
        max_steps: int = 200,
        delta_xyz: float = 0.02,
        delta_rpy: float = math.radians(5),
    ) -> None:
        """Initialise the Gazebo-RX150 environment.

        Args:
            prompt: Language description of the task.
            img_size: Side length of square RGB observations.
            max_steps: Episode length before *truncation*.
            delta_xyz: Maximum translational change per step (metres).
            delta_rpy: Maximum rotational change per step (radians).
        """
        super().__init__()
        self.img_size = img_size
        self.max_steps = max_steps
        self.dpos = np.array([delta_xyz] * 3 + [delta_rpy] * 3, np.float32)

        # --- ROS 2 ------------------------------------------------------- #
        if not rclpy.ok():
            rclpy.init()
        self.node: Node = rclpy.create_node("gazebo_rx150_env")
        self.bridge = CvBridge()

        self.current_image: np.ndarray | None = None
        self.current_depth: np.ndarray | None = None
        self.cam_model = PinholeCameraModel()

        self.node.create_subscription(CameraInfo, CAM_INFO_TOPIC, self._cam_info_cb, qos_profile_sensor_data)
        self.node.create_subscription(Image, RGB_TOPIC, self._image_cb, qos_profile_sensor_data)
        self.node.create_subscription(Image, DEPTH_TOPIC, self._depth_cb, qos_profile_sensor_data)

        self.arm_pub = self.node.create_publisher(JointTrajectory, "/rx150/arm_controller/joint_trajectory", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        # --- Gym spaces -------------------------------------------------- #
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 255, shape=(img_size, img_size, 3), dtype=np.uint8)

        # --- ML components ---------------------------------------------- #
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric_reward = MetricReward(prompt, device)
        self.grasp_net = ETRGA(
            repo_root="ETRG-RGS",
            cfg="config/OCID-VLG/etrg_r50.yaml",
            ckpt="ckpts/etrg_r50.pth",
            device=device,
        )

        # --- bookkeeping ------------------------------------------------- #
        self.current_step = 0
        self.debug_dir = Path("./reward_debug_images").resolve()
        self.debug_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------- #
    # ROS callbacks                                                       #
    # ------------------------------------------------------------------- #

    def _image_cb(self, msg: Image) -> None:
        """Store latest RGB frame from *msg*."""
        self.current_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def _depth_cb(self, msg: Image) -> None:
        """Store latest depth frame from *msg* (metres)."""
        depth = self.bridge.imgmsg_to_cv2(msg)
        if msg.encoding == "16UC1":
            depth = depth.astype(np.float32) / 1000.0
        self.current_depth = depth

    def _cam_info_cb(self, msg: CameraInfo) -> None:
        """Update internal :pyclass:`~image_geometry.PinholeCameraModel`."""
        self.cam_model.fromCameraInfo(msg)

    # ------------------------------------------------------------------- #
    # Coordinate transforms                                               #
    # ------------------------------------------------------------------- #

    def transform_camera_to_world(self, pose_cam: np.ndarray) -> np.ndarray:
        """Transform *pose_cam* (camera optical) to world coordinates.

        Args:
            pose_cam: ``(6,)`` array ``[x, y, z, roll, pitch, yaw]`` in metres
                and radians w.r.t. *camera_link_optical*.

        Returns:
            Same pose in *world* frame. If TF lookup fails a zero vector is
            returned instead.
        """
        try:
            roll, pitch, yaw = pose_cam[3:6]
            quat_c = quaternion_from_euler(roll, pitch, yaw, axes="sxyz")

            pose_msg = PoseStamped()
            pose_msg.header.frame_id = CAM_OPT_FRAME
            pose_msg.header.stamp = self.node.get_clock().now().to_msg()
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = pose_cam[:3]
            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quat_c

            trans = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                CAM_OPT_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            pose_world = do_transform_pose_stamped(pose_msg, trans)

            

            return np.array([
                pose_world.pose.position.x,
                pose_world.pose.position.y,
                pose_world.pose.position.z,
                math.pi,  # roll - keep gripper pointing down
                0.0,      # pitch
                yaw,      # maintain model yaw
            ], np.float32)
        except Exception as exc:  # pylint: disable=broad-except
            self.node.get_logger().warn(f"[transform_camera_to_world] {exc}")
            return np.zeros(6, np.float32)

    # ------------------------------------------------------------------- #
    # Robot actuation                                                     #
    # ------------------------------------------------------------------- #

    def _publish_arm(self, joints: List[float]) -> None:
        """Publish a single "move in 1s" waypoint to the arm controller."""
        jt = JointTrajectory()
        jt.joint_names = ["shoulder", "elbow", "wrist_angle", "wrist_rotate", "waist"]
        pt = JointTrajectoryPoint()
        pt.positions = joints
        pt.time_from_start.sec = 1
        jt.points.append(pt)
        self.arm_pub.publish(jt)

    # Gymnasium API                                                       #

    def _eef_pose(self) -> np.ndarray:
        """
        Get the current end-effector pose in the world frame.

        Returns:
            np.ndarray of shape (6,) and dtype float32: 
            [x, y, z, roll, pitch, yaw].
            If TF lookup fails, returns zeros.
        """
        try:
            # Lookup transform from world → end-effector
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                EE_FRAME,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1),
            )
            # Translation
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            # Rotation → Euler angles
            q = trans.transform.rotation
            roll, pitch, yaw = euler_from_quaternion(
                [q.x, q.y, q.z, q.w], axes="sxyz"
            )

            return np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)

        except Exception as exc:
            self.node.get_logger().warn(f"[ _eef_pose ] TF lookup failed: {exc}")
            return np.zeros(6, dtype=np.float32)

    def _quality_at_eef(self, eef_world: np.ndarray, q_map: np.ndarray) -> float:
        """Return ETRGA quality **Q(u,v)** at the current end‑effector tip.

        Projects the 3‑D end‑effector point from *world* into the
        camera‑optical frame using TF and the ROS pinhole model, then samples
        the provided *q_map* (already at camera resolution).  If projection
        fails or lands outside the image bounds the function returns 0.0 so
        the SAC agent still receives a consistent reward signal.
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                "camera_link_optical", BASE_FRAME,
                rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1)
            )
            pt_w = np.array([*eef_world[:3], 1.0])
            T = tf2_ros.transformations.quaternion_matrix([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ])
            T[:3, 3] = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ]
            p_cam = T @ pt_w
            u, v = self.cam_model.project3dToPixel(p_cam[:3])
            u_i, v_i = int(round(u)), int(round(v))
            if 0 <= v_i < q_map.shape[0] and 0 <= u_i < q_map.shape[1]:
                return float(q_map[v_i, u_i])
        except Exception:
            pass  
        return 0.0

    @torch.no_grad()
    def step(self, residual_action: np.ndarray):  
        """Single SAC interaction step following Eq. (4).

        *Quality term* – we now sample **Q(u,v)** *at the actual end‑effector
        pixel* instead of relying on a scalar stored by the wrapper. The wrapper
        exposes an up‑scaled *quality map* matching the camera resolution as
        ``self.grasp_net.quality_map``.
        """
        β = 10.0
        self.current_step += 1

        #  Δp from SAC (scale to   ±5 cm / ±10°)
        dpos = np.array([0.05, 0.05, 0.05, math.radians(10), math.radians(10), math.radians(10)], np.float32)
        delta = residual_action.astype(np.float32) * dpos

        #  Desired EE pose
        prev_eef = self._eef_pose()
        desired_pose_world = prev_eef + delta
        joints = self._pose_to_joints(desired_pose_world)
        self._publish_arm(joints)
        rclpy.spin_once(self.node)
        eef = self._eef_pose()                       

        #  Observation after motion
        obs_rgb   = self.current_image.copy()
        obs_depth = self.current_depth.copy()

        # Critic pass – produces full quality map in camera frame
        _ = self.grasp_net.predict(obs_rgb, obs_depth, "pick up the red cube")
        q_map = self.grasp_net.quality_map            

        # quality at the actual EE pixel
        q_t1 = self._quality_at_eef(eef, q_map)

        clip_term = 0.1 * self.metric_reward(obs_rgb)

        reward = motion_penalty + β * q_t1 + clip_term
        if self._cube_lifted():
            reward += 5.0

        #  Termination, logging, return                                   
        truncated = self.current_step >= self.max_steps
        info = dict(eef_pose=eef, Δp=eef-prev_eef, q=q_t1, clip=clip_term)

        
        # Debug snapshot 
        if random.random() < 0.05:
            ts = int(time.time() * 1e3)
            PILImage.fromarray(obs_rgb).save(self.debug_dir / f"{ts}_rgb.png")
            depth_mm = (obs_depth * 1000).astype(np.uint16)
            cv2.imwrite(str(self.debug_dir / f"{ts}_depth.png"), depth_mm)
            (self.debug_dir / f"{ts}.txt").write_text(
                f"reward {reward:.4f}\nquality {q_t1:.4f}\nΔp {delta.tolist()}\n"\
                f"eef {eef.tolist()}\nprev {prev_eef.tolist()}\n")

        return obs_rgb, reward, False, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (not used)
            
        Returns:
            observation: Initial RGB image
            info: Empty dictionary
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.current_image = None
        
        # Wait for first camera image/depth
        while self.current_image is None and self.current_depth is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
        return self.current_image.copy(), {}

    def close(self):
        """
        Clean up resources when environment is closed.
        """
        self.node.destroy_node()
        rclpy.shutdown()