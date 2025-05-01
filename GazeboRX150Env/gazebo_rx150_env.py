import os, time, math, random, numpy as np, cv2, torch, open_clip, rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.node import Node
from PIL import Image as PIL
import gymnasium as gym
from gymnasium import spaces

class GazeboRX150Env(gym.Env):
    def __init__(self, img_size=224, centroid_file="centroids.pt"):
        super().__init__()
        self.img_size, self.max_steps, self.current_step = img_size, 200, 0
        rclpy.init(args=None); self.node = Node("gazebo_rx150_env")
        self.bridge, self.current_image = CvBridge(), None
        self.node.create_subscription(Image, "/camera1/camera1/image_raw",
                                      self._cb, 10)
        # publishers
        self.arm_pub = self.node.create_publisher(
            JointTrajectory, "/rx150/arm_controller/joint_trajectory", 10)
        self.grip_pub = self.node.create_publisher(
            JointTrajectory, "/rx150/gripper_controller/joint_trajectory", 10)

        self.action_space = spaces.Box(-1., 1., (6,), np.float32)
        self.observation_space = spaces.Box(0, 255,
                                            (img_size, img_size, 3), np.uint8)

        # CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, _, self.prep = open_clip.create_model_and_transforms(
            "ViT-L-14-quickgelu", pretrained="openai", device=self.device)
        self.clip.eval()

        # centroids
        if not os.path.exists(centroid_file):
            raise FileNotFoundError("Run build_centroids.py first")
        self.cents = torch.load(centroid_file, map_location=self.device)
        self.weights = {"success":10, "partial_close":4,
                        "partial_far":2, "fail":-10}

        self.debug_dir = "./reward_debug_images"; os.makedirs(self.debug_dir, exist_ok=True)

    def _cb(self, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    @torch.no_grad()
    def _reward(self, img):
        if img is None: return 0.
        feat = self.clip.encode_image(self.prep(PIL.fromarray(img)).unsqueeze(0).to(self.device))
        feat = feat/feat.norm(dim=-1, keepdim=True)
        r, dbg = 0., {}

        for cls,w in self.weights.items():
            if cls not in self.cents: 
                continue
            s = (feat @ self.cents[cls].to(self.device).T).item()
            r += w*s
            dbg[cls]=s

        if random.random()<.05:
            ts=str(int(time.time()*1e3))
            PIL.fromarray(img).save(f"{self.debug_dir}/{ts}.png")
            with open(f"{self.debug_dir}/{ts}.txt","w") as f:
                for k,v in dbg.items(): 
                    f.write(f"{k} {v:.3f}\n")
                    f.write(f"reward {r:.3f}\n")
        return r

    def step(self, action):
        self.current_step += 1
        arm, grip = action[:5], action[5]
        # publish arm
        jt = JointTrajectory(); jt.joint_names=["shoulder","elbow","wrist_angle","wrist_rotate","waist"]
        pt = JointTrajectoryPoint()
        pt.positions=(np.clip(arm,-1,1)*(math.pi/2)).tolist()
        pt.time_from_start.sec=1
        jt.points.append(pt)
        self.arm_pub.publish(jt)
        # publish gripper
        gt = JointTrajectory()
        gt.joint_names=["left_finger","right_finger"]
        gp = JointTrajectoryPoint()
        p=float(np.clip(grip,-1,1)*0.3)
        gp.positions=[p,p]
        gp.time_from_start.sec=1
        gt.points.append(gp)
        self.grip_pub.publish(gt)

        rclpy.spin_once(self.node)
        obs = self.current_image.copy()
        rew=self._reward(obs)
        trunc = self.current_step>=self.max_steps
        return obs, rew, False, trunc, {}

    def reset(self,*_,**__):
        self.current_step=0; self.current_image=None
        while self.current_image is None: 
            rclpy.spin_once(self.node,timeout_sec=0.1)
        return self.current_image.copy(),{}

    def close(self): 
        self.node.destroy_node()
        rclpy.shutdown()
