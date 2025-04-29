import gymnasium as gym
from gymnasium import ObservationWrapper
import pybullet as p
from gymnasium.core import RewardWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from robot_env import RobotEnv
from clip_reward import CLIPReward
import cv2
import os
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

class PrintStuffCallback(BaseCallback):
    def __init__(self, verbose=2):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and isinstance(infos, list):
            info = infos[0]
            robot_id = info.get("robot_id", None) 
            if not robot_id or robot_id == -1:  
                print("Skipping invalid robot_id logging")
                return True
        return True

class FullLoggingCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_steps = 0
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos or not isinstance(infos, list):
            return True
        
        info = infos[0]
        robot_id = info.get("robot_id", None)
        
        if not robot_id or robot_id == -1:
            print("Skipping invalid robot_id logging")
            return True
        
        current_step = self.num_timesteps
        actions = self.locals.get("actions", None)

        # Extract positions from info
        gripper_pos = info.get("gripper_pos", (0,0,0))
        cube_pos = info.get("cube_pos", (0,0,0))
        
        # Calculate distance metrics
        horizontal_dist = np.linalg.norm(gripper_pos[:2] - cube_pos[:2])
        vertical_dist = abs(gripper_pos[2] - cube_pos[2])
        
        # Get joint states
        joint_states = p.getJointStates(robot_id, [0, 1, 2, 3])
        joint_positions = [s[0] for s in joint_states]

         # Log joint positions with URDF names
        joint_names = info.get("joint_names", [])
        for name in joint_names:
            pos = info.get(f"{name}_pos", 0.0)
            wandb.log({f"joint_positions/{name}": pos}, commit=False)

        # Log metrics to WandB
        wandb.log({
            "total_reward": info.get("total_reward", 0),
            "clip_goal_similarity": info.get("clip_goal_similarity", 0),
            "clip_init_similarity": info.get("clip_init_similarity", 0),
            "geometric_reward": info.get("geometric_reward", 0),
            "horizontal_distance": horizontal_dist,
            "vertical_distance": vertical_dist,
            "gripper_cube_dist": info.get("gripper_cube_dist", 0),
            "success": float(info.get("success", 0)),
            "joint_pos_0": joint_positions[0],
            "joint_pos_1": joint_positions[1],
            "joint_pos_2": joint_positions[2],
            "joint_pos_3": joint_positions[3],
            "action_norm": np.linalg.norm(actions[0]) if actions is not None else 0.0,
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_steps,
            "contact_reward": info["contact_reward"],
            "in_contact": info["in_contact"],
            "step": self.num_timesteps
        }, commit=True, step=current_step)

        # Update episode tracking
        self.episode_reward += info.get("total_reward", 0)
        self.episode_steps += 1

        if info.get("episode_done", False):
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_steps)
            self.episode_reward = 0.0
            self.episode_steps = 0

        return True

class PrintCallback(BaseCallback):
    def __init__(self, verbose=2):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        current_step = self.model.num_timesteps
        if current_step % 100 == 0:
            print(f"[PrintCallback] At timestep {current_step}")
            print(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards'][0]}")
        return True

class ResizeObservation(ObservationWrapper):
    def __init__(self, env, shape=(64, 64)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, *shape), dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return obs.transpose(2, 0, 1)

# Initialize environment and reward model
env = RobotEnv()
env.render_mode = "rgb_array" 

print("[TRAIN] Initializing CLIPReward...") #DEBUG
reward_model = CLIPReward(goal_text="a robot arm grasping a red block")
print("[TRAIN] CLIPReward initialized!") #DEBUG

class ClipRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model
        self.step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Save debug frame (BGR for OpenCV) while keeping obs as RGB for CLIP
        #cv2.imwrite(f"debug/frame_{self.step_count}.png", obs)
        #self.step_count += 1

        gripper_pos = info.get("gripper_pos", np.zeros(3))
        cube_pos = info.get("cube_pos", np.zeros(3))

        reward_dict = reward_model.compute_reward(obs, gripper_pos, cube_pos, info.get('success', False))
        total_reward = reward_dict['total_reward']

        info.update({
            "total_reward": total_reward,
            "clip_goal_similarity": reward_dict['goal_sim'],
            "clip_init_similarity": reward_dict['init_sim'],
            "geometric_reward": reward_dict['geometric_reward'],
            "gripper_pos": gripper_pos,
            "cube_pos": cube_pos
        })

        # Critical Fix: Preserve original termination signals
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_model.init_features = None
        self.reward_model.init_image = None
        #self.step_count = 0
        return obs, info

# Environment setup
env = ClipRewardWrapper(env, reward_model=reward_model)
env = ResizeObservation(env, shape=(64, 64))
vec_env = DummyVecEnv([lambda: env])

# Video recording
vec_env = VecVideoRecorder(
    vec_env,
    "logs/videos/",
    record_video_trigger=lambda x: x % 1000 == 0,
    video_length=200,
    name_prefix="rx150_training",
)

# Wandb initialization
wandb.init(
    project="reactor_x150_rl",
    config={
        "policy_type": "CnnPolicy",
        "total_timesteps": 1_000_000,
        "env_name": "ReactorX150Env",
        "clip_model": "ViT-B/32",
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 512,
        "hybrid_reward_ratio": 0.7,
    },
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

wandb.define_metric("step")
wandb.define_metric("*", step_metric="step")

def lr_schedule(progress_remaining: float) -> float:
    # progress_remaining ∈ [1.0 → 0.0]
    # Suppose you want 1e–4 for the first 80 k steps of a 100 k run,
    # then 1e–5 afterwards:
    threshold = 0.2  # bottom 20% of training
    return 1e-5 if progress_remaining < threshold else 1e-4

# Model setup
model = SAC(
    "CnnPolicy", 
    vec_env,
    policy_kwargs={'net_arch': [256, 256]}, # Default architecture works for 5D actions
    learning_rate=lr_schedule, 
    ent_coef="auto_0.01", # Enable entropy auto-tuning
    target_entropy  = "auto",
    verbose=2,
    buffer_size=100_000,
    batch_size=256, # Reduced from 512
    learning_starts=1000, #Increased from 100
    tensorboard_log="./logs/"
)


checkpoint_cb = CheckpointCallback(
    save_freq=10_000,
    save_path="models/",
    name_prefix="sac_103"
)

# Callbacks
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    model_save_path="models/",
    verbose=2
)

callback_list = CallbackList([
    wandb_callback,
    checkpoint_cb,
    PrintStuffCallback(verbose=2),
    PrintCallback(verbose=2),
    FullLoggingCallback()
])

# Training
print("Starting training...")
model.learn(
    total_timesteps=100_000,  # Reduced for initial testing
    callback=callback_list,
    log_interval=1,
    progress_bar=True
)
print("Finished training.")

wandb.finish()
model.save("sac_reactor_x150")