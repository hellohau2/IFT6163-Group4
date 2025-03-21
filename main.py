'''
in console (if using lightning.ai) to forward the display to browser : 

Xvfb :99 -screen 0 1920x1080x24 & export DISPLAY=:99
x11vnc -display :99 -rfbport 5901 -shared -forever &
websockify --web=/usr/share/novnc 8090 localhost:5901 
'''

from stable_baselines3 import DDPG,TD3,PPO,SAC
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import pybullet as p
import pybullet_data
import numpy as np
import time
import imageio
import torch

from rx150.rx150_env import RX150Env

urdf_path = "/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/rx150.urdf"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

def make_env():
    env = RX150Env(
        urdf_path=urdf_path, 
        headless=True,
        max_timesteps=1000,
        image_height=64,
        image_width=64
    )
    env = Monitor(env, log_dir)
    return env

if __name__ == "__main__":
    
    # Create one env per CPU CORE
    envs = SubprocVecEnv([make_env for _ in range(12)])

    # Create model
    model = SAC("CnnPolicy", envs, verbose=1, device=device)

    # Train
    model.learn(total_timesteps=500_000)

    # Save the trained model
    model.save("sac_rx150_expl")
