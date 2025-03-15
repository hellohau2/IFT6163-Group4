import pybullet as p
import pybullet_data
import numpy as np
import time
import os 
import imageio
import torch

from rx150_env import RX150Env

urdf_path = "/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/rx150.urdf"

'''
in console (if using lightning.ai) to forward the display to browser : 

Xvfb :99 -screen 0 1920x1080x24 & export DISPLAY=:99
x11vnc -display :99 -rfbport 5901 -shared -forever &
websockify --web=/usr/share/novnc 8090 localhost:5901 
'''

from stable_baselines3 import DDPG,TD3,PPO,SAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

rx_env = RX150Env(
    urdf_path=urdf_path, 
    headless=True,
    max_timesteps=1000,
    image_height=84,
    image_width=84
)

# Create model
model = SAC("MultiInputPolicy", rx_env, verbose=1, device=device)

# Train
model.learn(total_timesteps=50_000)

# Save the trained model
model.save("sac_rx150")

rx_env.close()