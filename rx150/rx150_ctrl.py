import pybullet as p
import pybullet_data
import numpy as np
import time
import os 
import imageio

from rx150.rx150_env import RX150Env
from stable_baselines3 import SAC

'''
    Basic code to view the trained SAC
    Press 'R' to reset the environment
'''

urdf_path = "/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/rx150.urdf"

model = SAC.load("sac_rx150")

total_it = 0

rx_env = RX150Env(
    urdf_path=urdf_path, 
    headless=False,
    max_timesteps=1_000_000,
    image_height=84,
    image_width=84
)

rx_env.reset()

while True : 

    keys = p.getKeyboardEvents()
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        rx_env.reset()

    total_it += 1

    ob = rx_env.get_obs()
    action = model.predict(ob)[0]

    _,rw,done,_,_ = rx_env.step(action)

    if total_it % 10 == 0 :
        print(f"reward : {rw}, sqr dist : {((np.array(rx_env.get_end_effector_pos()) - rx_env.target_pos)**2).sum()}")

    if done : 
        print("TARGET REACHED")
        break

    time.sleep(0.005)
        
rx_env.close()

