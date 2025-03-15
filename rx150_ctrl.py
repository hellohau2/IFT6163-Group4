import pybullet as p
import pybullet_data
import numpy as np
import time
import os 
import imageio

from rx150_env import RX150Env

'''
Basic code to test the simulation env with random actions
'''

urdf_path = "/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/rx150.urdf"

total_it = 0

rx_env = RX150Env(urdf_path=urdf_path,max_timesteps=10000000)
rx_env.reset()

while True : 
    total_it += 1

    if total_it % 250 == 0 :

        # action = np.random.choice([-1,0,1], size=5)
        # action = np.append(action,np.random.choice([0,1],size=1))

        action = np.random.uniform(-1,1,size=5)
        action = np.append(action,np.random.uniform(0,1,size=1))

        discrete_ac = np.zeros(6, dtype=int)

        for i in range(5):
            discrete_ac[i] = int(np.clip(round(action[i]),-1,1))
        discrete_ac[5] = int(np.clip(round(action[5]),0,1))

        print(f"New action generated : {action}\nDiscrete action : {discrete_ac}\nReceived floats : {isinstance(action[0], float)}")

    obs,rw,done,_ = rx_env.step(action)

    if total_it % 250 == 0 :
        print(f"obs : {obs} \nreward : {rw}\ndone : {done}")

    if done : break

    time.sleep(0.005)
        
rx_env.close()

