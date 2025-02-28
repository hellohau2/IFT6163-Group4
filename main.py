import pybullet as p
import pybullet_data
import time
import os

# Xvfb :99 -screen 0 1920x1080x24 & export DISPLAY=:99

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the URDF
urdf_path = os.path.join("/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/rx150.urdf")

robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])

for _ in range(10000):
    p.stepSimulation()
    time.sleep(0.01)

p.disconnect()



