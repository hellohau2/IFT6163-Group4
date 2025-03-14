import pybullet as p
import pybullet_data
import numpy as np
import time
import os 
import imageio
import gym
from gym.spaces import Box

'''
RX150 Joint list (from URDF) : 

    0 : b'waist' , LowerLimit : -3.141582653589793 , UpperLimit : 3.141582653589793, Joint type : JOINT_REVOLUTE
    1 : b'shoulder' , LowerLimit : -1.8500490071139892 , UpperLimit : 1.7453292519943295, Joint type : JOINT_REVOLUTE
    2 : b'elbow' , LowerLimit : -1.7802358370342162 , UpperLimit : 1.6580627893946132, Joint type : JOINT_REVOLUTE
    3 : b'wrist_angle' , LowerLimit : -1.7453292519943295 , UpperLimit : 2.1467549799530254, Joint type : JOINT_REVOLUTE
    4 : b'wrist_rotate' , LowerLimit : -3.141582653589793 , UpperLimit : 3.141582653589793, Joint type : JOINT_REVOLUTE
    
    5 : b'ee_arm' , LowerLimit : 0.0 , UpperLimit : -1.0, Joint type : JOINT_FIXED
    6 : b'gripper' , LowerLimit : 0.0 , UpperLimit : -1.0, Joint type : JOINT_REVOLUTE
    7 : b'gripper_bar' , LowerLimit : 0.0 , UpperLimit : -1.0, Joint type : JOINT_FIXED
    8 : b'ee_bar' , LowerLimit : 0.0 , UpperLimit : -1.0, Joint type : JOINT_FIXED

    9 : b'left_finger' , LowerLimit : 0.015 , UpperLimit : 0.037, Joint type : JOINT_PRISMATIC
    10 : b'right_finger' , LowerLimit : -0.037 , UpperLimit : -0.015, Joint type : JOINT_PRISMATIC
    
    11 : b'ee_gripper' , LowerLimit : 0.0 , UpperLimit : -1.0, Joint type : JOINT_FIXED
'''


class RX150Env(gym.Env):

    def __init__(self,urdf_path, epsilon_dist=0.1, max_timesteps=1000, step_size=0.1, headless=False):
        '''
            urdf_path (str) : path to urdf for robot arm
            epsilon_dist (float) : min distance from target position to consider it done
            max_timesteps (int) : max number of timesteps for the episode
            step_size (float) : the incremental value of the joint angle (radian)
        '''

        self.action_space = Box(low=-1, high=1, shape=(6,))
        self.observation_space = Box(low=-4,high=4,shape=(10,))

        self.urdf_path = urdf_path
        self.joint_list = [0,1,2,3,4,9,10]
        self.epsilon_dist = epsilon_dist
        self.sqr_epsilon_dist = epsilon_dist**2
        self.max_timesteps = max_timesteps
        self.step_size = step_size

        self.t = 0

        if headless : p.connect(p.DIRECT)
        else : p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    
    def reset(self):
        '''
        Resets the environment and generates a target pos (displayed as a red sphere)
        The current sphere pos generator is using rejection sampling and is very inefficient
        '''

        p.resetSimulation()
        p.setGravity(0,0,-9.81)

        it = 0
        while True:
            it += 1
            target_pos = np.random.uniform(-1.5,1.5,size=3)
            if np.linalg.norm(target_pos) >= 1 and np.linalg.norm(target_pos) <= 1.5 and target_pos[2] > 0.25 : 
                break
        
        print(f"Generated random target pos in {it} iterations")

        sphere_radius = 0.05
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=sphere_radius,
            rgbaColor=[1, 0, 0, 1] # RGBA : only red here
        )

        self.target_id = p.createMultiBody(
            baseMass=0,  # Mass = 0 so it's fixed
            baseVisualShapeIndex=visual_shape_id,
            basePosition=target_pos,
            baseCollisionShapeIndex=-1  # No collision
        )

        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_path,useFixedBase=True,globalScaling=4)

        self.target_pos = np.array(target_pos)

        self.t = 0

        return self.get_obs()

    def get_obs(self):
        '''
        Returns the observation of the current state.
        Arbitrarily chosen for now : 
            - the angle (radians) of each joint (i.e joints [0,1,2,3,4] and [9,10] for the gripper fingers)
            - The gripper radians can be changed later for a binary value (open or close gripper)
            - target (x,y,z) world position
        '''

        ob = []
        for joint_id in self.joint_list:
            curr_radian = p.getJointState(self.robot_id,joint_id)[0]
            ob.append(curr_radian)
        
        ob.extend(self.target_pos)

        return np.array(ob)

    def get_reward_and_terminal(self):
        '''
        Returns negative squared distance between the end effector and the sphere's (target) position
        and if the environment is done or not.
        '''
        sqr_dist_to_target = ((np.array(self.get_end_effector_pos()) - self.target_pos)**2).sum()

        done = (self.sqr_epsilon_dist >= sqr_dist_to_target) or (self.t >= self.max_timesteps)

        return -sqr_dist_to_target, done

    def get_end_effector_pos(self):
        ''' Returns world position of end effector 'ee_gripper' (joint_id 11) '''
        return p.getLinkState(self.robot_id, 11)[0]

    def step(self, action, skip_frames = 4):

        '''
        action is an np.array of 6 int values each being either [-1,0,1]
        -1 means decrease angle of joint, 0 means don't change angle, 1 increases angle
        The last index is the gripper, 0 means don't grip, 1 means grip.

            Index : 
                0 : Waist joint
                1 : Shoulder joint
                2 : Elbow joint
                3 : Wrist angle
                4 : Wrist rotate
                5 : Gripper

        skip_frames : int indicating the number of frames to 'skip' (apply the same action). The next observation and reward are obtained after skip_frames
        '''
        
        # Checks if received floats, in which case round them to nearest correct action values
        if isinstance(action[0], float):
            discrete_ac = np.zeros(6, dtype=int)

            for i in range(5):
                discrete_ac[i] = int(np.clip(round(action[i]),-1,1))
            discrete_ac[5] = int(np.clip(round(action[5]),0,1))

        else :
            discrete_ac = action

        for joint_id,ac in enumerate(discrete_ac):
            
            # If using gripper
            
            if joint_id == 5 : 
                
                # Close gripper : decrease left finger increase right finger
                # Open gripper : opposite

                if ac :
                    left_finger_step = -1
                    right_finger_step = 1
                else : 
                    left_finger_step = 1
                    right_finger_step = -1
                
                # Store these values in init instead of calling the api every frame
                current_pos_left_finger = p.getJointState(self.robot_id,9)[0]
                current_pos_right_finger = p.getJointState(self.robot_id,10)[0]

                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=9,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition= current_pos_left_finger + self.step_size * left_finger_step,
                    force=1
                )

                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=10,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition= current_pos_right_finger + self.step_size * right_finger_step,
                    force=1
                )

            else : 
                # Avoid unnecessary calls to API
                if ac == 0 : continue

                joint_max_force = p.getJointInfo(self.robot_id,joint_id)[10]

                current_pos = p.getJointState(self.robot_id,joint_id)[0]

                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=current_pos + self.step_size * ac,
                    force=joint_max_force
                )

        self.t += 1

        for _ in range(skip_frames) : 
            p.stepSimulation()

        reward,done = self.get_reward_and_terminal()
        info = {}

        return self.get_obs(), reward, done, info

    def close(self):
        p.disconnect()