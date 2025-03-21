import pybullet as p
import pybullet_data
import numpy as np
import time
import os 
import imageio
import gymnasium as gym
from gymnasium.spaces import Box,Dict

from transformers import CLIPProcessor, CLIPModel
import torch
import rx_utils

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

    '''
    Basic env for the RX150 robot arm. The goal is simply for the end effector (joint 11 : 'ee_gripper') to reach a target position (randomly generated around the center).
    More information about the observations, actions and rewards are explained in the functions below.
    '''

    def __init__(
        self,
        urdf_path, 
        epsilon_dist=0.1, 
        max_timesteps=1000, 
        step_size=0.1, 
        headless=False, 
        image_width=480, 
        image_height=640, 
        goal_prompt = "A 3D model of a robot arm and a red dot with a green end-effector. The robot arm's green end effector is not touching the red dot",
        baseline_prompt = "A 3D model of a robot arm and a red dot with a green end-effector.",
        clip_reg_alpha = 0.5,
        frame_skip = 4,
        use_intrinsic=True,
        image_only=True
        ):

        '''
            urdf_path (str) : path to urdf for robot arm
            epsilon_dist (float) : min distance from target position to consider it done
            max_timesteps (int) : max number of timesteps for the episode
            step_size (float) : the incremental value of the joint angle (radian)
        '''

        self.urdf_path = urdf_path
        self.joint_list = [0,1,2,3,4,9,10]
        self.epsilon_dist = epsilon_dist
        self.sqr_epsilon_dist = epsilon_dist**2
        self.max_timesteps = max_timesteps
        self.step_size = step_size
        self.image_height = image_height
        self.image_width = image_width
        self.clip_reg_alpha = clip_reg_alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_skip = frame_skip
        self.image_only=image_only
        self.use_intrinsic=use_intrinsic

        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.25, 0, -0.25],
            distance=2.5,
            yaw=90,
            pitch=-40,
            roll=0,
            upAxisIndex=2
        )

        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=90,
            aspect=float(image_width)/image_height,
            nearVal=0.1,
            farVal=100.0
        )

        if not self.use_intrinsic :
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # Goal / Baseline 
            goal_text_input = self.clip_processor(text=[goal_prompt], return_tensors="pt", padding=True).to(self.device)
            baseline_text_input = self.clip_processor(text=[baseline_prompt], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                goal_text_features = self.clip_model.get_text_features(**{k: goal_text_input[k] for k in ["input_ids", "attention_mask"]})
                baseline_text_features = self.clip_model.get_text_features(**{k: baseline_text_input[k] for k in ["input_ids", "attention_mask"]})

            self.goal_norm_text_features = goal_text_features / goal_text_features.norm(dim=-1, keepdim=True)
            self.baseline_norm_text_features = baseline_text_features / baseline_text_features.norm(dim=-1, keepdim=True)

            self.goal_baseline_line = self.goal_norm_text_features - self.baseline_norm_text_features
            self.goal_baseline_norm_line = self.goal_baseline_line / self.goal_baseline_line.norm(dim=-1, keepdim=True)

        # Use RND to get an exploration reward as seen in class
        else : 
            self.expl_net = rx_utils.ExplNet(out_dim=64,lr=1e-3).to(self.device)

        self.action_space = Box(low=-1, high=1, shape=(6,))

        if self.image_only : 
            self.observation_space = Box(low=0, high=255, shape=(self.image_height, self.image_width, 3), dtype=np.uint8)
        else :             
            self.observation_space = Dict({
                "image": Box(low=0, high=255, shape=(self.image_height, self.image_width, 3), dtype=np.uint8),
                "joint_states": Box(low=-4, high=4, shape=(len(self.joint_list),), dtype=np.float32)
            })


        self.t = 0

        if headless : 
            p.connect(p.DIRECT)
            self.renderer = p.ER_TINY_RENDERER
        else : 
            p.connect(p.GUI)
            self.renderer = p.ER_BULLET_HARDWARE_OPENGL

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.last_state = {}
    
    def reset(self, seed = None, option = None):
        '''
        Resets the environment and generates a target pos (displayed as a red sphere)
        The current sphere pos generator is using rejection sampling and is very inefficient
        '''

        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0,0,-9.81)

        # it = 0
        # while True:
        #     it += 1
        #     target_pos = np.random.uniform(-1.5,1.5,size=3)
        #     if np.linalg.norm(target_pos) >= 1 and np.linalg.norm(target_pos) <= 1.5 and target_pos[2] > 0.25 : 
        #         break

        target_pos = np.array([1.5,1,0.5])
        
        # print(f"Generated random target pos in {it} iterations")

        sphere_radius = 0.05
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=sphere_radius,
            rgbaColor=[1, 0, 0, 1] # RGBA : only red here
        )

        self.target_id = p.createMultiBody(
            baseMass=0,  # Mass = 0 so it doesn't fall
            baseVisualShapeIndex=visual_shape_id,
            basePosition=target_pos,
            baseCollisionShapeIndex=-1  # No collision
        )

        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_path,useFixedBase=True,globalScaling=4)

        # Recolors the robot arm
        num_joints = p.getNumJoints(self.robot_id)

        # Loop over all links to recolor them
        for link_idx in range(-1,num_joints):
    
            color = [0, 0, 0, 1]

            # If end effector, make it green
            if link_idx >= 6 :
                color = [0, 1, 0, 1]

            p.changeVisualShape(
                objectUniqueId=self.robot_id,
                linkIndex=link_idx,
                rgbaColor=color
            )

        self.target_pos = np.array(target_pos)

        self.t = 0

        return self.get_obs(), {}

    def render(self):

        _,_,rgbPixels,_,_ = p.getCameraImage(
            width=self.image_width,
            height = self.image_height,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            renderer= self.renderer
        )

        return np.reshape(np.array(rgbPixels),(self.image_height,self.image_width,4))[:,:,:3]

    def get_obs(self):
        '''
        Returns the observation of the current state.
        Arbitrarily chosen for now : 
            - the angle (radians) of each joint (i.e joints [0,1,2,3,4] and [9,10] for the gripper fingers)
            - The gripper radians can be changed later for a binary value (open or close gripper)
            - image of the rendering shape (self.image_height,self.image_width,3)
        '''
        
        if self.image_only:
            return self.render()

        ob = []
        for joint_id in self.joint_list:
            curr_radian = p.getJointState(self.robot_id,joint_id)[0]
            ob.append(curr_radian)
        
        # Also add the target's position
        # ob.extend(self.target_pos)

        return {
            "image": self.render(), 
            "joint_states": np.array(ob),  
        }

    def get_reward_and_terminal(self,ob = None):
        '''
        Uses the CLIP model to get the CLIP-Reg (from paper) rewards 
        and if the environment is done or not (if gripper is within epsilon distance from target pos).
        '''
        
        sqr_dist_to_target = ((np.array(self.get_end_effector_pos()) - self.target_pos)**2).sum()
        done = (self.sqr_epsilon_dist >= sqr_dist_to_target) or (self.t >= self.max_timesteps)

        if ob is None :
            # Get image, pass it through the clip model, compute similarity
            rx_img = self.render()
        else :
            if self.image_only :
                rx_img = ob
            else : 
                rx_img = ob['image']

        # Use exploration RND as reward
        if self.use_intrinsic:
            
            x = torch.tensor(rx_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            x = x.to(self.device)

            reward = self.expl_net(x).item()
            self.expl_net.update(x)

        # Otherwise just use CLIP rewards (not effective)
        else : 
            image_inputs = self.clip_processor(images=rx_img, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**{k: image_inputs[k] for k in ["pixel_values"]})
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Projection of s (image) to the line spanned by goal - baseline
            proj_img = (self.goal_baseline_norm_line @ image_features.T) * self.goal_baseline_norm_line

            # Clip regularized with the projection
            clip_reg = 1 - 0.5 * torch.sum(((self.clip_reg_alpha * proj_img + (1 - self.clip_reg_alpha) * image_features - self.goal_baseline_norm_line)) ** 2, dim=-1)
            
            # similarity_score = (self.goal_norm_text_features @ image_features.T).squeeze(0)

            # return -sqr_dist_to_target, done
            # return similarity_score.item(), done
            
            # print(f"Goal-baseline line norm : {self.goal_baseline_line.norm(dim=-1, keepdim=True).item()} , Projection norm : {proj_img.norm(dim=-1, keepdim=True).item()} , Reg 1 : {(((self.clip_reg_alpha * proj_img + (1 - self.clip_reg_alpha) * self.goal_baseline_line).norm(dim=-1, keepdim=True))**2)}, Reg 2. : {torch.sum(((self.clip_reg_alpha * proj_img + (1 - self.clip_reg_alpha) * self.goal_baseline_line)) ** 2, dim=-1)}")
            reward = clip_reg.item()
        
        return reward , done

    def get_end_effector_pos(self):
        ''' Returns world position of end effector 'ee_gripper' (joint_id 11) '''
        return p.getLinkState(self.robot_id, 11)[0]

    def step(self, action, simulation_steps = 4):

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

        simulation_steps : int indicating the number of frames to simulate (apply the same action). The next observation and reward are obtained after simulation_steps
        '''
        self.t += 1
        
        # Checks if received floats, in which case round them to nearest correct action values
        if isinstance(action[0], float):
            discrete_ac = np.zeros(6, dtype=int)

            for i in range(5):
                discrete_ac[i] = int(np.clip(round(action[i]),-1,1))
            discrete_ac[5] = int(np.clip(round(action[5]),0,1))

        else :
            discrete_ac = action

        # Apply the action to each joint
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

        # Let the simulation roll for a few frames to allow movement
        for _ in range(simulation_steps) : 
            p.stepSimulation()

        ob = self.get_obs()
        reward,done = self.get_reward_and_terminal(ob)
        truncated = self.t >= self.max_timesteps
        info = {}

        return ob , reward, done,truncated, info

    def close(self):
        p.disconnect()