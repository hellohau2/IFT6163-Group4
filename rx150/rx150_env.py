import pybullet as p
import pybullet_data
import numpy as np
import time
import os 
import imageio
import gymnasium as gym
from gymnasium.spaces import Box,Dict

# from transformers import CLIPProcessor, CLIPModel
import torch

import rx150.rx_utils as rx_utils
# import rx_utils

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

# class RX150Env(gym.Env):

#     '''
#     Basic env for the RX150 robot arm. The goal is simply for the end effector (joint 11 : 'ee_gripper') to reach a target position (randomly generated around the center).
#     More information about the observations, actions and rewards are explained in the functions below.
#     '''

#     def __init__(
#         self,
#         urdf_path, 
#         epsilon_dist=0.1, 
#         max_timesteps=1000, 
#         step_size=0.1, 
#         headless=False, 
#         image_width=64, 
#         image_height=64, 
#         goal_prompt = "A 3D model of a robot arm and a red dot with a green end-effector. The robot arm's green end effector is not touching the red dot",
#         baseline_prompt = "A 3D model of a robot arm and a red dot with a green end-effector.",
#         clip_reg_alpha = 0.5,
#         frame_skip = 4,
#         use_intrinsic=True,
#         image_only=True,
#         model=None,
#         ):

#         '''
#             urdf_path (str) : path to urdf for robot arm
#             epsilon_dist (float) : min distance from target position to consider it done
#             max_timesteps (int) : max number of timesteps for the episode
#             step_size (float) : the incremental value of the joint angle (radian)
#         '''

#         self.urdf_path = urdf_path
#         self.joint_list = [0,1,2,3,4,9,10]
#         self.epsilon_dist = epsilon_dist
#         self.sqr_epsilon_dist = epsilon_dist**2
#         self.max_timesteps = max_timesteps
#         self.step_size = step_size
#         self.image_height = image_height
#         self.image_width = image_width
#         self.clip_reg_alpha = clip_reg_alpha
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.frame_skip = frame_skip
#         self.image_only=image_only
#         self.use_intrinsic=use_intrinsic

#         self.model = model

#         self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
#             cameraTargetPosition=[0.25, 0, -0.25],
#             distance=1.75,
#             yaw=90,
#             pitch=-50,
#             roll=0,
#             upAxisIndex=2
#         )

#         self.projectionMatrix = p.computeProjectionMatrixFOV(
#             fov=90,
#             aspect=float(image_width)/image_height,
#             nearVal=0.1,
#             farVal=100.0
#         )

#         if not self.use_intrinsic :
#             self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
#             self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#             # Goal / Baseline 
#             goal_text_input = self.clip_processor(text=[goal_prompt], return_tensors="pt", padding=True).to(self.device)
#             baseline_text_input = self.clip_processor(text=[baseline_prompt], return_tensors="pt", padding=True).to(self.device)

#             with torch.no_grad():
#                 goal_text_features = self.clip_model.get_text_features(**{k: goal_text_input[k] for k in ["input_ids", "attention_mask"]})
#                 baseline_text_features = self.clip_model.get_text_features(**{k: baseline_text_input[k] for k in ["input_ids", "attention_mask"]})

#             self.goal_norm_text_features = goal_text_features / goal_text_features.norm(dim=-1, keepdim=True)
#             self.baseline_norm_text_features = baseline_text_features / baseline_text_features.norm(dim=-1, keepdim=True)

#             self.goal_baseline_line = self.goal_norm_text_features - self.baseline_norm_text_features
#             self.goal_baseline_norm_line = self.goal_baseline_line / self.goal_baseline_line.norm(dim=-1, keepdim=True)

#         # Use RND to get an exploration reward as seen in class
#         else : 
#             self.expl_net = rx_utils.ExplNet(out_dim=64,lr=1e-3).to(self.device)

#         self.action_space = Box(low=-1, high=1, shape=(6,))

#         if self.image_only : 
#             self.observation_space = Box(low=0, high=255, shape=(self.image_height, self.image_width, 3), dtype=np.uint8)
#         else :             
#             self.observation_space = Dict({
#                 "image": Box(low=0, high=255, shape=(self.image_height, self.image_width, 3), dtype=np.uint8),
#                 "joint_states": Box(low=-4, high=4, shape=(len(self.joint_list),), dtype=np.float32)
#             })


#         self.t = 0

#         if headless : 
#             p.connect(p.DIRECT)
#             self.renderer = p.ER_TINY_RENDERER
#         else : 
#             p.connect(p.GUI)
#             self.renderer = p.ER_BULLET_HARDWARE_OPENGL

#         p.setAdditionalSearchPath(pybullet_data.getDataPath())

#         self.last_state = {}
    
#     def set_model(self,model):
#         self.model = model

#     def reset(self, seed = None, option = None):
#         '''
#         Resets the environment and generates a target pos (displayed as a red sphere)
#         The current sphere pos generator is using rejection sampling and is very inefficient
#         '''

#         if seed is not None:
#             np.random.seed(seed)

#         p.resetSimulation()
#         p.setGravity(0,0,-9.81)

#         # it = 0
#         # while True:
#         #     it += 1
#         #     target_pos = np.random.uniform(-1.5,1.5,size=3)
#         #     if np.linalg.norm(target_pos) >= 1 and np.linalg.norm(target_pos) <= 1.5 and target_pos[2] > 0.25 : 
#         #         break

#         target_pos = np.array([0.75,0.75,0.25])
        
#         # print(f"Generated random target pos in {it} iterations")

#         sphere_radius = 0.05
#         visual_shape_id = p.createVisualShape(
#             shapeType=p.GEOM_SPHERE,
#             radius=sphere_radius,
#             rgbaColor=[1, 0, 0, 1] # RGBA : only red here
#         )

#         self.target_id = p.createMultiBody(
#             baseMass=0,  # Mass = 0 so it doesn't fall
#             baseVisualShapeIndex=visual_shape_id,
#             basePosition=target_pos,
#             baseCollisionShapeIndex=-1  # No collision
#         )

#         self.plane_id = p.loadURDF("plane.urdf")
#         self.robot_id = p.loadURDF(self.urdf_path,useFixedBase=True,globalScaling=4)

#         # Recolors the robot arm
#         num_joints = p.getNumJoints(self.robot_id)

#         # Loop over all links to recolor them
#         for link_idx in range(-1,num_joints):
    
#             color = [0, 0, 0, 1]

#             # If end effector, make it green
#             if link_idx >= 6 :
#                 color = [0, 1, 0, 1]

#             p.changeVisualShape(
#                 objectUniqueId=self.robot_id,
#                 linkIndex=link_idx,
#                 rgbaColor=color
#             )

#         self.target_pos = np.array(target_pos)

#         self.t = 0

#         return self.get_obs(), {}

#     def render(self):

#         _,_,rgbPixels,_,_ = p.getCameraImage(
#             width=self.image_width,
#             height = self.image_height,
#             viewMatrix=self.viewMatrix,
#             projectionMatrix=self.projectionMatrix,
#             renderer= self.renderer
#         )

#         return np.reshape(np.array(rgbPixels),(self.image_height,self.image_width,4))[:,:,:3]

#     def get_obs(self):
#         '''
#         Returns the observation of the current state.
#         Arbitrarily chosen for now : 
#             - the angle (radians) of each joint (i.e joints [0,1,2,3,4] and [9,10] for the gripper fingers)
#             - The gripper radians can be changed later for a binary value (open or close gripper)
#             - image of the rendering shape (self.image_height,self.image_width,3)
#         '''
        
#         if self.image_only:
#             return self.render()

#         ob = []
#         for joint_id in self.joint_list:
#             curr_radian = p.getJointState(self.robot_id,joint_id)[0]
#             ob.append(curr_radian)
        
#         # Also add the target's position
#         # ob.extend(self.target_pos)

#         return {
#             "image": self.render(), 
#             "joint_states": np.array(ob),  
#         }

#     def get_reward_and_terminal(self,ob = None):
#         '''
#         Uses the CLIP model to get the CLIP-Reg (from paper) rewards 
#         and if the environment is done or not (if gripper is within epsilon distance from target pos).
#         '''
        
#         sqr_dist_to_target = ((np.array(self.get_end_effector_pos()) - self.target_pos)**2).sum()
#         done = (self.sqr_epsilon_dist >= sqr_dist_to_target) or (self.t >= self.max_timesteps)

#         if ob is None :
#             # Get image, pass it through the clip model, compute similarity
#             rx_img = self.render()
#         else :
#             if self.image_only :
#                 rx_img = ob
#             else : 
#                 rx_img = ob['image']

#         # Use exploration RND as reward
#         if self.use_intrinsic:
            
#             # RND
#             x = torch.tensor(rx_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
#             x = x.to(self.device)

#             reward = self.expl_net(x).item()
#             self.expl_net.update(x)

#             # Entropy-based (from PEBBLE paper : "https://arxiv.org/pdf/2106.05091")
#             # if self.model.replay_buffer.size() < 10 : 
#             #     # print("not enough transitions in buffer")
#             #     return 0.0, done

#             # # If last dim is channel, reshape
#             # if rx_img.shape[-1] == 3: 
#             #     rx_img = rx_img.transpose(2, 0, 1)  # to (3, 64, 64)
#             # else:
#             #     rx_img = rx_img

#             # sampled_obs = self.model.replay_buffer.sample(1000).observations.detach().cpu().numpy()
#             # sampled_obs = sampled_obs.reshape(1000,-1)

#             # self.k_nn.fit(sampled_obs)

#             # rx_img = (rx_img[None]).reshape(1,-1)

#             # distances,_ = self.k_nn.kneighbors(rx_img)

#             # reward = np.log(distances[0][-1] + 1e-9)


#         # Otherwise just use CLIP rewards (not effective)
#         else : 
#             image_inputs = self.clip_processor(images=rx_img, return_tensors="pt", padding=True).to(self.device)

#             with torch.no_grad():
#                 image_features = self.clip_model.get_image_features(**{k: image_inputs[k] for k in ["pixel_values"]})
            
#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#             # Projection of s (image) to the line spanned by goal - baseline
#             proj_img = (self.goal_baseline_norm_line @ image_features.T) * self.goal_baseline_norm_line

#             # Clip regularized with the projection
#             clip_reg = 1 - 0.5 * torch.sum(((self.clip_reg_alpha * proj_img + (1 - self.clip_reg_alpha) * image_features - self.goal_baseline_norm_line)) ** 2, dim=-1)
            
#             # similarity_score = (self.goal_norm_text_features @ image_features.T).squeeze(0)

#             # return -sqr_dist_to_target, done
#             # return similarity_score.item(), done
            
#             # print(f"Goal-baseline line norm : {self.goal_baseline_line.norm(dim=-1, keepdim=True).item()} , Projection norm : {proj_img.norm(dim=-1, keepdim=True).item()} , Reg 1 : {(((self.clip_reg_alpha * proj_img + (1 - self.clip_reg_alpha) * self.goal_baseline_line).norm(dim=-1, keepdim=True))**2)}, Reg 2. : {torch.sum(((self.clip_reg_alpha * proj_img + (1 - self.clip_reg_alpha) * self.goal_baseline_line)) ** 2, dim=-1)}")
#             reward = clip_reg.item()
        
#         return reward , done

#     def get_end_effector_pos(self):
#         ''' Returns world position of end effector 'ee_gripper' (joint_id 11) '''
#         return p.getLinkState(self.robot_id, 11)[0]

#     def step(self, action, simulation_steps = 4):

#         '''
#         action is an np.array of 6 int values each being either [-1,0,1]
#         -1 means decrease angle of joint, 0 means don't change angle, 1 increases angle
#         The last index is the gripper, 0 means don't grip, 1 means grip.

#             Index : 
#                 0 : Waist joint
#                 1 : Shoulder joint
#                 2 : Elbow joint
#                 3 : Wrist angle
#                 4 : Wrist rotate
#                 5 : Gripper

#         simulation_steps : int indicating the number of frames to simulate (apply the same action). The next observation and reward are obtained after simulation_steps
#         '''
#         self.t += 1
        
#         # Checks if received floats, in which case round them to nearest correct action values
#         if isinstance(action[0], float):
#             discrete_ac = np.zeros(6, dtype=int)

#             for i in range(5):
#                 discrete_ac[i] = int(np.clip(round(action[i]),-1,1))
#             discrete_ac[5] = int(np.clip(round(action[5]),0,1))

#         else :
#             discrete_ac = action

#         # Apply the action to each joint
#         for joint_id,ac in enumerate(discrete_ac):
            
#             # If using gripper
            
#             if joint_id == 5 : 
                
#                 # Close gripper : decrease left finger increase right finger
#                 # Open gripper : opposite

#                 if ac :
#                     left_finger_step = -1
#                     right_finger_step = 1
#                 else : 
#                     left_finger_step = 1
#                     right_finger_step = -1
                
#                 # Store these values in init instead of calling the api every frame
#                 current_pos_left_finger = p.getJointState(self.robot_id,9)[0]
#                 current_pos_right_finger = p.getJointState(self.robot_id,10)[0]

#                 p.setJointMotorControl2(
#                     bodyUniqueId=self.robot_id,
#                     jointIndex=9,
#                     controlMode=p.POSITION_CONTROL,
#                     targetPosition= current_pos_left_finger + self.step_size * left_finger_step,
#                     force=1
#                 )

#                 p.setJointMotorControl2(
#                     bodyUniqueId=self.robot_id,
#                     jointIndex=10,
#                     controlMode=p.POSITION_CONTROL,
#                     targetPosition= current_pos_right_finger + self.step_size * right_finger_step,
#                     force=1
#                 )

#             else : 
#                 # Avoid unnecessary calls to API
#                 if ac == 0 : continue

#                 joint_max_force = p.getJointInfo(self.robot_id,joint_id)[10]

#                 current_pos = p.getJointState(self.robot_id,joint_id)[0]

#                 p.setJointMotorControl2(
#                     bodyUniqueId=self.robot_id,
#                     jointIndex=joint_id,
#                     controlMode=p.POSITION_CONTROL,
#                     targetPosition=current_pos + self.step_size * ac,
#                     force=joint_max_force
#                 )

#         # Let the simulation roll for a few frames to allow movement
#         for _ in range(simulation_steps) : 
#             p.stepSimulation()

#         ob = self.get_obs()
#         reward,done = self.get_reward_and_terminal(ob)
#         truncated = self.t >= self.max_timesteps
#         info = {}

#         return ob , reward, done,truncated, info

#     def close(self):
#         p.disconnect()

class RX150Env2(gym.Env):

    '''
        Same env as the first one, just cleaned up to use either RND or a parametric reward function (No CLIP in this)
    '''

    def __init__(
        self,
        urdf_path, 
        epsilon_dist=0.1, 
        max_timesteps=1000, 
        step_size=0.1, 
        headless=False, 
        image_width=64, 
        image_height=64, 
        frame_skip = 4,
        use_intrinsic=True,
        model=None,
        image_only=True,
        reward_model=None,
        use_sparse_rewards=True,
        task="reach"
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_skip = frame_skip
        self.use_intrinsic=use_intrinsic
        self.image_only = image_only
        self.task = task

        self.model = model
        self.expl_net = None
        self.reward_model = reward_model

        # self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
        #     cameraTargetPosition=[0.75, 0.25, -0.25],
        #     distance=1.25,
        #     yaw=270,
        #     pitch=-65,
        #     roll=0,
        #     upAxisIndex=2
        # )

        # self.viewMatrix = [
        #     0.2133,  0.761,  -0.6127, 0.0,
        #    -0.977,   0.1661, -0.1338, 0.0,
        #     0.0,     0.6272,  0.7789, 0.0,
        #    -0.4704, -0.9945, -1.2405, 1.0
        # ]

        self.viewMatrix = [
    0.4318,  0.7025, -0.5657,  0.0,
   -0.9020,  0.3363, -0.2708,  0.0,
    0.0,     0.6272,  0.7789,  0.0,
   -0.8590, -0.6357, -0.9405,  1.0
   ]
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=50,
            aspect=float(image_width)/image_height,
            nearVal=0.1,
            farVal=100.0
        )

        self.use_sparse_rewards = use_sparse_rewards

        # Use RND to get an exploration reward as seen in class
        if self.use_intrinsic : 
            if self.image_only : 
                self.expl_net = rx_utils.ExplNet(out_dim=64,lr=1e-3).to(self.device)
            else : 
                self.expl_net = rx_utils.ExplNet(state_in_dim=len(self.joint_list),out_dim=64,lr=1e-3,image_only=self.image_only).to(self.device)

        low = np.array([-1, -1, -1, -1, -1, 0])
        high = np.array([1, 1, 1, 1, 1, 1])

        self.action_space = Box(low=low, high=high, dtype=np.float32)

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
    
    def step_simulation(self,num_steps=10):
        for _ in range(num_steps):
            p.stepSimulation()  

    def set_model(self,model):
        self.model = model

    def set_reward_model(self,reward_model):
        self.reward_model=reward_model

    def reset(self, seed = None, options = None):
        '''
        Resets the environment and generates a target pos (displayed as a red sphere)
        The current sphere pos generator is using rejection sampling and is very inefficient
        '''

        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0,0,-9.81)

        if self.task == 'reach' :
            
            # Generate random position in front of random (fixed y coord)
            target_pos = np.array([0.75,np.random.uniform(-1, 1),0.25])
            
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
            
            self.target_pos = np.array(target_pos)

        elif self.task == 'stack':
            # Generate two random pos
            # Create two cubes

            # rand_z = np.random.uniform(-1, 1)
            # rand_offset = np.random.uniform(-0.05, 0.05)
            # random_pos_1 = np.array([0.75,rand_z,0.25])
            # random_pos_2 = np.array([0.75+rand_offset,rand_z+rand_offset,0.5])

            random_pos_1 = np.array([np.random.uniform(1, 1.5),np.random.uniform(0, -1),0.25])
            random_pos_2 = np.array([np.random.uniform(1, 1.5),np.random.uniform(0, -1),0.25])

            self.target_pos = random_pos_2

            self.lower_cube = p.loadURDF("cube.urdf",basePosition=random_pos_1, globalScaling=0.1)
            self.upper_cube = p.loadURDF("cube.urdf",basePosition=random_pos_2, globalScaling=0.1)

            p.changeVisualShape(objectUniqueId=self.lower_cube, rgbaColor = [1,0,0,1], linkIndex = -1)
            p.changeVisualShape(objectUniqueId=self.upper_cube, rgbaColor = [0,0,1,1], linkIndex = -1)

        else : 
            print("No task specified for env")

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

        self.t = 0

        return self.get_obs(), {}

    def are_cubes_stacked(self):
        pos1, _ = p.getBasePositionAndOrientation(self.lower_cube)
        pos2, _ = p.getBasePositionAndOrientation(self.upper_cube)


        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

        # Define thresholds
        xy_threshold = 0.1
        z_threshold = 0.05
        cube_height = 0.1

        xy_close = np.linalg.norm(pos1[:2] - pos2[:2]) < xy_threshold

        # Check Z difference (upper is above lower)
        z_diff = pos2[2] - pos1[2]
        z_stacked = abs(z_diff - cube_height) < z_threshold

        return xy_close and z_stacked

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

    def get_reward_and_terminal(self,obs):
        '''
        Uses the CLIP model to get the CLIP-Reg (from paper) rewards 
        and if the environment is done or not (if gripper is within epsilon distance from target pos).
        Does not expect an observation as input.
        '''
        
        sqr_dist_to_target = ((np.array(self.get_end_effector_pos()) - self.target_pos)**2).sum()
        
        if self.task == 'reach' :
            # Use true distance just to check if task complete
            task_completed = (self.sqr_epsilon_dist >= sqr_dist_to_target)

        elif self.task == 'stack':
            task_completed = self.are_cubes_stacked()
        
        done = task_completed or (self.t >= self.max_timesteps)

        # return -sqr_dist_to_target,done
        if self.image_only : 
            obs = torch.from_numpy(obs/255).permute(2,0,1).unsqueeze(0).float().to(self.device)
        else : 
            obs_tensor = torch.from_numpy(obs['image']/255).permute(2,0,1).unsqueeze(0).float().to(self.device)
            state_tensor = torch.from_numpy(obs["joint_states"]).unsqueeze(0).float().to(self.device)

        # Use exploration RND as reward
        if self.use_intrinsic:
            
            # RND
            if self.image_only : 
                reward = self.expl_net(obs)
                self.expl_net.update(obs)  
            else : 
                reward = self.expl_net(obs_tensor,state_tensor)
                self.expl_net.update(obs_tensor,state_tensor)
        
            '''
            Other method (not implemented yet)
            Entropy-based (from PEBBLE paper : "https://arxiv.org/pdf/2106.05091")
                - Sample n (500) samples from the replay buffer
                - Compute the distance (euclidian) of current ob to sampled observations
                - K-th closest sample from current ob is picked as s_t^k
                - Reward is log(||s_t - s_t^k||) = log(dist(ob,ob_k))
            '''
            
        # Otherwise use the parametric (cnn) reward model
        else : 
            if self.image_only : 
                reward = self.reward_model(obs)
            else : 
                reward = self.reward_model(obs_tensor,state_tensor)
        
        if self.use_sparse_rewards:
            # reward = reward.item() + (10.0 * task_completed) - (0.5 * sqr_dist_to_target)
            reward = reward.item() + (10.0 * task_completed)

        return reward , done

    def get_end_effector_pos(self):
        ''' Returns world position of end effector 'ee_gripper' (joint_id 11) '''
        return p.getLinkState(self.robot_id, 11)[0]

    def apply_action(self,action):

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

    def step(self, action, simulation_steps = 4):

        self.apply_action(action)
        
        # Let the simulation roll for a few frames to allow movement
        self.step_simulation(simulation_steps)

        ob = self.get_obs()
        reward,done = self.get_reward_and_terminal(ob)
        truncated = self.t >= self.max_timesteps
        info = {}

        return ob , reward, done,truncated, info        

    def close(self):
        p.disconnect()

# CLIP for tokenization
import clip
# LIV imports
from liv import load_liv
import torchvision.transforms as T
from PIL import Image

class RX150EnvLiv(gym.Env):
    """
    RX150 environment with 'reach', 'grasp', and 'stack' tasks.
    Observations include both image and joint states for MultiInputPolicy.
    Rewards are computed via LIV similarity against task-specific prompts.
    """
    def __init__(
        self,
        urdf_path,
        task="reach",
        epsilon_dist=0.1,
        max_timesteps=1000,
        step_size=0.1,
        headless=False,
        image_width=64,
        image_height=64,
        frame_skip=4,
        use_liv=True,
        reach_prompt="A 3D model of a robot arm touching a red sphere",
        grasp_prompt="A 3D model of a robot arm grasping a red cube",
        stack_prompt="A 3D model of a robot arm stacking a blue cube onto a red cube",
    ):
        super().__init__()
        self.urdf_path = urdf_path
        self.task = task
        self.epsilon_dist = epsilon_dist
        self.sqr_epsilon_dist = epsilon_dist ** 2
        self.max_timesteps = max_timesteps
        self.step_size = step_size
        self.image_width = image_width
        self.image_height = image_height
        self.frame_skip = frame_skip
        self.use_liv = use_liv

        # Joints to control and observe
        self.joint_list = [0, 1, 2, 3, 4, 9, 10]
        self.n_joints = len(self.joint_list)

        self.last_liv_score = 0.0

        # Load LIV model if using it
        if use_liv:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.liv = load_liv().to(self.device)
            self.liv.eval()
            self.liv_transform = T.Compose([T.ToTensor()])
            # Select prompt based on task
            if task == 'reach':
                prompt = reach_prompt
            elif task == 'grasp':
                prompt = grasp_prompt
            elif task == 'stack':
                prompt = stack_prompt
            else:
                raise ValueError(f"Unknown task '{task}' for LIV env")
            text_inputs = clip.tokenize([prompt]).to(self.device)
            with torch.no_grad():
                self.text_embedding = self.liv(input=text_inputs, modality="text")

        # Action: 6 discrete commands [-1,0,1] (last element for gripper open/close)
        self.action_space = Box(low=-1, high=1, shape=(6,), dtype=np.int32)

        # Observation: image + joint_states
        self.observation_space = Dict({
            "image": Box(low=0, high=255,
                         shape=(self.image_height, self.image_width, 3), dtype=np.uint8),
            "joint_states": Box(low=-4, high=4,
                                 shape=(self.n_joints,), dtype=np.float32)
        })

        # Connect to PyBullet
        if headless:
            p.connect(p.DIRECT)
            self.renderer = p.ER_TINY_RENDERER
        else:
            p.connect(p.GUI)
            self.renderer = p.ER_BULLET_HARDWARE_OPENGL
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Camera matrices
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.25, 0, -0.25],
            distance=1.75, yaw=90, pitch=-50, roll=0, upAxisIndex=2
        )
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=float(self.image_width)/self.image_height,
            nearVal=0.1, farVal=100.0
        )

        self.t = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Spawn environment objects based on task
        if self.task == 'reach':
            # red sphere target
            target_pos = np.array([0.75, 0.75, 0.25])
            sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1,0,0,1])
            self.target_id = p.createMultiBody(0, sphere, -1, target_pos)
            self.target_pos = target_pos

        elif self.task == 'grasp':
            # single cube to grasp
            start = [0.6, 0, 0.1]
            self.object_id = p.loadURDF("cube.urdf", basePosition=start, globalScaling=0.1)
            p.changeVisualShape(self.object_id, -1, rgbaColor=[1,0,0,1])
            pos, _ = p.getBasePositionAndOrientation(self.object_id)
            self.object_init_z = pos[2]

        elif self.task == 'stack':
            # two cubes to stack
            pos1 = np.array([np.random.uniform(0.75,1.5), np.random.uniform(-1,1), 0.25])
            pos2 = pos1 + np.array([0.1, 0.0, 0.25])
            self.lower_cube = p.loadURDF("cube.urdf", basePosition=pos1.tolist(), globalScaling=0.1)
            self.upper_cube = p.loadURDF("cube.urdf", basePosition=pos2.tolist(), globalScaling=0.1)
            p.changeVisualShape(self.lower_cube, -1, rgbaColor=[1,0,0,1])
            p.changeVisualShape(self.upper_cube, -1, rgbaColor=[0,0,1,1])
            self.target_pos = pos2

        else:
            raise ValueError(f"Unknown task '{self.task}'")

        # Load plane & robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_path, useFixedBase=True, globalScaling=4)
        num_j = p.getNumJoints(self.robot_id)
        for link in range(-1, num_j):
            color = [0,0,0,1]
            if link >= 6:
                color = [0,1,0,1]
            p.changeVisualShape(self.robot_id, link, rgbaColor=color)

        self.t = 0
        return self.get_obs(), {}

    def render(self):
        _,_,rgb,_,_ = p.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            renderer=self.renderer
        )
        arr = np.array(rgb).reshape(self.image_height, self.image_width, 4)
        return arr[:, :, :3]

    def get_obs(self):
        img = self.render()
        js = [p.getJointState(self.robot_id, j)[0] for j in self.joint_list]
        return {"image": img, "joint_states": np.array(js, dtype=np.float32)}

    def are_cubes_stacked(self):
        low, _ = p.getBasePositionAndOrientation(self.lower_cube)
        up, _ = p.getBasePositionAndOrientation(self.upper_cube)
        low = np.array(low); up = np.array(up)
        xy_close = np.linalg.norm(low[:2] - up[:2]) < 0.1
        z_diff = up[2] - low[2]
        z_close = abs(z_diff - 0.1) < 0.05
        return xy_close and z_close

    def compute_handcrafted_reward(self):
        ee = np.array(self.get_end_effector_pos())

        if self.task == "grasp":
            obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
            obj_pos = np.array(obj_pos)
            d = np.linalg.norm(ee - obj_pos)
            bonus_lift = 1.0 if obj_pos[2] >= self.object_init_z + 0.05 else 0.0
            return -d + bonus_lift

        elif self.task == "stack":
            low_pos, _ = p.getBasePositionAndOrientation(self.lower_cube)
            up_pos,  _ = p.getBasePositionAndOrientation(self.upper_cube)
            low_pos = np.array(low_pos)
            up_pos  = np.array(up_pos)

            # distance to lower cube
            d_ee_low   = np.linalg.norm(ee - low_pos)
            # distance between cubes
            d_cubes    = np.linalg.norm(up_pos - low_pos)
            # bonus if lifted (upper cube z > lower cube z + 0.1) set to zero for now
            # bonus_lift = 1.0 if (up_pos[2] - low_pos[2]) >= 0.5 else 0.0
            bonus_lift = 0.0
            # bonus if stacked
            bonus_stack = 1.0 if self.are_cubes_stacked() else 0.0

            return -d_ee_low - d_cubes + bonus_lift + bonus_stack

        else:
            return 0.0

    def get_reward_and_terminal(self, rx_img):
        done = False
        if self.task == 'reach':
            d2 = np.sum((np.array(self.get_end_effector_pos()) - self.target_pos)**2)
            done = (d2 <= self.sqr_epsilon_dist) or (self.t >= self.max_timesteps)
        elif self.task == 'grasp':
            pos, _ = p.getBasePositionAndOrientation(self.object_id)
            done = (pos[2] >= self.object_init_z + 0.1) or (self.t >= self.max_timesteps)
        else:  # stack
            done = self.are_cubes_stacked() or (self.t >= self.max_timesteps)

        if self.use_liv : 
            # Compute LIV similarity reward
            pil = Image.fromarray(rx_img)
            img_t = self.liv_transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_emb = self.liv(input=img_t, modality="vision")
                score = self.liv.module.sim(img_emb, self.text_embedding)

            current_liv_score = score.item()
            final_score = current_liv_score - self.last_liv_score

            self.last_liv_score = current_liv_score

            return final_score, done
        else : 
            reward = self.compute_handcrafted_reward()
            return reward, done

    def step(self, action, simulation_steps=4):
        self.t += 1
        # Discretize floats
        if isinstance(action[0], float):
            a = np.clip(np.round(action), -1, 1).astype(int)
            a[5] = int(np.clip(round(action[5]), 0, 1))
        else:
            a = action
        # Apply
        for i, ac in enumerate(a):
            if i == 5:
                # gripper
                ls = -1 if ac else 1
                rs = 1 if ac else -1
                lp = p.getJointState(self.robot_id, 9)[0]
                rp = p.getJointState(self.robot_id, 10)[0]
                p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL,
                                        lp + self.step_size * ls, force=1)
                p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL,
                                        rp + self.step_size * rs, force=1)
            else:
                if ac == 0:
                    continue
                mf = p.getJointInfo(self.robot_id, i)[10]
                cp = p.getJointState(self.robot_id, i)[0]
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                        cp + self.step_size * ac, force=mf)
        for _ in range(simulation_steps):
            p.stepSimulation()

        obs = self.get_obs()
        reward, done = self.get_reward_and_terminal(obs["image"])
        truncated = self.t >= self.max_timesteps
        return obs, reward, done, truncated, {}

    def get_end_effector_pos(self):
        return p.getLinkState(self.robot_id, 11)[0]

    def close(self):
        p.disconnect()
