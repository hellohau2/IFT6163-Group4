import pybullet as p
import pybullet_data

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import cv2

import os
os.makedirs("debug", exist_ok=True)

class RobotEnv(gym.Env):
    def __init__(self):
        self.max_steps = 200  #Adjust max episode steps
        self.current_step = 0

        super(RobotEnv, self).__init__()
        # Connect to PyBullet
        p.connect(p.DIRECT)
        #p.connect(p.GUI)  # Use p.DIRECT for headless
        p.setGravity(0, 0, -9.8)

        self.render_mode = "rgb_array"
        
        # Define observation spaces (customize robot)
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,))  # Joint torques: 4 arm + 1 gripper
        self.observation_space = spaces.Box(low=0, high=255, shape=(512, 512, 3), dtype=np.uint8)  # RGB image


    def render(self):
        # Return the observation directly (since observation is already an image)
        return self._get_observation() # Must return (H, W, 3) RGB array


    def step(self, action):

        if isinstance(action, np.generic):  # handles np.float32, np.int64, etc.
            action = np.array([float(action)] * 4, dtype=np.float32)
        elif isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = np.array([action.item()] * 4, dtype=np.float32)
            elif action.ndim == 2 and action.shape[0] == 1:
                action = action[0]
        elif isinstance(action, (float, int)):
            action = np.array([action] * 4, dtype=np.float32)

        assert action.shape == (5,), f"[ERROR] Invalid action shape: {action.shape}"

        #print("[DEBUG] Action shape:", action.shape)

        #print(f"Action (scaled): {action * 150}")
        #print("Robot ID at step:", getattr(self, "robot_id", "NOT SET"))

        #Debug
        #prev_cube_pos = p.getBasePositionAndOrientation(self.cube_id)
        
        # Get positions
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        gripper_pos = p.getLinkState(self.robot_id, 3)[0]

        # Apply action to joints
        p.setJointMotorControlArray(
            self.robot_id,
            jointIndices=range(4),  # Shoulder, upper_arm, forearm, wrist
            controlMode=p.TORQUE_CONTROL,
            forces=action[:4] * 250  # Scale arm actions
        )
        
        # Gripper joint (index 8: fingers_link)
        p.setJointMotorControl2(
            self.robot_id,
            jointIndex=8,  # Gripper open/close
            controlMode=p.POSITION_CONTROL,  # Grippers use position control
            targetPosition=action[4] * 0.1  # Scale gripper action
        )
        # Step simulation 20x to allow movement
        for _ in range(50):
            p.stepSimulation()

        
        # Log actual joint positions
        joint_states = p.getJointStates(self.robot_id, range(4))
        joint_positions = [s[0] for s in joint_states]
        #print("[DEBUG] Joint positions:", joint_positions)  


        # Calculate reward (we’ll add CLIP later)
        reward = 0.0
        done = False
    
        # Add termination logic (example)
        terminated = self._check_success()  # Implement success condition
        truncated = False  # No time limit yet

        # Approx gripper position 
        gripper_pos = p.getLinkState(self.robot_id, 3)[0]
        #print(f"Gripper position: {gripper_pos}")

        # Debug: Apply gravity to cube
        # p.applyExternalForce(self.cube_id, -1, 
        #                 forceObj=[0, 0, -1],  # Small downward force
        #                 posObj=[0, 0, 0],
        #                 flags=p.WORLD_FRAME)

        # Compute Euclidean distance
        gripper_cube_distance = np.linalg.norm(np.array(gripper_pos) - np.array(cube_pos))
        #print(f"Gripper-Cube Distance: {gripper_cube_distance}")

        # Capture image observation
        obs = self._get_observation()

        # Apply post-processing
        #obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        obs = cv2.GaussianBlur(obs, (3, 3), 0)          # Reduce aliasing
        obs = cv2.convertScaleAbs(obs, alpha=1.2, beta=20)  # Contrast/brightness
        

       # Check contact for left/right fingers and gripper tip
        left_contact = p.getContactPoints(self.robot_id, self.cube_id, linkIndexA=9)  # Left finger
        right_contact = p.getContactPoints(self.robot_id, self.cube_id, linkIndexA=10)  # Right finger
        tip_contact = p.getContactPoints(self.robot_id, self.cube_id, linkIndexA=11)  # Gripper tip

        # Reward if any part of the gripper touches the cube
        contact_reward = 1.0 if (len(left_contact) > 0 or 
                                len(right_contact) > 0 or 
                                len(tip_contact) > 0) else 0.0
        
        info = {
            "last_step_reward": reward,
            "robot_id": self.robot_id,
            "episode_done": terminated or truncated,        
            "gripper_cube_dist": gripper_cube_distance,
            "gripper_pos": np.array(gripper_pos),
            "cube_pos": np.array(cube_pos),
            "contact_reward": contact_reward,
            "in_contact": float(contact_reward >0),
            "success" : float(terminated)  }

        
        # Save only on success/failure (optimize logging)
        if done:
            cv2.imwrite(f"logs/output/{'success' if reward > 0 else 'failed'}.png", obs)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps  # Force reset

        # Get joint states for the controlled joints
        joint_states = p.getJointStates(self.robot_id, self.controlled_joint_indices)
        joint_positions = [s[0] for s in joint_states]

        # Add named joint positions to info dict
        for name, pos in zip(self.joint_names, joint_positions):
            info[f"{name}_pos"] = pos 

        return obs, reward, terminated, truncated, info

    def _get_observation(self):

       # Improved camera parametersx
        camera_pos = [0.35, 0.0, 1.2]  #Just in front and slightly above
        target_pos = [0.35, 0.0, 0.62]       #Aim at where the gripper interacts

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 1, 0]
        )

        # Widen FOV to 90 degrees
        proj_matrix = p.computeProjectionMatrixFOV(
                    fov=55, #zoom in
                    aspect=1.0, 
                    nearVal=0.01, 
                    farVal=3.0
        )
        
        # Enable anti-aliasing and higher resolution
        _, _, rgb, _, _ = p.getCameraImage(
            width=512,  # Resolution
            height=512,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL  # Better renderer
        )

        frame = rgb[:, :, :3]
        cv2.imwrite("debug/current_view.png", frame)

        return rgb[:, :, :3]  # Shape (224, 224, 3)


    def reset(self, seed=None, options=None):
        self.current_step = 0  # Reset step counter
        p.resetSimulation()
    
        # Add directional light (simulates ambient + shadows)
        light_position = [1, 1, 1]  # Position of the light source
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Enable shadows
        p.setPhysicsEngineParameter(enableFileCaching=0)  # smoother visuals
        p.addUserDebugParameter("light_distance", 1, 3, 2) # Optional ambient light
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")  # Add ground plane for reflections

        # Load decorative objects
        p.loadURDF("table/table.urdf", [0, 0, 0])

        #Random cube position 
        x = np.random.uniform(0.35, 0.5)
        y = np.random.uniform(-0.1, 0.1)
        z = 0.65
        cube_pos = [x, y, z]

        self.cube_id = p.loadURDF("cube_small.urdf",
                    cube_pos, 
                    globalScaling=0.4,
                    useFixedBase=True,
                    #flags=p.URDF_USE_INERTIA_FROM_FILE) # Ensure proper physics
        )

        p.changeDynamics(
            self.cube_id, -1,
            mass=2.0,  # Explicitly set mass (kg)
            #lateralFriction=0.5  # Prevent sliding
        )
        #print("[RESET] Cube ID:", self.cube_id)
        #print("[RESET] Cube initial pos:", p.getBasePositionAndOrientation(self.cube_id))


        p.changeVisualShape(self.cube_id, -1, rgbaColor=[1, 0.0, 0.0, 1])
        
        # Load robot URDF after connecting to PyBullet
        urdf_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "reactor_x150",
            "urdf",
            "reactor_x150.urdf"
        )

        self.robot_id = p.loadURDF(
                    urdf_path,
                    basePosition=[0, 0, 0.62],  #robot on table
                    useFixedBase=True
                )

        self.controlled_joint_indices = range(4)  # Joints 0-3 (adjust if needed)
        self.joint_names = []
        for i in self.controlled_joint_indices:
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode("utf-8")  # Joint name is at index 1
            self.joint_names.append(joint_name)

        #print("\n=== Robot Joint/Link Indices ===")
        #num_joints = p.getNumJoints(self.robot_id)
        #for i in range(num_joints):
        #    joint_info = p.getJointInfo(self.robot_id, i)
        #    print(f"Index {i}: {joint_info[12].decode('utf-8')}")

        #print("[RESET] Robot ID:", self.robot_id)
        #print("[RESET] Robot base pos:", p.getBasePositionAndOrientation(self.robot_id)[0])

        prev_cube_pos = p.getBasePositionAndOrientation(self.cube_id)[0]

        for _ in range(10):
            p.stepSimulation()

        #Debug
        new_cube_pos = p.getBasePositionAndOrientation(self.cube_id)[0]
        #print(f"Cube moved: {prev_cube_pos} -> {new_cube_pos}")
        print(f"[RESET] Robot ID: {self.robot_id} (valid?)")  # Debug

        return self._get_observation(), {"robot_id": self.robot_id}
    
    def _check_success(self):
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        gripper_pos = p.getLinkState(self.robot_id, 3)[0]
        
        # Success condition: gripper within 5cm of cube and vertical distance < 3cm
        horizontal_dist = np.linalg.norm(np.array(cube_pos[:2]) - np.array(gripper_pos[:2]))
        vertical_dist = abs(cube_pos[2] - gripper_pos[2])
        
        success = horizontal_dist < 0.03 and vertical_dist < 0.03
        
        return success
