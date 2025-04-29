import os
import numpy as np
import pybullet as p
import torch
import wandb
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from rx150.rx150_env import RX150EnvLiv
from wandb.integration.sb3 import WandbCallback

def compute_handcrafted_reward(env):
    ee = np.array(env.get_end_effector_pos())

    if env.task == "grasp":
        obj_pos, _ = p.getBasePositionAndOrientation(env.object_id)
        obj_pos = np.array(obj_pos)
        d = np.linalg.norm(ee - obj_pos)
        bonus_lift = 1.0 if obj_pos[2] >= env.object_init_z + 0.05 else 0.0
        return -d + bonus_lift

    elif env.task == "stack":
        # get current cube poses
        low_pos, _ = p.getBasePositionAndOrientation(env.lower_cube)
        up_pos, _ = p.getBasePositionAndOrientation(env.upper_cube)
        low_pos = np.array(low_pos)
        up_pos = np.array(up_pos)

        # distance EE to lower cube
        d_ee_low = np.linalg.norm(ee - low_pos)
        # distance between cubes
        d_cubes = np.linalg.norm(up_pos - low_pos)
        # bonus if lifted (upper cube z > lower cube z + 0.1)
        # bonus_lift = 1.0 if (up_pos[2] - low_pos[2]) >= 0.5 else 0.0
        bonus_lift = 0.0
        # bonus if properly stacked
        bonus_stack = 1.0 if env.are_cubes_stacked() else 0.0

        return -d_ee_low - d_cubes + bonus_lift + bonus_stack

    else:
        return 0.0

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, liv_reward, done, truncated, info = self.env.step(action)
        info["handcrafted_reward"] = compute_handcrafted_reward(self.env)
        
        return obs, liv_reward, done, truncated, info

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.liv_rewards = []
        self.hand_rewards = []

    def _on_step(self) -> bool:
        r_liv = float(self.locals["rewards"][0])
        r_hand = self.locals["infos"][0].get("handcrafted_reward")
        self.liv_rewards.append(r_liv)
        self.hand_rewards.append(r_hand)
        wandb.log({
            "liv_reward": r_liv,
            "handcrafted_reward": r_hand,
        }, step=self.num_timesteps)
        return True

def make_env_fn(rank, urdf_path, task):
    def _init():
        env = RX150EnvLiv(
            urdf_path = urdf_path,
            task = task,
            epsilon_dist = 0.1,
            max_timesteps = 1000,
            step_size = 0.1,
            headless = True,
            image_width = 64,
            image_height = 64,
            frame_skip = 4,
            use_liv = True,
        )
        return RewardWrapper(env)
    return _init

def main():

    wandb.init(project="IFT6163-Project", name="new_LIV")

    urdf_path = "/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/rx150.urdf"
    task = "stack"

    num_envs = 12
    env_fns = [make_env_fn(i, urdf_path, task) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    model = SAC(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./tb_logs",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    callback = RewardLoggingCallback()
    model.learn(
            total_timesteps=1_500_000, 
            callback=
                [
                    callback,
                    WandbCallback(gradient_save_freq=1000,verbose=2)
                ]
    )

    # Get correlations
    corr = np.corrcoef(callback.liv_rewards, callback.hand_rewards)[0, 1]
    wandb.run.summary["reward_correlation"] = float(corr)
    wandb.run.summary["reward_correlation_matrix"] = np.corrcoef(
        callback.liv_rewards, callback.hand_rewards
    ).tolist()

    model.save("sac_handcrafted_multiinput")

if __name__ == "__main__":
    main()
