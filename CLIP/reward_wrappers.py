import gymnasium as gym
import numpy as np

class ClipRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model
        self.step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        gripper_pos = info.get("gripper_pos", np.zeros(3))
        cube_pos = info.get("cube_pos", np.zeros(3))

        reward_dict = self.reward_model.compute_reward(obs, gripper_pos, cube_pos, info.get('success', False))
        total_reward = reward_dict['total_reward']

        info.update({
            "total_reward": total_reward,
            "clip_goal_similarity": reward_dict['goal_sim'],
            "clip_init_similarity": reward_dict['init_sim'],
            "geometric_reward": reward_dict['geometric_reward'],
            "gripper_pos": gripper_pos,
            "cube_pos": cube_pos
        })

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_model.init_features = None
        self.reward_model.init_image = None
        return obs, info

