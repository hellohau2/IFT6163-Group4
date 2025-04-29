from stable_baselines3 import SAC, PPO
from rx150.preference import QwenPref, GPTPref, GeminiPref
from rx150.rx150_env import RX150Env2
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import rx150.rx_utils as rx_utils
import torch.nn as nn
import torch
import numpy as np
import wandb
import torch.optim as optim
import torch.multiprocessing as mp
import pybullet as p
import gc

from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import Wrapper

# logging callback
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.learned_reward = []
        self.hand_rewards = []
        self.current_step = 0

    def _on_step(self) :
        
        r_learned  = float(self.locals["rewards"][0])
        r_hand = self.locals["infos"][0].get("handcrafted_reward", 0.0)
        self.learned_reward.append(r_learned)
        self.hand_rewards.append(r_hand)
        self.current_step += 1
        
        # wandb.log({
        #     "learned_reward": r_learned,
        #     "handcrafted_reward": r_hand,
        # }, step=self.num_timesteps)
        if r_hand == 0 : return True
        
        wandb.log({
            "learned_reward": r_learned,
            "handcrafted_reward": r_hand,
        })

        if self.current_step % 10000 == 0:
            corr = np.corrcoef(self.learned_reward, self.hand_rewards)[0,1]
            print(f"[RewardLoggingCallback] step {self.current_step}: reward correlation = {corr:.4f}")
        
        return True

def compute_handcrafted_reward(env):
    ee = np.array(env.get_end_effector_pos())
    if env.task == "reach":
        d = np.linalg.norm(ee - env.target_pos)
        return -d
    elif env.task == "stack":
        # distance end effector to lower cube
        low_pos, _ = p.getBasePositionAndOrientation(env.lower_cube)
        low_pos = np.array(low_pos)
        d_ee_low = np.linalg.norm(ee - low_pos)
        
        # distance between cubes
        up_pos, _ = p.getBasePositionAndOrientation(env.upper_cube)
        up_pos = np.array(up_pos)
        d_cubes = np.linalg.norm(up_pos - low_pos)
        
        # bonus if properly stacked
        bonus_stack = 1.0 if env.are_cubes_stacked() else 0.0
        return -d_ee_low - d_cubes + bonus_stack
    else:
        return 0.0

class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        obs, learned_reward, done, truncated, info = self.env.step(action)
        info["handcrafted_reward"] = compute_handcrafted_reward(self.env)
        return obs, learned_reward, done, truncated, info

def make_expl_env(urdf_path, task, image_only, image_width, image_height):
    def _init():
        env = RX150Env2(
            urdf_path=urdf_path,
            headless=True,
            use_intrinsic=True,
            max_timesteps=1000,
            task=task,
            image_only=image_only,
            image_height=image_height,
            image_width=image_width
        )
        return Monitor(env)
    return _init

def make_env(urdf_path, reward_model, task, image_only, image_width, image_height):
    def _init():
        env = RewardWrapper(RX150Env2(
            urdf_path=urdf_path,
            headless=True,
            max_timesteps=1000,
            reward_model=reward_model,
            use_intrinsic=False,
            task=task,
            image_only=image_only,
            image_height=image_height,
            image_width=image_width
        ))
        return Monitor(env)
    return _init


class RXTrainer:
    def __init__(self,
                 urdf_path,
                 openai_api_key=None,
                 gemini_api_key=None,
                 task='reach',
                 image_only=True,
                 replay_buffer_size=1_000_000,
                 image_width=64,
                 image_height=64,
                 agent_method="SAC"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.current_iter = 0

        if self.task == 'reach':
            self.goal_description = "to reach the red ball with the green end-effector, describe the orientation of the robot arm and the position of the red ball"
        elif self.task == 'stack':
            self.goal_description = "to stack the blue cube vertically on top of the red cube using the green end-effector, the goal can be decomposed into sub-goals : reach blue cube, grasp blue cube, reach red cube while grasping blue cube, stack blue on top of red."
        else:
            raise ValueError("Given task is not implemented")

        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key

        if openai_api_key is not None:
            self.gpt_pref = GPTPref(
                api_key=openai_api_key,
                model_name="gpt-4o-2024-11-20",
                detail='high',
                goal_desc=self.goal_description,
                max_tokens=512
            )
        if gemini_api_key is not None:
            self.gemini_pref = GeminiPref(
                api_key=gemini_api_key,
                model_name="gemini-2.0-flash",
                goal_desc=self.goal_description
            )

        self.num_parallel_envs = 10
        self.pair_sample_size = 64
        self.n_iter = 100
        self.image_only = image_only
        self.agent_method = agent_method
        self.image_width = image_width
        self.image_height = image_height

        # reward model
        self.reward_model = rx_utils.RewardNet(
            image_only=image_only,
            state_in_dim=7
        ).to(self.device)
        for param in self.reward_model.parameters():
            param.share_memory_()

        # initialize wandb
        wandb.init(
            project="IFT6163-Project",
            name="Run_1",
        )
        wandb.config.update({
            "agent_method": self.agent_method,
            "task": self.task,
            "image_only": self.image_only,
            "replay_buffer_size": replay_buffer_size,
            "num_parallel_envs": self.num_parallel_envs,
            "pair_sample_size": self.pair_sample_size,
            "n_iter": self.n_iter,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "reward_model_arch": str(self.reward_model),
        })
        wandb.run.log_code('.')

        # environments
        print("Creating exploration envs")
        self.expl_envs = SubprocVecEnv([make_expl_env(urdf_path, task, image_only, image_width, image_height) for _ in range(self.num_parallel_envs)])
        print("Creating regular envs")
        self.envs = SubprocVecEnv([make_env(urdf_path, self.reward_model, task, image_only, image_width, image_height) for _ in range(self.num_parallel_envs)])

        # policy kwargs
        policy_kwargs = dict(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs=dict(weight_decay=1e-4),
        )

        # instantiate agent
        if image_only:
            policy_type = "CnnPolicy"
        else:
            policy_type = "MultiInputPolicy"

        if self.agent_method == "SAC":
            self.agent = SAC(
                policy_type,
                env=self.expl_envs,
                verbose=1,
                policy_kwargs=policy_kwargs,
                buffer_size=replay_buffer_size,
                learning_starts=1_000
            )
        else:
            self.agent = PPO(
                policy_type,
                env=self.expl_envs,
                verbose=1,
                policy_kwargs=policy_kwargs,
                batch_size=256,
                learning_starts=1_000
            )

        self.all_labeled_pairs = []

    def reset_params(self, module: nn.Module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def reset_agent_critics(self):
        self.agent.policy.critic.apply(self.reset_params)
        self.agent.policy.critic_target.apply(self.reset_params)

    def reassign_rewards(self, batch_size=256):
        buffer = self.agent.replay_buffer
        total_size = buffer.size()

        if self.image_only:
            obs_img = buffer.observations[:total_size]
        else:
            obs_img = buffer.observations["image"][:total_size]
            obs_state = buffer.observations["joint_states"][:total_size]
            obs_state_flat = obs_state.reshape(-1, 7)

        obs_img_flat = obs_img.reshape(-1, 3, self.image_height, self.image_width)
        new_rewards_flat = np.zeros((obs_img_flat.shape[0], 1), dtype=np.float32)

        start_idx = 0
        while start_idx < obs_img_flat.shape[0]:
            end_idx = min(start_idx + batch_size, obs_img_flat.shape[0])
            batch_obs_img = torch.tensor(
                obs_img_flat[start_idx:end_idx],
                dtype=torch.float32
            ).to(self.device)

            if self.image_only:
                reward_batch = self.reward_model(batch_obs_img).detach().cpu().numpy()
            else:
                batch_obs_state = torch.tensor(
                    obs_state_flat[start_idx:end_idx],
                    dtype=torch.float32
                ).to(self.device)
                reward_batch = self.reward_model(
                    batch_obs_img,
                    batch_obs_state
                ).detach().cpu().numpy()

            new_rewards_flat[start_idx:end_idx] = reward_batch
            start_idx = end_idx

        new_rewards = new_rewards_flat.reshape(total_size, self.num_parallel_envs)
        buffer.rewards[:total_size] = new_rewards

        # log histogram of reward outputs
        wandb.log({
            "reward_model/output_hist": wandb.Histogram(new_rewards_flat.flatten().tolist()),
            "iter": getattr(self, "current_iter", 0)
        })

    def sample_pairs_entropy(self, candidate_multiplier=10, eps=1e-8):
        if self.agent.replay_buffer.size() < self.pair_sample_size * candidate_multiplier * 2:
            print("Not enough samples for entropy sampling, falling back to uniform sampling.")
            return self.sample_pairs()

        candidate_num = self.pair_sample_size * candidate_multiplier
        samples = self.agent.replay_buffer.sample(candidate_num * 2).observations

        if self.image_only:
            obs = torch.tensor(samples).float().div(255.0).to(self.device)
            rewards = self.reward_model(obs).squeeze()
        else:
            imgs = torch.tensor(samples["image"]).float().div(255.0).to(self.device)
            states = torch.tensor(samples["joint_states"]).to(self.device)
            rewards = self.reward_model(imgs, states).squeeze()

        r0, r1 = rewards[:candidate_num], rewards[candidate_num:]
        p1 = torch.sigmoid(r1 - r0)
        p0 = 1 - p1
        entropy = -(p1 * torch.log2(p1 + eps) + p0 * torch.log2(p0 + eps))
        _, top_idx = torch.topk(entropy, k=self.pair_sample_size)
        top_idx = top_idx.cpu().numpy()

        pairs = []
        for idx in top_idx:
            if self.image_only:
                ob1 = samples[idx]
                ob2 = samples[idx + candidate_num]
            else:
                ob1 = {"image": samples["image"][idx], "joint_states": samples["joint_states"][idx]}
                ob2 = {"image": samples["image"][idx + candidate_num], "joint_states": samples["joint_states"][idx + candidate_num]}
            pairs.append({"ob_1": ob1, "ob_2": ob2})
        return pairs

    def sample_pairs(self):
        pairs = []
        samples = self.agent.replay_buffer.sample(self.pair_sample_size * 2).observations
        for i in range(self.pair_sample_size):
            if self.image_only:
                pairs.append({"ob_1": samples[i], "ob_2": samples[i + self.pair_sample_size]})
            else:
                ob1 = {"image": samples["image"][i], "joint_states": samples["joint_states"][i]}
                ob2 = {"image": samples["image"][i + self.pair_sample_size], "joint_states": samples["joint_states"][i + self.pair_sample_size]}
                pairs.append({"ob_1": ob1, "ob_2": ob2})
        return pairs

    def get_preferences(self):
        pairs = self.sample_pairs_entropy()
        count_pref = [0, 0, 0]

        for pair in pairs:
            if self.image_only:
                y = self.gpt_pref.get_preference(pair["ob_1"], pair["ob_2"])
            else:
                y = self.gpt_pref.get_preference(pair["ob_1"]["image"], pair["ob_2"]["image"])
            pair['y'] = y
            count_pref[y + 1] += 1

        num_pairs = len(pairs)
        wandb.log({
            "preferences/minus_one": count_pref[0] / num_pairs,
            "preferences/neutral": count_pref[1] / num_pairs,
            "preferences/plus_one": count_pref[2] / num_pairs,
            "preferences/total_pairs": num_pairs,
            "iter": self.current_iter
        })

        return pairs

    def update_reward_model(self, new_pairs, epochs=15):
        batch_size = self.pair_sample_size
        self.all_labeled_pairs.extend(new_pairs)

        for epoch in range(epochs):
            np.random.shuffle(self.all_labeled_pairs)
            epoch_losses = []
            for i in range(0, len(self.all_labeled_pairs), batch_size):
                batch = self.all_labeled_pairs[i:i + batch_size]
                loss = self.reward_model.update(batch)
                epoch_losses.append(loss)
                wandb.log({
                    "reward_model/train_loss_batch": loss,
                    "reward_model/epoch": epoch,
                    "reward_model/batch_idx": i // batch_size,
                    "iter": self.current_iter
                })
            wandb.log({
                "reward_model/epoch_mean_loss": np.mean(epoch_losses),
                "reward_model/epoch_std_loss": np.std(epoch_losses),
                "iter": self.current_iter
            })

    def train(self):

        callback = RewardLoggingCallback()

        # Pretrain on intrinsic rewards
        print("Starting Pretraining")
        self.agent.learn(
            total_timesteps=150_000,
            callback=[callback]
        )

        # Prepare for real training 
        self.reset_agent_critics()
        self.reassign_rewards()

        print("Shutting down exploration envs to free RAM")
        self.expl_envs.close()
        del self.expl_envs
        gc.collect() 

        self.agent.set_env(self.envs)
        self.agent.learning_starts = 0

        for it in range(self.n_iter):
            self.current_iter = it
            print(f"\n---- Iteration {it} ----")

            print("Labeling data (preference)")
            labeled = self.get_preferences()
            self.update_reward_model(labeled)
            self.reassign_rewards()

            print("Training Agent")
            self.agent.learn(
                total_timesteps=200_000,
                reset_num_timesteps=False,
                callback=[callback]
            )

        # correlation
        corr = np.corrcoef(callback.learned_reward, callback.hand_rewards)[0,1]
        corr_matrix = np.corrcoef(callback.learned_reward, callback.hand_rewards).tolist()
        wandb.run.summary["reward_correlation"] = float(corr)
        wandb.run.summary["reward_correlation_matrix"] = corr_matrix

        # Save final policy
        self.agent.save(f"{self.agent_method}_rx150_preference_final")
        print(f"Final learned vs handcrafted reward corr = {corr:.4f}")

        return corr, corr_matrix