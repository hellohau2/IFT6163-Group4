import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
from stable_baselines3 import SAC

from robot_env import RobotEnv
from clip_reward import CLIPReward
from reward_wrappers import ClipRewardWrapper
from pad_gripper import PadGripper

# ActionWrapper to pad 4D actions into the 5D action space expected by RobotEnv
def make_env():
    raw = RobotEnv()
    reward_model = CLIPReward(goal_text="a robot arm grasping a red block")
    wrapped = ClipRewardWrapper(raw, reward_model=reward_model)
    resized = ResizeObservation(wrapped, shape=(64, 64))
    return resized


def load_model(model_path, device="gpu"):
    model = SAC.load(model_path, device=device)
    return model


def record_episode(env, model, video_path=None, max_steps=200):
    frames = []
    total_reward = 0
    obs, info = env.reset()
    terminated = False
    truncated = False
    step = 0
    

    while not terminated and not truncated and step < max_steps:
        if video_path is not None:
            frame = env.render()
            frames.append(frame)

        obs_pi = np.transpose(obs, (2,0,1))[None]  # batch
        action4d, _ = model.predict(obs_pi, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action4d[0])
        reward = info.get("total_reward", reward)
        total_reward += reward
        step += 1

    if video_path is not None:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = imageio.get_writer(video_path, fps=30)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"Saved {video_path}")

    return total_reward


def evaluate_model(model, env, n_episodes=20, video_prefix=None):
    rewards = []
    for idx in range(n_episodes):
        video_file = None
        if video_prefix is not None:
            video_file = os.path.join("videos", f"{video_prefix}_ep{idx}.mp4")
        total_reward = record_episode(env, model, video_file)
        rewards.append(total_reward)
        print(f"Episode {idx} reward: {total_reward}")
    return rewards

def export_results_to_csv(results_dict, filename="results.csv"):
    # Convert the results dict into a DataFrame
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_dict.items()]))
    
    # Save to CSV
    df.to_csv(filename, index_label="Episode")
    print(f"Saved results to {filename}")

def plot_comparison(results_dict):
    plt.figure()
    for label, rewards in results_dict.items():
        plt.plot(rewards, marker='o', label=label)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig('Model_Comparison.png')
    plt.show()


def main():
    env = make_env()

    # Load both models
    model1 = load_model("models/SAC_102/sac_rx150_ckpt_80000_steps", device="cuda" )
    model2 = load_model("models/SAC_103/sac_103_80000_steps", device="cuda")

    # Evaluate models
    rewards_model1 = evaluate_model(model1, env, n_episodes=21, video_prefix="model1")
    rewards_model2 = evaluate_model(model2, env, n_episodes=21, video_prefix="model2")

    # Plot results
    results = {
        "Model 1": rewards_model1,
        "Model 2": rewards_model2
    }
    plot_comparison(results)
    export_results_to_csv(results, filename="results.csv")


    env.close()


if __name__ == "__main__":
    main()