#!/usr/bin/env python3
import os, argparse, torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from GazeboRX150Env.gazebo_rx150_env import GazeboRX150Env

def make_env():
    return Monitor(GazeboRX150Env())

class PrintStepCallback(EvalCallback):
    def _on_step(self):  
        return super()._on_step()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="resume from grasping.zip if present")
    args = ap.parse_args()

    env = VecTransposeImage(DummyVecEnv([make_env]))
    if args.resume and os.path.exists("grasping.zip"):
        print(" Resuming model…"); model = SAC.load("grasping", env=env, device="auto")
    else:
        print("Starting new model…")
        model = SAC("CnnPolicy", env, verbose=1, buffer_size=500_000,
                    tensorboard_log="./tb")

    eval_cb = EvalCallback(env, best_model_save_path="./best",
                           eval_freq=5_000, n_eval_episodes=5)

    model.learn(total_timesteps=50_000, callback=eval_cb)
    model.save("grasping")
