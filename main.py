from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
from stable_baselines3 import SAC  
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from gazebo_rx150_env_clean import GazeboRX150Env  
from encoders import R3MExtractor                 # frozen vision backbone


# Environment factory                                                         

def make_env() -> Callable[[], Monitor]:
    """Return a *callable* that builds one wrapped env for DummyVecEnv."""
    def _init() -> Monitor:
        env = GazeboRX150Env()
        return Monitor(env)

    return _init


# Simple periodic print‑callback                                              

class PrintStepCallback(BaseCallback):
    """Print a tick every *n* calls when learning."""

    def __init__(self, total_steps: int, interval: int = 100):
        super().__init__(verbose=0)
        self.total = total_steps
        self.interval = interval

    def _on_step(self) -> bool:  # noqa: D401
        if self.n_calls % self.interval == 0:
            print(f"step {self.n_calls:>6}/{self.total}")
        return True


# Main training routine                                                       

def main() -> None:  # noqa: D401
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = VecTransposeImage(DummyVecEnv([make_env()]))

    # R3M frozen feature extractor + 2‑layer 256‑unit MLP for π and Q‑networks
    policy_kwargs = dict(
        features_extractor_class=R3MExtractor,
        net_arch=[256, 256],
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        device=device,
        batch_size=64,
        buffer_size=100_000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    total_steps = 100_000
    model.learn(
        total_timesteps=total_steps,
        callback=[PrintStepCallback(total_steps)],
        tb_log_name="SAC_R3M_ETRGA",
        progress_bar=True,
    )

    out_dir = Path("models"); out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "rx150_sac_etrga_r3m"
    model.save(model_path)
    print(" training complete - model saved to", model_path)


if __name__ == "__main__":
    main()
