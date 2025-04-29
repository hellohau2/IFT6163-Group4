# Robotic Grasping with CLIP & Geometric Rewards

A reinforcement-learning pipeline to train a simulated robotic arm (XR150) to grasp a small cube using both geometric and CLIP-based vision rewards.

---

## File Structure

clip_reward.py
robot_env.py
train.py
test_policies.py
cube_small.urdf
Model_Comparison.png
results.csv
models/
  ├── sac_103_80000_steps.zip #CLIP + Geometric Reward Model
  ├── sac_rx150_ckpt_80000_steps.zip #CLIP Reward Model


## Script Overview

### `robot_env.py`
- Defines a custom Gymnasium environment
- Loads `cube_small.urdf` in PyBullet
- Exposes observations: robot joint states + camera images
- Supports both geometric and CLIP reward modes

### `clip_reward.py`
- Wraps the environment to compute a CLIP-based similarity reward
- Compares current camera frame against target image features

### `train.py`
- Sets up the SAC agent from Stable-Baselines3
- Registers the CLIP/geometry reward wrappers
- Handles logging and checkpointing

### `test_policies.py`
- Loads a trained policy
- Runs evaluation rollouts
- Saves videos for qualitative inspection