import os
from rx150.rx_trainer import RXTrainer
import torch
import gc
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from rx150.rx150_env import RX150Env2
from stable_baselines3.common.monitor import Monitor


# torch.cuda.empty_cache()
# gc.collect()

'''
pkill -f python
rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub 
'''

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

def make_sparse_env(urdf_path):
    def _init():
        env = RX150Env2(
            urdf_path=urdf_path, 
            headless=True,
            use_intrinsic=True,
            max_timesteps=1000,
            use_sparse_rewards=True,
            image_only=False,
            task='stack'
        )
        env = Monitor(env, log_dir)
        return env
    return _init

if __name__ == "__main__":
    urdf_path = "/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/rx150.urdf"
    # trainer = RXTrainer(urdf_path=urdf_path,gemini_api_key="",openai_api_key="",task='stack',image_height=64,image_width=64,replay_buffer_size=1_000_000, image_only=False)
    trainer = RXTrainer(urdf_path=urdf_path,openai_api_key="",task='stack',image_height=64,image_width=64,replay_buffer_size=1_000_000, image_only=False)
    trainer.train()

    # sparse_envs = SubprocVecEnv([make_sparse_env(urdf_path) for _ in range(10)])
    
    # model = SAC("MultiInputPolicy", env=sparse_envs, verbose=1)
    # # model = SAC("CnnPolicy", env=sparse_envs, verbose=1)
    # model.learn(total_timesteps=1_500_000)

    # model.save("SAC_sparse")