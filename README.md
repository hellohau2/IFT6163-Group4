# IFT6163-Group4
Repos for the IFT6163's project. 

The project aims to implement the method described in the paper : Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning (https://arxiv.org/pdf/2310.12921) and test it in the real-world on the Robot arm Reactor X150.

The simulated environment generates a robot arm RX150 from the urdf file. The rewards are obtained from the CLIP-Reg ([arxiv](https://arxiv.org/abs/2310.12921)) using the image provided from pybullet (or gazebo, we implemented in both) simulator. This image and the joint angles (in radians) are given as input to the model.

We also experimented with preference-based learning ([arxiv](https://arxiv.org/abs/2402.03681)) using preferences from a VLM (GPT-4o was chosen after testing for correct preferences on a small dataset where we manually indicated preferences).

As it seemed that the basic CLIP model struggled with understanding depth and distances for robotics tasks, we attempted to use fine-tuned models (LIV : [arxiv](https://arxiv.org/abs/2306.00958), PEFT : [arxiv](https://arxiv.org/abs/2409.19457))

Each method is assigned its own branch to simplify reading and understanding the code.

