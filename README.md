# IFT6163-Group4
Repos for the IFT6163's project. 

The project aims to implement the method described in the paper : Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning (https://arxiv.org/pdf/2310.12921) and test it in the real-world on the Robot arm Reactor X150.

The simulated environment generates a robot arm RX150 from the urdf file. The rewards are obtained from the CLIP-Reg (from the paper) using the image provided from pybullet simulator. This image and the joint angles (in radians) are given as input to the model.

The current goal is to reach a target position randomly generated within a 1.5 unit radius from the center (the base of the arm). This position is shown using a red sphere. Thus the goal of the arm is to reach the red sphere.

Due to memory limitation, the images are required to be downscaled to 84x84x3 pixels, which essentially removes depth, the red sphere also becomes a couple of pixels big.

As such, a current main limitation is that, even for a human observer, it is very hard to tell the depth and position of the red sphere, as well as how far/close the green end-effector is from the target position.

Here is an example of what the model and CLIP see : 
![image](https://github.com/user-attachments/assets/6108018f-6ed7-4847-9530-d57b875fb4a7)
