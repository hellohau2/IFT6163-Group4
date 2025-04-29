import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import torch.nn.utils as utils

# Network that takes as input an image and outputs  
def create_cnn(out_dim=64):
    '''
    The network assume that the input is an image of shape 64x64x3
    '''

    model = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=3), # Outputs 32x20x20
        nn.ReLU(),
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=3), # Outputs 64x6x6
        nn.ReLU(),
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1), # Outputs 64x4x4
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=1024,out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256,out_features=out_dim)
    )

    return model

def create_mlp(input_dim, out_dim):

    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, out_dim),
    )

    return model

class MultiInputNet(nn.Module):
    def __init__(self,mlp_input_dim,out_dim=64,cnn_out_dim=128,mlp_out_dim=64, combined_out_dim=64):
        super().__init__()

        self.cnn_out_dim=cnn_out_dim
        self.mlp_out_dim=mlp_out_dim
        self.mlp_input_dim=mlp_input_dim

        self.cnn_model = create_cnn(out_dim=cnn_out_dim)
        self.mlp_model = create_mlp(input_dim=mlp_input_dim,out_dim=mlp_out_dim)

        self.combined_model = create_mlp(input_dim= cnn_out_dim + mlp_out_dim, out_dim=combined_out_dim)

    def forward(self, image,state):

        # img_out = F.relu(self.cnn_model(image))
        # state_out = F.relu(self.mlp_model(state))

        img_out = self.cnn_model(image)
        state_out = self.mlp_model(state)

        return self.combined_model(torch.cat([img_out,state_out],dim=-1))

class ExplNet(nn.Module):
    def __init__(self,state_in_dim=None,out_dim=64,lr=1e-4,image_only=True, max_grad_norm=1):
        super().__init__()

        self.image_only = image_only
        self.out_dim=out_dim
        self.lr = lr
        self.max_grad_norm = max_grad_norm

        if image_only : 
            self.predictor = create_cnn(out_dim=self.out_dim)
            self.frozen_model = create_cnn(out_dim=self.out_dim)
        else : 
            self.predictor = MultiInputNet(mlp_input_dim=state_in_dim,combined_out_dim=self.out_dim)
            self.frozen_model = MultiInputNet(mlp_input_dim=state_in_dim,combined_out_dim=self.out_dim)

        # Freeze params for fixed model
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=self.lr, weight_decay=1e-4)

    def forward(self,image,state=None):

        if state is None : 
            error = torch.linalg.norm((self.predictor(image) - self.frozen_model(image)), dim=-1).detach()
        else :
            error = torch.linalg.norm((self.predictor(image,state) - self.frozen_model(image,state)), dim=-1).detach()

        return error

    def update(self,image,state=None):
        self.optimizer.zero_grad()
        
        if state is None : 
            output = self.predictor(image)
            target = self.frozen_model(image)
        else : 
            output = self.predictor(image,state)
            target = self.frozen_model(image,state)

        loss = self.loss(output,target)
        loss.backward()
        utils.clip_grad_norm_(self.predictor.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        return loss.item()

class RewardNet(nn.Module):
    def __init__(self,state_in_dim=None,lr=1e-4,image_only=True, max_grad_norm=1.0):
        super().__init__()

        self.image_only = image_only
        self.max_grad_norm = max_grad_norm
        
        self.lr = lr
        if image_only : 
            self.model = create_cnn(out_dim=1)
        else : 
            self.model = MultiInputNet(mlp_input_dim=state_in_dim,combined_out_dim=1)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.lr,weight_decay=1e-4)

    def forward(self,image, state=None):
        if state == None :
            return F.tanh(self.model(image).detach())
            # return (self.model(image).detach()).clamp(min=-10.0, max=10.0)
        else : 
            return F.tanh(self.model(image, state).detach())
            # return (self.model(image, state).detach()).clamp(min=-10.0, max=10.0)


    def update(self, pairs):

        '''
        Follows the reward loss from preference algorithm (Bradley-Terry).
        Expect array of pairs : {"ob_1","ob_2","y"}
            P(ob_2 > ob_1) = exp(r(ob_2)) / (exp(r(ob_2)) + exp(r(ob_1)))
            P(ob_1 > ob_2) = exp(r(ob_1)) / (exp(r(ob_2)) + exp(r(ob_1)))
        L_r = -E[y[0]*P(ob_1 > ob_2) + y[1]*P(ob_2 > ob_1)]
        '''

        self.optimizer.zero_grad()
        N = len(pairs)

        if self.image_only : 
            imgs_0 = [pair["ob_1"] for pair in pairs]
            imgs_1 = [pair["ob_2"] for pair in pairs]
            imgs_stack = torch.stack(imgs_0 + imgs_1).float().to("cuda")

            # reward model call
            # reward_stack = F.tanh(self.model(imgs_stack).squeeze().to("cuda"))
            reward_stack = (self.model(imgs_stack).squeeze().to("cuda")).clamp(min=-10.0,max=10.0)

        else : 
            # Might have to change that in case not enough memory
            # Stacks images so we only call model once
            imgs_0 = [pair["ob_1"]["image"] for pair in pairs]
            imgs_1 = [pair["ob_2"]["image"] for pair in pairs]
            imgs_stack = torch.stack(imgs_0 + imgs_1).float().to("cuda")

            states_0 = [pair["ob_1"]["joint_states"] for pair in pairs]
            states_1 = [pair["ob_2"]["joint_states"] for pair in pairs]
            states_stack = torch.stack(states_0 + states_1).float().to("cuda")

            # reward model call
            # reward_stack = F.tanh(self.model(imgs_stack, states_stack).squeeze().to("cuda"))
            reward_stack = (self.model(imgs_stack, states_stack).squeeze().to("cuda")).clamp(min=-10.0,max=10.0)

        # Exp rewards
        exp_reward_0 = torch.exp(reward_stack[:N])
        exp_reward_1 = torch.exp(reward_stack[N:])

        exp_reward_0_1 = exp_reward_0 + exp_reward_1 + 1e-8

        # Probs
        P_0_1 = exp_reward_0 / exp_reward_0_1
        P_1_0 = exp_reward_1 / exp_reward_0_1

        # correct preference logic
        y_val = torch.tensor([pair["y"] for pair in pairs]).to("cuda")
        y_none = (y_val == -1)
        
        y_0 = (y_val == 0) + 0.5 * y_none
        y_1 = (y_val == 1) + 0.5 * y_none

        # Final loss
        loss = -(y_0 * torch.log(P_0_1 + 1e-8) + y_1 * torch.log(P_1_0 + 1e-8)).mean()

        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()

        return loss.item()
