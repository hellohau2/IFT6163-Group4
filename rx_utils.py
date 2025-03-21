import torch
import torch.nn as nn
import numpy as np

# Network that takes as input an image and outputs  
def create_expl_net(out_dim=64, freeze_params=False):
    '''
    The shapes assume that the input is an image of shape 64x64x3
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

    if freeze_params : 
        for param in model.parameters():
            param.requires_grad = False

    return model

class ExplNet(nn.Module):
    def __init__(self,out_dim=64,lr=1e-3):
        super().__init__()

        self.out_dim=out_dim
        self.lr = lr

        self.predictor = create_expl_net(out_dim=self.out_dim)
        self.frozen_model = create_expl_net(out_dim=self.out_dim,freeze_params=True)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.lr)

    def forward(self,input):
        error = torch.linalg.norm((self.predictor(input) - self.frozen_model(input)), dim=-1).detach()
        return error

    def update(self,input):
        self.optimizer.zero_grad()

        output = self.predictor(input)
        target = self.frozen_model(input)

        loss = self.loss(output,target)
        loss.backward()
        self.optimizer.step()

        return loss.item()
        