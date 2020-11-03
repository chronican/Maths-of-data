import torch
import numpy as np

from torch import nn


class LinearGenerator(nn.Module):
    def __init__(self, noise_dim=2, output_dim=2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(noise_dim, output_dim))
        self.b = nn.Parameter(2 * torch.randn(output_dim))

    def forward(self, z):
        """
        Evaluate on a sample. The variable z contains one sample per row
        """
        # TODO\
        #return zw+b
        return torch.add(torch.matmul(z,self.W),self.b)


class LinearDualVariable(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.v = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        """
        Evaluate on a sample. The variable x contains one sample per row
        """
        # TODO
        return torch.matmul(x,self.v)


    def enforce_lipschitz(self):
        """Enforce the 1-Lipschitz condition of the function"""
        with torch.no_grad():
            #make the gradient of f less than 1, which make the value of v less than 1
           self.v.data=self.v.data/torch.norm(self.v.data)








