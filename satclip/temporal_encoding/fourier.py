import torch
from torch import nn
import numpy as np

"""
Direct encoding
"""
class Fourier(nn.Module):
    def __init__(self, k: int):
        super(Fourier, self).__init__()

        self.K = int(k)
        # adding this class variable is important to determine
        # the dimension of the follow-up neural network
        self.embedding_dim = 2*self.K

    def forward(self, t):
        """
        Assumes t is of shape (..., 1)
        """

        fk = []
        for k in range(self.K):
            fk.append((1./np.sqrt(2))*torch.sin(k * torch.pi * t / 2.))
            fk.append((1./np.sqrt(2))*torch.cos(k * torch.pi * t / 2.))
        # return torch.stack(fk, dim=-1)
        return torch.cat(fk, dim=-1)