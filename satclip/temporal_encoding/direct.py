import torch
from torch import nn
import numpy as np

"""
Direct encoding
"""
class Direct(nn.Module):
    def __init__(self):
        super(Direct, self).__init__()

        # adding this class variable is important to determine
        # the dimension of the follow-up neural network
        self.embedding_dim = 1

    def forward(self, t):
        # just directly map time
        return t
