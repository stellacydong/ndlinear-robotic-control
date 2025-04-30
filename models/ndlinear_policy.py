# ndlinear_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ndlinear import NdLinear

class NdLinearPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256), rank=32):
        super(NdLinearPolicy, self).__init__()
        self.fc1 = NdLinear((obs_dim,), (hidden_sizes[0],), rank)
        self.fc2 = NdLinear((hidden_sizes[0],), (hidden_sizes[1],), rank)
        self.out = NdLinear((hidden_sizes[1],), (action_dim,), rank)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
