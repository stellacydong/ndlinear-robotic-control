# baseline_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselinePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256)):
        super(BaselinePolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

