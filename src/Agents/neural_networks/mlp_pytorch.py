import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicMLP(nn.Module):
    """
    Basic MLP that is used in the hybrid network
    """
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.fc1 = nn.Linear(num_input_features, 128)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        self.fc2 = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.0)
        self.fc3 = nn.Linear(64, num_output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x