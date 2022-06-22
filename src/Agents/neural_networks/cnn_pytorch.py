import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    """
    Basic CNN that is used in the hybrid network
    """

    def __init__(self, num_feature_maps: int, num_output_features: int):
        super().__init__()
        # board size is 21
        self.conv1 = nn.Conv2d(num_feature_maps, 32, 3)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0.0)
        self.conv2 = nn.Conv2d(32, 64, 3)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0.0)
        self.conv3 = nn.Conv2d(64, 64, 2)
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        nn.init.constant_(self.conv3.bias, 0.0)
        # resulting shape: (batch_size, 64, 8, 8)
        self.fc1 = nn.Linear(16 * 16 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_output_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
