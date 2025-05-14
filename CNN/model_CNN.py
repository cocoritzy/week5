import torch 
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class UrbanCNN(nn.Module):  # Capital M in Module!
    def __init__(self, num_classes=10):
        super(UrbanCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  # You can reuse this layer

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the flattened size after 3 poolings:
        # Input: (1, 128, 128) → after 3 poolings → (64, 16, 16)
        self.fc1 = nn.Linear(64 * 16 * 15, 128)  # update size based on input!
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (B, 16, 64, 64)
        x = self.pool(self.relu(self.conv2(x)))  # (B, 32, 32, 32)
        x = self.pool(self.relu(self.conv3(x)))  # (B, 64, 16, 16)

        x = self.flatten(x)                      # (B, 64*16*16)
        x = self.relu(self.fc1(x))               # (B, 128)
        x = self.fc2(x)                          # (B, 10)
        return x
