import torch
import torch.nn as nn
import torch.nn.functional as F

# discriminator for 64x64 images
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)

        self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))
        out = F.leaky_relu(self.conv5(out))
        out = F.leaky_relu(self.conv6(out))
        out = F.leaky_relu(self.conv7(out))
        out = F.leaky_relu(self.conv8(out))
        out = out.reshape(-1, 256)
        out = F.leaky_relu(self.fc1(out))
        out = self.fc2(out)
        return out