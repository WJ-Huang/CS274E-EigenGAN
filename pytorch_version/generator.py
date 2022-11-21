import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, size=64):
        super().__init__()
        self.fc = nn.Linear(size, size * 4 * 4)