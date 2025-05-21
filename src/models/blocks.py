import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
