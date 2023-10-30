import torch
import torch.nn as nn
from .spatial import SpatialAttention
from ..hat_base import ChannelAttention

class SMA(nn.Module):
    def __init__(self, inc, ouc):
        super().__init__()
        d = ouc//2
        self.sab = SpatialAttention()
        self.cab = ChannelAttention(d)
        self.conv_first = nn.Sequential(
            nn.Conv2d(1, d, 1, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d( d, d, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d( d ),
            nn.ReLU(),
            nn.Conv2d( d, d, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d( d ),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d( d, d, kernel_size = 3, padding = 2, dilation = 2),
            nn.BatchNorm2d( d ),
            nn.ReLU(),
            nn.Conv2d( d, d, kernel_size = 3, padding = 2, dilation = 2),
            nn.BatchNorm2d( d ),
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(d , ouc, 1, 1),
            nn.BatchNorm2d( ouc ),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        
        
     

    def forward(self,x):
        residual = x
        x2 = self.conv1_1(x)
        x4 = self.conv1_2(x)
        xs = self.relu(x2 + x4)
        # x6 = self.conv1(x)
        x7 = self.psa(x)

    def forward(self,x):
        residual = self.conv_first(x)
        ca = self.cab(residual)
        sa = ca * self.sab(ca)
        
        x2 = self.conv1_1(residual)
        x4 = self.conv1_2(residual)
        xs = self.relu(x2 + x4)
        
        out = self.relu(residual + sa + xs)
        return  self.conv_last(out)