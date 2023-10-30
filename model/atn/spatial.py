import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernelSize = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d( 2, 1, kernelSize, padding = 1)
        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax(dim =1)

    def forward(self, x):
        avgO = torch.mean( x, dim = 1, keepdim = True)
        maxO, _ = torch.max( x, dim = 1, keepdim = True)
        x = torch.cat([ avgO, maxO], dim = 1)
        x = self.conv(x)
        return x * self.sigmod(x)