import torch
import torch.nn as nn
from .blocks import Up, Down, DoubleConv

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.D1 = DoubleConv(config['channels']['c1'], config['channels']['c2'])
        self.D2 = Down(config['channels']['c2'], config['channels']['c3'])
        self.D3 = Down(config['channels']['c3'], config['channels']['c4'])
        self.D4 = Down(config['channels']['c4'], config['channels']['c5'])
        self.U1 = Up(config['channels']['c5'])
        self.U2 = Up(config['channels']['c4'])
        self.U3 = Up(config['channels']['c3'])
        self.U4 = nn.Conv2d(config['channels']['c2'], config['channels']['c6'], 1, dtype=torch.float32)
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['FFN']['in'], config['FFN']['hl1'], dtype=torch.float32),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.Linear(config['FFN']['hl1'], config['FFN']['hl2'], dtype=torch.float32),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.Linear(config['FFN']['hl2'], config['FFN']['hl3'], dtype=torch.float32),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.D1(x)
        x2 = self.D2(x1)
        x3 = self.D3(x2)
        x4 = self.D4(x3)
        x5 = self.U1(x4, x3)
        x6 = self.U2(x5, x2)
        x7 = self.U3(x6, x1)
        x8 = self.U4(x7)
        x9 = self.mlp(x8)
        
        return x9*self.config['out']