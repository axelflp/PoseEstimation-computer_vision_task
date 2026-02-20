import torch
import torch.nn as nn
from torchvision import transforms

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, bias=True, dtype=torch.float32),
            nn.BatchNorm2d(mid_channels, dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, bias=True, dtype=torch.float32),
            nn.BatchNorm2d(out_channels, dtype=torch.float32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Max_Pool = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.Max_Pool(x)

class Up(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.Ups = nn.ConvTranspose2d(in_channels, in_channels//2, 2, stride=2, dtype=torch.float32)
        self.conv = DoubleConv(in_channels, in_channels//2)

    def forward(self, x1, x2):
        x1 = self.Ups(x1)
        crp = transforms.CenterCrop((x1.shape[-2],x1.shape[-1]))
        x2 = crp(x2)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

