import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Conv1 = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()      
        self.Conv2 = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.Conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.Conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu2(out)
        return out

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, out_size):
        super().__init__()
        self.layer = nn.Sequential(
            BasicBlock(in_channels),
            BasicBlock(in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels, 
                kernel_size=3,
                stride=in_size // out_size,
                padding=1,
                bias=True,
                padding_mode="zeros",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.begin = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64, 
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True,
                padding_mode="zeros",
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        self.layer1 = Layer(64, 64, 56, 56)
        self.layer2 = Layer(64, 128, 56, 28)
        self.layer3 = Layer(128, 256, 28, 14)
        self.layer4 = Layer(256, 512, 14, 7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ln = nn.Linear(512, 27)
    
    def forward(self, x):
        out = self.begin(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.ln(out)
        return out

if __name__ == "__main__":
    model = ResNet()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)