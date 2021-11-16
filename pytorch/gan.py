import torch.nn as nn
import torch as torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            nn.ConvTranspose2d(100, 1024,4,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            # 4x4x1024
            nn.ConvTranspose2d(1024, 512,4,stride=2, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            # 8x8x512
            nn.ConvTranspose2d(512, 256,4,stride=2, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            # 16x16x256
            nn.ConvTranspose2d(256, 128,4,stride=2, bias=False, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            # 32x32x128
            nn.Conv2d(128, 1,5, bias=False),
            nn.Tanh()
            # 28x28x1
        )
    def forward(self, x):
        y = torch.reshape(x, (-1,100,1,1))
        y = self.seq(y)
        return y

class Disc(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(1,32,5,bias=False),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            
            nn.Conv2d(32,64,4,stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),

            nn.Conv2d(64,128,4,stride=2,bias=False),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),

            nn.Conv2d(128,1,4,bias=False),
            nn.Sigmoid(),
            nn.Flatten()
        )
    def forward(self, x):
        y = self.seq(x)
        return y
