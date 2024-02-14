import torch
from torch import nn
import torch.nn.functional as F
from models.SNGAN.res_blocks import GeneratorResidualBlock, DiscriminatorSNResidualBlock
from models.SNGAN.core_modules import ConditionalBatchNorm2d, SNEmbedding

class Generator(nn.Module):
    def __init__(self, latent_size, n_classes_g=0):
        super().__init__()
        self.dense = nn.Linear(latent_size, 3 * 3 * 256)
        self.block1 = GeneratorResidualBlock(256, 128, upsample=True, n_classes=n_classes_g)
        self.block2 = GeneratorResidualBlock(128, 64, upsample=True, n_classes=n_classes_g)
        self.block3 = GeneratorResidualBlock(64, 32, upsample=True, n_classes=n_classes_g)
        self.bn_out = ConditionalBatchNorm2d(32, n_classes_g) if n_classes_g > 0 else nn.BatchNorm2d(32)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs, y=None):
        x = self.dense(inputs).view(inputs.size(0), 256, 3, 3)
        x = self.block1(x, y)
        x = self.block2(x, y)
        x = self.block3(x, y)
        x = self.bn_out(x, y) if y is not None else self.bn_out(x)
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self, n_classes_d=0):
        super().__init__()
        self.block1 = DiscriminatorSNResidualBlock(3, 32, 2)
        self.block2 = DiscriminatorSNResidualBlock(32, 64, 2)
        self.block3 = DiscriminatorSNResidualBlock(64, 128, 2)
        self.block4 = DiscriminatorSNResidualBlock(128, 256, 2)
        self.dense = nn.Linear(256, 1)
        if n_classes_d > 0:
            self.sn_embedding = SNEmbedding(n_classes_d, 256)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.leaky_relu(x)
        features = torch.sum(x, dim=(2,3)) # gloobal sum pooling
        x = self.dense(features)
        return x