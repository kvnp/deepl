# -*- coding: utf-8 -*-
# @Date    : 3/26/20
# @Author  : Rene Breuning
# @Link    : None
# @Version : 0.0

import torch.nn as nn
import torch
import torch.nn.functional as F
from models.SNGAN.core_layers import SpectralNorm, ConditionalBatchNorm2d

class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), upsample=False, n_classes=0):
        super(GeneratorResidualBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)

        # hier für bildgröße
        # self.up = nn.Sequential(nn.ConvTranspose2d(in_channels,in_channels, kernel_size=4, stride=1),
        #                         nn.ConvTranspose2d(in_channels,out_channels, kernel_size=3, stride=3))
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=4)
        
        self.b1 = ConditionalBatchNorm2d(in_channels, self.n_classes)
        self.b2 = ConditionalBatchNorm2d(hidden_channels, self.n_classes)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    # def upsample_conv(self, x, conv):
        # return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))
    

    def residual(self, x, label_onehots= None):
        h = x
        h = self.b1(h, label_onehots)
        h = self.activation(h)
        #h = self.upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.up(h) if self.upsample else self.c1(h)
        h = self.b2(h, label_onehots)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            # x = self.upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            x= self.up(x) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, label_onehots= None):
        return self.residual(x, label_onehots) + self.shortcut(x)
    


class DiscriminatorSNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsampling):
        super().__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.downsampling = downsampling
        if in_ch != out_ch or downsampling > 1:
            self.shortcut_conv = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0))
        else:
            self.shortcut_conv = None

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)

    def forward(self, inputs):
        x = F.leaky_relu(inputs)
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        # short cut
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
        else:
            shortcut = inputs
        if self.downsampling > 1:
            x = F.avg_pool2d(x, kernel_size=self.downsampling)
            shortcut = F.avg_pool2d(shortcut, kernel_size=self.downsampling)
        # residual add
        return x + shortcut