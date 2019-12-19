""" Parts of the U-Net model """

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv => ReLU => BN ) * 2 + Dropout"""
    
    def __init__(self, in_channels, out_channels, padding, batch_norm=False, leaky=False, dropout=0.):
        super(DoubleConv, self).__init__()
        
        block = []
        
        # First conv
        block += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding)),
                  nn.LeakyReLU(0.1) if leaky else nn.ReLU()]
        
        # batchnorm
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        
        if dropout > 0:
            block.append(nn.Dropout2d(dropout))
        
        block += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)),
                  nn.LeakyReLU(0.1) if leaky else nn.ReLU()]
        
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        
        self.double_conv = nn.Sequential(*block)
    
    def forward(self, x):
        
        return self.double_conv(x)


class Down(nn.Module):
    """
    Component of contracting path (ends with maxpooling)
    """
    
    def __init__(self, in_channels, out_channels, padding, batch_norm=False, leaky=False, dropout=0.):
        super(Down, self).__init__()
        
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels,
                               padding=padding, batch_norm=batch_norm, leaky=leaky, dropout=dropout)
        self.max_pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        conved = self.conv(x)
        bridge = conved.clone()
        return self.max_pool(conved), bridge


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, padding, batch_norm=False, leaky=False):
        super(Up, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels,
                               padding=padding, batch_norm=batch_norm, leaky=leaky)
    
    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv(out)
        
        return out


class OutConv(nn.Module):
    """Last layer"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
