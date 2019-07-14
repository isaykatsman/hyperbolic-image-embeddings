import torch.nn as nn
import hyptorch.nn as hypnn
import hyptorch.pmath as pmath

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = nn.MaxPool2d(5)(x)
        x = x.view(x.size(0), -1)
        return x

# Basic Hyperbolic ConvNet with Pooling layer
class HypConvNetEncoder(nn.Module):

    def __init__(self, c, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.c = c

        self.c1 = hypnn.HypConv(x_dim, hid_dim, 3, c, padding=1)
        self.b1 = nn.BatchNorm2d(hid_dim)
        # relu and maxpool(2) on forward pass
        self.c2 = hypnn.HypConv(hid_dim, hid_dim, 3, c, padding=1)
        self.b2 = nn.BatchNorm2d(hid_dim)
        # relu and maxpool(2) on forward pass
        self.c3 = hypnn.HypConv(hid_dim, hid_dim, 3, c, padding=1)
        self.b3 = nn.BatchNorm2d(hid_dim)
        # relu and maxpool(2) on forward pass
        self.c4 = hypnn.HypConv(hid_dim, z_dim, 3, c, padding=1)
        self.b4 = nn.BatchNorm2d(z_dim)

    def forward(self, x, c=None):
        if c is None:
            c = self.c

        # BLOCK 1

        x = self.c1(x, c=c)
        # batch norm
        x = pmath.logmap0(x, c=c)
        x = self.b1(x)
        x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)
        
        # blocked relu and maxpool 2
        x = pmath.logmap0(x, c=c)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)

        # BLOCK 2
        
        x = self.c2(x, c=c)
        # batch norm
        x = pmath.logmap0(x, c=c)
        x = self.b2(x)
        x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)
        
        # blocked relu and maxpool 2
        x = pmath.logmap0(x, c=c)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)

        # BLOCK 3
        
        x = self.c3(x, c=c)
        # batch norm
        x = pmath.logmap0(x, c=c)
        x = self.b3(x)
        x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)
        
        # blocked relu and maxpool 2
        x = pmath.logmap0(x, c=c)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)
        
        # BLOCK 4
        
        x = self.c4(x, c=c)
        # batch norm
        x = pmath.logmap0(x, c=c)
        x = self.b4(x)
        x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)

        # blocked relu and maxpool 2
        x = pmath.logmap0(x, c=c)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)

        # final pool
        x = nn.MaxPool2d(5)(x)
        x = x.view(x.size(0), -1)
        return x   
    
