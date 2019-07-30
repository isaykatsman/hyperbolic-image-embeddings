import torch.nn as nn
import hyptorch.nn as hypnn
import hyptorch.pmath as pmath

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    # removed batchnorm after conv: nn.BatchNorm2d(out_channels),
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
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

    def __init__(self, c, x_dim=3, hid_dim=64, z_dim=64, normalization='perchannel'):
        super().__init__()
        self.c = c
        self.normalization = normalization

        print('getting here')

        self.c1 = hypnn.HypConv3(x_dim, hid_dim, 3, c, padding=1, normalization=normalization)
#         self.b1 = nn.BatchNorm2d(hid_dim)
        # relu and maxpool(2) on forward pass
        self.c2 = hypnn.HypConv3(hid_dim, hid_dim, 3, c, padding=1, normalization=normalization)
#         self.b2 = nn.BatchNorm2d(hid_dim)
        # relu and maxpool(2) on forward pass
        self.c3 = hypnn.HypConv3(hid_dim, hid_dim, 3, c, padding=1, normalization=normalization)
#         self.b3 = nn.BatchNorm2d(hid_dim)
        # relu and maxpool(2) on forward pass
        self.c4 = hypnn.HypConv3(hid_dim, z_dim, 3, c, padding=1, normalization=normalization)
#         self.b4 = nn.BatchNorm2d(z_dim)

    def forward(self, x, c=None):
        if c is None:
            c = self.c

        # do proper normalization of euclidean data
        x = pmath.project(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())

        # BLOCK 1

        x = self.c1(x, c=c)
        # batch norm
#         x = pmath.logmap0(x, c=c)
#         x = self.b1(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # blocked relu and maxpool 2
#         x = pmath.logmap0(x, c=c)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # separate relu and maxpool 2
        x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
        x = nn.ReLU()(x)
        x = pmath.expmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())

#         xbrp = x
#         print(f'norm after relu: {x.norm(dim=-1, keepdim=True, p=2)[0]}')
        x = pmath.project(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         print(f'norm after projection relu: {x.norm(dim=-1, keepdim=True, p=2)[0]}')
#         print(f'diff: {xbrp[0]-x[0]}')
#         print(f'diff sum: {sum(sum(sum(xbrp[0]-x[0])))}')
#         print(f'x after relu project the same: {xbrp.equal(x)}')

        x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
        x = nn.MaxPool2d(2)(x)
        x = pmath.expmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())

#         xbpp = x
        x = pmath.project(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         print(f'x after pool project the same: {xbpp.equal(x)}')

#         # BLOCK 2

#         x = self.c2(x, c=c)
#         # batch norm
# #         x = pmath.logmap0(x, c=c)
# #         x = self.b2(x)
# #         x = pmath.expmap0(x, c=c)
# #         x = pmath.project(x, c=c)

#         # blocked relu and maxpool 2
# #         x = pmath.logmap0(x, c=c)
# #         x = nn.ReLU()(x)
# #         x = nn.MaxPool2d(2)(x)
# #         x = pmath.expmap0(x, c=c)
# #         x = pmath.project(x, c=c)

#         # separate relu and maxpool 2
#         x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = nn.ReLU()(x)
#         x = pmath.expmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = pmath.project(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())

#         x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = pmath.project(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())

#         # BLOCK 3

#         x = self.c3(x, c=c)
#         # batch norm
# #         x = pmath.logmap0(x, c=c)
# #         x = self.b3(x)
# #         x = pmath.expmap0(x, c=c)
# #         x = pmath.project(x, c=c)

#         # blocked relu and maxpool 2
# #         x = pmath.logmap0(x, c=c)
# #         x = nn.ReLU()(x)
# #         x = nn.MaxPool2d(2)(x)
# #         x = pmath.expmap0(x, c=c)
# #         x = pmath.project(x, c=c)

#         # separate relu and maxpool 2
#         x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = nn.ReLU()(x)
#         x = pmath.expmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = pmath.project(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())

#         x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = pmath.project(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())

#         # BLOCK 4

#         x = self.c4(x, c=c)
#         # batch norm
# #         x = pmath.logmap0(x, c=c)
# #         x = self.b4(x)
# #         x = pmath.expmap0(x, c=c)
# #         x = pmath.project(x, c=c)

#         # blocked relu and maxpool 2
# #         x = pmath.logmap0(x, c=c)
# #         x = nn.ReLU()(x)
# #         x = nn.MaxPool2d(2)(x)
# #         x = pmath.expmap0(x, c=c)
# #         x = pmath.project(x, c=c)

#         # separate relu and maxpool 2
#         x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = nn.ReLU()(x)
#         x = pmath.expmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = pmath.project(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())

#         x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
#         x = pmath.project(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())

        # final pool
        x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
        x = nn.MaxPool2d(5)(x)
        x = pmath.expmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
        x = pmath.project(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())

        # print(x.size()), currently N x 512 x 1 x 1

        # currently I believe this step may mess with the geometry
        # what would be a natural replacement? A: view as eucl vector, then do expmap to go back to hyperbolic space
        x = x.view(x.size(0), -1)
        x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)
        return x


# Encoder with some hyperbolic layers
class EncoderWithSomeHyperbolic(nn.Module):

    def __init__(self, c, x_dim=3, hid_dim=64, z_dim=64, args=None):
        super().__init__()
        self.c = c

        self.c1 = nn.Conv2d(x_dim, hid_dim, 3, padding=1)
#         self.b1 = nn.BatchNorm2d(hid_dim)
        # relu and maxpool(2) on forward pass
        self.c2 = nn.Conv2d(hid_dim, hid_dim, 3, padding=1)
#         self.b2 = nn.BatchNorm2d(hid_dim)
        # relu and maxpool(2) on forward pass
        self.c3 = nn.Conv2d(hid_dim, hid_dim, 3, padding=1)
#         self.b3 = nn.BatchNorm2d(hid_dim)
        # relu and maxpool(2) on forward pass
        self.c4 = hypnn.HypConv(hid_dim, z_dim, 3, c, padding=1)
#         self.b4 = nn.BatchNorm2d(z_dim)

        self.e2p = hypnn.ToPoincare(c=args.c, train_c=args.train_c, train_x=args.train_x)

    def get_c(self):
        return self.e2p.c
        
    def forward(self, x, c=None):
        if c is None:
            c = self.c

        # BLOCK 1

        x = self.c1(x)
        # batch norm
#         x = pmath.logmap0(x, c=c)
#         x = self.b1(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)
        
        # blocked relu and maxpool 2
#         x = pmath.logmap0(x, c=c)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # separate relu and maxpool 2
#         x = pmath.logmap0(x, c=c)
        x = nn.ReLU()(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

#         x = pmath.logmap0(x, c=c)
        x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # BLOCK 2

        x = self.c2(x)
        # batch norm
#         x = pmath.logmap0(x, c=c)
#         x = self.b2(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # blocked relu and maxpool 2
#         x = pmath.logmap0(x, c=c)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # separate relu and maxpool 2
#         x = pmath.logmap0(x, c=c)
        x = nn.ReLU()(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

#         x = pmath.logmap0(x, c=c)
        x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # BLOCK 3

        x = self.c3(x)
        # batch norm
#         x = pmath.logmap0(x, c=c)
#         x = self.b3(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # blocked relu and maxpool 2
#         x = pmath.logmap0(x, c=c)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # separate relu and maxpool 2
#         x = pmath.logmap0(x, c=c)
        x = nn.ReLU()(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

#         x = pmath.logmap0(x, c=c)
        x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # BLOCK 4, to hyperbolic

        x = self.e2p(x)

        x = self.c4(x, c=c)
        # batch norm
#         x = pmath.logmap0(x, c=c)
#         x = self.b4(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # blocked relu and maxpool 2
#         x = pmath.logmap0(x, c=c)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x, c=c)
#         x = pmath.project(x, c=c)

        # separate relu and maxpool 2
#         x = pmath.logmap0(x, c=c)
        x = nn.ReLU()(x)
#         x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)

#         x = pmath.logmap0(x, c=c)
        x = nn.MaxPool2d(2)(x)
#         x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)

        # final pool
#         x = pmath.logmap0(x, c=c)
        x = nn.MaxPool2d(5)(x)
#         x = pmath.expmap0(x, c=c)
        x = pmath.project(x, c=c)

        # print(x.size()), currently N x 512 x 1 x 1

        # currently I believe this step mess with the geometry - what would be a natural replacement? 
        x = x.view(x.size(0), -1)
        return x   

    