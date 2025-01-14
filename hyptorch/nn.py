import math

import torch
import torch.nn as nn
import torch.nn.init as init

import hyptorch.pmath as pmath


class HyperbolicMLR(nn.Module):
    r"""
    Module which performs softmax classification
    in Hyperbolic space.
    """
    def __init__(self, ball_dim, n_classes, c):
        super(HyperbolicMLR, self).__init__()
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.c = c
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.reset_parameters()

    def forward(self, x, c=None):
        if c is None:
            c = torch.as_tensor(self.c).type_as(x)
        else:
            c = torch.as_tensor(c).type_as(x)
        p_vals_poincare = pmath.expmap0(self.p_vals, c=c)
        conformal_factor = (1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True))
        a_vals_poincare = self.a_vals * conformal_factor
        logits = pmath._hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits


    def extra_repr(self):
        return 'Poincare ball dim={}, n_classes={}, c={}'.format(
            self.ball_dim, self.n_classes, self.c
        )


    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))


class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        mv = pmath.mobius_matvec(self.weight, x, c=c)
        if self.bias is None:
            return pmath.project(mv, c=c)
        else:
            bias = pmath.expmap0(self.bias, c=c)
            return pmath.project(pmath.mobius_add(mv, bias), c=c)


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )

# hyp linear using exp-log, goes from H^n to H^n
# (log to go from H^n to R^n, then do conv, then exp to go from R^n to H^n)
class HypLinear2(nn.Module):
    def __init__(self, in_features, out_features, c, bias=True):
        super(HypLinear2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        #self.reset_parameters()


#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)


    def forward(self, x, c=None):
        if c is None:
            c = self.c

        # note that logmap and exmap are happening with respect to origin
        x_eucl = pmath.logmap0(x, c=c)
        out = self.lin(x_eucl)
        x_hyp = pmath.expmap0(out, c=c)
        x_hyp_proj = pmath.project(x_hyp, c=c)

        return x_hyp_proj


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )

# hyperbolic conv via doing regular conv in exp-log wrapper, 
# and isolating bias to be in H^n
# normalization = "vector" | "perchannel"
class HypConv(nn.Module):
    def __init__(self, in_channels, out_channels, ker_size, c, bias=True, padding=0, normalization='perchannel'):
        super(HypConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c = c
        self.normalization = normalization
        self.conv = nn.Conv2d(in_channels, out_channels, ker_size, bias=False, padding=padding)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c

        # do cast back x to R^n, do conv, then cast the result back to H space
        x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
        out = self.conv(x)
        out = pmath.expmap0(out.view(out.size(0) * out.size(1), -1), c=c).view(out.size())

        # now add the H^n bias
        if self.bias is None:
            return pmath.project(out.view(out.size(0) * out.size(1), -1), c=c).view(out.size())
        else:
            bias = pmath.expmap0(self.bias, c=c)
            bias = bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(out)
            # print dimensions
#             print(out.size())
#             print(bias.size())
            # conventional vector normalization
            interm = pmath.mobius_add(out.contiguous().view(out.size(0) * out.size(1), -1), bias.contiguous().view(bias.size(0) * bias.size(1), -1), c=c).view(out.size())
            normed = pmath.project(interm.view(interm.size(0) * interm.size(1), -1), c=c).view(interm.size())
            return normed


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )
    
# hyperbolic conv via exp-log
# (log to go from H^n to R^n, then do conv, then exp to go from R^n to H^n)
class HypConv2(nn.Module):
    def __init__(self, in_channels, out_channels, ker_size, c, bias=True, padding=0):
        super(HypConv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c = c
        self.conv = nn.Conv2d(in_channels, out_channels, ker_size, bias=bias, padding=padding)
        #self.reset_parameters()


#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)


    def forward(self, x, c=None):
        if c is None:
            c = self.c

        x_eucl = pmath.logmap0(x, c=c)
        out = self.conv(x_eucl)
        x_hyp = pmath.expmap0(out, c=c)
        x_hyp_proj = pmath.project(x_hyp, c=c)

        return x_hyp_proj


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )

# convolves each of c_in channels (of dim m x m) with the
# respective c_in(th) k x k kernel to produce first a set of
# c_in (m-k+1) x (m-k+1) ouput channels that are then
# added gyrovectors to produce a single (m-k+1) x (m-k+1) output
# "vector"
# Inputs:
#   channels: c_in x m1 x m2
#   kers: c_in x k x k
# Output:
#   out_mat: (m-k+1) x (m-k+1)
# def kers_by_channels(channels, kers, c=None, padding=0):
#     c_in, m1, m2 = channels.size()
#     k = kers.size(1)
#     out_mat = torch.zeros(m1-k+1 + 2*padding, m2-k+1 + 2*padding).view(-1).cuda()
#     for i in range(c_in):
#         temp_ker = ker_by_channel(channels[i, :, :], kers[i, :, :], c=c, padding=padding).view(-1) # temp_ker = in final version
#         out_mat = pmath.mobius_add(out_mat, temp_ker, c=c) # final version
#         out_mat = pmath.project(out_mat, c=c) # final version

#     out_mat = out_mat.view(m1-k+1 + 2*padding, m2-k+1 + 2*padding)

#     return out_mat

# convolves a m x m channel with c_out k x k kernel
# to produce a c_out x (m-k+1) x (m-k+1) output
# channel: bs x m x m
# ker: c_out x k x k
# output: bs x c_out x (m-k+1)^2
def ker_by_channel(channel, ker, c=None, padding=0):
    channel = nn.ConstantPad2d(padding, 0)(channel)
    c_out, kernel_size, _ = ker.size()
    bs, m1, m2 = channel.size()
    channel = pmath.logmap0(channel.view(bs, -1), c=c).view(bs, m1, m2)
    channel = nn.functional.conv2d(channel.unsqueeze(1), ker.unsqueeze(1), bias=None).view(bs * c_out, -1)
    channel = pmath.expmap0(channel, c=c)
    channel = pmath.project(channel, c=c)
    return channel

# convolves each of the c_out kers_full_weight kernel volumes
# with channels, thereby producing c_out volumes of
# dimension (m-k+1) x (m-k+1)
# Inputs:
#   channels: bs x c_in x m x m
#   kers_full_weight: c_out x c_in x k x k
# Output:
#   out_mat: bs x c_out x (m-k+1) x (m-k+1)
def full_conv(channels, kers_full_weight, c=None, padding=0):
    bs, c_in, m1, m2 = channels.size()
    c_out, _, _, k = kers_full_weight.size()
    out_mat = None # torch.zeros(bs, c_out, m1-k+1 + 2*padding, m2-k+1 + 2*padding).cuda()

    for j in range(c_in):
        temp_ker = ker_by_channel(channels[:, j, :, :], kers_full_weight[:, j, :, :], c=c, padding=padding)
        # temp_ker : bs * c_out x (m-k+1)^2
        if j == 0:
            out_mat = temp_ker
        else:
            out_mat = pmath.mobius_add(out_mat, temp_ker, c=c)
            out_mat = pmath.project(out_mat, c=c)

    return out_mat.view(bs, c_out, m1-k+1 + 2 * padding, m2-k+1 + 2*padding)

# hyperbolic conv via explicit doubly blocked circulant matrix kernel
# multiplication, viewing each output channel as a vector
class HypConv3(nn.Module):
    def __init__(self, in_channels, out_channels, ker_size, c, bias=True, padding=0, normalization='perchannel'):
        super(HypConv3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c = c
        self.normalization = normalization
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, ker_size, ker_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c

        # do cast back x to R^n, do conv, then cast the result back to H space
#         x = pmath.logmap0(x.view(x.size(0) * x.size(1), -1), c=c).view(x.size())
        out = full_conv(x, self.weight, c=c, padding=self.padding)
#         out = pmath.expmap0(out.view(out.size(0) * out.size(1), -1), c=c).view(out.size())

        # now add the H^n bias
        if self.bias is None:
            return pmath.project(out.view(out.size(0) * out.size(1), -1), c=c).view(out.size())
        else:
            bias = pmath.expmap0(self.bias, c=c)
            bias = bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(out)
            # print dimensions
#             print(out.size())
#             print(bias.size())
            # conventional vector normalization
            interm = pmath.mobius_add(out.contiguous().view(out.size(0) * out.size(1), -1), bias.contiguous().view(bias.size(0) * bias.size(1), -1), c=c).view(out.size())
            normed = pmath.project(interm.view(interm.size(0) * interm.size(1), -1), c=c).view(interm.size())
            return normed


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )

    
class ConcatPoincareLayer(nn.Module):
    def __init__(self, d1, d2, d_out, c):
        super(ConcatPoincareLayer, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out

        self.l1 = HypLinear(d1, d_out, bias=False, c=c)
        self.l2 = HypLinear(d2, d_out, bias=False, c=c)
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.mobius_add(self.l1(x1), self.l2(x2), c=c)


    def extra_repr(self):
        return 'dims {} and {} ---> dim {}'.format(
            self.d1, self.d2, self.d_out
        )


class HyperbolicDistanceLayer(nn.Module):
    def __init__(self, c):
        super(HyperbolicDistanceLayer, self).__init__()
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.dist(x1, x2, c=c, keepdim=True)

    def extra_repr(self):
        return 'c={}'.format(self.c)


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    """
    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError("if train_x=True, ball_dim has to be integer, got {}".format(ball_dim))
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter('xp', None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_x = train_x

    def forward(self, x):
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c)
        return pmath.project(pmath.expmap0(x, c=self.c), c=self.c)

    def extra_repr(self):
        return 'c={}, train_x={}'.format(self.c, self.train_x)


class FromPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Poincare ball
    to n-dim Euclidean space
    """
    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):

        super(FromPoincare, self).__init__()

        if train_x:
            if ball_dim is None:
                raise ValueError("if train_x=True, ball_dim has to be integer, got {}".format(ball_dim))
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter('xp', None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_c = train_c
        self.train_x = train_x

    def forward(self, x):
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return pmath.logmap(xp, x, c=self.c)
        return pmath.logmap0(x, c=self.c)

    def extra_repr(self):
        return 'train_c={}, train_x={}'.format(self.train_c, self.train_x)


