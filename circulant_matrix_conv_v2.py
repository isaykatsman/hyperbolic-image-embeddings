import torch
import torch.nn as nn

# convolves a m x m channel with a k x k kernel
# to produce a (m-k+1) x (m-k+1) output
# (currently assumes NO padding)
# channel: m x m
# ker: k x k
def ker_by_channel(channel, ker, padding=0):
    channel = nn.ConstantPad2d(padding, 0)(channel)
    m1, m2 = channel.size()
    k, _ = ker.size() # assumes square kernel
    channel_v = channel.view(-1)
    dbl_circulant = torch.zeros((m1-k+1) * (m2-k+1), m1 * m2)

    shift = 0 # controls shift when convolution switches to new row
    for i in range(dbl_circulant.size(0)):

        ker_flat = ker.view(-1)
        row = torch.zeros(m1 * m2)

        ix = 0
        for j in range(i + shift, dbl_circulant.size(1)):
            if (j-shift-i) % (m2) < k and ix <= ker_flat.size(0) - 1:
                row[j] = ker_flat[ix]
                ix += 1

        # (shift is k-1 every m1-k+1 rows)
        if (i+1) % (m2-k+1) == 0:
            shift += k-1

        # load a row
        dbl_circulant[i] = row

#     return dbl_circulant, channel_v
    # bug test version
    return dbl_circulant @ channel_v

    # now have a doubly blocked circulant matrix for convolution
    # perform matrix vector in hyp space
#     return pmath.project(pmath.mobius_matvec(dbl_circulant, channel_v, c=c) # pmath.mobius_matvec(self.weight, x, c=c), c=c)


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
def kers_by_channels(channels, kers, padding=0):
    c_in, m1, m2 = channels.size()
    k = kers.size(1)
    out_mat = torch.zeros(m1-k+1 + 2*padding, m2-k+1 + 2*padding).view(-1)
    for i in range(c_in):
        temp_ker = ker_by_channel(channels[i, :, :], kers[i, :, :], padding=padding).view(-1) # temp_ker = in final version
        out_mat = out_mat + temp_ker
#         out_mat = mobius_add(out_mat, temp_ker) # final version
#         out_mat = pmath.project(out_mat, c=c) # final version

    out_mat = out_mat.view(m1-k+1 + 2*padding, m2-k+1 + 2*padding)
    return out_mat

# convolves each of the c_out kers_full_weight kernel volumes
# with channels, thereby producing c_out volumes of
# dimension (m-k+1) x (m-k+1)
# Inputs:
#   channels: bs x c_in x m x m
#   kers_full_weight: c_out x c_in x k x k
# Output:
#   out_mat: bs x c_out x (m-k+1) x (m-k+1)
def full_conv(channels, kers_full_weight, padding=0):
    bs, c_in, m1, m2 = channels.size()
    c_out, _, _, k = kers_full_weight.size()
    out_mat = torch.zeros(bs, c_out, m1-k+1 + 2*padding, m2-k+1 + 2*padding)
    for b in range(bs):
        for i in range(c_out):
            temp_ker = kers_by_channels(channels[b], kers_full_weight[i, :, :, :], padding=padding)
            out_mat[b, i, :, :] = temp_ker

    return out_mat

