import torch

def ker_by_channel(channel, ker):
    m, _ = channel.size()
    k, _ = ker.size() # assumes square kernel
    channel_v = channel.view(-1)
    dbl_circulant = torch.zeros((m-k+1)**2, m**2)

    shift = 0 # controls shift when convolution switches to new row
    for i in range(dbl_circulant.size(0)):

        ker_flat = ker.view(-1)
        row = torch.zeros(m**2)

        ix = 0
        for j in range(i + shift, dbl_circulant.size(1)):
            if (j-shift-i) % (m) < k and ix <= ker_flat.size(0) - 1:
                row[j] = ker_flat[ix]
                ix += 1

        # (shift is k-1 every n-m+1 rows)
        if (i+1) % (m-k+1) == 0:
            shift += k-1

        # load a row
        dbl_circulant[i] = row

    # return dbl_circulant, channel_v
    return dbl_circulant @ channel_v.unsqueeze(1)

def kers_by_channels(channels, kers):
    c_in, m, _ = channels.size()
    k = kers.size(1)
    out_mat = torch.zeros(m-k+1, m-k+1).view(-1)
    for i in range(c_in):
        temp_ker = ker_by_channel(channels[i, :, :], kers[i, :, :]).view(-1) # temp_ker = in final version
        out_mat = out_mat + temp_ker
#         out_mat = mobius_add(out_mat, temp_ker) # final version
#         out_mat = pmath.project(out_mat, c=c) # final version

    out_mat = out_mat.view(m-k+1, m-k+1)
    return out_mat

def full_conv(channels, kers_full_weight):
    c_in, m, _ = channels.size()
    c_out, _, _, k = kers_full_weight.size()
    out_mat = torch.zeros(c_out, m-k+1, m-k+1)
    for i in range(c_out):
        temp_ker = kers_by_channels(channels, kers_full_weight[i, :, :, :])
        out_mat[i, :, :] = temp_ker

    return out_mat
