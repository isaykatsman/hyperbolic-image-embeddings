# nn.py legacy 

# convolves a m x m channel with a k x k kernel
# to produce a (m-k+1) x (m-k+1) output 
# (currently assumes NO padding)
# channel: m x m
# ker: k x k
# sparse version, doesn't fully work (can pretty easily rever back to non-sparse if needed)
def ker_by_channel_sparse(channel, ker, c=None, padding=0):
    channel = nn.ConstantPad2d(padding, 0)(channel)
    m1, m2 = channel.size()
    k, _ = ker.size() # assumes square kernel
    ker_flat = ker.view(-1)

    # final dbl_circulant will be [(m1-k+1) * (m2-k+1)] x [m1 * m2]
    output_size = (m1-k+1) * (m2-k+1)
    intput_size = m1 * m2

    channel_v = channel.view(-1)
    # dbl_circulant = torch.zeros((m1-k+1) * (m2-k+1), m1 * m2).cuda()
    val_matrix = ker_flat.repeat((m1-k+1) * (m2-k+1))
    ix_matrix = torch.zeros(2, ker_flat.size(0) * (m1-k+1) * (m2-k+1)).long().cuda() # torch.cuda.LongTensor((2, k * (m1-k+1) * (m2-k+1)))

    shift = 0 # controls shift when convolution switches to new row
    for i in range(output_size):
        ix = 0
        for j in range(i + shift, intput_size):
            if (j - shift - i) % (m2) < k and ix <= ker_flat.size(0) - 1:
                ix_matrix[0, i + ix] = i
                ix_matrix[1, i + ix] = j
                ix += 1
                # row[j] = ker_flat[ix].item() # shouldn't really be a problem but actually is: ker_flat[ix] (prolly pt bug)
            if ix == ker_flat.size(0):
                break

        # (shift is k-1 every m2-k+1 rows)
        if (i+1) % (m2-k+1) == 0:
            shift += k-1

        # load a row
#         dbl_circulant[i] = row
        
#     return dbl_circulant, channel_v
    # bug test version
#     return dbl_circulant @ channel_v

    # now have a doubly blocked circulant matrix for convolution
    # perform matrix vector in hyp space
    dbl_circulant = torch.sparse.FloatTensor(ix_matrix, val_matrix, torch.Size([(m1-k+1) * (m2-k+1),m1 * m2]))
    out = pmath.project(pmath.mobius_matvec(dbl_circulant, channel_v, c=c), c=c) #pmath.mobius_matvec(self.weight, x, c=c), c=c)
    return out