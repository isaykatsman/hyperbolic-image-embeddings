import torch
torch.set_printoptions(precision=20)
import torch.nn as nn
import numpy as np

from circulant_matrix_conv_v2 import ker_by_channel, kers_by_channels, full_conv

tests = 20
passed = 0
for test in range(tests):
    if test % 10 == 0:
        print(test)
    c_in = np.random.randint(3,10)
    c_h = np.random.randint(10,20) # 10,20
    c_w = np.random.randint(10,20)
    c_out = np.random.randint(3,10)
    ker_size = np.random.randint(2,4)
    padding = np.random.randint(3)

    chs = torch.randn(c_in,c_h,c_w)
    kers_full = torch.randn(c_out,c_in,ker_size,ker_size)
    our_out = full_conv(chs.unsqueeze(0), kers_full, padding=padding).double()
    conv1 = nn.Conv2d(c_in,c_out,ker_size, padding=padding, bias=False)
    conv1.weight = nn.Parameter(kers_full)
    their_out = conv1(chs.unsqueeze(0)).double()
    print(our_out.norm())
    print(their_out.norm())
    res = torch.sub(their_out.detach(), our_out)
    print(f'max of sub: {res.abs().max()}')
    print(f'min of sub: {res.abs().min()}')
    print(torch.sub(their_out.detach(), our_out).norm())
    print(torch.sub(their_out.detach(), our_out).norm().item() < 1e-4)
    passed += torch.sub(their_out.detach(), our_out).norm().item() < 1e-4
    print(f'torch equal : {torch.equal(our_out, their_out)}')

print(f'Passed {passed} out of {tests} tests ran')

