import torch
import torch.nn as nn
import numpy as np

from circulant_matrix_conv_v2 import ker_by_channel, kers_by_channels, full_conv

tests = 20
passed = 0
for test in range(tests):
    if test % 10 == 0:
        print(test)
    c_in = np.random.randint(3,10)
    c_dim = np.random.randint(10,20)
    c_out = np.random.randint(3,10)
    ker_size = np.random.randint(2,5)

    chs = torch.ones(c_in,c_dim,c_dim)
    kers_full = torch.ones(c_out,c_in,ker_size,ker_size)
    our_out = full_conv(chs.unsqueeze(0), kers_full)
    conv1 = nn.Conv2d(c_in,c_out,ker_size, bias=False)
    conv1.weight = nn.Parameter(kers_full)
    their_out = conv1(chs.unsqueeze(0))
    passed += (their_out.detach()-our_out).norm().item() == 0

print(f'Passed {passed} out of {tests} tests ran')

