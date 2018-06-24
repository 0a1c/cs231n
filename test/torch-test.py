from __future__ import print_function
import torch

x = torch.ones(5, 3, dtype=torch.float)
x = torch.randn_like(x)
print(x)
print(x.size()[0])
