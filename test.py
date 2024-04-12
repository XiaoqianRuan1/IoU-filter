import torch.nn as nn
import torch
import torch.nn.functional as F

input = torch.randn(2, 3)
output = F.softmax(input, -1)
for o in output:
    print(o.reshape(-1))