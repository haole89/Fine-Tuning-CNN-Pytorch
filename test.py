import torch

t = torch.rand([4, 3, 244, 224])

print(t.size(dim=0))

x = t.view(t.size(0), -1)

print(x.size())