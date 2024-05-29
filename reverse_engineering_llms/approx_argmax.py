import torch

l = torch.tensor([0.1, 0.2, 0.15, 0.05])

print("real argmax:", torch.argmax(l))

p = torch.softmax(l, dim=-1)
print("softmax t=1:", p)
print("approximated argmax:", torch.sum(p * torch.arange(4).float()))

p = torch.softmax(l/0.1, dim=-1)
print("softmax t=0.1:", p)
print("approximated argmax:", torch.sum(p * torch.arange(4).float()))

p = torch.softmax(l/0.01, dim=-1)
print("softmax t=0.01:", p)
print("approximated argmax:", torch.sum(p * torch.arange(4).float()))

p = torch.softmax(l/0.001, dim=-1)
print("softmax t=0.001:", p)
print("approximated argmax:", torch.sum(p * torch.arange(4).float()))
