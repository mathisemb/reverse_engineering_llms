import torch

"""
input (Tensor) he source tensor
dim (int) The axis along which to index
index (LongTensor) The indices of elements to gather

Index tensor must have the same number of dimensions as input tensor but the dimensions sizes can be different. 
"""

probabilities = torch.tensor([ [[0.1, 0.6, 0.3],
                                [0.1, 0.6, 0.3],
                                [0.1, 0.6, 0.3],
                                [0.1, 0.6, 0.3]],
                               
                               [[0.1, 0.6, 0.3],
                                [0.1, 0.6, 0.3],
                                [0.1, 0.6, 0.3],
                                [0.1, 0.6, 0.3]] ])

indices = torch.tensor([ [0, 1, 1, 2],
                         [1, 1, 2, 0] ])

indices = indices.unsqueeze(-1)

print(torch.gather(probabilities, -1, indices).squeeze(-1))
