# Neural Network Reconstruction

This project reconstructs the correct ordering of a scrambled residual neural network using optimization techniques including Hungarian matching, beam search, and simulated annealing.

---

## Problem

A trained neural network was broken into individual linear layers.  
The goal is to recover the correct ordering of these layers using only:

- layer weights  
- historical input/output data  
- knowledge of the residual block structure  

The residual block architecture is:

```python
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        residual = x
        x = self.inp(x)
        x = self.activation(x)
        x = self.out(x)
        return residual + x


class LastLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.layer(x)
