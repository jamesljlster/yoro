import torch
from torch.nn import Module, LeakyReLU, Identity
from torch.nn import functional as F


# Alias
leaky = LeakyReLU
linear = Identity


# Implementations
class mish(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.softplus(x).tanh()
