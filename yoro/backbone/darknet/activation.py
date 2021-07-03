import torch
from torch.nn import Module, LeakyReLU
from torch.nn import functional as F


# Alias
leaky = LeakyReLU


# Implementations
class linear(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class mish(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.softplus(x).tanh()
