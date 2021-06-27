import torch
from torch.nn import Module
from torch.nn import LeakyReLU


class linear(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


leaky = LeakyReLU
