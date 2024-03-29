import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Conv2d, BatchNorm2d, Upsample, ZeroPad2d, MaxPool2d

from typing import List

from . import activation as _activ


class ACTIVATION(Module):

    def __init__(self, input_size=None, activation='linear'):
        super().__init__()
        self.module = _activ.__dict__[activation]()

    def forward(self, x: List[Tensor]) -> Tensor:
        assert len(x) == 1
        return self.module(x[0])


class BATCHNORM(Module):

    def __init__(self, input_size):
        super().__init__()
        assert len(input_size) == 1
        self.module = BatchNorm2d(num_features=input_size[0][1])

    def forward(self, x: List[Tensor]) -> Tensor:
        assert len(x) == 1
        return self.module(x[0])


class MAXPOOL(Module):

    def __init__(self, input_size=None, size=2, stride=2):
        super().__init__()
        assert len(input_size) == 1

        maxPool = MaxPool2d(size, stride=stride, padding=int((size - 1) / 2))
        if size == 2 and stride == 1:
            self.module = Sequential(ZeroPad2d((0, 1, 0, 1)), maxPool)
        else:
            self.module = maxPool

    def forward(self, x: List[Tensor]) -> Tensor:
        assert len(x) == 1
        return self.module(x[0])


class CONVOLUTIONAL(Module):

    def __init__(self, input_size,
                 filters=1,
                 size=1,
                 stride=1,
                 pad=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 batch_normalize=0,
                 activation='leaky'
                 ):
        super().__init__()

        assert len(input_size) == 1
        in_channels = input_size[0][1]

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=size,
            stride=stride,
            padding=int(size / 2) if pad else padding,
            dilation=dilation,
            groups=groups,
            bias=(not batch_normalize)
        )

        if batch_normalize:
            self.bn = BatchNorm2d(num_features=filters)
        else:
            self.bn = None

        self.activ = _activ.__dict__[activation]()

    def forward(self, x: List[Tensor]) -> Tensor:
        assert len(x) == 1

        x = self.conv(x[0])
        if self.bn is not None:
            x = self.bn(x)
        x = self.activ(x)

        return x


class SHORTCUT(Module):

    def __init__(self, input_size=None, activation='linear'):
        super().__init__()
        self.activ = ACTIVATION(activation=activation)

    def forward(self, x: List[Tensor]) -> Tensor:
        ret = x[0]
        for i in range(1, len(x)):
            ret += x[i]

        return self.activ([ret])


class ROUTE(Module):

    def __init__(self, input_size, groups=1, group_id=0):
        super().__init__()

        channels = input_size[0][1]
        assert (channels % groups) == 0
        assert group_id < groups

        self.groupSize = int(channels / groups)
        self.groupID = group_id

    def forward(self, x: List[Tensor]) -> Tensor:
        x = torch.cat(x, dim=1)
        return torch.split(x, self.groupSize, dim=1)[self.groupID]


class UPSAMPLE(Module):

    def __init__(self, input_size=None, stride=1):
        super().__init__()
        self.module = Upsample(scale_factor=stride)

    def forward(self, x: List[Tensor]) -> Tensor:
        assert len(x) == 1
        return self.module(x[0])
