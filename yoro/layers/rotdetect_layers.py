import torch
from torch.nn import Module, Parameter
from torch.nn import functional as F


def correlation_coefficient(predict, targets):

    vP = predict - torch.mean(predict)
    vT = targets - torch.mean(targets)
    corr = (torch.sum(vP * vT) /
            (torch.sqrt(torch.sum(vP ** 2)) * torch.sqrt(torch.sum(vT ** 2))))

    return corr


def get_degrees(targets, dtype, device):
    return torch.tensor([inst[0]['degree'] for inst in targets],
                        dtype=dtype, device=device)


class RotRegressor(Module):

    __constants__ = ['width', 'height', 'base', 'scale']

    def __init__(self, width: int, height: int, deg_min, deg_max):

        super(RotRegressor, self).__init__()

        # Save parameters
        self.width = width
        self.height = height

        self.base = (deg_max + deg_min) / 2
        self.scale = (deg_max - deg_min) / 2

    def forward(self, inputs):
        return (inputs * self.scale + self.base).squeeze(-1)

    @torch.jit.unused
    def loss(self, inputs, targets):

        device = inputs.device
        dtype = inputs.dtype

        # Predict
        predict = self.forward(inputs)

        # Build target
        targets = get_degrees(targets, dtype, device)

        # Find loss
        loss = F.mse_loss(
            inputs, (targets.unsqueeze(-1) - self.base) / self.scale)

        # Find correlation coefficient
        corr = correlation_coefficient(predict, targets)

        return (loss, 1), {'corr': (corr.item(), 1)}


class RotClassifier(Module):

    __constants__ = ['width', 'height']

    def __init__(self, width: int, height: int, deg_min, deg_max, deg_step=1):

        super(RotClassifier, self).__init__()

        # Save parameters
        self.width = width
        self.height = height

        self.degs = Parameter(torch.arange(
            start=deg_min, end=deg_max + deg_step, step=deg_step), requires_grad=False)

    def forward(self, inputs):
        return self.degs[torch.argmax(inputs, dim=1)].to(inputs.dtype)

    @torch.jit.unused
    def loss(self, inputs, targets):

        device = inputs.device
        dtype = inputs.dtype

        # Predict
        predict = self.forward(inputs)

        # Build target
        targets = get_degrees(targets, dtype, device)
        targets = torch.argmin(
            torch.abs(targets.unsqueeze(-1) - self.degs), dim=1)

        # Find loss
        loss = F.cross_entropy(inputs, targets)

        # Find correlation coefficient
        targets = self.degs[targets].to(dtype)
        corr = correlation_coefficient(predict, targets)

        return (loss, 1), {'corr': (corr.item(), 1)}


class RotAnchor(Module):

    __constants__ = ['width', 'height']

    def __init__(self, width: int, height: int, deg_min, deg_max, deg_part_size):

        super(RotAnchor, self).__init__()

        # Save parameters
        self.width = width
        self.height = height

        self.degMin = deg_min
        self.degMax = deg_max
        self.degRange = deg_max - deg_min

        self.degPartSize = deg_part_size
        self.degValueScale = float(deg_part_size) / 2.0
        self.degAnchor = Parameter(
            torch.arange(
                start=deg_min, end=deg_max + deg_part_size, step=deg_part_size),
            requires_grad=False)

        self.degPartDepth = len(self.degAnchor)
        self.degValueDepth = len(self.degAnchor)

    def forward(self, inputs):
        idx = torch.argmax(inputs[:, :self.degPartDepth], dim=1)
        return (self.degAnchor[idx] +
                torch.gather(
                    inputs[:, self.degPartDepth:], 1, idx.unsqueeze(-1)).squeeze(-1) *
                self.degValueScale)

    @ torch.jit.unused
    def loss(self, inputs, targets):

        device = inputs.device
        dtype = inputs.dtype

        # Predict
        predict = self.forward(inputs)

        # Build target
        targets = get_degrees(targets, dtype, device)

        degDiff = targets.unsqueeze(-1) - self.degAnchor
        degPartIdx = torch.argmin(torch.abs(degDiff), dim=1).unsqueeze(-1)
        degShiftValue = torch.gather(
            degDiff, 1, degPartIdx) / self.degValueScale

        # Find loss
        degPartLoss = F.cross_entropy(
            inputs[:, :self.degPartDepth], degPartIdx.squeeze(-1))

        degShiftLoss = F.mse_loss(torch.gather(
            inputs[:, self.degPartDepth:], 1, degPartIdx), degShiftValue)
        loss = degPartLoss + degShiftLoss

        # Find correlation coefficient
        corr = correlation_coefficient(predict, targets)

        return (loss, 1), {'corr': (corr.item(), 1)}
