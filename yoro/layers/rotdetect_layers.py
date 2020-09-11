import torch
from torch.nn import Module, Parameter
from torch.nn import functional as F


def correlation_coefficient(predict, targets):

    vP = predict - torch.mean(predict)
    vT = targets - torch.mean(targets)
    corr = (torch.sum(vP * vT) /
            (torch.sqrt(torch.sum(vP ** 2)) * torch.sqrt(torch.sum(vT ** 2))))

    return corr


class RotRegressor(Module):

    __constants__ = ['base', 'scale']

    def __init__(self, deg_min, deg_max):

        super(RotRegressor, self).__init__()

        # Save parameters
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
        targets = torch.tensor(
            [inst[0]['degree'] for inst in targets],
            dtype=dtype, device=device)

        # Find loss
        loss = F.mse_loss(
            inputs, (targets.unsqueeze(-1) - self.base) / self.scale)

        # Find correlation coefficient
        corr = correlation_coefficient(predict, targets)

        return (loss, 1), {'corr': (corr.item(), 1)}


class RotClassifier(Module):

    def __init__(self, deg_min, deg_max, deg_step=1):

        super(RotClassifier, self).__init__()

        # Save parameters
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
        targets = torch.tensor(
            [inst[0]['degree'] for inst in targets],
            dtype=dtype, device=device).unsqueeze(-1)
        targets = torch.argmin(torch.abs(targets - self.degs), dim=1)

        # Find loss
        loss = F.cross_entropy(inputs, targets)

        # Find correlation coefficient
        targets = self.degs[targets].to(dtype)
        corr = correlation_coefficient(predict, targets)

        return (loss, 1), {'corr': (corr.item(), 1)}


class RotAnchor(Module):

    def __init__(self, deg_min, deg_max, deg_part_size):

        super(RotAnchor, self).__init__()

        # Save parameters
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
        targets = torch.tensor(
            [inst[0]['degree'] for inst in targets],
            dtype=dtype, device=device)

        degDiff = targets.unsqueeze(-1) - self.degAnchor
        degPartIdx = torch.argmin(torch.abs(degDiff), dim=1).unsqueeze(-1)
        degShiftValue = torch.gather(degDiff, 1, degPartIdx)

        # Find loss
        degPartLoss = F.cross_entropy(
            inputs[:, :self.degPartDepth], degPartIdx.squeeze(-1))

        degShiftLoss = F.mse_loss(torch.gather(
            inputs[:, self.degPartDepth:], 1, degPartIdx), degShiftValue)
        loss = degPartLoss + degShiftLoss

        # Find correlation coefficient
        corr = correlation_coefficient(predict, targets)

        return (loss, 1), {'corr': (corr.item(), 1)}


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from yoro.datasets import RBoxSample, rbox_collate_fn
    from yoro.transforms import Rot_ToTensor
    from torchvision.models import resnet18

    data = RBoxSample('~/dataset/PlateShelf_Mark_Test',
                      '~/dataset/PlateShelf_Mark_Test/data.names',
                      transform=Rot_ToTensor())
    dataLoader = DataLoader(data, batch_size=8, collate_fn=rbox_collate_fn)
    dataIter = iter(dataLoader)
    inputs, targets = dataIter.next()

    # Test rotation regressor
    rotDetect = RotRegressor(deg_min=-45, deg_max=45)
    net = resnet18(num_classes=1)
    out = net(inputs)
    print(rotDetect(out))
    loss, info = rotDetect.loss(out, targets)
    print(loss, info)
    print()

    # Test rotation classifier
    rotDetect = RotClassifier(deg_min=-45, deg_max=45, deg_step=1)
    net = resnet18(num_classes=91)
    out = net(inputs)
    print(rotDetect(out))
    loss, info = rotDetect.loss(out, targets)
    print(loss, info)
    print()

    # Test rotation anchor
    rotDetect = RotAnchor(deg_min=-45, deg_max=45, deg_part_size=10)
    net = resnet18(num_classes=20)
    out = net(inputs)
    print(rotDetect(out))
    loss, info = rotDetect.loss(out, targets)
    print(loss, info)
    print()
