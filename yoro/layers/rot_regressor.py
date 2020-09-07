import torch
from torch.nn import Module, SmoothL1Loss
from torch.nn import functional as F


class RotRegressor(Module):

    def __init__(self, degMin, degMax):

        super(RotRegressor, self).__init__()

        # Save parameters
        self.base = (degMax + degMin) / 2
        self.scale = (degMax - degMin) / 2

        # Loss layer
        self.regLoss = SmoothL1Loss()

    def forward(self, inputs):
        return inputs * self.scale + self.base

    @torch.jit.unused
    def loss(self, inputs, targets):

        # Predict
        predict = self.forward(inputs)

        # Build target
        targets = torch.FloatTensor(
            [inst[0]['degree'] for inst in targets]).unsqueeze(-1)

        # Find loss
        loss = self.regLoss(inputs, (targets - self.base) / self.scale)

        # Find correlation coefficient
        vP = predict - torch.mean(predict)
        vT = targets - torch.mean(targets)
        corr = (torch.sum(vP * vT) /
                (torch.sqrt(torch.sum(vP ** 2)) * torch.sqrt(torch.sum(vT ** 2))))

        return loss, {'corr': corr.item()}


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from yoro.datasets import RBoxSample, rbox_collate_fn
    from yoro.transforms import Rot_ToTensor
    from torchvision.models import resnet18

    rotReg = RotRegressor(degMin=-45, degMax=45)

    data = RBoxSample('~/dataset/PlateShelf_Mark_Test',
                      '~/dataset/PlateShelf_Mark_Test/data.names',
                      transform=Rot_ToTensor())
    dataLoader = DataLoader(data, batch_size=12, collate_fn=rbox_collate_fn)
    dataIter = iter(dataLoader)
    inputs, targets = dataIter.next()

    net = resnet18(num_classes=1)
    out = net(inputs)
    loss, info = rotReg.loss(out, targets)
    print(loss, info)
