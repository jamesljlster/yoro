from torch import Tensor
from torch.nn import Module, AdaptiveAvgPool2d
from torchvision.models import resnet
from typing import List


class ResNet_FCN(Module):
    def __init__(self, model_name, **kwargs):
        super(ResNet_FCN, self).__init__()

        # Load ResNet model from torchvision
        self.base = resnet.__dict__[model_name](**kwargs)

        # Drop average pooling layer and fully connected layer
        del self.base.avgpool
        del self.base.fc

        # Reset base forward method (reserve feature blocks only)
        self.base.forward = super(ResNet_FCN, self).forward

    def forward(self, x: Tensor) -> List[Tensor]:

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        return [x]


class ResNet_Feature(Module):

    def __init__(self, model_name, pool_size=(1, 1), **kwargs):
        super(ResNet_Feature, self).__init__()
        self.resnet = ResNet_FCN(model_name, **kwargs)
        self.avgpool = AdaptiveAvgPool2d(pool_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet(x)[0]
        x = self.avgpool(x)
        return x
