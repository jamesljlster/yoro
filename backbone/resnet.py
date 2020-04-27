from torch.nn import Module
from torchvision.models import resnet


class ResNet(Module):
    def __init__(self, model_name, **kwargs):
        super(ResNet, self).__init__()
        self.base = resnet.__dict__[model_name](**kwargs)

    def forward(self, x):

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        return x
