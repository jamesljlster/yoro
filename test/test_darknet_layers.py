import torch

from yoro.backbone.darknet import layers

if __name__ == '__main__':

    src = torch.randn(1, 3, 224, 224)

    module = torch.jit.script(layers.CONVOLUTIONAL(
        input_size=src.size()
    ))
    print(module(src).size())

    module = torch.jit.script(layers.SHORTCUT())
    print(module([src, src]).size())

    module = torch.jit.script(layers.ROUTE())
    print(module([src, src]).size())

    module = torch.jit.script(layers.UPSAMPLE(stride=2))
    print(module(src).size())
