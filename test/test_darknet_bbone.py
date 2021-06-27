import torch
from yoro.backbone.darknet import YOLOv3


if __name__ == '__main__':

    width = 416
    height = 416
    channels = 3
    src = torch.randn(2, channels, height, width)

    module = torch.jit.script(
        YOLOv3(width=width, height=height, channels=channels))
    out = module(src)
    for ten in out:
        print(ten.size())
    print()

    module = YOLOv3(width=width, height=height, channels=channels)
    out = module(src)
    for ten in out:
        print(ten.size())
    print()
