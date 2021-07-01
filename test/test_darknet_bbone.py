import torch
from yoro.backbone import darknet


if __name__ == '__main__':

    width = 416
    height = 416
    channels = 3
    src = torch.randn(2, channels, height, width)

    for model in [darknet.YOLOv3_Tiny, darknet.YOLOv3]:

        module = model(width=width, height=height, channels=channels)
        out = module(src)
        for ten in out:
            print(ten.size())
        print()

        module = torch.jit.script(
            model(width=width, height=height, channels=channels))
        out = module(src)
        for ten in out:
            print(ten.size())
        print()
