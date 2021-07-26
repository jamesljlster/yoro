import torch
from yoro.backbones import darknet


if __name__ == '__main__':

    width = 416
    height = 416
    channels = 3
    src = torch.randn(2, channels, height, width)

    for model in [
            darknet.YOLOv4,
            darknet.YOLOv4_CSP,
            darknet.YOLOv4_Tiny,
            darknet.YOLOv3,
            darknet.YOLOv3_Tiny
    ]:

        print('=== Test for %s ===' % model.__name__)
        module = model(width=width, height=height, channels=channels)
        out = module(src)
        for ten in out:
            print(ten.size())
        print()

        print('=== Test TorchScript Compatibility for %s ===' % model.__name__)
        module = torch.jit.script(
            model(width=width, height=height, channels=channels))
        out = module(src)
        for ten in out:
            print(ten.size())
        print()
