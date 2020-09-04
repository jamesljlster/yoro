from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import Compose

from yoro.transforms import *
from yoro.datasets import RBoxSample, rbox_collate_fn
from yoro.visual import rbox_draw


def imshow(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':

    # Coating rbox image
    data = RBoxSample('~/dataset/coating_test',
                      '~/dataset/coating_test/coating.names')
    print('numClasses:', data.numClasses)
    print('classNames:', data.classNames)
    dataLoader = DataLoader(data, batch_size=12, collate_fn=rbox_collate_fn)
    for images, annos in dataLoader:
        images = make_grid(rbox_draw(images, annos, to_tensor=True), nrow=4)
        imshow(images.numpy().transpose(1, 2, 0))

    # Coating rbox tensor
    data = RBoxSample('~/dataset/coating_test',
                      '~/dataset/coating_test/coating.names',
                      transform=RBox_ToTensor())
    dataLoader = DataLoader(data, batch_size=12, collate_fn=rbox_collate_fn)
    for images, annos in dataLoader:
        images = make_grid(rbox_draw(images, annos, to_tensor=True), nrow=4)
        imshow(images.numpy().transpose(1, 2, 0))

    # Coating rbox with transforms
    transform = Compose([
        RBox_PadToSquare(),
        RBox_ColorJitter(0.3, 0.3, 0.3, 0.1),
        RBox_RandomAffine(30, [0.3, 0.3], [0.7, 1.3]),
        RBox_ToTensor()
    ])

    data = RBoxSample('~/dataset/coating_test',
                      '~/dataset/coating_test/coating.names',
                      transform=transform)
    dataLoader = DataLoader(data, batch_size=12, collate_fn=rbox_collate_fn)
    for images, annos in dataLoader:
        images = make_grid(rbox_draw(images, annos, to_tensor=True), nrow=4)
        imshow(images.numpy().transpose(1, 2, 0))

    # PlateShelf rotation dataset
    transform = Compose([
        Rot_ColorJitter(0.3, 0.3, 0.3, 0.1),
        Rot_RandomAffine([0.3, 0.3], [0.7, 1.3]),
        Rot_ToTensor()
    ])

    data = RBoxSample('~/dataset/PlateShelf_Mark_Test',
                      '~/dataset/PlateShelf_Mark_Test/data.names',
                      transform=transform)
    dataLoader = DataLoader(data, batch_size=12, collate_fn=rbox_collate_fn)
    for images, annos in dataLoader:
        images = make_grid(images, nrow=4)
        imshow(images.numpy().transpose(1, 2, 0))
