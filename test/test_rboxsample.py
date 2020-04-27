from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from yoro.transforms import RBox_ToTensor
from yoro.datasets import RBoxSample, rbox_collate_fn
from yoro.visual import rbox_draw


def imshow(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':

    # data = RBoxSample('~/dataset/coating_test',
    #                   '~/dataset/coating_test/coating.names')
    data = RBoxSample('~/dataset/coating_test',
                      '~/dataset/coating_test/coating.names',
                      transform=RBox_ToTensor())
    print('numClasses:', data.numClasses)
    print('classNames:', data.classNames)
    dataLoader = DataLoader(data, batch_size=2, collate_fn=rbox_collate_fn)
    for images, annos in dataLoader:
        images = rbox_draw(images, annos)
        for image in images:
            imshow(image)
        exit()
