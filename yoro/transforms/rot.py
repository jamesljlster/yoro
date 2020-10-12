from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor, RandomAffine, ColorJitter

from .rbox import RBox_ColorJitter, RBox_ToTensor, pad_to_aspect_param


class Rot_PadToAspect(object):

    def __init__(self, aspectRatio, fill=0, padding_mode='constant'):
        self.aspectRatio = aspectRatio
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample):

        image, anno = sample
        image = F.pad(image,
                      pad_to_aspect_param(image.size, self.aspectRatio),
                      fill=self.fill, padding_mode=self.padding_mode)

        return (image, anno.copy())


class Rot_ColorJitter(RBox_ColorJitter):
    pass


class Rot_RandomAffine(object):

    def __init__(self, translate=None, scale=None, resample=False, fillcolor=0):

        self.randAffine = RandomAffine(
            0, translate=translate, scale=scale, shear=None,
            resample=resample, fillcolor=fillcolor)

    def __call__(self, sample):

        image, anno = sample
        image = self.randAffine(image)

        return (image, anno.copy())


class Rot_Resize(object):

    def __init__(self, size, interpolation=2):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):

        image, anno = sample
        image = F.resize(image, self.size, self.interpolation)

        return (image, anno.copy())


class Rot_ToTensor(RBox_ToTensor):
    pass
