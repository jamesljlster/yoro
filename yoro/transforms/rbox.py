from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor, RandomAffine, ColorJitter
from PIL import Image, __version__ as PILLOW_VERSION
import numpy as np
import torch
import math


def rbox_affine(sample, degree, translate, scale, resample=0, fillcolor=None):

    image, anno = sample
    width, height = image.size

    # Find center point for transformation
    ctrX = float(width) / 2.0
    ctrY = float(height) / 2.0

    # Apply image transformation
    matrix = F._get_inverse_affine_matrix(
        (ctrX, ctrY), -degree, translate, scale, 0)

    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] >= '5' else {}
    image = image.transform(image.size, Image.AFFINE,
                            matrix, resample, **kwargs)

    # Find annotation transformation matrix
    rad = math.radians(degree)
    tx, ty = translate
    s = scale
    cos = math.cos
    sin = math.sin

    matrix = np.float32([
        [s*cos(rad), s*sin(rad), -ctrX*s*cos(rad) -
         ctrY*s*sin(rad) + ctrX + tx],
        [-s*sin(rad), s*cos(rad), -ctrY*s*cos(rad) +
         ctrX*s*sin(rad) + ctrY + ty],
        [0, 0, 1]
    ])

    # Apply annotation transformation
    newAnno = []
    for i in range(len(anno)):

        tmpAnno = anno[i].copy()

        pt = np.float32([tmpAnno['x'], tmpAnno['y'], 1])
        pt = np.matmul(pt, matrix.transpose())

        x = pt[0].item()
        y = pt[1].item()
        if 0 <= x < width and 0 <= y < height:
            tmpAnno['x'] = x
            tmpAnno['y'] = y
            tmpAnno['w'] *= scale
            tmpAnno['h'] *= scale

            deg = tmpAnno['degree'] + degree - 180
            while deg < 0:
                deg += 360
            tmpAnno['degree'] = deg - 180

            newAnno.append(tmpAnno)

    return (image, newAnno)


class RBox_ColorJitter(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.colorJitter = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):

        image, anno = sample
        image = self.colorJitter(image)
        return (image, anno.copy())


class RBox_RandomAffine(object):

    def __init__(self, degrees, translate=None, scale=None, resample=False, fillcolor=0):

        self.randAffine = RandomAffine(
            degrees, translate=translate, scale=scale, shear=None,
            resample=resample, fillcolor=fillcolor)

    def __call__(self, sample):

        degree, translate, scale, _ = self.randAffine.get_params(
            self.randAffine.degrees, self.randAffine.translate, self.randAffine.scale,
            self.randAffine.shear, sample[0].size)

        return rbox_affine(
            sample, degree, translate, scale,
            resample=self.randAffine.resample, fillcolor=self.randAffine.fillcolor)


class RBox_Resize(object):

    def __init__(self, size, interpolation=2):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):

        image, anno = sample

        width, height = image.size
        newHeight, newWidth = self.size
        wScale = float(newWidth) / float(width)
        hScale = float(newHeight) / float(height)

        newAnno = []
        for i in range(len(anno)):

            tmpAnno = anno[i].copy()

            tmpAnno['x'] *= wScale
            tmpAnno['w'] *= wScale
            tmpAnno['y'] *= hScale
            tmpAnno['h'] *= hScale

            newAnno.append(tmpAnno)

        image = F.resize(image, self.size, self.interpolation)

        return (image, newAnno)


class RBox_PadToSquare(object):

    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample):

        image, anno = sample

        tarSize = np.max(image.size)
        width, height = image.size

        wPad = float(tarSize - width) / 2.0
        hPad = float(tarSize - height) / 2.0

        lPad = int(np.floor(wPad))
        rPad = int(np.ceil(wPad))
        tPad = int(np.floor(hPad))
        bPad = int(np.ceil(hPad))

        newAnno = []
        for i in range(len(anno)):

            tmpAnno = anno[i].copy()

            tmpAnno['x'] += lPad
            tmpAnno['y'] += tPad

            newAnno.append(tmpAnno)

        image = F.pad(image, (lPad, tPad, rPad, bPad),
                      fill=self.fill, padding_mode=self.padding_mode)

        return (image, newAnno)


class RBox_ToTensor(object):

    def __init__(self):

        # Image transform
        self.toTensor = ToTensor()

    def __call__(self, sample):

        image, anno = sample

        # Apply image transform
        image = self.toTensor(image)

        return (image, anno)
