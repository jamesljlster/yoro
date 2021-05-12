from torchvision.transforms import functional as F
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms import ToTensor, RandomAffine, ColorJitter
from PIL import Image, __version__ as PILLOW_VERSION
import numpy as np
import torch
import math

from typing import List, Dict, Tuple


def rbox_affine(sample, degree, translate, scale,
                interpolation=F.InterpolationMode.NEAREST, fill=0):

    image, anno = sample
    width, height = image.size

    # Find center point for transformation
    ctrX = float(width) / 2.0
    ctrY = float(height) / 2.0

    # Apply image transformation
    matrix = F._get_inverse_affine_matrix(
        [ctrX, ctrY], -degree, translate, scale, [0, 0])

    pil_interpolation = F.pil_modes_mapping[interpolation]
    image = F_pil.affine(
        image, matrix=matrix, interpolation=pil_interpolation, fill=fill)

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

    def __init__(self, degrees, translate=None, scale=None,
                 interpolation=F.InterpolationMode.NEAREST, fill=0):

        self.randAffine = RandomAffine(
            degrees, translate=translate, scale=scale, shear=None,
            interpolation=interpolation, fill=fill)

    def __call__(self, sample):

        degree, translate, scale, _ = self.randAffine.get_params(
            degrees=self.randAffine.degrees,
            translate=self.randAffine.translate,
            scale_ranges=self.randAffine.scale,
            shears=self.randAffine.shear,
            img_size=sample[0].size)

        return rbox_affine(
            sample, degree, translate, scale,
            interpolation=self.randAffine.interpolation,
            fill=self.randAffine.fill)


class RBox_Resize(object):

    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
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


def pad_to_aspect_param(imgSize, aspectRatio):

    width, height = imgSize
    imSize = np.array([width, height])
    cand1 = np.array([width, int(round(width / aspectRatio))])
    cand2 = np.array([int(round(height * aspectRatio)), height])
    tarSize = cand1 if ((cand1 - imSize) < 0).sum() == 0 else cand2

    wPad = float(tarSize[0] - width) / 2.0
    hPad = float(tarSize[1] - height) / 2.0

    lPad = int(np.floor(wPad))
    rPad = int(np.ceil(wPad))
    tPad = int(np.floor(hPad))
    bPad = int(np.ceil(hPad))

    return (lPad, tPad, rPad, bPad)


class RBox_PadToAspect(object):

    def __init__(self, aspectRatio, fill=0, padding_mode='constant'):
        self.aspectRatio = aspectRatio
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample):

        image, anno = sample
        (lPad, tPad, rPad, bPad) = pad_to_aspect_param(
            image.size, self.aspectRatio)

        newAnno = []
        for i in range(len(anno)):

            tmpAnno = anno[i].copy()

            tmpAnno['x'] += lPad
            tmpAnno['y'] += tPad

            newAnno.append(tmpAnno)

        image = F.pad(image, (lPad, tPad, rPad, bPad),
                      fill=self.fill, padding_mode=self.padding_mode)

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


class TargetBuilder(object):

    def __init__(self, anchor_list, obj_dims, grid_size, deg_anchor, deg_scale):

        self.anchorList = anchor_list
        self.objDims = obj_dims
        self.gridSize = grid_size
        self.degAnchor = deg_anchor
        self.degScale = deg_scale

    def __call__(self, anno_list):

        # Build objectness mask
        objs = [torch.zeros(size, dtype=torch.bool) for size in self.objDims]

        # Target storage
        def _build_target_storage():
            return [[] for _ in range(len(self.objDims))]

        acrIdxT = _build_target_storage()
        xIdxT = _build_target_storage()
        yIdxT = _build_target_storage()
        clsT = _build_target_storage()

        xT = _build_target_storage()
        yT = _build_target_storage()
        wT = _build_target_storage()
        hT = _build_target_storage()

        degPartT = _build_target_storage()
        degShiftT = _build_target_storage()

        # Tensorlize
        boxesSize = torch.tensor(
            [[anno['w'], anno['h']] for anno in anno_list],
            dtype=torch.float)

        boxesCoord = torch.tensor(
            [[anno['x'], anno['y']] for anno in anno_list],
            dtype=torch.float)

        labels = torch.tensor([anno['label'] for anno in anno_list])
        degrees = torch.tensor([anno['degree'] for anno in anno_list])

        # Find anchor score for boxes
        ret = self.anchor_score(boxesSize)
        if ret is not None:
            headInd, acrInd, acrScores = ret

            # Anchor matching
            while True:

                maxScore = torch.max(acrScores)
                if maxScore < 0:
                    break

                maxInd = torch.where(acrScores == maxScore)
                rowInd = maxInd[0][0]
                colInd = maxInd[1][0]

                headIdx = headInd[colInd]
                acrIdx = acrInd[colInd]

                normCoord = boxesCoord[rowInd] / self.gridSize[headIdx]
                xIdx, yIdx = torch.floor(normCoord).to(torch.long)

                if objs[headIdx][acrIdx, yIdx, xIdx]:
                    acrScores[rowInd, colInd] = -1
                else:
                    objs[headIdx][acrIdx, yIdx, xIdx] = True
                    acrScores[rowInd, :] = -1

                    # Target encoding
                    acrIdxT[headIdx].append(acrIdx)
                    xIdxT[headIdx].append(xIdx)
                    yIdxT[headIdx].append(yIdx)
                    clsT[headIdx].append(labels[rowInd])

                    xT[headIdx].append((normCoord[0] - xIdx + 0.5) / 2.0)
                    yT[headIdx].append((normCoord[1] - yIdx + 0.5) / 2.0)

                    normSize = torch.sqrt(
                        boxesSize[rowInd] / self.anchorList[headIdx][acrIdx]) / 2.0
                    wT[headIdx].append(normSize[0])
                    hT[headIdx].append(normSize[1])

                    degDiff = degrees[rowInd] - self.degAnchor
                    degPartIdx = torch.argmin(torch.abs(degDiff))
                    degPartT[headIdx].append(degPartIdx)
                    degShiftT[headIdx].append(
                        degDiff[degPartIdx] / self.degScale)

        targets = [
            [torch.tensor(elem, dtype=dtype) for (elem, dtype) in
                zip(tup, [torch.long, torch.long, torch.long, torch.long,
                          torch.float, torch.float, torch.float, torch.float,
                          torch.long, torch.float
                          ])]
            for tup in zip(acrIdxT, xIdxT, yIdxT, clsT,
                           xT, yT, wT, hT,
                           degPartT, degShiftT)]

        return objs, targets

    def anchor_score(self, box):

        if box.size(0) == 0:
            return None

        box = box.view(box.size(0), -1, 2)

        headIndices = torch.tensor([], dtype=torch.long)
        anchorIndices = torch.tensor([], dtype=torch.long)
        for i, anchor in enumerate(self.anchorList):

            anchorSize = anchor.size(0)
            headIndices = torch.cat(
                [headIndices, torch.full((anchorSize,), i)])
            anchorIndices = torch.cat(
                [anchorIndices, torch.arange(anchorSize)])

        anchor = torch.cat(
            [anchor for anchor in self.anchorList]).unsqueeze(0)

        bw, bh = box[..., 0], box[..., 1]
        aw, ah = anchor[..., 0], anchor[..., 1]

        interArea = torch.min(bw, aw) * torch.min(bh, ah)
        unionArea = bw * bh + aw * ah - interArea

        return headIndices, anchorIndices, (interArea / (unionArea + 1e-8))


class RBox_ToTensor(object):

    def __init__(self, anchor_list, obj_dims, grid_size, deg_anchor, deg_scale):

        # Image transform
        self.toTensor = ToTensor()

        # Annotation transform
        self.tgtBuilder = TargetBuilder(
            anchor_list, obj_dims, grid_size, deg_anchor, deg_scale)

    def __call__(self, sample):

        image, anno = sample

        # Apply transformation
        image = self.toTensor(image)
        anno = self.tgtBuilder(anno)

        return (image, anno)
