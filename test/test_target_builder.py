import numpy as np
import cv2 as cv

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from yoro.datasets import RBoxSample
from yoro.layers import YOROLayer
from yoro.transforms.rbox import TargetBuilder, RBox_RandomAffine, RBox_PadToAspect, RBox_Resize
from yoro.visual import rbox_draw

width = 416
height = 416
num_classes = 2

if __name__ == '__main__':

    inputs = [torch.randn(2, 256, 26, 26),
              torch.randn(2, 512, 13, 13)]
    input_shapes = [ten.size() for ten in inputs]

    anchor = [[[21.39, 15.10], [55.22, 55.26], [64.18, 45.31]],
              [[78.89, 78.94], [91.68, 64.73]]]

    yoro = YOROLayer(width, height, num_classes, input_shapes, anchor,
                     deg_min=-90, deg_max=90, deg_part_size=10)
    predTemp = yoro.predict(inputs)

    objDims = [tup[0].size()[-3:] for tup in yoro(inputs)]
    gridSize = [torch.tensor([w, h])
                for (w, h) in zip(yoro.gridWidth, yoro.gridHeight)]

    tgtBuilder = TargetBuilder(
        yoro.anchorList, objDims, gridSize, yoro.degAnchor[0].clone(), yoro.degValueScale)

    # Dataset
    transform = Compose([
        RBox_PadToAspect((width / height)),
        RBox_Resize((height, width)),
        RBox_RandomAffine(30, None, [0.5, 1.5])
    ])

    data = RBoxSample('~/dataset/coating/valid',
                      '~/dataset/coating/coating.names',
                      transform=transform)
    for image, anno in data:

        objMask, target = tgtBuilder(anno)

        preds = list(tuple(torch.zeros_like(ten) for ten in pred)
                     for pred in predTemp)

        for headIdx, (acrIdxT, xIdxT, yIdxT, clsT, xT, yT, wT, hT, degPartT, degShiftT) \
                in enumerate(target):

            (obj, cls, x, y, w, h, degPart, degShift) = preds[headIdx]
            obj[0, acrIdxT, yIdxT, xIdxT] = 1.0
            cls[0, acrIdxT, yIdxT, xIdxT, clsT] = 1.0
            x[0, acrIdxT, yIdxT, xIdxT] = xT
            y[0, acrIdxT, yIdxT, xIdxT] = yT
            w[0, acrIdxT, yIdxT, xIdxT] = wT
            h[0, acrIdxT, yIdxT, xIdxT] = hT
            degPart[0, acrIdxT, yIdxT, xIdxT, degPartT] = 1.0
            degShift[0, acrIdxT, yIdxT, xIdxT, degPartT] = degShiftT

            print('Head', headIdx)
            print('  acrIdxT:', acrIdxT)
            print('  xIdxT:', xIdxT)
            print('  yIdxT:', yIdxT)
            print('  wT:', wT)
            print('  hT:', hT)
            print('  degPartT:', degPartT)
            print('  degShiftT:', degShiftT)
            print()
        print()

        outputs = yoro.decode(preds)

        results = []
        confTh = 0.7

        for i, (conf, label, boxes, degree) in enumerate(outputs):

            # Convert result
            results += [
                {
                    'label': label[n, a, h, w].item(),
                    'x': boxes[n, a, h, w, 0].item(),
                    'y': boxes[n, a, h, w, 1].item(),
                    'w': boxes[n, a, h, w, 2].item(),
                    'h': boxes[n, a, h, w, 3].item(),
                    'degree': degree[n, a, h, w].item()
                }
                for n in range(conf.size(0))
                for a in range(conf.size(1))
                for h in range(conf.size(2))
                for w in range(conf.size(3))
                if conf[n, a, h, w] >= confTh
            ]

        print('GT:', anno)
        print('DT:', results)
        print()

        # Draw result
        result = rbox_draw([np.array(image)[..., ::-1]], [results])
        cv.imshow('result', result[0])
        if cv.waitKey(0) == 27:
            break
