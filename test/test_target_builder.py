import torch

from yoro.layers import YOROLayer
from yoro.transforms.rbox import TargetBuilder

width = 640
height = 480
num_classes = 2

if __name__ == '__main__':

    inputs = [torch.randn(2, 256, 40, 30),
              torch.randn(2, 512, 20, 15)]
    input_shapes = [ten.size() for ten in inputs]

    anchor = [[[21.39, 15.10], [55.22, 55.26], [64.18, 45.31]],
              [[78.89, 78.94], [91.68, 64.73]]]

    yoro = YOROLayer(width, height, num_classes, input_shapes, anchor,
                     deg_min=-90, deg_max=90, deg_part_size=10)

    objDims = [tup[0].size()[-3:] for tup in yoro(inputs)]
    gridSize = [torch.tensor([w, h])
                for (w, h) in zip(yoro.gridWidth, yoro.gridHeight)]

    tgtBuilder = TargetBuilder(
        yoro.anchorList, objDims, gridSize, yoro.degAnchor, yoro.degValueScale)

    gt = [
        {'label': 1,
         'degree': 51.542092828142785,
         'x': 239.01401711274869,
         'y': 180.60637177461822,
         'w': 139.38557740471791,
         'h': 99.401066096449185},
        {'label': 0,
         'degree': -37.663634282283738,
         'x': 446.21821037540292,
         'y': 290.40284360189571,
         'w': 121.75447942176599,
         'h': 121.65217738106354
         }
    ]

    objMask, target = tgtBuilder(gt)
    for headIdx, (acrIdxT, xIdxT, yIdxT, xT, yT, wT, hT, degPartT, degShiftT) \
            in enumerate(target):
        print('Head', headIdx)
        print('  acrIdxT:', acrIdxT)
        print('  xIdxT:', xIdxT)
        print('  yIdxT:', yIdxT)
        print('  wT:', wT)
        print('  hT:', hT)
        print('  degPartT:', degPartT)
        print('  degShiftT:', degShiftT)
        print()
