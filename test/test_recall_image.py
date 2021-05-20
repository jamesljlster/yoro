import sys

import torch
import cv2 as cv
import numpy as np

from yoro.visual import rbox_draw
# from yoro.api import non_maximum_suppression as nms

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: %s <model_path> <image_path>' % sys.argv[0])
        exit()

    modelPath = sys.argv[1]
    imgPath = sys.argv[2]

    # Load model and image
    model = torch.jit.load(modelPath)
    netWidth = model.suffix.width
    netHeight = model.suffix.height
    model.eval()

    img = cv.imread(imgPath, cv.IMREAD_COLOR)

    # Padding image to square
    height, width, channel = img.shape
    tarSize = max(height, width)

    startX = int((tarSize - width) / 2)
    startY = int((tarSize - height) / 2)

    mat = np.zeros((tarSize, tarSize, channel), dtype=img.dtype)
    mat[startY:startY+height, startX:startX+width, :] = img
    mat = cv.resize(mat, (netWidth, netHeight))
    mat = cv.cvtColor(mat, cv.COLOR_BGR2RGB)
    mat = mat.transpose(2, 0, 1)

    # Predict
    inputs = (torch.FloatTensor(mat) / 255).unsqueeze(0)
    outputs = model(inputs)

    # Convert results
    results = []
    confTh = 0.7

    shift = torch.tensor(
        [[[startX, startY]]], dtype=inputs.dtype, device=inputs.device)
    for i, (conf, label, boxes, degree) in enumerate(outputs):

        # Denormalize
        boxes.mul_(tarSize / netWidth)
        boxes[..., 0:2].sub_(shift)

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

    for inst in results:
        print(inst)

    # Draw result
    result = rbox_draw([img], [results])
    cv.imshow('result', result[0])
    cv.waitKey(0)

    """
    # Apply non-maximum suppression
    confTh = 0.9
    nmsTh = 0.7
    nmsOut = nms(outputs, confTh, nmsTh)

    # Draw result
    print(nmsOut)
    result = rbox_draw([img], nmsOut)
    cv.imshow('result', result[0])
    cv.waitKey(0)
    """
