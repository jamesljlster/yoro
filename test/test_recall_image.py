import sys

import torch
import cv2 as cv
import numpy as np

from yoro.visual import rbox_draw
from yoro.api import non_maximum_suppression as nms

netWidth = 224
netHeight = 224

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: %s <model_path> <image_path>' % sys.argv[0])
        exit()

    modelPath = sys.argv[1]
    imgPath = sys.argv[2]

    # Load model and image
    model = torch.jit.load(modelPath)
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
    (pred_conf, pred_class, pred_class_conf, pred_boxes, pred_deg) = outputs

    # Denormalize
    shift = torch.tensor([[[startX, startY]]],
                         dtype=inputs.dtype, device=inputs.device)
    pred_boxes.mul_(tarSize / netWidth)
    pred_boxes[..., 0:2].sub_(shift)

    # Apply non-maximum suppression
    confTh = 0.9
    nmsTh = 0.7
    nmsOut = nms(outputs, confTh, nmsTh)

    # Draw result
    print(nmsOut)
    result = rbox_draw([img], nmsOut)
    cv.imshow('result', result[0])
    cv.waitKey(0)
