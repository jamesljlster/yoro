import torch
import cv2 as cv
import numpy as np

from yoro.visual import rbox_draw

netWidth = 224
netHeight = 224

if __name__ == '__main__':

    # Load model and image
    model = torch.jit.load('coating_epoch_30000.zip')
    img = cv.imread(
        './test/test_image/CamToolbox_20200121_154634.jpg', cv.IMREAD_COLOR)

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
    (pred_conf, pred_class, pred_boxes, pred_deg) = model(inputs)

    predSize = pred_conf.size()
    batch = predSize[0]
    boxes = predSize[1]

    scale = tarSize / netWidth
    labels = []
    for n in range(batch):
        anno = []
        for i in range(boxes):
            if pred_conf[n, i] >= 0.9:
                anno.append({
                    'label': pred_class[n, i].item(),
                    'degree': pred_deg[n, i].item(),
                    'x': pred_boxes[n, i, 0].item() * scale - startX,
                    'y': pred_boxes[n, i, 1].item() * scale - startY,
                    'w': pred_boxes[n, i, 2].item() * scale,
                    'h': pred_boxes[n, i, 3].item() * scale
                })

        print(anno)
        labels.append(anno)

    result = rbox_draw([img], labels)
    cv.imshow('result', result[0])
    cv.waitKey(0)
