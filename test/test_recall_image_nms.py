import torch
import cv2 as cv
import numpy as np

from yoro.visual import rbox_draw
from yoro.api import non_maximum_suppression as nms

netWidth = 224
netHeight = 224

if __name__ == '__main__':

    # Load model and image
    model = torch.jit.load('coating_epoch_30000.zip')
    model.eval()

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
    outputs = model(inputs)
    (pred_conf, pred_class, pred_class_conf, pred_boxes, pred_deg) = outputs

    # Denormalize
    shift = torch.tensor([[[-startX, -startY, 0, 0]]],
                         dtype=inputs.dtype, device=inputs.device)
    pred_boxes.mul_(tarSize / netWidth)
    pred_boxes.add_(shift)

    # Future conversion for non-maximum suppression
    predSize = pred_conf.size()
    batch = predSize[0]
    boxes = predSize[1]

    predList = []
    for n in range(batch):
        pred = []
        for i in range(boxes):
            pred.append([
                pred_conf[n, i].item(),
                pred_class[n, i].item(),
                pred_class_conf[n, i].item(),
                pred_deg[n, i].item(),
                pred_boxes[n, i, 0].item(),
                pred_boxes[n, i, 1].item(),
                pred_boxes[n, i, 2].item(),
                pred_boxes[n, i, 3].item()
            ])

        predList.append(pred)

    # Convert degrees to radians
    def deg2rad(deg):
        return deg * 3.1415927410125732 / 180.0

    # Convert bounding box to corners
    def bbox_to_corners(bbox):
        corners = torch.zeros_like(bbox)
        corners[:, 0] = bbox[:, 0] - bbox[:, 2] / 2.0
        corners[:, 1] = bbox[:, 1] - bbox[:, 3] / 2.0
        corners[:, 2] = bbox[:, 0] + bbox[:, 2] / 2.0
        corners[:, 3] = bbox[:, 1] + bbox[:, 3] / 2.0
        return corners

    # Non-maximum suppression
    def rbox_similarity(pred1, pred2):

        # BBox to corners
        corners1 = bbox_to_corners(pred1[:, 4:8])
        corners2 = bbox_to_corners(pred2[:, 4:8])

        # Find IoU scores
        interX1 = torch.max(corners1[:, 0], corners2[:, 0])
        interY1 = torch.max(corners1[:, 1], corners2[:, 1])
        interX2 = torch.min(corners1[:, 2], corners2[:, 2])
        interY2 = torch.min(corners1[:, 3], corners2[:, 3])
        interArea = torch.clamp(interX2 - interX1, min=0) * \
            torch.clamp(interY2 - interY1, min=0)
        unionArea = pred1[:, 6] * pred[:, 7] + \
            pred2[:, 6] * pred[:, 7] - interArea
        ious = interArea / (unionArea + 1e-4)

        # Find degree similarity
        rad1 = deg2rad(pred1[:, 3])
        rad2 = deg2rad(pred2[:, 3])
        ang1 = torch.stack([torch.sin(rad1), torch.cos(rad1)], dim=1)
        ang2 = torch.stack([torch.sin(rad2), torch.cos(rad2)], dim=1)
        angSim = (torch.matmul(ang1, ang2.t()) + 1.0) / 2.0

        return ious * angSim

    confTh = 0.9
    nmsTh = 0.7

    batch = len(predList)
    nmsOut = [None] * batch
    for n in range(batch):
        pred = torch.FloatTensor(predList[n])
        pred = pred[pred[:, 0] >= confTh]

        inst = []
        while pred.size(0):

            confScore = pred[:, 0] * pred[:, 2]
            pred = pred[confScore.argsort(descending=True)]

            labelMatch = (pred[0, 1] == pred[:, 1])
            highSim = (rbox_similarity(pred[0, :].unsqueeze(0), pred) > nmsTh)

            rmIdx = (labelMatch & highSim).squeeze(0)
            weight = confScore[rmIdx].unsqueeze(0)

            tmpInst = torch.matmul(weight, pred[rmIdx, 3:8]) / weight.sum()
            tmpInst = tmpInst.squeeze(0).tolist()
            inst.append({
                'label': int(pred[rmIdx, 1][0].item()),
                'degree': tmpInst[0],
                'x': tmpInst[1],
                'y': tmpInst[2],
                'w': tmpInst[3],
                'h': tmpInst[4]
            })

            pred = pred[~rmIdx]

        nmsOut[n] = inst

    # Draw result
    print(nmsOut)
    print(nms(outputs, confTh, nmsTh))
    result = rbox_draw([img], nmsOut)
    cv.imshow('result', result[0])
    cv.waitKey(0)
