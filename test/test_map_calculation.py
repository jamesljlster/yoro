import torch
import numpy as np
import cv2 as cv

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from yoro import api
from yoro.datasets import RBoxSample, rbox_collate_fn
from yoro.transforms import RBox_PadToAspect, RBox_ToTensor
from yoro.visual import rbox_draw

confTh = 0.7
confTh = 0.5
nmsTh = 0.9
simTh = 0.7


def bbox_to_corners(bbox):

    corners = torch.zeros_like(bbox)
    corners[..., 0] = bbox[..., 0] - bbox[..., 2] / 2.0
    corners[..., 1] = bbox[..., 1] - bbox[..., 3] / 2.0
    corners[..., 2] = bbox[..., 0] + bbox[..., 2] / 2.0
    corners[..., 3] = bbox[..., 1] + bbox[..., 3] / 2.0

    return corners


def get_bbox(pred):
    return torch.tensor([
        [inst.x, inst.y, inst.w, inst.h] for inst in pred])


def deg2rad(deg):
    return deg * 3.1415927410125732 / 180.0


def get_degree(pred):
    return torch.tensor([inst.degree for inst in pred])


def rbox_similarity(pred1, pred2):

    # Get bounding boxes
    bbox1 = get_bbox(pred1)
    bbox2 = get_bbox(pred2)

    # Bounding boxes to corners
    corners1 = bbox_to_corners(bbox1)
    corners2 = bbox_to_corners(bbox2)

    # Find IoU scores
    interX1 = torch.max(corners1[..., 0], corners2[..., 0])
    interY1 = torch.max(corners1[..., 1], corners2[..., 1])
    interX2 = torch.min(corners1[..., 2], corners2[..., 2])
    interY2 = torch.min(corners1[..., 3], corners2[..., 3])

    interArea = (torch.clamp(interX2 - interX1, 0) *
                 torch.clamp(interY2 - interY1, 0))
    unionArea = (bbox1[..., 2] * bbox1[..., 3] +
                 bbox2[..., 2] * bbox2[..., 3] -
                 interArea)
    ious = interArea / (unionArea + 1e-4)

    # Find degree similarity
    rad1 = deg2rad(get_degree(pred1))
    rad2 = deg2rad(get_degree(pred2))
    ang1 = torch.stack([torch.sin(rad1), torch.cos(rad1)], 1)
    ang2 = torch.stack([torch.sin(rad2), torch.cos(rad2)], 1)
    angSim = (torch.matmul(ang1, ang2.t()) + 1.0) / 2.0

    return ious * angSim


if __name__ == '__main__':

    """
    # Load dataset and detector
    detector = api.YORODetector('coating_epoch_30000.zip')
    data = RBoxSample('~/dataset/coating/valid',
                      '~/dataset/coating/coating.names')
    classNames = data.classNames

    # Save all prediction and ground truths
    evalData = []
    for image, anno in data:
        image = np.uint8(image)[..., ::-1]
        pred = detector.detect(image, confTh, nmsTh)
        evalData.append(([inst.to_dict() for inst in pred], anno))

    torch.save({
        'classNames': classNames,
        'evalData': evalData
    }, 'evalDict.pth')

    exit()
    """

    # Load data for evaluation
    evalDict = torch.load('evalDict.pth')
    classNames = evalDict['classNames']
    evalData = evalDict['evalData']

    predPair = {idx: [] for idx in range(len(classNames))}
    gts = {}

    gtCounter = 0
    for dataInd, (pred, gt) in enumerate(evalData):

        # Record ground truth
        gtTmp = {}
        for inst in gt:

            gtId = gtCounter
            gtCounter += 1

            gt = api.RBox()
            gt.conf = 1.0
            gt.label = inst['label']
            gt.degree = inst['degree']
            gt.x = inst['x']
            gt.y = inst['y']
            gt.w = inst['w']
            gt.h = inst['h']

            gtTmp[gtId] = gt

        gts.update(gtTmp)

        # Compare prediction with ground truths
        for inst in pred:

            dt = api.RBox()
            dt.conf = inst['conf']
            dt.label = inst['label']
            dt.degree = inst['degree']
            dt.x = inst['x']
            dt.y = inst['y']
            dt.w = inst['w']
            dt.h = inst['h']

            # Find similarities between prediction and ground truths
            rboxSim = rbox_similarity(
                [dt], [gtTmp[gtKey] for gtKey in gtTmp])
            highSim = (rboxSim >= simTh).squeeze(0)

            dtLabel = dt.label
            dtConf = dt.conf

            if highSim.sum() > 0:
                for gtKey, match in zip(gtTmp, highSim):
                    if match:
                        predPair[dtLabel].append({
                            'conf': dtConf,
                            'pred': dtLabel,
                            'label': gtTmp[gtKey].label,
                            'gtId': gtKey,
                            'dataInd': dataInd
                        })

            else:
                predPair[dtLabel].append({
                    'conf': dtConf,
                    'pred': dtLabel,
                    'label': -1,
                    'gtId': -1,
                    'dataInd': dataInd
                })

    for cId in predPair:

        results = predPair[cId]
        results = sorted(results, key=lambda inst: inst['conf'], reverse=True)

        # Find all ground truth keys with given class ID
        gtKeys = [key for key in gts if gts[key].label == cId]

        gtHit = []
        predHit = 0
        apTable = []
        for i, inst in enumerate(results):

            gtId = inst['gtId']
            if gtId >= 0:
                if gtId not in gtHit:
                    gtHit.append(inst['gtId'])
                if inst['pred'] == inst['label']:
                    predHit += 1

            recall = len(gtHit) / len(gtKeys)
            precision = predHit / (i + 1)

            apTable.append([
                inst['dataInd'],
                inst['conf'],
                inst['pred'],
                inst['label'],
                precision,
                recall
            ])

            if recall >= 1.0:
                break

        for inst in apTable:
            print(inst)
        print()

    exit()

    results = []
    gts = {}

    gtCounter = 0
    for dataInd, (image, anno) in enumerate(data):

        # Record ground truth
        gtTmp = {}
        for inst in anno:

            gtId = gtCounter
            gtCounter += 1

            gt = api.RBox()
            gt.conf = 1.0
            gt.label = inst['label']
            gt.degree = inst['degree']
            gt.x = inst['x']
            gt.y = inst['y']
            gt.w = inst['w']
            gt.h = inst['h']

            gtTmp[gtId] = gt

        gts.update(gtTmp)

        # Convert image to opencv format
        image = np.uint8(image)[..., ::-1]

        # Predict and record
        pred = detector.detect(image, confTh, nmsTh)
        for inst in pred:

            # Find similarities between prediction and ground truths
            rboxSim = rbox_similarity(
                [inst], [gtTmp[gtKey] for gtKey in gtTmp])
            highSim = (rboxSim >= simTh).squeeze(0)

            if highSim.sum() > 0:
                for gtKey, match in zip(gtTmp, highSim):
                    if match:
                        results.append({
                            'conf': inst.conf,
                            'pred': inst.label,
                            'label': gtTmp[gtKey].label,
                            'gtId': gtKey,
                            'dataInd': dataInd
                        })

            else:
                results.append({
                    'conf': inst.conf,
                    'pred': inst.label,
                    'label': -1,
                    'gtId': -1,
                    'dataInd': dataInd
                })

    # Sort predict result with confidence
    results = sorted(results, key=lambda inst: inst['conf'], reverse=True)

    # Append precision and recall
    gtHit = []
    predHit = 0
    apTable = []
    for i, inst in enumerate(results):

        recall = 0
        precision = 0

        # Find recall and precision
        gtId = inst['gtId']
        if gtId >= 0:
            if gtId not in gtHit:
                gtHit.append(inst['gtId'])
            if inst['pred'] == inst['label']:
                predHit += 1

        recall = len(gtHit) / len(gts)
        precision = predHit / (i + 1)

        apTable.append([
            inst['dataInd'],
            inst['conf'],
            inst['pred'],
            inst['label'],
            precision,
            recall
        ])

        if recall >= 1.0:
            break

    for inst in apTable:
        print(inst)

    """
    image, anno = data[0]
    image = np.uint8(image)[..., ::-1]
    pred = detector.detect(image, confTh, nmsTh)
    print(len(pred))

    print(rbox_similarity([pred[0]], pred))
    for inst in pred:
        print(inst)

    resultImg = rbox_draw([image], [pred])[0]
    cv.imshow('Result', resultImg)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
