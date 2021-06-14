import math

import torch
from torch import Tensor

from typing import Tuple


def ciou_loss(boxes1: Tensor, boxes2: Tensor,
              v_thresh: float = 0.5, reduction: str = 'sum') -> Tuple[Tensor, Tensor]:
    """
    From: https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py
    """

    w1, h1 = boxes1[:, 2], boxes1[:, 3]
    w2, h2 = boxes2[:, 2], boxes2[:, 3]
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1, center_y1 = boxes1[:, 0], boxes1[:, 1]
    center_x2, center_y2 = boxes2[:, 0], boxes2[:, 1]

    inter_l = torch.max(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * \
        torch.clamp((inter_b - inter_t), min=0)

    c_l = torch.min(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + \
        torch.clamp((c_b - c_t), min=0) ** 2

    union = area1 + area2 - inter_area
    u = inter_diag / c_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * \
        torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = (iou > v_thresh).float()
        alpha = S * v / (1 - iou + v)
    cious = iou - u - alpha * v
    cious = torch.clamp(cious, min=-1.0, max=1.0)

    loss = None
    if reduction == 'mean':
        loss = torch.mean(1 - cious)
    else:
        loss = torch.sum(1 - cious)

    return loss, iou.detach()
