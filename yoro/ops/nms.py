import torch
from torch import Tensor

from typing import List, Tuple, Dict, Union


def bbox_to_corners(bbox: Tensor) -> Tensor:
    """
    Input bbox: [[center_x, center_y, w, h]]
    Output corners: [[left_x, top_y, right_x, bottom_y]]
    """

    corners = torch.zeros_like(bbox)
    corners[..., 0] = bbox[..., 0] - bbox[..., 2] / 2.0
    corners[..., 1] = bbox[..., 1] - bbox[..., 3] / 2.0
    corners[..., 2] = bbox[..., 0] + bbox[..., 2] / 2.0
    corners[..., 3] = bbox[..., 1] + bbox[..., 3] / 2.0

    return corners


def rbox_similarity(rbox1: Tensor, rbox2: Tensor, eps: float = 1e-4) -> Tensor:
    """
    Input: [[deg, x, y, w, h]]
    Reference: https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """

    # BBox to corners
    corners1 = bbox_to_corners(rbox1[..., 1:5])
    corners2 = bbox_to_corners(rbox2[..., 1:5])

    # Find IoU scores
    lt = torch.max(corners1[:, None, :2], corners2[:, :2])
    rb = torch.min(corners1[:, None, 2:], corners2[:, 2:])
    wh = torch.clamp(rb - lt, 0)

    inter = wh[..., 0] * wh[..., 1]
    union = ((rbox1[..., 3] * rbox1[..., 4])[:, None] +
             rbox2[..., 3] * rbox2[..., 4] - inter)
    ious = inter / (union + eps)

    # Find degree similarity
    rad1 = torch.deg2rad(rbox1[..., 0])
    rad2 = torch.deg2rad(rbox2[..., 0])
    ang1 = torch.stack([torch.sin(rad1), torch.cos(rad1)], 1)
    ang2 = torch.stack([torch.sin(rad2), torch.cos(rad2)], 1)
    angSim = (torch.matmul(ang1, ang2.t()) + 1.0) / 2.0

    return ious * angSim


def flatten_prediction(pred_list: List[Tuple[Tensor, Tensor, Tensor]]) \
        -> Tuple[Tensor, Tensor, Tensor]:

    totalConf: List[Tensor] = []
    totalLabel: List[Tensor] = []
    totalRBox: List[Tensor] = []

    for (conf, label, rbox) in pred_list:
        batch = conf.size(0)
        totalConf.append(conf.view(batch, -1))
        totalLabel.append(label.view(batch, -1))
        totalRBox.append(rbox.view(batch, -1, 5))

    return (
        torch.cat(totalConf, 1), torch.cat(totalLabel, 1), torch.cat(totalRBox, 1))


def non_maximum_suppression(
    pred: Tuple[Tensor, Tensor, Tensor], conf_th: float, nms_th: float) \
        -> List[List[Dict[str, Union[int, float]]]]:

    predConf, predClass, predRBox = pred

    batch = predConf.size(0)
    nmsOut = [[] for _ in range(batch)]

    # Processing NMS on mini-batch
    for n in range(batch):

        # Confidence filtering
        mask = (predConf[n] >= conf_th)
        conf = predConf[n, mask]
        cls = predClass[n, mask]
        rbox = predRBox[n, mask]
        sim = rbox_similarity(rbox, rbox)

        if conf.size(-1) == 0:
            continue

        while True:

            # Start with the maximum confident instance
            maxConf, ind = torch.max(conf, dim=-1)
            if maxConf < conf_th:
                break

            # Merge instances with high similarity
            curClass = cls[ind]
            candMask = (
                (conf >= conf_th) & (cls == curClass) & (sim[ind] >= nms_th))
            if candMask.sum() == 0:
                candMask[ind] = 1.0

            weight = conf[candMask]
            resultRBox = torch.matmul(weight, rbox[candMask]) / weight.sum()
            resultConf = torch.matmul(weight, weight)

            # Clear merged RBox
            conf[candMask] = -1

            # Append result
            nmsOut[n].append({
                'conf': resultConf.item(),
                'label': curClass.item(),
                'degree': resultRBox[0].item(),
                'x': resultRBox[1].item(),
                'y': resultRBox[2].item(),
                'w': resultRBox[3].item(),
                'h': resultRBox[4].item()
            })

    return nmsOut
