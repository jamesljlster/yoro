import torch
from torch import Tensor

from typing import List, Tuple, Dict, Union

from .. import api


def bbox_to_corners(bbox: Tensor) -> Tensor:
    """
    Input bbox: [[center_x, center_y, w, h]]
    Output corners: [[left_x, top_y, right_x, bottom_y]]
    """
    return api.bbox_to_corners(bbox)


def rbox_similarity(rbox1: Tensor, rbox2: Tensor) -> Tensor:
    """
    Input: [[deg, x, y, w, h]]
    Reference: https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    return api.rbox_similarity(rbox1, rbox2)


def flatten_prediction(pred_list: List[Tuple[Tensor, Tensor, Tensor]]) \
        -> Tuple[Tensor, Tensor, Tensor]:
    return api.flatten_prediction(pred_list)


def non_maximum_suppression(
    pred: Tuple[Tensor, Tensor, Tensor], conf_th: float, nms_th: float) \
        -> List[List[Dict[str, Union[int, float]]]]:
    return [[rbox.to_dict() for rbox in dts]
            for dts in api.non_maximum_suppression(pred, conf_th, nms_th)]
