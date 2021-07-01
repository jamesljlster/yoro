import torch
from torch import Tensor
from torch.nn import Module, ModuleDict

from typing import List, Dict
from collections import OrderedDict

from .config_parser import parse_config
from . import layers, config


class DarknetBBone(Module):

    # Attribute annotation
    layerLink: Dict[str, Dict[str, List[str]]]

    def __init__(self, config_path, input_size):
        super().__init__()

        # Parse network config
        netCfg = []
        for layerCfg in parse_config(config_path):

            # Bypass head layers
            if layerCfg['type'] == 'yolo':
                netCfg.pop()
                netCfg[-1]['to'] = []
            else:
                netCfg.append(layerCfg)

        # Construct layers
        layerOut = {-1: torch.randn(input_size)}

        layerLink = {}
        moduleDict = OrderedDict()
        for layerCfg in netCfg:

            layerIdx = layerCfg['layer_idx']
            layerType = layerCfg['type'].upper()

            # Resolve source layer indices
            fromInd = layerCfg['from']
            if len(fromInd) == 0:
                fromInd = [-1]

            # Resolve source tensor and size
            inputs = [layerOut[idx] for idx in fromInd]
            inputSize = [ten.size() for ten in inputs]

            # Construct layer module
            layerClass = layers.__dict__[layerType]
            layerObj = layerClass(input_size=inputSize, **layerCfg['param'])
            layerOut[layerIdx] = layerObj(inputs)

            layerLink[str(layerIdx)] = {
                'from': [str(idx) for idx in fromInd],
                'to': [str(idx) for idx in layerCfg['to']]
            }

            moduleDict[str(layerIdx)] = layerObj

        self.layerLink = layerLink
        self.moduleDict = ModuleDict(moduleDict)

    def forward(self, x: Tensor):

        layerOut: Dict[str, Tensor] = {str(-1): x}
        netOut: List[Tensor] = []
        for key, module in self.moduleDict.items():

            fromInd = self.layerLink[key]['from']
            toInd = self.layerLink[key]['to']

            inputs: List[Tensor] = [layerOut[idx] for idx in fromInd]
            outputs = module(inputs)

            layerOut[key] = outputs
            if len(toInd) == 0:
                netOut.append(outputs)

        return netOut


def YOLOv3(width, height, channels):
    return DarknetBBone(config.yolov3(), (1, channels, height, width))


def YOLOv3_Tiny(width, height, channels):
    return DarknetBBone(config.yolov3_tiny(), (1, channels, height, width))
