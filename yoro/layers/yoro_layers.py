import math

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, ParameterList, Parameter, Conv2d
from torch.nn import functional as F

from typing import List, Dict, Tuple

from .functional import correlation_coefficient


class YOROLayer(Module):

    """
    __constants__ = [
        'numClasses', 'width', 'height', 'gridWidth', 'gridHeight',
        'anchorSize',
        'degMin', 'degMax', 'degRange',
        'degPartSize', 'degValueScale',
        'degPartDepth', 'degValueDepth',
        'bboxDepth', 'classDepth', 'objDepth',
        'groupDepth', 'fmapDepth'
    ]

    def __init__(self, in_channels, num_classes,
                 width: int, height: int, fmap_width, fmap_height, anchor,
                 deg_min=-180, deg_max=180, deg_part_size=10, conv_params={}):
    """

    def __init__(self,
                 width: int, height: int, num_classes: int,
                 input_shapes: List[torch.Size], anchor: List[List[list]],
                 deg_min: int = -180, deg_max: int = 180, deg_part_size: int = 10,
                 conv_params: List[Dict] = [{}], loss_norm: Dict = {}
                 ):

        super(YOROLayer, self).__init__()

        # Find head size
        if isinstance(input_shapes, torch.Size):
            input_shapes = [input_shapes]

        headSize = len(input_shapes)

        # Check convolution parameters
        if isinstance(conv_params, dict):
            conv_params = [conv_params] * headSize

        assert len(conv_params) != headSize, \
            'Numbers of \"conv_params\" does not match with desire head size.'

        # Save parameters
        self.numClasses = num_classes
        self.width = width
        self.height = height

        self.headSize = headSize

        self.gridWidth: List[float] = [
            (width / size[3]) for size in input_shapes]
        self.gridHeight: List[float] = [
            (height / size[2]) for size in input_shapes]

        # Loss normalizer
        self.objNorm = loss_norm.get('obj', 1.0)
        self.nobjNorm = loss_norm.get('no_obj', 1.0)
        self.clsNorm = loss_norm.get('class', 1.0)
        self.boxNorm = loss_norm.get('box', 1.0)
        self.degPartNorm = loss_norm.get('degree_part', 1.0)
        self.degShiftNorm = loss_norm.get('degree_shift', 1.0)

        # Build anchor
        if len(anchor) == 1:
            anchor = [anchor] * headSize

        self.anchorList: List[torch.Tensor] = [
            torch.tensor(anc) for anc in anchor]
        self.anchorSizeList: List[int] = []

        for i, anc in enumerate(self.anchorList):
            size, dim = anc.size()

            # Check anchor format
            assert dim == 2, \
                'Anchor last dimension size is not 2 (width, height)'
            self.anchorSizeList.append(size)

        # Feature map specification construction: degree
        self.degMin = deg_min
        self.degMax = deg_max
        self.degRange = deg_max - deg_min

        self.degPartSize = deg_part_size
        self.degValueScale = float(deg_part_size) / 2.0
        self.degAnchor: List[torch.Tensor] = [torch.arange(
            start=deg_min, end=deg_max + deg_part_size, step=deg_part_size)]

        self.degPartDepth = self.degAnchor[0].size(0)
        self.degValueDepth = self.degAnchor[0].size(0)
        self.degDepth = self.degPartDepth + self.degValueDepth

        # Feature map specification construction: bbox
        self.bboxDepth = 4  # x, y, w, h

        # Feature map specification construction: class
        self.classDepth = self.numClasses

        # Feature map specification construction: objectness
        self.objDepth = 1

        # Feature map specification construction: final
        self.groupDepth = (
            self.objDepth + self.classDepth + self.bboxDepth + self.degDepth)
        self.fmapDepthList: List[int] = [
            self.groupDepth * ancSize for ancSize in self.anchorSizeList]

        # Build regressor
        if len(conv_params) == 1:
            conv_params = conv_params * headSize

        regressor = []
        for i, conv_param in enumerate(conv_params):

            conv_param.pop('in_channels', None)
            conv_param.pop('out_channels', None)
            conv_param = {
                'in_channels': input_shapes[i][1],
                'out_channels': self.fmapDepthList[i],
                'kernel_size': 1,
                'padding': 0,
                **conv_param
            }

            regressor.append(Conv2d(**conv_param))

        self.regressor = ModuleList(regressor)

    @torch.jit.export
    def head_regression(self, inputs: List[Tensor]) -> List[Tensor]:

        outputs: List[Tensor] = []
        for i, module in enumerate(self.regressor):
            outputs.append(module(inputs[i]))

        return outputs

    @torch.jit.export
    def head_slicing(self, inputs: List[Tensor]) -> List[Tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:

        outputs: List[Tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        for i, head in enumerate(inputs):

            # Get tensor dimensions
            batch = head.size(0)
            fmapHeight = head.size(2)
            fmapWidth = head.size(3)

            # Rearange predict tensor
            head = head.view(
                batch, self.anchorSizeList[i], self.groupDepth, fmapHeight, fmapWidth)
            head = head.permute(0, 1, 3, 4, 2).contiguous()

            # Get outputs: objectness
            base = 0
            obj = torch.sigmoid(head[..., base])

            # Get outputs: class
            base += self.objDepth
            cls = head[..., base:base + self.classDepth]

            # Get outputs: bounding box
            base += self.classDepth
            boxes = torch.sigmoid(head[..., base:base+self.bboxDepth])
            x = boxes[..., 0]
            y = boxes[..., 1]
            w = boxes[..., 2]
            h = boxes[..., 3]

            # Get outputs: degree partition
            base += self.bboxDepth
            degPart = head[..., base:base + self.degPartDepth]

            # Get outputs: degree value shift
            base += self.degPartDepth
            degShift = head[..., base:base + self.degValueDepth]

            outputs.append((obj, cls, x, y, w, h, degPart, degShift))

        return outputs

    @torch.jit.export
    def predict(self, inputs: List[Tensor]) -> List[Tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:

        x = self.head_regression(inputs)
        x = self.head_slicing(x)

        return x

    @torch.jit.export
    def decode(self, inputs: List[Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]) \
            -> List[Tuple[Tensor, Tensor, Tensor]]:

        outputs: List[Tuple[Tensor, Tensor, Tensor]] = []
        for i, (obj, cls, x, y, w, h, degPart, degShift) in enumerate(inputs):

            # Detach tensors
            obj = obj.detach()
            cls = cls.detach()
            x = x.detach()
            y = y.detach()
            w = w.detach()
            h = h.detach()
            degPart = degPart.detach()
            degShift = degShift.detach()

            # Cache dtype, device and dimensions
            device = obj.device
            dtype = obj.dtype
            batch, anchorSize, fmapHeight, fmapWidth = obj.size()

            # Cache anchor
            anchor = self.anchorList[i].to(device)
            degAnchor = self.degAnchor[0].to(device)

            # Find grid x, y
            gridY = (torch.arange(fmapHeight, dtype=dtype, device=device)
                     .view(1, 1, fmapHeight, 1))
            gridX = (torch.arange(fmapWidth, dtype=dtype, device=device)
                     .view(1, 1, 1, fmapWidth))

            # Decoding
            label = torch.argmax(cls, dim=4)
            labelConf = torch.softmax(cls, dim=4).gather(
                4, label.unsqueeze(-1))
            conf = obj * labelConf.squeeze(-1)

            idx = torch.argmax(degPart, dim=4)
            degree = (degAnchor[idx] +
                      torch.gather(degShift, 4, idx.unsqueeze(-1)).squeeze(-1) *
                      self.degValueScale)

            rboxes = torch.zeros(
                batch, anchorSize, fmapHeight, fmapWidth, 5, device=device)
            rboxes[..., 0] = degree
            rboxes[..., 1] = ((x * 2 - 0.5) + gridX) * self.gridWidth[i]
            rboxes[..., 2] = ((y * 2 - 0.5) + gridY) * self.gridHeight[i]
            rboxes[..., 3] = \
                torch.pow(w * 2, 2) * anchor[:, 0].view(1, -1, 1, 1)
            rboxes[..., 4] = \
                torch.pow(h * 2, 2) * anchor[:, 1].view(1, -1, 1, 1)

            outputs.append((conf, label, rboxes))

        return outputs

    def forward(self, inputs: List[Tensor]) -> List[Tuple[Tensor, Tensor, Tensor]]:

        x = self.predict(inputs)
        x = self.decode(x)

        return x

    @torch.jit.unused
    def loss(self, inputs, targets):

        device = inputs[0].device
        dtype = inputs[0].dtype

        # Predict
        predList = self.predict(inputs)

        # Mask of feature map
        objMaskList = targets[0]

        # Compute loss for rotated bounding box
        objLoss = torch.tensor([0], dtype=dtype, device=device)
        nobjLoss = torch.tensor([0], dtype=dtype, device=device)
        clsLoss = torch.tensor([0], dtype=dtype, device=device)
        boxLoss = torch.tensor([0], dtype=dtype, device=device)
        degPartLoss = torch.tensor([0], dtype=dtype, device=device)
        degShiftLoss = torch.tensor([0], dtype=dtype, device=device)

        for headIdx, objMask in enumerate(objMaskList):

            nobjMask = (objMask == False)
            objT = objMask.float()

            # Extract tensor
            #   Predict: [conf, cls, x, y, w, h, degPart, degShift]
            #   Index:   [   0,   1, 2, 3, 4, 5,       6,        7]
            (obj, cls, x, y, w, h, degPart, degShift) = predList[headIdx]

            # Accumulate loss
            if objMask.sum() > 0:

                # Objectness loss
                objLoss += self.objNorm * F.binary_cross_entropy(
                    obj[objMask], objT[objMask].to(device), reduction='sum')

                # Extract targets
                batchT, acrIdxT, xIdxT, yIdxT, clsT, xT, yT, wT, hT, degPartT, degShiftT = \
                    targets[1][headIdx]

                # Class loss
                clsLoss += self.clsNorm * F.cross_entropy(
                    cls[batchT, acrIdxT, yIdxT, xIdxT], clsT.to(device), reduction='sum')

                # Bounding box loss
                xLoss = F.mse_loss(
                    x[batchT, acrIdxT, yIdxT, xIdxT], xT.to(device), reduction='sum')
                yLoss = F.mse_loss(
                    y[batchT, acrIdxT, yIdxT, xIdxT], yT.to(device), reduction='sum')
                wLoss = F.mse_loss(
                    w[batchT, acrIdxT, yIdxT, xIdxT], wT.to(device), reduction='sum')
                hLoss = F.mse_loss(
                    h[batchT, acrIdxT, yIdxT, xIdxT], hT.to(device), reduction='sum')

                boxLoss += self.boxNorm * (xLoss + yLoss + wLoss + hLoss)

                # Degree loss
                degPartLoss += self.degPartNorm * F.cross_entropy(
                    degPart[batchT, acrIdxT, yIdxT, xIdxT],
                    degPartT.to(device), reduction='sum')
                degShiftLoss += self.degShiftNorm * F.mse_loss(
                    degShift[batchT, acrIdxT, yIdxT, xIdxT, degPartT],
                    degShiftT.to(device), reduction='sum')

            if nobjMask.sum() > 0:

                # Objectness loss
                nobjLoss += F.binary_cross_entropy(
                    obj[nobjMask], objT[nobjMask].to(device), reduction='sum')

        # Total loss
        loss = (
            objLoss + nobjLoss + clsLoss + boxLoss + degPartLoss + degShiftLoss)

        return (loss, 1), {}
