import torch
from torch import Tensor

from torch.nn import Module, ModuleList, Parameter, Conv2d
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
                 xy_scale: float = 1.05,
                 conv_params: List[Dict] = [{}]
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

        self.xy_scale = xy_scale

        # Build anchor
        if len(anchor) == 1:
            anchor = [anchor] * headSize

        self.anchorList: List[torch.nn.parameter.Parameter] = [
            Parameter(torch.tensor(anc), requires_grad=False)
            for anc in anchor]
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
        self.degAnchor = Parameter(
            torch.arange(
                start=deg_min, end=deg_max + deg_part_size, step=deg_part_size),
            requires_grad=False)

        self.degPartDepth = len(self.degAnchor)
        self.degValueDepth = len(self.degAnchor)
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
            x = torch.sigmoid(head[..., base + 0])
            y = torch.sigmoid(head[..., base + 1])
            w = head[..., base + 2]
            h = head[..., base + 3]

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
            -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:

        outputs: List[Tuple[Tensor, Tensor, Tensor, Tensor]] = []
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

            # Find grid x, y
            gridY = (torch.arange(fmapHeight, dtype=dtype, device=device)
                     .view(1, 1, fmapHeight, 1))
            gridX = (torch.arange(fmapWidth, dtype=dtype, device=device)
                     .view(1, 1, 1, fmapWidth))

            # Decoding
            label = torch.argmax(cls, dim=4)
            labelConf = \
                torch.softmax(cls, dim=4).gather(4, label.unsqueeze(-1))
            conf = obj * labelConf.squeeze(-1)

            boxes = torch.zeros(
                batch, anchorSize, fmapHeight, fmapWidth, 4, device=device)
            boxes[..., 0] = (x * self.xy_scale + gridX) * self.gridWidth[i]
            boxes[..., 1] = (y * self.xy_scale + gridY) * self.gridHeight[i]
            boxes[..., 2] = \
                torch.exp(w) * self.anchorList[i][:, 0].view(1, -1, 1, 1)
            boxes[..., 3] = \
                torch.exp(h) * self.anchorList[i][:, 1].view(1, -1, 1, 1)

            idx = torch.argmax(degPart, dim=4)
            degree = (self.degAnchor[idx] +
                      torch.gather(degShift, 4, idx.unsqueeze(-1)).squeeze(-1) *
                      self.degValueScale)

            outputs.append((conf, label, boxes, degree))

        return outputs

    def forward(self, inputs: List[Tensor]) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:

        x = self.predict(inputs)
        x = self.decode(x)

        return x

    @torch.jit.unused
    def loss(self, inputs, targets):

        device = inputs.device
        dtype = inputs.dtype

        batch = inputs.size(0)

        # Build target
        tList = []
        for n in range(batch):
            for anno in targets[n]:

                degDiff = anno['degree'] - self.degAnchor
                degPartIdx = torch.argmin(torch.abs(degDiff))
                degShiftValue = degDiff[degPartIdx] / self.degValueScale

                tList.append([
                    n, anno['label'],
                    anno['x'] / self.gridWidth,  # FIXME
                    anno['y'] / self.gridHeight,  # FIXME
                    anno['w'], anno['h'],
                    degPartIdx.item(), degShiftValue.item()
                ])

        targets = torch.tensor(tList, dtype=dtype, device=device)
        objs = targets.size(0)

        # Predict
        (conf, cls, x, y, w, h, degPart, degShift) = self.predict(inputs)

        # Mask of feature map
        objMask = torch.empty(
            conf.size(), dtype=torch.bool, device=device).fill_(False)
        nobjMask = torch.empty(
            conf.size(), dtype=torch.bool, device=device).fill_(True)

        # Get targets
        if objs:

            n = targets[:, 0].long()
            clsT = targets[:, 1].long()

            boxT = targets[:, 2:6]
            xT = boxT[:, 0]
            yT = boxT[:, 1]
            wT = boxT[:, 2]
            hT = boxT[:, 3]

            degPartT = targets[:, 6].long()
            degShiftT = targets[:, 7]

            # Get anchor scores
            acrScore = self.anchor_score(boxT[:, 2:])  # FIXME
            _, acrIdx = acrScore.max(dim=1)

            # Set mask
            xIdx = xT.long()
            yIdx = yT.long()

            objMask[n, acrIdx, yIdx, xIdx] = True
            nobjMask[n, acrIdx, yIdx, xIdx] = False

            # Ignore objectness if anchor score greater than threshold
            # for i, score in enumerate(acrScore):
            #    nobjMask[n[i], (score > self.ignThres),
            #             xIdx[i], yIdx[i]] = False

        # Build loss
        confT = objMask.float()

        objLoss = F.binary_cross_entropy(conf[objMask], confT[objMask])
        nobjLoss = F.binary_cross_entropy(conf[nobjMask], confT[nobjMask])

        if objs:

            # Class loss
            clsLoss = F.cross_entropy(
                cls[n, acrIdx, yIdx, xIdx], clsT, reduction='sum')

            # Bounding box loss
            xLoss = F.mse_loss(
                x[n, acrIdx, yIdx, xIdx], xT - xT.floor(), reduction='sum')  # FIXME
            yLoss = F.mse_loss(
                y[n, acrIdx, yIdx, xIdx], yT - yT.floor(), reduction='sum')  # FIXME
            wLoss = F.mse_loss(
                w[n, acrIdx, yIdx, xIdx],
                torch.log(wT / self.anchor[acrIdx, 0]),  # FIXME
                reduction='sum')
            hLoss = F.mse_loss(
                h[n, acrIdx, yIdx, xIdx],
                torch.log(hT / self.anchor[acrIdx, 1]),  # FIXME
                reduction='sum')

            boxLoss = xLoss + yLoss + wLoss + hLoss

            # Degree loss
            degPartLoss = F.cross_entropy(
                degPart[n, acrIdx, yIdx, xIdx], degPartT, reduction='sum')
            degShiftLoss = F.mse_loss(
                degShift[n, acrIdx, yIdx, xIdx, degPartT], degShiftT, reduction='sum')

        else:

            clsLoss = torch.tensor([0], dtype=dtype, device=device)
            boxLoss = torch.tensor([0], dtype=dtype, device=device)
            degPartLoss = torch.tensor([0], dtype=dtype, device=device)
            degShiftLoss = torch.tensor([0], dtype=dtype, device=device)

        loss = ((objLoss + nobjLoss + clsLoss +
                 boxLoss + degPartLoss + degShiftLoss), 1)

        # Estimation
        objConf = conf[objMask].mean().item()
        nobjConf = conf[nobjMask].mean().item()

        clsAccu = 0
        degCorr = 0
        if objs:

            # Class accuracy
            clsAccu = (torch.argmax(
                cls[n, acrIdx, yIdx, xIdx], dim=1) == clsT).sum().float().item()

            # Degree correlation coefficient
            degTarget = \
                self.degAnchor[degPartT] + degShiftT * self.degValueScale
            degIdx = torch.argmax(degPart[n, acrIdx, yIdx, xIdx], dim=1)
            degPredict = (self.degAnchor[degIdx] +
                          degShift[n, acrIdx, yIdx, xIdx, degIdx] *
                          self.degValueScale)
            degCorr = correlation_coefficient(degTarget, degPredict)

        # Summarize accuracy
        info = {
            'obj': (objConf, 1),
            'nobj': (nobjConf, 1),
            'cls': (clsAccu, objs),
            'deg': (degCorr, 1) if degCorr is not None else (0, 0)
        }

        return loss, info

    @torch.jit.unused
    def anchor_score(self, box):  # FIXME

        box = box.view(box.size()[0], -1, 2)
        anchor = self.anchor.to(box.device).unsqueeze(0)

        bw, bh = box[..., 0], box[..., 1]
        aw, ah = anchor[..., 0], anchor[..., 1]

        interArea = torch.min(bw, aw) * torch.min(bh, ah)
        unionArea = bw * bh + aw * ah - interArea

        return interArea / (unionArea + 1e-8)
