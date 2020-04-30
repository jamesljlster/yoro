import torch
from torch.nn import Module, Parameter, Conv2d
from torch.nn import functional as F


class YOROLayer(Module):

    __constants__ = [
        'numClasses', 'gridWidth', 'gridHeight',
        'anchorSize',
        'degMin', 'degMax', 'degRange',
        'degPartSize', 'degValueScale',
        'degPartDepth', 'degValueDepth',
        'bboxDepth', 'classDepth', 'objDepth',
        'groupDepth', 'fmapDepth'
    ]

    def __init__(self, in_channels, num_classes,
                 width, height, fmap_width, fmap_height, anchor,
                 deg_min=-180, deg_max=180, deg_part_size=10):

        super(YOROLayer, self).__init__()

        # Save parameters
        self.numClasses = num_classes
        self.gridWidth = width / fmap_width
        self.gridHeight = height / fmap_height

        # Build anchor
        self.anchor = Parameter(torch.FloatTensor(anchor), requires_grad=False)
        self.anchorSize = self.anchor.size()[0]

        self.anchor[:, 0] /= self.gridWidth
        self.anchor[:, 1] /= self.gridHeight

        self.anchorW = Parameter(
            self.anchor[:, 0].view(1, -1, 1, 1).clone(), requires_grad=False)
        self.anchorH = Parameter(
            self.anchor[:, 1].view(1, -1, 1, 1).clone(), requires_grad=False)

        # Feature map specification construction: degree
        self.degMin = deg_min
        self.degMax = deg_max
        self.degRange = deg_max - deg_min

        self.degPartSize = deg_part_size
        self.degValueScale = float(deg_part_size) / 2.0
        self.degAnchor = Parameter(torch.arange(
            0, self.degRange+1, deg_part_size, dtype=torch.float) + deg_min,
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
        self.groupDepth = self.objDepth + self.classDepth + self.bboxDepth + self.degDepth
        self.fmapDepth = self.groupDepth * self.anchorSize

        # Build regressor
        self.regressor = Conv2d(in_channels=in_channels, out_channels=self.fmapDepth,
                                kernel_size=3, padding=1)

    @torch.jit.export
    def predict(self, inputs):

        # Get convolution outputs
        inputs = self.regressor(inputs)

        # Get tensor dimensions
        batch = inputs.size()[0]
        fmapHeight = inputs.size()[2]
        fmapWidth = inputs.size()[3]

        # Rearange predict tensor
        pred = inputs.view(batch, self.anchorSize, self.groupDepth,
                           fmapHeight, fmapWidth).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs: objectness
        base = 0
        conf = torch.sigmoid(pred[..., base])

        # Get outputs: class
        base += self.objDepth
        cls = pred[..., base:base + self.classDepth]

        # Get outputs: bounding box
        base += self.classDepth
        x = torch.sigmoid(pred[..., base + 0])
        y = torch.sigmoid(pred[..., base + 1])
        w = pred[..., base + 2]
        h = pred[..., base + 3]

        # Get outputs: degree partition
        base += self.bboxDepth
        degPart = pred[..., base:base + self.degPartDepth]

        # Get outputs: degree value shift
        base += self.degPartDepth
        degShift = pred[..., base:base + self.degValueDepth]

        return (conf, cls, x, y, w, h, degPart, degShift)

    def forward(self, inputs):

        (conf, cls, x, y, w, h, degPart, degShift) = self.predict(inputs)

        # Get tensor dimensions
        batch = inputs.size()[0]
        fmapHeight = inputs.size()[2]
        fmapWidth = inputs.size()[3]

        # Find grid x, y
        gridY = (torch.arange(fmapHeight, dtype=inputs.dtype, device=inputs.device)
                 .view(1, 1, fmapHeight, 1))
        gridX = (torch.arange(fmapWidth, dtype=inputs.dtype, device=inputs.device)
                 .view(1, 1, 1, fmapWidth))

        # Decoding
        pred_conf = conf.data
        pred_class = torch.argmax(cls.data, dim=4)

        size = conf.size()
        pred_boxes = torch.zeros(
            size[0], size[1], size[2], size[3], 4, device=inputs.device)
        pred_boxes[..., 0] = (x.data + gridX) * self.gridWidth
        pred_boxes[..., 1] = (y.data + gridY) * self.gridHeight
        pred_boxes[..., 2] = \
            torch.exp(w.data) * self.anchorW * self.gridWidth
        pred_boxes[..., 3] = \
            torch.exp(h.data) * self.anchorH * self.gridHeight

        idx = torch.argmax(degPart.data, dim=4)
        pred_deg = (self.degAnchor[idx] +
                    torch.gather(degShift.data, 4, idx.unsqueeze(-1)).squeeze(-1) *
                    self.degValueScale)

        pred_conf = pred_conf.view(batch, -1)
        pred_class = pred_class.view(batch, -1)
        pred_boxes = pred_boxes.view(batch, -1, 4)
        pred_deg = pred_deg.view(batch, -1)

        return (pred_conf, pred_class, pred_boxes, pred_deg)

    @torch.jit.unused
    def loss(self, inputs, targets):

        device = inputs.device
        dtype = inputs.dtype

        batch = inputs.size()[0]

        # Predict
        (conf, cls, x, y, w, h, degPart, degShift) = self.predict(inputs)

        # Build target
        tList = []
        for n in range(batch):
            for anno in targets[n]:

                degDiff = anno['degree'] - self.degAnchor
                degPartIdx = torch.argmin(torch.abs(degDiff))
                degShiftValue = degDiff[degPartIdx] / self.degValueScale

                tList.append([
                    n, anno['label'],
                    anno['x'] / self.gridWidth,
                    anno['y'] / self.gridHeight,
                    anno['w'] / self.gridWidth,
                    anno['h'] / self.gridHeight,
                    degPartIdx.item(), degShiftValue.item()
                ])

        targets = torch.tensor(tList, dtype=dtype, device=device)
        objs = targets.size()[0]

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
            acrScore = self.anchor_score(boxT[:, 2:])
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
                x[n, acrIdx, yIdx, xIdx], xT - xT.floor(), reduction='sum')
            yLoss = F.mse_loss(
                y[n, acrIdx, yIdx, xIdx], yT - yT.floor(), reduction='sum')
            wLoss = F.mse_loss(
                w[n, acrIdx, yIdx, xIdx],
                torch.log(wT / self.anchor[acrIdx, 0]),
                reduction='sum')
            hLoss = F.mse_loss(
                h[n, acrIdx, yIdx, xIdx],
                torch.log(hT / self.anchor[acrIdx, 1]),
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

        loss = (objLoss + nobjLoss + clsLoss +
                boxLoss + degPartLoss + degShiftLoss)

        # Estimation
        objConf = conf[objMask].mean().item()
        nobjConf = conf[nobjMask].mean().item()

        clsAccu = 0
        degPartAccu = 0
        if objs:

            # Class accuracy
            clsAccu = (torch.argmax(
                cls[n, acrIdx, yIdx, xIdx], dim=1) == clsT).sum().float().item()

            # Degree partitioin accuracy
            degPartAccu = (torch.argmax(
                degPart[n, acrIdx, yIdx, xIdx], dim=1) == degPartT).sum().float().item()

        # Summarize accuracy
        accu = {
            'obj': (objConf, 1),
            'nobj': (nobjConf, 1),
            'cls': (clsAccu, objs),
            'deg': (degPartAccu, objs)
        }

        return loss, accu

    @torch.jit.unused
    def anchor_score(self, box):

        box = box.view(box.size()[0], -1, 2)
        anchor = self.anchor.to(box.device).unsqueeze(0)

        bw, bh = box[..., 0], box[..., 1]
        aw, ah = anchor[..., 0], anchor[..., 1]

        interArea = torch.min(bw, aw) * torch.min(bh, ah)
        unionArea = bw * bh + aw * ah - interArea

        return interArea / (unionArea + 1e-8)
