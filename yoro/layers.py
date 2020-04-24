import torch
from torch.nn import Module, Parameter
from torch.nn import functional as F


@torch.jit.script
class YOROScalarParam:
    """
    Warning: TorchScript class is currently experimental
    """

    def __init__(self):

        self.anchorSize = 0
        self.groupDepth = 0

        self.objDepth = 0
        self.classDepth = 0
        self.bboxDepth = 0
        self.degDepth = 0
        self.degPartDepth = 0
        self.degValueDepth = 0
        self.degValueScale = 0.0

        self.gridWidth = 0
        self.gridHeight = 0


class YOROInference(Module):

    def __init__(self):
        super(YOROInference, self).__init__()

        # Placeholder for uninitialized scalar parameters
        self.sParam = YOROScalarParam()

        # Placeholder for uninitialized tensor parameters
        self.anchorW = Parameter(torch.zeros(1, 1, 1, 1), requires_grad=False)
        self.anchorH = Parameter(torch.zeros(1, 1, 1, 1), requires_grad=False)
        self.degAnchor = Parameter(torch.zeros(1), requires_grad=False)

    def predict(self, inputs):

        # Get tensor dimensions
        batch = inputs.size()[0]
        fmapHeight = inputs.size()[2]
        fmapWidth = inputs.size()[3]

        # Rearange predict tensor
        pred = inputs.view(batch, self.sParam.anchorSize, self.sParam.groupDepth,
                           fmapHeight, fmapWidth).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs: objectness
        base = 0
        conf = torch.sigmoid(pred[..., base])

        # Get outputs: class
        base += self.sParam.objDepth
        cls = pred[..., base:base + self.sParam.classDepth]

        # Get outputs: bounding box
        base += self.sParam.classDepth
        x = torch.sigmoid(pred[..., base + 0])
        y = torch.sigmoid(pred[..., base + 1])
        w = pred[..., base + 2]
        h = pred[..., base + 3]

        # Get outputs: degree partition
        base += self.sParam.bboxDepth
        degPart = pred[..., base:base + self.sParam.degPartDepth]

        # Get outputs: degree value shift
        base += self.sParam.degPartDepth
        degShift = pred[..., base:base + self.sParam.degValueDepth]

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
        pred_boxes[..., 0] = (x.data + gridX) * self.sParam.gridWidth
        pred_boxes[..., 1] = (y.data + gridY) * self.sParam.gridHeight
        pred_boxes[..., 2] = \
            torch.exp(w.data) * self.anchorW * self.sParam.gridWidth
        pred_boxes[..., 3] = \
            torch.exp(h.data) * self.anchorH * self.sParam.gridHeight

        idx = torch.argmax(degPart.data, dim=4)
        pred_deg = (self.degAnchor[idx] +
                    torch.gather(degShift.data, 4, idx.unsqueeze(-1)).squeeze(-1) *
                    self.sParam.degValueScale)

        pred_conf = pred_conf.view(batch, -1)
        pred_class = pred_class.view(batch, -1)
        pred_boxes = pred_boxes.view(batch, -1, 4)
        pred_deg = pred_deg.view(batch, -1)

        return (pred_conf, pred_class, pred_boxes, pred_deg)


class YOROTraining(YOROInference):

    def __init__(self, backbone, numClasses, width=224, height=224):
        super(YOROTraining, self).__init__()


if __name__ == '__main__':

    yoro = YOROInference()

    numClasses = 2
    degMin = -180
    degMax = 180
    degRange = degMax - degMin
    degPartSize = 10

    anchor = torch.FloatTensor([
        [42.48, 45.48],
        [46.14, 34.85],
    ])

    anchor[:, 0] /= 32
    anchor[:, 1] /= 32

    yoro.anchorW.data = anchor[:, 0].view(1, -1, 1, 1).clone()
    yoro.anchorH.data = anchor[:, 1].view(1, -1, 1, 1).clone()
    yoro.degAnchor.data = (torch.arange(
        0, degRange + 1, degPartSize, dtype=torch.float) + degMin).clone()

    sParam = YOROScalarParam()

    sParam.gridWidth = 32
    sParam.gridHeight = 32

    sParam.anchorSize = 2
    sParam.bboxDepth = 4
    sParam.classDepth = numClasses
    sParam.objDepth = 1

    sParam.degValueScale = float(degPartSize) / 2.0
    sParam.degPartDepth = len(yoro.degAnchor)
    sParam.degValueDepth = len(yoro.degAnchor)
    sParam.degDepth = sParam.degPartDepth + sParam.degValueDepth

    sParam.groupDepth = sParam.objDepth + \
        sParam.classDepth + sParam.bboxDepth + sParam.degDepth

    yoro.sParam = sParam

    src = torch.randn(2, yoro.sParam.groupDepth * yoro.sParam.anchorSize, 7, 7)
    out = yoro(src)
    # print(out)

    #scp = torch.jit.trace(yoro, src)

    scp = torch.jit.script(YOROInference())
    scp.sParam = yoro.sParam
    params = {key: param for key, param in yoro.named_parameters()}
    for key, param in scp.named_parameters():
        param.data = params[key].data.clone()

    scpOut = scp(src)
    for i in range(len(scpOut)):
        print(torch.equal(out[i], scpOut[i]))

    yoro = yoro.to('cuda')
    scp = scp.to('cuda')
    # for param in scp.named_parameters():
    #    print(param)

    src = src.to('cuda')
    out = scp(src)
    out = yoro(src)
