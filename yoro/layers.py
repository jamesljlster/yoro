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


if __name__ == '__main__':

    anchor = [[42.48, 45.48],
              [46.14, 34.85]]
    yoro = YOROLayer(in_channels=512, num_classes=2, width=224, height=224,
                     fmap_width=7, fmap_height=7, anchor=anchor)

    src = torch.randn(2, 512, 7, 7)
    out = yoro(src)

    scp = torch.jit.script(
        YOROLayer(in_channels=512, num_classes=2, width=224, height=224,
                  fmap_width=7, fmap_height=7, anchor=anchor)
    )

    #params = {key: param for key, param in yoro.named_parameters()}
    # for key, param in scp.named_parameters():
    #    param.data = params[key].data.clone()
    scp.load_state_dict(yoro.state_dict())

    #print('=== Parameters in yoro script module ===')
    # for name, param in scp.named_parameters():
    #    print(name, param)
    # print()

    #print('=== Modules in yoro script module ===')
    # for name, module in scp.named_modules():
    #    print(name, module)
    # print()

    print('=== Tensor comparison result (CPU) ===')
    scpOut = scp(src)
    for i in range(len(scpOut)):
        print(torch.equal(out[i], scpOut[i]))
    print()

    yoro = yoro.to('cuda')
    scp = scp.to('cuda')

    src = src.to('cuda')
    scpOut = scp(src)
    out = yoro(src)

    print('=== Tensor comparison result (CUDA) ===')
    scpOut = scp(src)
    for i in range(len(scpOut)):
        print(torch.equal(out[i], scpOut[i]))
    print()
