import torch

from yoro.layers import YOROLayer
from yoro.ops import rbox_similarity, flatten_prediction, non_maximum_suppression
from yoro.ops import ciou_loss

width = 224
height = 224
num_classes = 2

if __name__ == '__main__':

    # Test ops for rotated bounding box
    rbox1 = torch.tensor([
        [45., 10., 20., 25., 35.],
        [75., 5., 25., 40., 50.],
    ])
    rbox2 = torch.tensor([
        [0., 15., 20., 50., 60.],
        [-30., 20., 20., 30., 40.],
    ])

    print(rbox_similarity(rbox1, rbox1))
    print(rbox_similarity(rbox1, rbox2))

    inputs = [torch.randn(2, 256, 26, 26),
              torch.randn(2, 512, 13, 13)]
    input_shapes = [ten.size() for ten in inputs]

    anchor = [[[21.39, 15.10], [55.22, 55.26], [64.18, 45.31]],
              [[78.89, 78.94], [91.68, 64.73]]]

    yoro = YOROLayer(width, height, num_classes, input_shapes, anchor,
                     deg_min=-45, deg_max=45, deg_part_size=10)
    preds = yoro(inputs)

    preds = flatten_prediction(preds)
    for pred in preds:
        print(pred.size())
    print()
    preds = non_maximum_suppression(preds, 0.5, 1.0)

    # Test CIoU loss
    bbox1 = rbox1[:, 1:]
    bbox2 = rbox2[:, 1:]
    loss, iou = ciou_loss(bbox1, bbox2)
    print('Loss:', loss)
    print('IoU:', iou)
