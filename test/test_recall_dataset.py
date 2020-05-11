import torch

from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from yoro.utils.train_util import YOROTrain
from yoro.visual import rbox_draw


def imshow(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':

    tc = YOROTrain('config/example.yaml')
    model = torch.jit.load('coating_epoch_30000.zip').to(tc.dev)
    model.eval()

    for images, annos in tc.tstLoader:

        inputs = images.to(tc.dev)
        (pred_conf, pred_class, pred_class_conf,
         pred_boxes, pred_deg) = model(inputs)

        predSize = pred_conf.size()
        batch = predSize[0]
        boxes = predSize[1]

        labels = []
        for n in range(batch):
            anno = []
            for i in range(boxes):
                if pred_conf[n, i] >= 0.9:
                    anno.append({
                        'label': pred_class[n, i].item(),
                        'degree': pred_deg[n, i].item(),
                        'x': pred_boxes[n, i, 0].item(),
                        'y': pred_boxes[n, i, 1].item(),
                        'w': pred_boxes[n, i, 2].item(),
                        'h': pred_boxes[n, i, 3].item()
                    })

            labels.append(anno)

        images = make_grid(rbox_draw(images, labels, to_tensor=True), nrow=4)
        imshow(images.numpy().transpose(1, 2, 0))
