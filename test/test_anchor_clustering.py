import yaml
import numpy as np
import torch
from random import shuffle
from yoro.utils.train_util import YOROTrain

numOfAnchor = 2
maxIter = 1000
stopMovDist = 0.5


def get_rect_list(loader):

    rectList = []
    loop = iter(loader)
    for inst in loop:
        for batch in inst[1]:
            for rbox in batch:
                rectList.append([rbox['w'], rbox['h']])

    return rectList


if __name__ == '__main__':

    tc = YOROTrain('config/example.yaml')
    device = torch.device('cpu')

    # Select initial anchors
    rectList = get_rect_list(tc.traLoader)
    shuffle(rectList)

    anchor = (torch.tensor(rectList[:numOfAnchor], device=device)
              .view(numOfAnchor, 1, 2))

    # Run K-Means clustering
    for it in range(maxIter):

        # Get new point list with anchors to prevent empty cluster
        rectTen = torch.tensor(
            get_rect_list(tc.traLoader) + anchor.view(numOfAnchor, 2).tolist(),
            device=device)

        # Find new cluster center
        dist = torch.square(anchor - rectTen).sum(-1)
        belongIdx = torch.argmin(dist, dim=0)
        newAnchor = torch.zeros_like(anchor)
        for i in range(numOfAnchor):
            newAnchor[i] = rectTen[belongIdx == i].mean(dim=0).unsqueeze(0)

        # Estimate average moving distance for cluster centers
        avgMovDist = torch.sqrt(
            torch.square(newAnchor - anchor).sum(-1)).mean().item()
        anchor = newAnchor

        print('Iter %d. Average moving distance: %f' % (it + 1, avgMovDist))
        if avgMovDist <= stopMovDist:
            break

    anchorInfo = {'anchor': anchor.view(numOfAnchor, 2).tolist()}
    print(anchorInfo)
