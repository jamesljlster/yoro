#!@Python_EXECUTABLE@

import argparse
import torch

from random import shuffle

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from yoro.transforms import RBox_Resize, RBox_PadToSquare
from yoro.datasets import RBoxSample, rbox_collate_fn


def get_rect_list(loader):

    rectList = []
    loop = iter(loader)
    for inst in loop:
        for batch in inst[1]:
            for rbox in batch:
                rectList.append([rbox['w'], rbox['h']])

    return rectList


if __name__ == '__main__':

    # Parse arguments
    argp = argparse.ArgumentParser(
        description='K-Means Anchor Clustering Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argp.add_argument('path', metavar='dataset_path', type=str,
                      help='The folder contains image and annotation files')
    argp.add_argument('num', metavar='num_of_anchors', type=int,
                      help='Number of anchors')

    clustArg = argp.add_argument_group('K-Means clustering arguments')
    clustArg.add_argument('--iter', type=int, default=100,
                          help='Maximum iteration')
    clustArg.add_argument('--stop-dist', type=float, default=0.001,
                          help=('Stop algorithm after centeroid average' +
                                'moving distance equal or less then given value'))
    clustArg.add_argument('--device', type=str, default='cpu',
                          help='Running algorithm on given device')

    netReso = argp.add_argument_group('Network resolution')
    netReso.add_argument('--width', type=int, default=224,
                         help='Network input width')
    netReso.add_argument('--height', type=int, default=224,
                         help='Network input height')

    args = argp.parse_args()

    # Load dataset
    dataLoader = DataLoader(
        RBoxSample(args.path, transform=Compose([
            RBox_PadToSquare(),
            RBox_Resize((args.height, args.width))
        ])),
        shuffle=True,
        collate_fn=rbox_collate_fn
    )

    # Task setup
    numOfAnchor = args.num
    maxIter = args.iter
    stopMovDist = args.stop_dist
    device = torch.device(args.device)

    # Select initial anchors
    rectList = get_rect_list(dataLoader)
    shuffle(rectList)

    anchor = (torch.tensor(rectList[:numOfAnchor], device=device)
              .view(numOfAnchor, 1, 2))

    # Run K-Means clustering
    for it in range(maxIter):

        # Get new point list with anchors to prevent empty cluster
        rectTen = torch.tensor(
            get_rect_list(dataLoader) + anchor.view(numOfAnchor, 2).tolist(),
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

    # Show result
    anchorList = anchor.view(numOfAnchor, 2).tolist()
    print()
    print('anchor:')
    for inst in anchorList:
        print('  -', inst)
