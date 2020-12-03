#!@Python_EXECUTABLE@

import argparse
import torch

from os.path import basename, splitext

if __name__ == '__main__':

    # Parse arguments
    argp = argparse.ArgumentParser(
        description='Export pretrain weight from existing model or backup file')

    argp.add_argument('source', type=str, help='Target model or backup file')
    argp.add_argument('destination', type=str, help='Output path')

    args = argp.parse_args()

    # Import pretrain
    src = args.source
    srcExt = splitext(basename(src))[1]
    if srcExt == '.zip':
        sDict = torch.jit.load(src).backbone.state_dict()
    else:
        sDict = torch.load(
            src, map_location=torch.device('cpu'))['bbone_state_dict']

    # Save pretrain weight
    torch.save({'bbone_state_dict': sDict}, args.destination)
