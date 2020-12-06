#!@Python_EXECUTABLE@

import argparse
import torch

from os.path import basename, splitext

if __name__ == '__main__':

    # Parse arguments
    argp = argparse.ArgumentParser(
        description='Export pretrained weight from existing model or backup file')

    argp.add_argument('source', type=str, help='Target model or backup file')
    argp.add_argument('destination', type=str, help='Output path')

    opt = argp.add_argument_group('Optional arguments')
    opt.add_argument('--best', type=bool, default=True,
                     help=('Export pretrained weight with best model state. '
                           '(default: %(default)s)'))

    args = argp.parse_args()

    # Import pretrain
    src = args.source
    srcExt = splitext(basename(src))[1]
    if srcExt == '.zip':
        sDict = torch.jit.load(src).backbone.state_dict()
    else:
        bak = torch.load(src, map_location=torch.device('cpu'))
        if args.best:
            sDict = bak['best_state']['backbone']
        else:
            sDict = bak['model_state_dict']['backbone']

    # Save pretrain weight
    torch.save({'bbone_state_dict': sDict}, args.destination)
