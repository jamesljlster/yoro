#!@Python_EXECUTABLE@

import os
import sys
import argparse
import yaml

import torch
from yoro.utils.train_util import \
    YOROTrain, RotAnchorTrain, RotClassifierTrain, RotRegressorTrain


trainMode = {
    'yoro': YOROTrain,
    'rotanc': RotAnchorTrain,
    'rotcls': RotClassifierTrain,
    'rotreg': RotRegressorTrain
}


if __name__ == '__main__':

    # Append current working directory to path for custom backbone
    sys.path.append(os.getcwd())

    # Parse arguments
    argp = argparse.ArgumentParser(description='General Trainer')

    argp.add_argument('config', type=str, help='Configuration file path')

    trainOpt = argp.add_argument_group('Optional trainer arguments')
    trainOpt.add_argument('--no-restore', action='store_true',
                          help='Disable auto restore training progress from backups.')
    trainOpt.add_argument('--no-export', action='store_true',
                          help='Disable model exporting at the end of training progress.')
    trainOpt.add_argument('--pretrain', default=None,
                          help='Training with given pretrained weight. '
                          'If auto restoring is performed, '
                          'the pretrained weight will be overwritten with backup weight.')

    args = argp.parse_args()

    # Task setup
    config = args.config
    cfg = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)

    TrainClass = trainMode.get(cfg['mode'], None)
    if TrainClass is None:
        print('Invalid training mode:', cfg['mode'])
        exit()

    autoRestore = not args.no_restore
    autoExport = not args.no_export

    # Initialize training handler
    tc = TrainClass(config)

    # Load pretrained weight
    if args.pretrain is not None:
        sDict = torch.load(
            args.pretrain, map_location=tc.dev)['bbone_state_dict']
        tc.backbone.load_state_dict(sDict)

    # Perform auto restoring
    if autoRestore:
        tc.restore()

    # Run training progress
    tc.valid()
    tc.train()

    # Export model
    if autoExport:
        tc.export_model()
