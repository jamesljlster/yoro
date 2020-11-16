#!@Python_EXECUTABLE@

import os
import sys
import argparse
from yoro.utils.train_util import \
    YOROTrain, RotAnchorTrain, RotClassifierTrain, RotRegressorTrain


trainMode = {
    'yoro': YOROTrain,
    'rotanc': RotAnchorTrain,
    'rotcls': RotClassifierTrain,
    'rotreg': RotRegressorTrain
}


modeHelp = '''\
Training mode corresponding to the given config.
    yoro:    Train a rotated object detector.
    rotanc:  Train a rotation detector with anchor encoding.
    rotcls:  Train a rotation detector with class encoding.
    rotreg:  Train a rotation detector with regression encoding.
'''


if __name__ == '__main__':

    # Append current working directory to path for custom backbone
    sys.path.append(os.getcwd())

    # Parse arguments
    argp = argparse.ArgumentParser(
        description='General Trainer',
        formatter_class=argparse.RawTextHelpFormatter
    )

    argp.add_argument('mode', metavar='mode', type=str,
                      choices=trainMode.keys(), help=modeHelp)
    argp.add_argument('config', type=str, help='Configuration file path')

    trainOpt = argp.add_argument_group('Optional trainer arguments')
    trainOpt.add_argument('--no-restore', action='store_true',
                          help='Disable auto restore training progress from backups')
    trainOpt.add_argument('--no-export', action='store_true',
                          help='Disable model exporting at the end of training progress')

    args = argp.parse_args()

    # Task setup
    TrainClass = trainMode.get(args.mode, None)
    config = args.config
    autoRestore = not args.no_restore
    autoExport = not args.no_export

    # Run training progress
    tc = TrainClass(config)
    if autoRestore:
        tc.restore()
    tc.valid()
    tc.train()
    if autoExport:
        tc.export_model()
