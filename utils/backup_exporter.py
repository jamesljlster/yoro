#!@Python_EXECUTABLE@

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

    # Parse arguments
    argp = argparse.ArgumentParser(
        description='Export model from given config and backup',
        formatter_class=argparse.RawTextHelpFormatter
    )

    argp.add_argument('mode', metavar='mode', type=str,
                      choices=trainMode.keys(), help=modeHelp)
    argp.add_argument('config', type=str, help='Configuration file path')
    argp.add_argument('backup', type=str, help='Backup file path')

    trainOpt = argp.add_argument_group('Optional exporter arguments')
    trainOpt.add_argument('--out-name', type=str, default=None,
                          help='Output model name')

    args = argp.parse_args()

    # Export model from backup
    TrainClass = trainMode.get(args.mode, None)
    tc = TrainClass(args.config)
    tc.restore(args.backup)
    tc.export_model(args.out_name)
