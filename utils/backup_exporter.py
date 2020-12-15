#!@Python_EXECUTABLE@

import argparse
import yaml

from yoro.utils.train_util import \
    YOROTrain, RotAnchorTrain, RotClassifierTrain, RotRegressorTrain


trainMode = {
    'yoro': YOROTrain,
    'rotanc': RotAnchorTrain,
    'rotcls': RotClassifierTrain,
    'rotreg': RotRegressorTrain
}


if __name__ == '__main__':

    # Parse arguments
    argp = argparse.ArgumentParser(
        description='Export model from given config and backup',
        formatter_class=argparse.RawTextHelpFormatter
    )

    argp.add_argument('config', type=str, help='Configuration file path')
    argp.add_argument('backup', type=str, help='Backup file path')

    trainOpt = argp.add_argument_group('Optional exporter arguments')
    trainOpt.add_argument('--out-name', type=str, default=None,
                          help='Output model name')

    args = argp.parse_args()

    # Task setup
    config = args.config
    cfg = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)

    TrainClass = trainMode.get(cfg['mode'], None)
    if TrainClass is None:
        print('Invalid training mode:', args.mode)
        exit()

    # Export model from backup
    tc = TrainClass(args.config)
    tc.restore(args.backup)
    tc.export_model(args.out_name)
