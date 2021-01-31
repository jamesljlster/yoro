#!@Python_EXECUTABLE@

import argparse
import numpy as np
from tqdm import tqdm

from yoro.api import DeviceType, YORODetector
from yoro.datasets import RBoxSample
from yoro.utils.train_util.yoro_train import YOROEvaluator

device = {
    'auto': DeviceType.Auto,
    'cuda': DeviceType.CUDA,
    'cpu': DeviceType.CPU
}

if __name__ == '__main__':

    # Parse arguments
    argp = argparse.ArgumentParser(description='YORO mAP Evaluator')

    argp.add_argument('model', type=str, help='Exported model path')
    argp.add_argument('dataset', type=str, help='Dataset path')

    opt = argp.add_argument_group('Optional arguments')
    opt.add_argument('--device', type=str,
                     choices=device.keys(), default='auto',
                     help='Running on specific device. (default: %(default)s)')
    opt.add_argument('--conf', type=float, default=0.7,
                     help=('Confidence filtering threshold. ' +
                           '(default: %(default)s)'))
    opt.add_argument('--nms', type=float, default=0.9,
                     help=('NMS filtering threshold. ' +
                           '(default: %(default)s)'))
    opt.add_argument('--sim', type=float, default=0.5,
                     help='Rotated bounding box similarity threshold')

    args = argp.parse_args()

    # Load model and dataset
    detector = YORODetector(args.model, device.get(args.device, None))
    dataset = RBoxSample(args.dataset)

    # Perform evaluation
    evaluator = YOROEvaluator(
        dataset.numClasses, args.conf, args.nms, args.sim)

    predPair = []
    for image, target in tqdm(dataset):

        image = np.array(image)[..., ::-1]
        preds = detector.detect(image, args.conf, args.nms)
        predPair.append((preds, target))

    print()
    print('=== Evaluation Result ===')
    print('mAP:', evaluator.evaluate(predPair)['mAP'])
