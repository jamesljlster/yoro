import argparse

import torch
import math

import numpy as np
import cv2 as cv

from yoro.api import DeviceType, YORODetector, RotationDetector
from yoro.visual import rbox_draw

mode = {
    'yoro': YORODetector,
    'rot': RotationDetector
}

modeHelp = '''\
Recall mode corresponding to the given model.
    yoro: Recall with rotated object detector.
    rot:  Recall with rotation detector.
'''

device = {
    'auto': DeviceType.Auto,
    'cuda': DeviceType.CUDA,
    'cpu': DeviceType.CPU
}


def recall(detector, args, img):
    if args.mode == 'yoro':
        return detector.detect(img, args.conf, args.nms)
    else:
        return detector.detect(img)


def draw_result(result, args, img):

    if args.mode == 'yoro':
        return rbox_draw([img], [result])[0]

    else:

        # Draw rotation vector
        height, width, _ = img.shape
        innerSize = min(height, width)
        ctrX = int(width / 2)
        ctrY = int(height / 2)

        rad = math.radians(result)
        sinVal = math.sin(rad)
        cosVal = math.cos(rad)
        rotMat = np.float32([
            [cosVal, -sinVal],
            [sinVal, cosVal]
        ])

        dirVec = np.zeros((1, 2), dtype=np.float32)
        dirVec[0, 0] = 0
        dirVec[0, 1] = -innerSize * 3 / 8

        dirVec = np.matmul(dirVec, rotMat)
        dirVec = (dirVec + np.float32([ctrX, ctrY])).astype(np.int)

        cv.arrowedLine(img, (ctrX, ctrY),
                       (dirVec[0, 0], dirVec[0, 1]),
                       (255, 0, 0), 2, cv.LINE_AA)

        return img


def get_detection_image(detector, args, img):
    result = recall(detector, args, img)
    return draw_result(result, args, img)


if __name__ == '__main__':

    # Parse arguments
    argp = argparse.ArgumentParser(
        description='General Recaller',
        formatter_class=argparse.RawTextHelpFormatter
    )

    argp.add_argument('mode', type=str, choices=mode.keys(), help=modeHelp)
    argp.add_argument('model', type=str, help='Exported model path')
    argp.add_argument('source', type=str, choices=['image', 'video'],
                      help='Use image or video as input source')
    argp.add_argument('file', type=str, nargs='+', help='Source files')

    opt = argp.add_argument_group('Optional recaller arguments')
    opt.add_argument('--device', type=str,
                     choices=device.keys(), default='auto',
                     help='Recall on specific device. (default: %(default)s)')
    opt.add_argument('--conf', type=float, default=0.7,
                     help=('Confidence filtering threshold. ' +
                           '(default: %(default)s, for YORO only)'))
    opt.add_argument('--nms', type=float, default=0.9,
                     help=('NMS filtering threshold. ' +
                           '(default: %(default)s, for YORO only)'))

    args = argp.parse_args()

    # Load model
    print('Loading %s model from:' % args.mode, args.model)
    detClass = mode.get(args.mode, None)
    detector = detClass(args.model, device.get(args.device, None))

    # Load sources
    for src in args.file:

        if args.source == 'image':

            print('Loading image from:', src)
            img = cv.imread(src, cv.IMREAD_COLOR)
            img = get_detection_image(detector, args, img)
            cv.imshow('Press ESC to terminate', img)
            if cv.waitKey(0) == 27:
                exit()

        else:

            print('Loading video from:', src)
            cap = cv.VideoCapture(src)
            while True:
                ret, img = cap.read()
                if not ret:
                    break

                img = get_detection_image(detector, args, img)
                cv.imshow('Press ESC to stop', img)
                if cv.waitKey(1) == 27:
                    cap.release()
                    break

    cv.destroyAllWindows()
