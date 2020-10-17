import torch
import math
import numpy as np
import cv2 as cv


__all__ = ['rbox_draw']


def rbox_draw(images, annos, to_tensor=False):

    # Convert source
    imType = type(images[0]).__name__
    if imType == 'Tensor':
        images = images.clone().permute(0, 2, 3, 1).numpy()
    else:
        cvtImages = []
        for image in images:
            cvtImages.append(np.array(image).copy())
        images = np.float32(cvtImages) / 255.0

    # Draw annotations
    for i in range(images.shape[0]):

        mat = cv.cvtColor(images[i], cv.COLOR_RGB2BGR)
        anno = annos[i]

        # Draw rotated bounding box
        for inst in anno:

            if inst.__class__.__name__ == 'RBox':
                inst = inst.to_dict()

            # Get instance detail
            label = inst['label']
            x = inst['x']
            y = inst['y']
            w = inst['w']
            h = inst['h']

            if 'degree' in inst.keys():
                th = inst['degree']
            else:
                th = 0

            # Get rotate matrix
            sinVal = math.sin(math.radians(th))
            cosVal = math.cos(math.radians(th))
            rotMat = np.float32([
                [cosVal, -sinVal],
                [sinVal, cosVal]
            ])

            # Draw rotate bounding box
            origMat = np.float32([x, y])

            pts = np.zeros((4, 2), dtype=np.float32)
            pts[0] = np.matmul(np.float32([-w / 2, -h / 2]), rotMat) + origMat
            pts[1] = np.matmul(np.float32([+w / 2, -h / 2]), rotMat) + origMat
            pts[2] = np.matmul(np.float32([+w / 2, +h / 2]), rotMat) + origMat
            pts[3] = np.matmul(np.float32([-w / 2, +h / 2]), rotMat) + origMat

            cv.polylines(mat, np.int32([pts]), 1,
                         (0, 0, 0), 2, lineType=cv.LINE_AA)

            # Draw arraw
            arrPt = np.matmul(np.float32(
                [0, float(-h) / 2.0]), rotMat) + origMat
            cv.line(mat, (int(x), int(y)), (int(arrPt[0]), int(arrPt[1])),
                    (0, 0, 0), 2, lineType=cv.LINE_AA)

            arrPt = np.matmul(np.float32(
                [float(w) / 2.0, 0]), rotMat) + origMat
            cv.line(mat, (int(x), int(y)), (int(arrPt[0]), int(arrPt[1])),
                    (0, 0, 0), 2, lineType=cv.LINE_AA)

            cv.putText(mat, str(label), (int(origMat[0] - 20), int(origMat[1] + 20)),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, lineType=cv.LINE_AA)

        # Assign result
        images[i] = cv.cvtColor(mat, cv.COLOR_BGR2RGB)

    # Convert to tensor format
    if to_tensor:
        images = torch.tensor(images).permute(0, 3, 1, 2)

    return images
