import math

import numpy as np
import cv2 as cv

import torch
from torch import Tensor

from . import api


def rbox_draw(images, annos, class_names=None, to_tensor=False):

    # Convert source
    cvtImages = []
    for image in images:

        # Type convert and check
        if isinstance(image, torch.Tensor):
            image = image.cpu().permute(1, 2, 0).numpy()
        if not isinstance(image, np.ndarray):
            raise ValueError(
                'Unsupported image data type for drawing: %s' % str(type(image)))

        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255).astype(np.uint8)

        cvtImages.append(image.copy())

    # Draw annotations
    retImages = []
    for (mat, anno) in zip(cvtImages, annos):

        # Draw rotated bounding box
        for inst in anno:

            if isinstance(inst, api.RBox):
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

            # Calculate points and vectors for drawing
            origMat = np.float32([x, y])

            pts = np.zeros((4, 2), dtype=np.float32)
            pts[0] = np.matmul(np.float32([-w / 2, -h / 2]), rotMat) + origMat
            pts[1] = np.matmul(np.float32([+w / 2, -h / 2]), rotMat) + origMat
            pts[2] = np.matmul(np.float32([+w / 2, +h / 2]), rotMat) + origMat
            pts[3] = np.matmul(np.float32([-w / 2, +h / 2]), rotMat) + origMat

            def to_unit_vector(vec):
                return vec / np.linalg.norm(vec)

            wVec = to_unit_vector(pts[1] - pts[0])
            hVec = to_unit_vector(pts[1] - pts[2])
            arrowCtr = (pts[0] + pts[1]) / 2

            # Drawing
            def _draw_rotate_bbox(
                    scalar, rbox_thickness, arrow_side_len, font_scalar, font_thickness):

                if font_scalar is None:
                    font_scalar = scalar

                # Draw rotate bounding box
                cv.polylines(
                    mat, np.int32([pts]), True, scalar, rbox_thickness, lineType=cv.LINE_AA)

                # Draw arrow
                arrowPts = np.zeros((3, 2), dtype=np.float32)
                arrowPts[0] = arrowCtr + wVec * arrow_side_len
                arrowPts[1] = arrowCtr - wVec * arrow_side_len
                arrowPts[2] = arrowCtr + hVec * arrow_side_len
                cv.fillPoly(
                    mat, np.int32([arrowPts]), scalar, lineType=cv.LINE_AA)

                # Draw text
                text = str(label)
                if class_names is not None:
                    text += ': %s' % class_names[label]
                fontFace = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 1

                (textWidth, textHeight), _ = cv.getTextSize(
                    text, fontFace, fontScale, 1)

                cv.putText(
                    mat, text,
                    (int(origMat[0] - textWidth / 2),
                     int(origMat[1] + textHeight / 2)),
                    fontFace, fontScale, font_scalar, font_thickness,
                    lineType=cv.LINE_AA)

            _draw_rotate_bbox((255, 255, 255), 3, 9, None, 3)
            _draw_rotate_bbox((0, 0, 0), 1, 7, (255, 128, 128), 1)

        # Assign result
        retImages.append(mat)

    # Convert to tensor format
    if to_tensor:
        retImages = torch.tensor(retImages).permute(0, 3, 1, 2)

    return retImages
