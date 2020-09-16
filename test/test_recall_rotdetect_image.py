import sys
import math
import torch
import cv2 as cv
import numpy as np


netWidth = 224
netHeight = 224

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: %s <model_path> <image_path>' % sys.argv[0])
        exit()

    modelPath = sys.argv[1]
    imgPath = sys.argv[2]

    # Load model and image
    model = torch.jit.load(modelPath)
    model.eval()

    img = cv.imread(imgPath, cv.IMREAD_COLOR)

    # Resize image and permute axis
    mat = cv.resize(img, (netWidth, netHeight))
    mat = cv.cvtColor(mat, cv.COLOR_BGR2RGB)
    mat = mat.transpose(2, 0, 1)

    # Predict
    inputs = (torch.FloatTensor(mat) / 255).unsqueeze(0)
    outputs = model(inputs)
    degree = -outputs[0].item()

    # Draw result
    print('Degree:', degree)
    theta = math.radians(degree)

    # Draw result
    imgHeight, imgWidth, _ = img.shape

    sinVal = math.sin(theta)
    cosVal = math.cos(theta)
    rotMat = np.float32([
        [cosVal, -sinVal],
        [sinVal, cosVal]
    ])

    dirVec = np.zeros((1, 2), dtype=np.float32)
    dirVec[0, 0] = 0
    dirVec[0, 1] = -(imgWidth + imgHeight) * 3 / 16

    cx = int(imgWidth / 2)
    cy = int(imgHeight / 2)
    dirVec = np.matmul(dirVec, rotMat)
    dirVec = dirVec + np.float32([cx, cy])

    cv.arrowedLine(img, (cx, cy), (int(
        dirVec[0, 0]), int(dirVec[0, 1])), (255, 0, 0), 2)

    cv.imshow('Rotate Detection', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
