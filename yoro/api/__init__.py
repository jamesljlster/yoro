import torch
import cv2

try:
    from .yoro_api_pym import *
except:
    raise AssertionError('YORO API Error: Missing or imcompatible binary')
