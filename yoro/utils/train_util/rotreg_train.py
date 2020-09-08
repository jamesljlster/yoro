import yaml
from tqdm.auto import tqdm
from collections import OrderedDict

from glob import glob
from os import makedirs
from os.path import join, exists, isdir, isfile, splitext, basename

import torch
from torch import optim
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ...datasets import RBoxSample, rbox_collate_fn
from ...transforms import \
    Rot_ColorJitter, Rot_RandomAffine, Rot_Resize, Rot_ToTensor
from ...layers import RotRegressor
from ..info_summarize import info_add, info_simplify, info_represent
from ..object_loader import load_object

from .base_train import BaseTrain


class RotRegressorTrain(BaseTrain):

    def __init__(self, config_path):

        cfg = super(RotRegressorTrain, self).__init__(config_path)

        # Get child config
        cfgCons = cfg['construct']
        cfgTParam = cfg['train_param']

        # Configure data augmentation
        tfPrefix = []
        tfSuffix = [Rot_Resize((cfgCons['height'], cfgCons['width'])),
                    Rot_ToTensor()]

        cfgTf = cfg['transform']
        tfContent = [
            Rot_ColorJitter(
                brightness=cfgTf['brightness'],
                contrast=cfgTf['contrast'],
                saturation=cfgTf['saturation'],
                hue=cfgTf['hue']),
            Rot_RandomAffine(
                translate=cfgTf['translate'],
                scale=cfgTf['scale'])
        ]

        tfTrain = Compose(tfPrefix + tfContent + tfSuffix)
        tfValid = tfTrain if cfgTf['apply_on_valid'] \
            else Compose(tfPrefix + tfSuffix)

        # Configure dataset
        cfgData = cfg['dataset']
        trainSet = RBoxSample(
            cfgData['train_dir'], cfgData['names_file'], transform=tfTrain)
        validSet = RBoxSample(
            cfgData['valid_dir'], cfgData['names_file'], transform=tfValid)

        self.traLoader = DataLoader(
            trainSet, shuffle=True, collate_fn=rbox_collate_fn,
            batch_size=cfgTParam['batch'],
            num_workers=cfgTParam['num_workers'])
        self.tstLoader = DataLoader(
            validSet, shuffle=False, collate_fn=rbox_collate_fn,
            batch_size=cfgTParam['batch'],
            num_workers=cfgTParam['num_workers'])

        # Configure backbone
        cfgBBone = cfgCons['backbone']
        self.bboneClass = load_object(cfgBBone['name'])
        self.bboneArgs = cfgBBone['args']
        self.backbone = self.bboneClass(**self.bboneArgs).to(self.dev)

        # Configure rotation regressor
        self.suffixClass = RotRegressor
        self.suffixArgs = {
            'deg_min': cfgCons['deg_min'],
            'deg_max': cfgCons['deg_max']
        }

        self.suffix = self.suffixClass(**self.suffixArgs).to(self.dev)

        # Configure optimizer
        cfgOptim = cfgTParam['optimizer']
        self.optimizer = optim.__dict__[cfgOptim['name']](
            [{'params': self.backbone.parameters()},
                {'params': self.suffix.parameters()}],
            **cfgOptim['args'])


if __name__ == '__main__':

    tc = RotRegressorTrain('config/rotation_regressor.yaml')
    tc.restore()
    tc.valid()
    tc.train()
    tc.export_model()
