import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ...datasets import RBoxSample, rbox_collate_fn
from ...transforms import \
    Rot_ColorJitter, Rot_RandomAffine, Rot_Resize, Rot_ToTensor, Rot_PadToAspect
from ...layers import RotRegressor, RotClassifier, RotAnchor
from ..object_loader import load_object

from .base_train import BaseTrain


class RotLayerTrain(BaseTrain):

    def __init__(self, config_path, rot_module, rot_arg_lambda):

        cfg = super(RotLayerTrain, self).__init__(config_path)

        # Get child config
        cfgCons = cfg['construct']
        cfgTParam = cfg['train_param']

        # Get network input size
        height = cfgCons['height']
        width = cfgCons['width']

        # Configure data augmentation
        tfPrefix = [Rot_PadToAspect(float(width) / height)]
        tfSuffix = [Rot_Resize((height, width)),
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
            num_workers=cfgTParam['num_workers'],
            pin_memory=cfgTParam['pin_memory'])
        self.tstLoader = DataLoader(
            validSet, shuffle=False, collate_fn=rbox_collate_fn,
            batch_size=cfgTParam['batch'],
            num_workers=cfgTParam['num_workers'],
            pin_memory=cfgTParam['pin_memory'])

        # Configure backbone
        cfgBBone = cfgCons['backbone']
        self.bboneClass = load_object(cfgBBone['name'])
        self.bboneArgs = cfgBBone['args']
        self.backbone = self.bboneClass(**self.bboneArgs).to(self.dev)

        # Configure rotation layer
        src = torch.randn(1, 3, height, width)
        out = torch.flatten(self.backbone(src.to(self.dev)), 1)

        self.suffixClass = rot_module
        self.suffixArgs = {
            'in_features': out.size(1),
            'width': width,
            'height': height,
            **rot_arg_lambda(cfgCons)
        }

        self.suffix = self.suffixClass(**self.suffixArgs).to(self.dev)

        # Configure optimizer
        cfgOptim = cfgTParam['optimizer']
        self.optimizer = optim.__dict__[cfgOptim['name']](
            [{'params': self.backbone.parameters()},
                {'params': self.suffix.parameters()}],
            **cfgOptim['args'])

        # Configure model KPI
        self.modelKpi = ['corr']


class RotRegressorTrain(RotLayerTrain):

    def __init__(self, config_path):

        super(RotRegressorTrain, self).__init__(
            config_path,
            RotRegressor,
            lambda cfgCons: {
                'deg_min': cfgCons['deg_min'],
                'deg_max': cfgCons['deg_max']
            }
        )


class RotClassifierTrain(RotLayerTrain):

    def __init__(self, config_path):

        super(RotClassifierTrain, self).__init__(
            config_path,
            RotClassifier,
            lambda cfgCons: {
                'deg_min': cfgCons['deg_min'],
                'deg_max': cfgCons['deg_max'],
                'deg_step': cfgCons['deg_step']
            }
        )


class RotAnchorTrain(RotLayerTrain):

    def __init__(self, config_path):

        super(RotAnchorTrain, self).__init__(
            config_path,
            RotAnchor,
            lambda cfgCons: {
                'deg_min': cfgCons['deg_min'],
                'deg_max': cfgCons['deg_max'],
                'deg_part_size': cfgCons['deg_part_size']
            }
        )
