import yaml
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets import RBoxSample, rbox_collate_fn
from ..transforms import \
    RBox_ColorJitter, RBox_RandomAffine, RBox_Resize, RBox_PadToSquare, RBox_ToTensor
from .. import backbone


class TrainClass(object):

    def __init__(self, config_path):

        # Load config
        cfg = yaml.load(
            open(config_path, 'r'), Loader=yaml.FullLoader)
        self.name = cfg['name']

        # Get child config
        cfgCons = cfg['construct']
        cfgTParam = cfg['train_param']

        # Configure data augmentation
        tfPrefix = [RBox_PadToSquare()]
        tfSuffix = [RBox_Resize((cfgCons['height'], cfgCons['width'])),
                    RBox_ToTensor()]

        cfgTf = cfg['transform']
        tfContent = [
            RBox_ColorJitter(
                brightness=cfgTf['brightness'],
                contrast=cfgTf['contrast'],
                saturation=cfgTf['saturation'],
                hue=cfgTf['hue']),
            RBox_RandomAffine(
                degrees=cfgTf['degrees'],
                translate=cfgTf['translate'],
                scale=cfgTf['scale'])
        ]

        transform = Compose(tfPrefix + tfContent + tfSuffix)

        # Configure dataset
        cfgData = cfg['dataset']
        trainSet = RBoxSample(
            cfgData['train_dir'], cfgData['names_file'], transform=transform)
        validSet = RBoxSample(
            cfgData['valid_dir'], cfgData['names_file'], transform=transform)

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
        self.backbone = backbone.__dict__[cfgBBone['name']](**cfgBBone['args'])
