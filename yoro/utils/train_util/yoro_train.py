import yaml
from tqdm.auto import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ...datasets import RBoxSample, rbox_collate_fn
from ...transforms import \
    RBox_ColorJitter, RBox_RandomAffine, RBox_Resize, RBox_PadToSquare, RBox_ToTensor
from ... import backbone
from ...layers import YOROLayer
from ..info_summarize import info_add, info_simplify, info_represent


class YOROTrain(object):

    def __init__(self, config_path):

        # Load config
        cfg = yaml.load(
            open(config_path, 'r'), Loader=yaml.FullLoader)
        self.name = cfg['name']

        # Set training device
        devStr = cfg['device']
        if devStr == 'auto':
            self.dev = torch.device('cuda') \
                if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.dev = torch.device(devStr)

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
        self.bboneArgs = cfgBBone['args']
        self.backbone = backbone.__dict__[cfgBBone['name']](**self.bboneArgs)
        self.backbone = self.backbone.to(self.dev)

        # Configure yoro layer
        height = cfgCons['height']
        width = cfgCons['width']

        src = torch.randn(1, 3, height, width)
        out = self.backbone(src.to(self.dev))
        fmapSize = out.size()

        self.yoroArgs = {
            'in_channels': fmapSize[1],
            'num_classes': trainSet.numClasses,
            'width': width,
            'height': height,
            'fmap_width': fmapSize[2],
            'fmap_height': fmapSize[3],
            'anchor': cfgCons['anchor'],
            'deg_min': cfgCons['deg_min'],
            'deg_max': cfgCons['deg_max'],
            'deg_part_size': cfgCons['deg_part_size']
        }

        self.yoroLayer = YOROLayer(**self.yoroArgs).to(self.dev)
        self.yoroLayer = self.yoroLayer.to(self.dev)

        # Configure optimizer
        cfgOptim = cfgTParam['optimizer']
        self.optimizer = optim.__dict__[cfgOptim['name']](
            [{'params': self.backbone.parameters()},
                {'params': self.yoroLayer.parameters()}],
            **cfgOptim['args'])

        # Training setting
        self.maxEpoch = cfgTParam['max_epoch']
        self.estiEpoch = cfgTParam['esti_epoch']
        self.bakEpoch = cfgTParam['bak_epoch']

        # Iterating index
        self.epoch = 0

        # Training log
        self.trainLog = {}
        self.validLog = {}

    def train(self, saveLog=True):

        while self.epoch < self.maxEpoch:

            runInfo = None
            runLoss = None

            loop = tqdm(self.traLoader)
            for inst in loop:

                # Get inputs and targets
                inputs, targets = inst
                inputs = inputs.to(self.dev)

                # Zero gradients
                self.optimizer.zero_grad()

                # Training on mini-batch
                out = self.backbone(inputs)
                loss, info = self.yoroLayer.loss(out, targets)
                loss[0].backward()
                self.optimizer.step()

                # Estimating
                runInfo = info_add(runInfo, info)
                runLoss = info_add(runLoss, loss)

                # Show training message
                loop.set_description('Epoch %d/%d' %
                                     (self.epoch + 1, self.maxEpoch))
                loop.set_postfix_str('loss: %s, %s' % (
                    info_represent(runLoss),
                    info_represent(runInfo)
                ))

            # Logging
            runLoss = info_simplify(runLoss)
            runInfo = info_simplify(runInfo)

            if saveLog:
                self.trainLog[self.epoch + 1] = {
                    'loss': runLoss,
                    'info': runInfo
                }

            # Increase iterating index
            self.epoch = self.epoch + 1
