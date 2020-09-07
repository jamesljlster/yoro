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


class RotRegressorTrain(object):

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
        self.regArgs = {
            'deg_min': cfgCons['deg_min'],
            'deg_max': cfgCons['deg_max']
        }

        self.regLayer = RotRegressor(**self.regArgs).to(self.dev)

        # Configure optimizer
        cfgOptim = cfgTParam['optimizer']
        self.optimizer = optim.__dict__[cfgOptim['name']](
            [{'params': self.backbone.parameters()},
                {'params': self.regLayer.parameters()}],
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

        # Check backup folder
        self.bakDir = self.name + '.backup'
        if exists(self.bakDir):
            if not isdir(self.bakDir):
                raise FileExistsError(
                    '\'{}\' exists and it is not a directory'.format(self.bakDir))
        else:
            makedirs(self.bakDir)

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
                loss, info = self.regLayer.loss(out, targets)
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
            if saveLog:
                self.trainLog[self.epoch + 1] = {
                    'loss': info_simplify(runLoss),
                    'info': info_simplify(runInfo)
                }

            # Increase iterating index
            self.epoch = self.epoch + 1

            # Validating
            if self.epoch % self.estiEpoch == 0:
                self.valid(saveLog=saveLog)

            # Backup
            if self.epoch % self.bakEpoch == 0:
                self.backup()

    def valid(self, saveLog=False):

        # Change to evaluate mode
        self.backbone.eval()
        self.regLayer.eval()

        # Show message header
        print()
        print('=== Validation on Epoch %d ===' % self.epoch)

        # Estimating
        runInfo = None
        runLoss = None

        loop = tqdm(self.tstLoader, leave=False)
        for inst in loop:

            # Get inputs and targets
            inputs, targets = inst
            inputs = inputs.to(self.dev)

            # Forward
            out = self.backbone(inputs)
            loss, info = self.regLayer.loss(out, targets)

            # Accumulate informations
            runInfo = info_add(runInfo, info)
            runLoss = info_add(runLoss, loss)

        # Show message
        print('Loss: %s' % info_represent(runLoss))
        print('Info: %s' % info_represent(runInfo))
        print()

        # Logging
        if saveLog:
            self.validLog[self.epoch] = {
                'loss': info_simplify(runInfo),
                'info': info_simplify(runLoss)
            }

    def backup(self):

        # Make backup
        bakName = 'epoch_' + str(self.epoch) + '.sdict'
        bakPath = join(self.bakDir, bakName)

        print('Backup to:', bakPath)
        torch.save({
            'epoch': self.epoch,

            'bbone_state_dict': self.backbone.state_dict(),
            'yoro_state_dict': self.regLayer.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),

            'trainLog': self.trainLog,
            'validLog': self.validLog
        }, bakPath)

    def restore(self, path=None):

        # Auto selection
        if path == None:
            bakFiles = \
                [f for f in glob(join(self.bakDir, '*.sdict')) if isfile(f)]
            selBase = \
                [int(splitext(basename(f))[0].split('_')[1]) for f in bakFiles]
            path = bakFiles[selBase.index(max(selBase))]

        # Load backup
        print('Restore from:', path)
        bak = torch.load(path)

        # Load parameters
        self.epoch = bak['epoch']

        self.backbone.load_state_dict(bak['bbone_state_dict'])
        self.regLayer.load_state_dict(bak['yoro_state_dict'])
        self.optimizer.load_state_dict(bak['optim_state_dict'])

        self.trainLog = bak['trainLog']
        self.validLog = bak['validLog']

    def export_model(self, path=None):

        # Auto selection
        if path == None:
            path = '_'.join([self.name, 'epoch', str(self.epoch)]) + '.zip'

        # Compose sequential model for torchscript
        model = torch.jit.script(Sequential(OrderedDict([
            ('backbone', self.bboneClass(**self.bboneArgs)),
            ('regLayer', YOROLayer(**self.yoroArgs))
        ])))

        model.backbone.load_state_dict(self.backbone.state_dict())
        model.regLayer.load_state_dict(self.regLayer.state_dict())

        # Save model
        print('Export model to:', path)
        model.save(path)

    def import_model(self, path):

        # Load model
        print('Import model from:', path)
        model = torch.jit.load(path)

        # Apply state dict
        self.backbone.load_state_dict(model.backbone.state_dict())
        self.regLayer.load_state_dict(model.regLayer.state_dict())


if __name__ == '__main__':

    tc = RotRegressorTrain('config/rotation_regressor.yaml')
    tc.valid()
    tc.train()
    tc.export_model()
