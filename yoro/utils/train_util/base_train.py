import yaml
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict

from glob import glob
from os import makedirs
from os.path import join, exists, isdir, isfile, splitext, basename

import torch
from torch import optim
from torch.nn import Sequential
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import Compose

from ...datasets import RBoxSample, rbox_collate_fn
from ...transforms import \
    Rot_ColorJitter, Rot_RandomAffine, Rot_Resize, Rot_ToTensor
from ..info_summarize import info_moving_avg, info_add, info_simplify, info_represent
from ..object_loader import load_object


def kpi_compare(old, new):

    greater = False
    for (ko, kn) in zip(old, new):
        if kn > ko:
            greater = True
        elif kn < ko:
            break

    return greater


class BaseEvaluator(object):

    def post_process(self, preds):
        return None

    def evaluate(self, dts_gts):
        return None


class BaseTrain(object):

    def __init__(self, config_path, **derived_kwargs):

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
        cfgTParam = cfg['train_param']

        # Training setting
        self.maxEpoch = cfgTParam['max_epoch']
        self.estiEpoch = cfgTParam['esti_epoch']
        self.bakEpoch = cfgTParam['bak_epoch']

        self.movingFactor = cfgTParam.get('moving_factor', 0.01)
        self.trainUnits = cfgTParam.get('train_units', 0)
        if self.trainUnits <= 0:
            self.trainUnits = min(self.estiEpoch, self.bakEpoch)

        batch = cfgTParam['batch']
        subdivision = cfgTParam.get('subdivision', 1)
        if batch % subdivision:
            raise ValueError(
                'Invalid subdivision or batch size. '
                'Batch size should be divisible by subdivision.')

        self.batch = batch
        self.subdivision = subdivision
        self.subbatch = int(batch / subdivision)
        self.fedCount = 0

        # Iterating index
        self.epoch = 0
        self.epochStart = 0

        # Evaluator callback
        self.evaluator = None

        # Training log
        self.trainLog = {}
        self.validLog = {}

        # Best model state
        self.modelKpi = []
        self.bestKpi = None
        self.bestState = None

        # Check backup folder
        self.bakDir = self.name + '.backup'
        if exists(self.bakDir):
            if not isdir(self.bakDir):
                raise FileExistsError(
                    '\'{}\' exists and it is not a directory'.format(self.bakDir))
        else:
            makedirs(self.bakDir)

        # Initialization for derived classes
        self.derived_init(cfg, **derived_kwargs)

        # Configure optimizer
        cfgOptim = cfgTParam['optimizer']
        self.optimizer = optim.__dict__[cfgOptim['name']](
            [{'params': self.backbone.parameters()},
             {'params': self.suffix.parameters()}],
            **cfgOptim['args'])

        # Gradient clipping
        self.gradClipNorm = cfgTParam.get('grad_clip_norm', None)
        if self.gradClipNorm is not None:
            print('Gradient is clipping by max norm of', self.gradClipNorm)
        else:
            print('Gradient clipping is disabled')

        # Configure learning rate scheduler
        cfgSched = cfgTParam.get('lr_scheduler', None)
        if cfgSched is not None:
            print('Learning rate scheduler is enabled')
            schedClass = load_object(cfgSched['name'])
            self.scheduler = schedClass(self.optimizer, **cfgSched['args'])
        else:
            print('Learning rate scheduler is disabled')
            self.scheduler = None

    def derived_init(self, config, **derived_kwargs):
        raise RuntimeError('derived_init is not implemented by derived class!')

    def train(self, saveLog=True):

        # Change to train mode
        self.backbone.train()
        self.suffix.train()

        while self.epoch < self.maxEpoch:

            runInfo = None
            runLoss = None

            loop = tqdm(self.traLoader)
            for inst in loop:

                # Get inputs and targets
                inputs, targets = inst
                inputs = inputs.to(self.dev)

                # Accumulate gradients
                out = self.backbone(inputs)
                loss, info = self.suffix.loss(out, targets)
                loss[0].backward()

                self.fedCount += inputs.size(0)

                # Update weight
                if self.fedCount >= self.batch:

                    if self.gradClipNorm is not None:
                        clip_grad_norm_(
                            self.backbone.parameters(), self.gradClipNorm)
                        clip_grad_norm_(
                            self.suffix.parameters(), self.gradClipNorm)

                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.optimizer.zero_grad()
                    self.fedCount = 0

                # Estimating
                runInfo = info_moving_avg(runInfo, info, self.movingFactor)
                runLoss = info_moving_avg(runLoss, loss, self.movingFactor)

                # Show training message
                if self.trainUnits > 1:
                    desc = 'Epoch [%d-%d]/%d' % (
                        self.epoch + 1, self.epoch + self.trainUnits, self.maxEpoch)
                else:
                    desc = 'Epoch %d/%d' % (self.epoch + 1, self.maxEpoch)

                loop.set_description(desc)
                loop.set_postfix_str('loss: %s, %s' % (
                    info_represent(info_simplify(runLoss)),
                    info_represent(info_simplify(runInfo))
                ))

            # Logging
            lossInfo = info_simplify(runLoss)
            estiInfo = info_simplify(runInfo)
            if saveLog:
                self.trainLog[self.epoch + 1] = {
                    'loss': lossInfo,
                    'info': estiInfo
                }

            # Increase iterating index
            self.epoch = self.epoch + self.trainUnits

            # Validating
            if (self.epoch - self.epochStart) % self.estiEpoch == 0:
                validLoss, validEsti = self.valid(saveLog=saveLog)

                # Compare kpi and save best weight
                kpiCmp = [validEsti.get(item) for item in self.modelKpi]
                kpiCmp += [-validLoss, -lossInfo]

                newBest = False
                if self.bestKpi is None:
                    newBest = True
                elif kpi_compare(self.bestKpi, kpiCmp):
                    newBest = True

                if newBest:
                    self.bestKpi = kpiCmp
                    self.bestState = deepcopy(self.model_state_dict())

                    print('New best weight was found!')
                    print()

            # Backup
            if (self.epoch - self.epochStart) % self.bakEpoch == 0:
                self.backup()

    def valid(self, saveLog=False):

        # Change to evaluate mode
        self.backbone.eval()
        self.suffix.eval()

        # Show message header
        print()
        print('=== Validation on Epoch %d ===' % self.epoch)

        # Estimating
        runInfo = None
        runLoss = None

        if self.evaluator is not None:
            predPair = []
        else:
            predPair = None

        loop = tqdm(self.tstLoader, leave=False)
        for inst in loop:

            # Get inputs and targets
            inputs, targets = inst
            inputs = inputs.to(self.dev)

            # Forward
            with torch.no_grad():

                out = self.backbone(inputs)
                loss, info = self.suffix.loss(out, targets)

                if self.evaluator is not None:
                    preds = self.suffix(out)
                    preds = self.evaluator.post_process(preds)
                    predPair.append((preds, targets))

            # Accumulate informations
            runInfo = info_add(runInfo, info)
            runLoss = info_add(runLoss, loss)

        # Evaluationg with prediction and ground truth
        evalInfo = None
        if self.evaluator is not None:
            evalInfo = self.evaluator.evaluate(predPair)

        # Show message
        lossInfo = info_simplify(runLoss)
        estiInfo = info_simplify(runInfo)
        if evalInfo is not None:
            estiInfo = {**evalInfo, **estiInfo}

        print('Loss: %s' % info_represent(lossInfo))
        print('Info: %s' % info_represent(estiInfo))
        print()

        # Logging
        if saveLog:
            self.validLog[self.epoch] = {
                'loss': lossInfo,
                'info': estiInfo
            }

        return lossInfo, estiInfo

    def model_state_dict(self):
        sDict = {
            'backbone': self.backbone.state_dict(),
            'suffix': self.suffix.state_dict(),
        }
        return sDict

    def apply_best_state(self):
        if self.bestState is None:
            print('No best state is avaliable!')
            return
        else:
            self.backbone.load_state_dict(self.bestState['backbone'])
            self.suffix.load_state_dict(self.bestState['suffix'])

    def backup(self):

        # Make backup
        bakName = 'epoch_' + str(self.epoch) + '.sdict'
        bakPath = join(self.bakDir, bakName)

        bakData = {
            'epoch': self.epoch,

            'model_state_dict': self.model_state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),

            'best_kpi': self.bestKpi,
            'best_state': self.bestState,

            'trainLog': self.trainLog,
            'validLog': self.validLog
        }

        if self.scheduler is not None:
            bakData['scheduler'] = self.scheduler.state_dict()

        print('Backup to:', bakPath)
        torch.save(bakData, bakPath)

    def restore(self, path=None):

        # Auto selection
        if path == None:
            bakFiles = [f for f in glob(
                join(self.bakDir, '*.sdict')) if isfile(f)]
            if len(bakFiles) == 0:
                print('No backup files were found for %s' % self.name)
                return

            selBase = [int(splitext(basename(f))[0].split('_')[1])
                       for f in bakFiles]
            path = bakFiles[selBase.index(max(selBase))]

        # Load backup
        print('Restore from:', path)
        bak = torch.load(path, map_location=self.dev)

        # Load parameters
        self.epoch = bak['epoch']
        self.epochStart = self.epoch

        modelState = bak['model_state_dict']
        self.backbone.load_state_dict(modelState['backbone'])
        self.suffix.load_state_dict(modelState['suffix'])
        self.optimizer.load_state_dict(bak['optim_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(bak['scheduler'])

        self.bestKpi = bak['best_kpi']
        self.bestState = bak['best_state']

        self.trainLog = bak['trainLog']
        self.validLog = bak['validLog']

    def export_model(self, best_state=True, path=None):

        # Auto selection
        if path == None:
            path = '_'.join([self.name, 'epoch', str(self.epoch)]) + '.zip'

        # Compose sequential model for torchscript
        model = torch.jit.script(Sequential(OrderedDict([
            ('backbone', self.bboneClass(**self.bboneArgs)),
            ('suffix', self.suffixClass(**self.suffixArgs))
        ])))

        if best_state and (self.bestState is not None):
            model.backbone.load_state_dict(self.bestState['backbone'])
            model.suffix.load_state_dict(self.bestState['suffix'])
        else:
            model.backbone.load_state_dict(self.backbone.state_dict())
            model.suffix.load_state_dict(self.suffix.state_dict())

        # Save model
        print('Export model to:', path)
        model.save(path)

    def import_model(self, path):

        # Load model
        print('Import model from:', path)
        model = torch.jit.load(path)

        # Apply state dict
        self.backbone.load_state_dict(model.backbone.state_dict())
        self.suffix.load_state_dict(model.suffix.state_dict())
