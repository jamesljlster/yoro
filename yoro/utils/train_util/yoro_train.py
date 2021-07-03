import yaml
import pprint

import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ... import ops
from ...datasets import RBoxSample, load_class_names
from ...transforms import \
    RBox_ColorJitter, RBox_RandomAffine, RBox_Resize, RBox_PadToAspect, RBox_ToTensor
from ...layers import YOROLayer
from ..object_loader import load_object

from .base_train import BaseTrain, BaseEvaluator


class YOROTrain(BaseTrain):

    def __init__(self, config_path):

        cfg = super(YOROTrain, self).__init__(config_path)

        # Get child config
        cfgCons = cfg['construct']
        cfgTParam = cfg['train_param']
        cfgData = cfg['dataset']

        # Get network input size
        height = cfgCons['height']
        width = cfgCons['width']

        # Configure backbone
        cfgBBone = cfgCons['backbone']
        self.bboneClass = load_object(cfgBBone['name'])
        self.bboneArgs = cfgBBone['args']
        self.backbone = self.bboneClass(**self.bboneArgs).to(self.dev)

        # Configure yoro layer
        numClasses = len(load_class_names(cfgData['names_file']))

        src = torch.randn(1, 3, height, width)
        out = self.backbone(src.to(self.dev))

        lossNorm = cfgTParam.get('loss_normalizer', {})
        if lossNorm != {}:
            pp = pprint.PrettyPrinter(indent=2)
            print('Using custom loss normalizer:')
            pp.pprint(lossNorm)
            print()

        self.suffixClass = YOROLayer
        self.suffixArgs = {
            'width': width,
            'height': height,
            'num_classes': numClasses,
            'input_shapes': [tensor.size() for tensor in out],
            'anchor': cfgCons['anchor'],
            'deg_min': cfgCons['deg_min'],
            'deg_max': cfgCons['deg_max'],
            'deg_part_size': cfgCons['deg_part_size'],
            'loss_norm': lossNorm
        }

        self.suffix = self.suffixClass(**self.suffixArgs).to(self.dev)

        # Configure data transformation
        tfPrefix = [RBox_PadToAspect(float(width) / height)]
        tfSuffix = [
            RBox_Resize((height, width)),
            RBox_ToTensor(
                self.suffix.anchorList,
                [tup[0].size()[-3:] for tup in self.suffix(out)],
                [torch.tensor([w, h])
                    for (w, h) in zip(self.suffix.gridWidth, self.suffix.gridHeight)],
                self.suffix.degAnchor[0].clone(),
                self.suffix.degValueScale)
        ]

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

        tfTrain = Compose(tfPrefix + tfContent + tfSuffix)
        tfValid = tfTrain if cfgTf['apply_on_valid'] \
            else Compose(tfPrefix + tfSuffix)

        # Configure dataset
        trainSet = RBoxSample(
            cfgData['train_dir'], cfgData['names_file'], transform=tfTrain,
            repeats=self.trainUnits)
        validSet = RBoxSample(
            cfgData['valid_dir'], cfgData['names_file'], transform=tfValid)

        # Collate function for data loader
        def rbox_tensor_collate(samples):

            # Target storage
            def _build_target_storage():
                return [[] for _ in range(self.suffix.headSize)]

            objs = _build_target_storage()
            batchT = _build_target_storage()

            acrIdxT = _build_target_storage()
            xIdxT = _build_target_storage()
            yIdxT = _build_target_storage()
            clsT = _build_target_storage()

            bboxT = _build_target_storage()
            degPartT = _build_target_storage()
            degShiftT = _build_target_storage()

            # Collate image and annotations
            images = []
            rawAnnos = []
            for batchIdx, (image, anno) in enumerate(samples):

                images.append(image)
                rawAnnos.append(anno[2])

                for headIdx in range(len(anno[0])):

                    objs[headIdx].append(anno[0][headIdx])
                    (acr, xIdx, yIdx, cls, bbox, degPart, degShift) = \
                        anno[1][headIdx]

                    instSize = acr.size(0)
                    batchT[headIdx].append(
                        torch.tensor([batchIdx] * instSize, dtype=torch.long))

                    acrIdxT[headIdx].append(acr)
                    xIdxT[headIdx].append(xIdx)
                    yIdxT[headIdx].append(yIdx)
                    clsT[headIdx].append(cls)
                    bboxT[headIdx].append(bbox)
                    degPartT[headIdx].append(degPart)
                    degShiftT[headIdx].append(degShift)

            images = torch.stack(images, 0)
            objs = [torch.stack(obj, 0) for obj in objs]

            def _gt_cat(gt_list, dtype=torch.float):
                return [torch.cat(tens) if len(tens) > 0 else torch.tensor([], dtype=dtype)
                        for tens in gt_list]

            batchT = _gt_cat(batchT, torch.long)
            acrIdxT = _gt_cat(acrIdxT, torch.long)
            xIdxT = _gt_cat(xIdxT, torch.long)
            yIdxT = _gt_cat(yIdxT, torch.long)
            clsT = _gt_cat(clsT, torch.long)
            bboxT = _gt_cat(bboxT)
            degPartT = _gt_cat(degPartT, torch.long)
            degShiftT = _gt_cat(degShiftT)

            targets = []
            for headIdx in range(self.suffix.headSize):
                targets.append((
                    batchT[headIdx],
                    acrIdxT[headIdx],
                    xIdxT[headIdx],
                    yIdxT[headIdx],
                    clsT[headIdx],
                    bboxT[headIdx],
                    degPartT[headIdx],
                    degShiftT[headIdx])
                )

            return images, (objs, targets, rawAnnos)

        # Configure data loader
        self.traLoader = DataLoader(
            trainSet, shuffle=True, collate_fn=rbox_tensor_collate,
            batch_size=self.subbatch,
            num_workers=cfgTParam['num_workers'],
            pin_memory=cfgTParam['pin_memory'])
        self.tstLoader = DataLoader(
            validSet, shuffle=False, collate_fn=rbox_tensor_collate,
            batch_size=self.subbatch,
            num_workers=cfgTParam['num_workers'],
            pin_memory=cfgTParam['pin_memory'])

        # Configure optimizer
        cfgOptim = cfgTParam['optimizer']
        self.optimizer = optim.__dict__[cfgOptim['name']](
            [{'params': self.backbone.parameters()},
                {'params': self.suffix.parameters()}],
            **cfgOptim['args'])

        # Configure evaluator and KPI
        self.evaluator = YOROEvaluator(
            num_classes=trainSet.numClasses,
            **cfgTParam['evaluator'])

        self.modelKpi = ['mAP']


class YOROEvaluator(BaseEvaluator):

    def __init__(self, num_classes, conf_threshold, nms_threshold, sim_threshold):
        self.numClasses = num_classes
        self.confTh = conf_threshold
        self.nmsTh = nms_threshold

        if isinstance(sim_threshold, float):
            self.simTh = [sim_threshold]

        elif isinstance(sim_threshold, list):
            assert len(sim_threshold) == 3, \
                'sim_threshold should be set to [start, end, step]'

            simStart, simEnd, simStep = sim_threshold
            numStep = int(round((simEnd - simStart + simStep) / simStep))
            self.simTh = np.linspace(simStart, simEnd, num=numStep)

        else:
            raise ValueError(
                'Invalid sim_threshold setting: ' + str(sim_threshold))

    def post_process(self, preds):
        preds = ops.flatten_prediction(preds)
        return ops.non_maximum_suppression(preds, self.confTh, self.nmsTh)

    def evaluate_with_sim_thres(self, dts_gts, sim_th):

        gtCountPerClass = [0] * self.numClasses
        apTable = [list() for _ in range(self.numClasses)]

        gtCounter = 0
        dtCounter = 0
        for predGroup, gtGroup in dts_gts:
            for dts, gts in zip(predGroup, gtGroup[2]):

                dtIds = torch.arange(len(dts))
                gtIds = torch.arange(len(dts))
                dtsPaired = []

                # Record ground truth count
                for inst in gts:
                    gtCountPerClass[inst['label']] += 1

                # Detection result pairing
                if len(dts) > 0 and len(gts) > 0:

                    scores = rbox_similarity(dts, gts)

                    while True:

                        # Get the highest score for current iteration
                        score = torch.max(scores)
                        if score < sim_th:
                            break

                        maxPos = torch.where(scores == score)
                        dtId = maxPos[0][0]
                        gtId = maxPos[1][0]

                        if dts[dtId]['label'] == gts[gtId]['label']:

                            # Record this pair as true positive
                            dtsPaired.append(dtId)
                            apTable[dts[dtId]['label']].append(
                                (dts[dtId]['conf'], True)
                            )

                            # Clear matched dt and gt
                            scores[dtId, :] = -1
                            scores[:, gtId] = -1

                        else:

                            # Clear this score due to label mismatch
                            scores[dtId, gtId] = -1

                # Record false positive
                for dtId in dtIds:
                    if dtId not in dtsPaired:
                        apTable[dts[dtId]['label']].append(
                            (dts[dtId]['conf'], False))

        # Sort by confidence
        for cId in range(self.numClasses):
            apTable[cId] = sorted(
                apTable[cId], key=lambda inst: inst[0], reverse=True)

        # Find mean average precision
        classAP = [0] * self.numClasses
        hasInst = [0] * self.numClasses
        for cId, chart in enumerate(apTable):

            if len(chart) > 0:
                hasInst[cId] = 1

                recallHold = 0
                recallCount = 0
                tpCount = 0
                fpCount = 0
                tmpAP = 0

                for conf, paired in chart:
                    if paired:
                        tpCount += 1
                        recallCount += 1
                    else:
                        fpCount += 1

                    def safe_division(n, d):
                        return n / d if d else 0.0

                    precision = safe_division(tpCount, tpCount + fpCount)
                    recall = safe_division(
                        tpCount, tpCount + gtCountPerClass[cId] - recallCount)

                    tmpAP += (recall - recallHold) * precision
                    recallHold = recall

                classAP[cId] = tmpAP

            else:
                hasInst[cId] = 0

        mAP = torch.tensor(0)
        if torch.sum(torch.tensor(hasInst)) > 0:
            mAP = (torch.sum(torch.tensor(classAP)) /
                   torch.sum(torch.tensor(hasInst)))

        return mAP.item()

    def evaluate(self, dts_gts):
        mapList = [
            self.evaluate_with_sim_thres(dts_gts, simTh) for simTh in self.simTh]
        return {'mAP': np.mean(mapList)}


def get_rbox_tensor(rbox):
    return [rbox['degree'], rbox['x'], rbox['y'], rbox['w'], rbox['h']]


def rbox_similarity(pred1, pred2):
    if (len(pred2) == 0) or (len(pred1) == 0):
        return None
    else:
        return ops.rbox_similarity(
            torch.tensor(
                [get_rbox_tensor(rbox) for rbox in pred1], dtype=torch.float32),
            torch.tensor(
                [get_rbox_tensor(rbox) for rbox in pred2], dtype=torch.float32))
