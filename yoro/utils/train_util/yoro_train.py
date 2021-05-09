import yaml

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ... import api
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

        self.suffixClass = YOROLayer
        self.suffixArgs = {
            'width': width,
            'height': height,
            'num_classes': numClasses,
            'input_shapes': [tensor.size() for tensor in out],
            'anchor': cfgCons['anchor'],
            'deg_min': cfgCons['deg_min'],
            'deg_max': cfgCons['deg_max'],
            'deg_part_size': cfgCons['deg_part_size']
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
                self.suffix.degAnchor.clone(),
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

            xT = _build_target_storage()
            yT = _build_target_storage()
            wT = _build_target_storage()
            hT = _build_target_storage()

            degPartT = _build_target_storage()
            degShiftT = _build_target_storage()

            # Collate image and annotations
            images = []
            annos = []
            for batchIdx, (image, anno) in enumerate(samples):
                images.append(image)
                for headIdx, obj in enumerate(anno[0]):
                    objs[headIdx].append(obj)

                acr, xIdx, yIdx, cls, x, y, w, h, degPart, degShift = anno[1][headIdx]

                instSize = acr.size(0)
                batchT[headIdx].append(
                    torch.tensor([batchIdx] * instSize, dtype=torch.long))

                acrIdxT[headIdx].append(acr)
                xIdxT[headIdx].append(xIdx)
                yIdxT[headIdx].append(yIdx)
                clsT[headIdx].append(cls)
                xT[headIdx].append(x)
                yT[headIdx].append(y)
                wT[headIdx].append(w)
                hT[headIdx].append(h)
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
            xT = _gt_cat(xT)
            yT = _gt_cat(yT)
            wT = _gt_cat(wT)
            hT = _gt_cat(hT)
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
                    xT[headIdx],
                    yT[headIdx],
                    wT[headIdx],
                    hT[headIdx],
                    degPartT[headIdx],
                    degShiftT[headIdx])
                )

            return images, (objs, targets)

        # Configure data loader
        self.traLoader = DataLoader(
            trainSet, shuffle=True, collate_fn=rbox_tensor_collate,
            batch_size=cfgTParam['batch'],
            num_workers=cfgTParam['num_workers'],
            pin_memory=cfgTParam['pin_memory'])
        self.tstLoader = DataLoader(
            validSet, shuffle=False, collate_fn=rbox_tensor_collate,
            batch_size=cfgTParam['batch'],
            num_workers=cfgTParam['num_workers'],
            pin_memory=cfgTParam['pin_memory'])

        # Configure optimizer
        cfgOptim = cfgTParam['optimizer']
        self.optimizer = optim.__dict__[cfgOptim['name']](
            [{'params': self.backbone.parameters()},
                {'params': self.suffix.parameters()}],
            **cfgOptim['args'])

        # Configure evaluator and KPI
        """
        self.evaluator = YOROEvaluator(
            num_classes=trainSet.numClasses,
            **cfgTParam['evaluator'])

        self.modelKpi = ['mAP']
        """


class YOROEvaluator(BaseEvaluator):

    def __init__(self, num_classes, conf_threshold, nms_threshold, sim_threshold):
        self.numClasses = num_classes
        self.confTh = conf_threshold
        self.nmsTh = nms_threshold
        self.simTh = sim_threshold

    def post_process(self, preds):
        return api.non_maximum_suppression(preds, self.confTh, self.nmsTh)

    def evaluate(self, dts_gts):

        # Record all detection truth and ground truth
        def _build_truth_storage():
            return (
                {idx: {} for idx in range(self.numClasses)},
                [list() for _ in range(len(dts_gts))]
            )

        dts, dtidPerImg = _build_truth_storage()
        gts, gtidPerImg = _build_truth_storage()
        gtidPerClass = [list() for _ in range(self.numClasses)]

        apTable = [list() for _ in range(self.numClasses)]

        gtCounter = 0
        dtCounter = 0
        for imgInd, (pred, gt) in enumerate(dts_gts):

            # Convert ground truth
            for inst in gt:

                gtId = gtCounter
                gtCounter += 1

                gt = api.RBox()
                gt.conf = 1.0
                gt.label = inst['label']
                gt.degree = inst['degree']
                gt.x = inst['x']
                gt.y = inst['y']
                gt.w = inst['w']
                gt.h = inst['h']

                gts[gtId] = gt
                gtidPerImg[imgInd].append(gtId)
                gtidPerClass[gt.label].append(gtId)

            # Record detection truth
            for dt in pred:

                dtId = dtCounter
                dtCounter += 1

                dts[dtId] = dt
                dtidPerImg[imgInd].append(dtId)

            # Detection result pairing
            dtsPaired = []

            dtIds = dtidPerImg[imgInd]
            gtIds = gtidPerImg[imgInd]

            if len(gtIds) > 0 and len(dtIds) > 0:
                scores = torch.stack([
                    rbox_similarity(dts[dtId], [gts[gtId] for gtId in gtIds])
                    for dtId in dtIds
                ])

                while torch.max(scores) > self.simTh:

                    # Get the highest score for current iteration
                    score = torch.max(scores)
                    maxPos = torch.where(scores == score)
                    rowInd = maxPos[0][0]
                    colInd = maxPos[1][0]

                    if dts[dtIds[rowInd]].label == gts[gtIds[colInd]].label:

                        # Record this pair as true positive
                        dtId = dtIds[rowInd]
                        gtId = gtIds[colInd]

                        dtsPaired.append(dtId)
                        apTable[dts[dtIds[rowInd]].label].append(
                            (dts[dtId].conf, True)
                        )

                        # Clear matched dt and gt
                        scores[rowInd, :] = -1
                        scores[:, colInd] = -1

                    else:

                        # Clear this score due to label mismatch
                        scores[rowInd, colInd] = -1

            # Record for false positive
            for dtId in dtIds:
                if dtId not in dtsPaired:
                    apTable[dts[dtId].label].append(
                        (dts[dtId].conf, False)
                    )

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

                    precision = float(tpCount) / (tpCount + fpCount)
                    recall = (
                        float(tpCount) / (tpCount + len(gtidPerClass[cId]) - recallCount))
                    tmpAP += (recall - recallHold) * precision

                    recallHold = recall

                classAP[cId] = tmpAP

            else:
                hasInst[cId] = 0

        mAP = (torch.sum(torch.tensor(classAP)) /
               torch.sum(torch.tensor(hasInst)))

        return {'mAP': mAP.item()}


def get_rbox_tensor(rbox):
    return [rbox.degree, rbox.x, rbox.y, rbox.w, rbox.h]


def rbox_similarity(pred1, pred2):
    if len(pred2) == 0:
        return None
    else:
        return api.rbox_similarity(
            torch.tensor([get_rbox_tensor(pred1)], dtype=torch.float32),
            torch.tensor(
                [get_rbox_tensor(rbox) for rbox in pred2],
                dtype=torch.float32)
        ).squeeze(0)
