import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ...datasets import RBoxSample, rbox_collate_fn
from ...transforms import \
    RBox_ColorJitter, RBox_RandomAffine, RBox_Resize, RBox_PadToAspect, RBox_ToTensor
from ...layers import YOROLayer
from ...api import non_maximum_suppression, RBox
from ..object_loader import load_object

from .base_train import BaseTrain, BaseEvaluator


class YOROTrain(BaseTrain):

    def __init__(self, config_path):

        cfg = super(YOROTrain, self).__init__(config_path)

        # Get child config
        cfgCons = cfg['construct']
        cfgTParam = cfg['train_param']

        # Get network input size
        height = cfgCons['height']
        width = cfgCons['width']

        # Configure data augmentation
        tfPrefix = [RBox_PadToAspect(float(width) / height)]
        tfSuffix = [RBox_Resize((height, width)),
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

        # Configure yoro layer
        src = torch.randn(1, 3, height, width)
        out = self.backbone(src.to(self.dev))
        fmapSize = out.size()

        self.suffixClass = YOROLayer
        self.suffixArgs = {
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

        self.suffix = self.suffixClass(**self.suffixArgs).to(self.dev)

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
        self.simTh = sim_threshold

    def post_process(self, preds):
        return non_maximum_suppression(preds, self.confTh, self.nmsTh)

    def evaluate(self, dts_gts):

        predPair = {idx: [] for idx in range(self.numClasses)}
        gts = {}

        gtCounter = 0
        for dataInd, (pred, gt) in enumerate(dts_gts):

            # Record ground truth
            gtTmp = {}
            for inst in gt:

                gtId = gtCounter
                gtCounter += 1

                gt = RBox()
                gt.conf = 1.0
                gt.label = inst['label']
                gt.degree = inst['degree']
                gt.x = inst['x']
                gt.y = inst['y']
                gt.w = inst['w']
                gt.h = inst['h']

                gtTmp[gtId] = gt

            gts.update(gtTmp)

            # Compare prediction with ground truths
            for dt in pred:

                # Find similarities between prediction and ground truths
                rboxSim = rbox_similarity(
                    [dt], [gtTmp[gtKey] for gtKey in gtTmp])
                highSim = (rboxSim >= self.simTh).squeeze(0)

                dtLabel = dt.label
                dtConf = dt.conf

                if highSim.sum() > 0:
                    for gtKey, match in zip(gtTmp, highSim):
                        if match:
                            predPair[dtLabel].append({
                                'conf': dtConf,
                                'pred': dtLabel,
                                'label': gtTmp[gtKey].label,
                                'gtId': gtKey,
                                'dataInd': dataInd
                            })

                else:
                    predPair[dtLabel].append({
                        'conf': dtConf,
                        'pred': dtLabel,
                        'label': -1,
                        'gtId': -1,
                        'dataInd': dataInd
                    })

        mAP = 0
        for cId in predPair:

            results = predPair[cId]
            results = sorted(
                results, key=lambda inst: inst['conf'], reverse=True)

            # Find all ground truth keys with given class ID
            gtKeys = [key for key in gts if gts[key].label == cId]

            gtHit = []
            predHit = 0
            r = [0.0]
            p = [0.0]
            for i, inst in enumerate(results):

                gtId = inst['gtId']
                if gtId >= 0:
                    if gtId not in gtHit:
                        gtHit.append(inst['gtId'])
                    if inst['pred'] == inst['label']:
                        predHit += 1

                recall = len(gtHit) / len(gtKeys)
                precision = predHit / (i + 1)

                if recall not in r:
                    r.append(recall)
                    p.append(precision)
                else:
                    p[-1] = max(p[-1], precision)

                if recall >= 1.0:
                    break

            ap = 0
            for i in range(1, len(r)):
                ap += (r[i] - r[i - 1]) * p[i]

            mAP += ap

        mAP /= len(predPair)
        return {'mAP': mAP}


def bbox_to_corners(bbox):

    corners = torch.zeros_like(bbox)
    corners[..., 0] = bbox[..., 0] - bbox[..., 2] / 2.0
    corners[..., 1] = bbox[..., 1] - bbox[..., 3] / 2.0
    corners[..., 2] = bbox[..., 0] + bbox[..., 2] / 2.0
    corners[..., 3] = bbox[..., 1] + bbox[..., 3] / 2.0

    return corners


def get_bbox(pred):
    return torch.tensor([
        [inst.x, inst.y, inst.w, inst.h] for inst in pred])


def deg2rad(deg):
    return deg * 3.1415927410125732 / 180.0


def get_degree(pred):
    return torch.tensor([inst.degree for inst in pred])


def rbox_similarity(pred1, pred2):

    # Get bounding boxes
    bbox1 = get_bbox(pred1)
    bbox2 = get_bbox(pred2)

    # Bounding boxes to corners
    corners1 = bbox_to_corners(bbox1)
    corners2 = bbox_to_corners(bbox2)

    # Find IoU scores
    interX1 = torch.max(corners1[..., 0], corners2[..., 0])
    interY1 = torch.max(corners1[..., 1], corners2[..., 1])
    interX2 = torch.min(corners1[..., 2], corners2[..., 2])
    interY2 = torch.min(corners1[..., 3], corners2[..., 3])

    interArea = (torch.clamp(interX2 - interX1, 0) *
                 torch.clamp(interY2 - interY1, 0))
    unionArea = (bbox1[..., 2] * bbox1[..., 3] +
                 bbox2[..., 2] * bbox2[..., 3] -
                 interArea)
    ious = interArea / (unionArea + 1e-4)

    # Find degree similarity
    rad1 = deg2rad(get_degree(pred1))
    rad2 = deg2rad(get_degree(pred2))
    ang1 = torch.stack([torch.sin(rad1), torch.cos(rad1)], 1)
    ang2 = torch.stack([torch.sin(rad2), torch.cos(rad2)], 1)
    angSim = (torch.matmul(ang1, ang2.t()) + 1.0) / 2.0

    return ious * angSim
