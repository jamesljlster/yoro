import yaml

from glob import glob
from os.path import join, expanduser, isfile, splitext
from PIL import Image

import torch
from torch.utils.data import Dataset


def rbox_collate_fn(samples):

    srcType = type(samples[0][0]).__name__

    images = []
    annos = []
    for image, anno in samples:
        images.append(image)
        annos.append(anno)

    if srcType == 'Tensor':
        images = torch.stack(images, 0)

    return images, annos


class RBoxSample(Dataset):

    def __init__(self, image_dir, names_file, transform=None):

        if image_dir[0] == '~':
            image_dir = expanduser(image_dir)
        if names_file[0] == '~':
            names_file = expanduser(names_file)

        # Load names
        self.classNames = yaml.load(
            open(names_file, 'r'), Loader=yaml.FullLoader)
        self.numClasses = len(self.classNames)

        # Load dataset
        instList = []
        markFiles = [f for f in glob(join(image_dir, '*.mark')) if isfile(f)]
        for markFile in markFiles:

            # Load annotation
            fRead = open(markFile, 'r')
            anno = yaml.load(fRead, Loader=yaml.FullLoader)

            # Make instance
            inst = {}
            inst['file'] = splitext(markFile)[0]
            inst['anno'] = anno
            instList.append(inst)

        # Assignment
        self.instList = instList
        self.transform = transform

    def __len__(self):
        return len(self.instList)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load instance
        image = Image.open(self.instList[idx]['file'])
        anno = self.instList[idx]['anno']
        sample = (image, anno)

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample
