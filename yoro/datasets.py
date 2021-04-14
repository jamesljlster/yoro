import yaml

from glob import glob
from os.path import join, expanduser, isfile, splitext
from PIL import Image

import torch
from torch.utils.data import Dataset


__all__ = [
    'rbox_collate_fn', 'RBoxSample'
]


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

    def __init__(self, image_dir, names_file=None, transform=None, repeats=1):

        # Check argument
        if not isinstance(repeats, int):
            repeats = int(round(repeats))
        assert repeats > 0, 'Parameter \"repeats\" should greater than 1'

        # Load dataset
        if image_dir[0] == '~':
            image_dir = expanduser(image_dir)

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

        # Load names
        if names_file:

            # Load names from file
            if names_file[0] == '~':
                names_file = expanduser(names_file)

            self.classNames = yaml.load(
                open(names_file, 'r'), Loader=yaml.FullLoader)
            self.numClasses = len(self.classNames)

        else:

            # Set names with labels
            maxLabel = 0
            for inst in instList:
                for anno in inst['anno']:
                    label = anno['label']
                    if label > maxLabel:
                        maxLabel = label

            self.numClasses = maxLabel + 1
            self.classNames = [str(label) for label in range(self.numClasses)]

        # Assignment
        self.repeats = repeats
        self.instList = instList
        self.transform = transform

    def actual_length(self):
        return len(self.instList)

    def __len__(self):
        return self.actual_length() * self.repeats

    def __getitem__(self, idx):

        idx = idx % self.actual_length()

        # Load instance
        image = Image.open(self.instList[idx]['file'])
        anno = self.instList[idx]['anno']
        sample = (image, anno)

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample
