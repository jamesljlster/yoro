import math
import numpy as np
from torch.utils.data import Sampler


class AlignedSampler(Sampler):

    def __init__(self, dataset, aligned_size, repeats=1, shuffle=False):

        self.dataset = dataset
        self.shuffle = shuffle

        self.sampleSize = int(math.ceil(
            len(dataset) * repeats / aligned_size) * aligned_size)
        self.sampleStart = 0

    def __len__(self):
        return self.sampleSize

    def __iter__(self):

        # Sampling
        sampleIdx = np.arange(0, self.sampleSize) + self.sampleStart
        if self.shuffle:
            np.random.shuffle(sampleIdx)
        for idx in sampleIdx:
            yield idx % len(self.dataset)

        # Reset sampling start index
        self.sampleStart = \
            (self.sampleStart + self.sampleSize) % len(self.dataset)
