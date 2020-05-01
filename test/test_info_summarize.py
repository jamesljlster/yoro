import torch
from yoro.util.info_summarize import *

if __name__ == '__main__':

    info = None
    info1 = {
        'loss': (torch.FloatTensor([0.7]), 1),
        'obj': ([0.8, 0.2], 4),
        'accu': ({'class': 0.4, 'deg': 0.6}, 5)
    }
    info2 = {
        'loss': (torch.FloatTensor([0.3]), 1),
        'obj': ([0.2, 0.8], 4),
        'accu': ({'class': 0.6, 'deg': 0.4}, 5)
    }

    info = info_add(info, info1)
    info = info_add(info, info2)

    print('info:', info)
    print('info (simplify):', info_simplify(info))
    print('info:', info)