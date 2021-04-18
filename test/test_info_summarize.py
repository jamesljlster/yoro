import torch
from yoro.utils.info_summarize import *

if __name__ == '__main__':

    print('=== Test quantized info ===')

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
    print('info (constantency):', info)
    print('info (represent):', info_represent(info_simplify(info)))
    print()

    print('=== Test scalar info ===')

    info = None
    info1 = (torch.FloatTensor([0.7]), 1)
    info2 = (torch.FloatTensor([0.3]), 1)

    info = info_add(info, info1)
    info = info_add(info, info2)

    print('info:', info)
    print('info (simplify):', info_simplify(info))
    print('info (constantency):', info)
    print('info (represent):', info_represent(info_simplify(info)))
    print()

    print('=== Test quantized info (moving average) ===')

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

    info = info_moving_avg(info, info1, 0.01)
    info = info_moving_avg(info, info2, 0.01)

    print('info:', info)
    print('info (simplify):', info_simplify(info))
    print('info (constantency):', info)
    print('info (represent):', info_represent(info_simplify(info)))
    print()

    print('=== Test scalar info (moving average) ===')

    info = None
    info1 = (torch.FloatTensor([0.7]), 1)
    info2 = (torch.FloatTensor([0.3]), 1)

    info = info_moving_avg(info, info1, 0.01)
    info = info_moving_avg(info, info2, 0.01)

    print('info:', info)
    print('info (simplify):', info_simplify(info))
    print('info (constantency):', info)
    print('info (represent):', info_represent(info_simplify(info)))
    print()
