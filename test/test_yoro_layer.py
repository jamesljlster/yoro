import torch

from yoro.layers import YOROLayer

width = 224
height = 224
num_classes = 2

if __name__ == '__main__':

    inputs = [torch.randn(2, 256, 26, 26),
              torch.randn(2, 512, 13, 13)]
    input_shapes = [ten.size() for ten in inputs]

    anchor = [[[21.39, 15.10], [55.22, 55.26], [64.18, 45.31]],
              [[78.89, 78.94], [91.68, 64.73]]]

    yoro = YOROLayer(width, height, num_classes, input_shapes, anchor,
                     deg_min=-45, deg_max=45, deg_part_size=10)

    x = inputs

    # Test conv_regression
    print('=== conv_regression ===')
    x = yoro.head_regression(x)
    for output in x:
        print(output.size())
    print('=======================')
    print()

    # Test head_slicing
    print('=== head_slicing ===')
    x = yoro.head_slicing(x)
    for i, tup in enumerate(x):
        for output in tup:
            print(i, output.size())
    print('====================')
    print()

    # Test predict
    print('=== predict ===')
    x = yoro.predict(inputs)
    for i, tup in enumerate(x):
        for output in tup:
            print(i, output.size())
    print('===============')
    print()

    # Test forward
    print('=== forward ===')
    x = yoro.forward(inputs)
    for i, tup in enumerate(x):
        for output in tup:
            print(i, output.size())
    print('===============')
    print()

    # Test TorchScript compatibility
    yoro = torch.jit.script(
        YOROLayer(width, height, num_classes, input_shapes, anchor,
                  deg_min=-45, deg_max=45, deg_part_size=10)
    )

    # Test loss
    annoList = [
        [
            {'label': 0, 'x': 100, 'y': 100, 'w': 150, 'h': 120, 'degree': 30},
            {'label': 1, 'x': 150, 'y': 50, 'w': 200, 'h': 150, 'degree': -40}
        ],
        [
            {'label': 1, 'x': 120, 'y': 150, 'w': 250, 'h': 100, 'degree': 10}
        ]
    ]
    loss, accu = yoro.loss(inputs, annoList)
    print('loss:', loss)
    print('accu:', accu)

    """
    anchor = [[42.48, 45.48],
              [46.14, 34.85]]
    yoro = YOROLayer(in_channels=512, num_classes=2, width=224, height=224,
                     fmap_width=7, fmap_height=7, anchor=anchor)

    src = torch.randn(2, 512, 7, 7)
    out = yoro(src)

    scp = torch.jit.script(
        YOROLayer(in_channels=512, num_classes=2, width=224, height=224,
                  fmap_width=7, fmap_height=7, anchor=anchor)
    )

    #params = {key: param for key, param in yoro.named_parameters()}
    # for key, param in scp.named_parameters():
    #    param.data = params[key].data.clone()
    scp.load_state_dict(yoro.state_dict())

    #print('=== Parameters in yoro script module ===')
    # for name, param in scp.named_parameters():
    #    print(name, param)
    # print()

    #print('=== Modules in yoro script module ===')
    # for name, module in scp.named_modules():
    #    print(name, module)
    # print()

    print('=== Tensor comparison result (CPU) ===')
    scpOut = scp(src)
    for i in range(len(scpOut)):
        print(torch.equal(out[i], scpOut[i]))
    print()

    yoro = yoro.to('cuda')
    scp = scp.to('cuda')

    src = src.to('cuda')
    scpOut = scp(src)
    out = yoro(src)

    print('=== Tensor comparison result (CUDA) ===')
    scpOut = scp(src)
    for i in range(len(scpOut)):
        print(torch.equal(out[i], scpOut[i]))
    print()

    annoList = [
        [
            {'label': 0, 'x': 100, 'y': 100, 'w': 150, 'h': 120, 'degree': 30},
            {'label': 1, 'x': 150, 'y': 50, 'w': 200, 'h': 150, 'degree': -40}
        ],
        [
            {'label': 1, 'x': 120, 'y': 150, 'w': 250, 'h': 100, 'degree': 10}
        ]
    ]
    loss, accu = yoro.loss(src, annoList)
    print('loss:', loss)
    print('accu:', accu)
    """
