import torch

from yoro.layers import YOROLayer

if __name__ == '__main__':

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
