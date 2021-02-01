import torch
from yoro.api import rbox_similarity

if __name__ == '__main__':

    src = torch.tensor([[15, 25, 15, 30, 10]], dtype=torch.float32)
    test = torch.tensor([[15, 25, 15, 30, 10]], dtype=torch.float32)
    sim = rbox_similarity(src, test)
    print(sim)

    src = torch.tensor([[15, 25, 15, 30, 10]], dtype=torch.float32)
    test = torch.tensor([[-45, 25, 15, 30, 10]], dtype=torch.float32)
    sim = rbox_similarity(src, test)
    print(sim)

    src = torch.tensor([[15, 25, 15, 30, 10]], dtype=torch.float32)
    test = torch.tensor([[15, 35, 25, 30, 20]], dtype=torch.float32)
    sim = rbox_similarity(src, test)
    print(sim)

    src = torch.tensor([[15, 25, 15, 30, 10]], dtype=torch.float32)
    test = torch.tensor([[-45, 35, 25, 30, 20]], dtype=torch.float32)
    sim = rbox_similarity(src, test)
    print(sim)

    src = torch.tensor([[15, 25, 15, 30, 10]], dtype=torch.float32)
    test = torch.tensor([
        [15, 25, 15, 30, 10],
        [-45, 25, 15, 30, 10],
        [15, 35, 25, 30, 20],
        [-45, 35, 25, 30, 20]
    ], dtype=torch.float32)
    sim = rbox_similarity(src, test)
    print(sim)
