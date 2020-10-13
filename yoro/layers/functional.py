import torch


def correlation_coefficient(predict, targets):

    vP = predict - torch.mean(predict)
    vT = targets - torch.mean(targets)
    corr = (torch.sum(vP * vT) /
            (torch.sqrt(torch.sum(vP ** 2)) * torch.sqrt(torch.sum(vT ** 2))))

    return corr
