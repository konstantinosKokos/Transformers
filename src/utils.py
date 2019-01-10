import torch
from torch import Tensor


def sigsoftmax(x: Tensor) -> Tensor:
    sigx = torch.sigmoid(x) * torch.exp(x)
    norm = torch.sum(sigx, dim=-1).unsqueeze(-1).repeat(1, 1, sigx.shape[-1])
    return sigx/norm