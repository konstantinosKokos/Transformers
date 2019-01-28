from typing import NamedTuple, Optional, Callable, Iterable, Any, Union, Tuple, List
from torch.nn import functional as F
from torch import nn
import torch
import math
import numpy as np


FloatTensor = Union[torch.cuda.FloatTensor, torch.FloatTensor]
LongTensor = Union[torch.cuda.LongTensor, torch.LongTensor]
Tensor = Union[FloatTensor, LongTensor]
tensor_map = Callable[[Any], Tensor]
tensor_maps = Iterable[tensor_map]

EncoderInput = NamedTuple('EncoderInput', [('encoder_input', FloatTensor),
                                           ('mask', Optional[LongTensor])])
DecoderInput = NamedTuple('DecoderInput', [('encoder_output', FloatTensor),
                                           ('decoder_input', FloatTensor),
                                           ('encoder_mask', Optional[LongTensor]),
                                           ('decoder_mask', LongTensor)])


#########################################################################################
# # # # # # # # # # # # # # # # # # # # # UTILS # # # # # # # # # # # # # # # # # # # # #
#########################################################################################


def argmax_top_k(x: FloatTensor, k: int) -> List[Tuple[FloatTensor, LongTensor]]:
    copy = x.clone().detach().requires_grad_(False)
    ret = []
    for repeat in range(k):
        values, indices = torch.max(copy, dim=-1)
        mask = torch.arange(x.size(-1), device=x.device).reshape(1, -1) == indices.unsqueeze(-1)
        copy[mask] = -float('inf')
        ret.append((values, indices))
    return ret


def sigsoftmax(x: FloatTensor) -> FloatTensor:
    sigx = torch.sigmoid(x) * torch.exp(x)
    rank = len(sigx.shape)
    norm = torch.sum(sigx, dim=-1).unsqueeze(-1).repeat([1 for _ in range(rank-1)] + [sigx.shape[-1]])
    return sigx/norm


def ScaledDotProduct(queries: FloatTensor, keys: FloatTensor, values: FloatTensor,
                     mask: Optional[LongTensor] = None) -> FloatTensor:
    b, _, dk = keys.shape
    weights = torch.bmm(queries, keys.transpose(2, 1)) / math.sqrt(dk)  # [B, M, N]
    if mask is not None:
        weights = weights.masked_fill(mask == 0, value=-1e10)
    weights = F.softmax(weights, dim=-1)  # [B, M, N]
    return torch.bmm(weights, values)


def MultiHeadAttentionFn(queries: FloatTensor, keys: FloatTensor, values: FloatTensor,
                         qts: tensor_maps, kts: tensor_maps, vts: tensor_maps, wo: tensor_map,
                         mask: Optional[LongTensor] = None) -> FloatTensor:
    qs = [qt(queries) for qt in qts]
    ks = [kt(keys) for kt in kts]
    vs = [vt(values) for vt in vts]
    outputs = [ScaledDotProduct(qs[i], ks[i], vs[i], mask) for i in range(len(qs))]
    outputs = torch.cat(outputs, dim=-1)
    return wo(outputs)


def PE(b: int, n: int, d_inp: int, d_model: int, freq: int = 10000, device: str='cpu') -> FloatTensor:
    pe = torch.zeros(n, d_model, device=device)
    position = torch.arange(0, n, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_inp, 2, device=device, dtype=torch.float) *
                         - (math.log(freq) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.repeat(b, 1, 1)


def Mask(size: Union[Tuple[int, int], Tuple[int, int, int]]) -> LongTensor:
    mask = np.triu(np.ones(size), k=1)
    return torch.from_numpy(mask) == 0


class CustomLRScheduler(object):
    def __init__(self, optimizer: torch.optim.Optimizer, update_fn: Callable[[int, Any], float],
                 **kwargs: Any) -> None:
        self.opt = optimizer
        self._step = 0
        self.update_fn = update_fn
        self.lr = None
        self.__dict__.update(kwargs)

    def step(self) -> None:
        self._step += 1
        self.lr = self.update(step=self._step, **{k: v for k, v in self.__dict__.items() if k not in
                             ('_step', 'opt', 'update_fn', 'lr')})
        for p in self.opt.param_groups:
            p['lr'] = self.lr
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()

    def update(self, step: int, **kwargs) -> float:
        return self.update_fn(step, **kwargs)


def noam_scheme(_step: int, d_model: int, warmup_steps: int) -> float:
    return d_model**-0.5 * min(_step**-0.5, _step*warmup_steps**-1.5)


class FuzzyLoss(object):
    def __init__(self, loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor], num_classes: int,
                 mass_redistribution: float) -> None:
        self.loss_fn = loss_fn
        self.nc = num_classes
        self.mass_redistribution = mass_redistribution

    def __call__(self, x: FloatTensor, y: LongTensor) -> FloatTensor:
        y_float = torch.zeros(x.shape[0], self.nc, x.shape[2], device=x.device, dtype=torch.float)
        y_float.fill_(self.mass_redistribution / (self.nc - 1))
        y_float.scatter_(1, y.unsqueeze(1), 1 - self.mass_redistribution)
        mask = y == 0
        y_float[mask.unsqueeze(1).repeat(1, self.nc, 1)] = 0
        return self.loss_fn(x, y_float)


def count_parameters(model: nn.Module) -> int:
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param

#########################################################################################
# # # # # # # # # # # # # # # # # # # CORE ATN LAYERS # # # # # # # # # # # # # # # # # #
#########################################################################################


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.q_transformations = nn.ModuleList([nn.Linear(in_features=d_model, out_features=d_k, bias=False)
                                                for _ in range(num_heads)])
        self.k_transformations = nn.ModuleList([nn.Linear(in_features=d_model, out_features=d_k, bias=False)
                                                for _ in range(num_heads)])
        self.v_transformations = nn.ModuleList([nn.Linear(in_features=d_model, out_features=d_v, bias=False)
                                                for _ in range(num_heads)])
        self.Wo = nn.Linear(in_features=num_heads * d_v, out_features=d_model, bias=False)

    def forward(self, queries: FloatTensor, keys: FloatTensor, values: FloatTensor,
                mask: Optional[LongTensor] = None) -> FloatTensor:
        return MultiHeadAttentionFn(queries, keys, values, self.q_transformations, self.k_transformations,
                                    self.v_transformations, self.Wo, mask)


class FFN(nn.Module):
    def __init__(self, d_intermediate: int, d_model: int) -> None:
        super(FFN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_intermediate),
            nn.ReLU(),
            nn.Linear(in_features=d_intermediate, out_features=d_model)
        )

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.network(x)