from typing import NamedTuple, Optional, Callable, Iterable, Any, Union, Tuple, List, Sequence
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

class gelu(nn.Module):
    def __init__(self):
        super(gelu, self).__init__()

    def forward(self, x: FloatTensor):
        return gelu_fn(x)


def gelu_fn(x: FloatTensor) -> FloatTensor:
    return 0.5 * x * (1 + torch.tanh(0.7978845608028654*(x+0.044715*x**3)))


def sigsoftmax(x: FloatTensor) -> FloatTensor:
    sigx = torch.sigmoid(x) * torch.exp(x)
    rank = len(sigx.shape)
    norm = torch.sum(sigx, dim=-1).unsqueeze(-1).repeat([1 for _ in range(rank-1)] + [sigx.shape[-1]])
    return sigx/norm


def argmax_top_k(x: FloatTensor, k: int) -> Tuple[LongTensor, FloatTensor]:
    copy = x.clone().detach().requires_grad_(False)
    retv, reti = [], []
    for repeat in range(k):
        values, indices = torch.max(copy, dim=-1)
        mask = torch.arange(x.size(-1), device=x.device).reshape(1, -1) == indices.unsqueeze(-1)
        copy[mask] = -float('inf')
        retv.append(values)
        reti.append(indices)
    retv, reti = torch.stack(retv), torch.stack(reti)
    return retv, reti


def ScaledDotProduct(queries: FloatTensor, keys: FloatTensor, values: FloatTensor,
                     mask: Optional[LongTensor] = None) -> FloatTensor:
    dk = keys.shape[-1]
    weights = torch.bmm(queries, keys.transpose(2, 1)) / math.sqrt(dk)  # [B, M, N]
    if mask is not None:
        weights = weights.masked_fill_(mask == 0, value=-1e10)
    weights = F.softmax(weights, dim=-1)  # [B, M, N] -- each m thing attends a probability distribution over N things
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
    # return pe.unsqueeze(0).expand(b, n, d_model)
    return pe.repeat(b, 1, 1)


def PT(b: int, t: int, n: int, d_inp: int, d_model: int, freq: int = 10000, device: str='cpu') -> FloatTensor:
    pe = torch.zeros(n, d_model, device=device)
    position = torch.arange(0, n, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_inp, 2, device=device, dtype=torch.float) *
                         - (math.log(freq) / d_model))
    times = torch.arange(0, t, device=device, dtype=torch.float).unsqueeze(1)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.repeat(t, 1, 1)

    pe[:, :, 0::2] = pe[:, :, 0::2] + torch.sin(times * div_term).unsqueeze(1).expand(t, n, d_inp//2)
    pe[:, :, 1::2] = pe[:, :, 1::2] + torch.cos(times * div_term).unsqueeze(1).expand(t, n, d_inp//2)
    return pe.unsqueeze(1).expand(t, b, n, d_model)


def Mask(size: Union[Tuple[int, int], Tuple[int, int, int]]) -> LongTensor:
    mask = np.triu(np.ones(size), k=1)
    return torch.from_numpy(mask) == 0


class CustomLRScheduler(object):
    def __init__(self, optimizer: torch.optim.Optimizer, update_fns: Sequence[Callable[[int, Any], float]],
                 **kwargs: Any) -> None:
        assert len(update_fns) == len(optimizer.param_groups)
        self.opt = optimizer
        self._step = 0
        self.update_fns = update_fns
        self.lrs = [None for _ in range(len(self.opt.param_groups))]
        self.__dict__.update(kwargs)

    def step(self) -> None:
        self._step += 1
        self.lrs = self.update(step=self._step, **{k: v for k, v in self.__dict__.items() if k not in
                                                   ('_step', 'opt', 'update_fns', 'lrs')})
        for i, p in enumerate(self.opt.param_groups):
            p['lr'] = self.lrs[i]
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()

    def update(self, step: int, **kwargs) -> List[float]:
        return [update_fn(step, **kwargs) for update_fn in self.update_fns]


def noam_scheme(_step: int, d_model: int, warmup_steps: int, batch_size: int=2048) -> float:
    return d_model**-0.5 * min(_step**-0.5, _step*warmup_steps**-1.5) * batch_size/2048


class FuzzyLoss(object):
    def __init__(self, loss_fn: Callable[[FloatTensor, FloatTensor], FloatTensor], num_classes: int,
                 mass_redistribution: float, ignore_index: int=0) -> None:
        self.loss_fn = loss_fn
        self.nc = num_classes
        self.mass_redistribution = mass_redistribution
        self.ignore_idx = ignore_index

    def __call__(self, x: FloatTensor, y: LongTensor) -> FloatTensor:
        y_float = torch.zeros(x.shape[0], self.nc, x.shape[2], device=x.device, dtype=torch.float)
        y_float.fill_(self.mass_redistribution / (self.nc - 1))
        y_float.scatter_(1, y.unsqueeze(1), 1 - self.mass_redistribution)
        mask = y == self.ignore_idx
        y_float[mask.unsqueeze(1).repeat(1, self.nc, 1)] = 0
        return self.loss_fn(x, y_float)


def infer_wrapper(transformer: nn.Module, encoder_output: FloatTensor, encoder_mask: FloatTensor, b: int) -> \
        Callable[[FloatTensor, int, Optional[LongTensor]], FloatTensor]:
    return lambda decoder_input, t, decoder_mask=None: \
        transformer.infer_one(encoder_output, encoder_mask, decoder_input, t, b, decoder_mask)


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
            gelu(),
            nn.Linear(in_features=d_intermediate, out_features=d_model)
        )

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.network(x)
