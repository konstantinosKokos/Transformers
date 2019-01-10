from typing import NamedTuple, Optional, Callable, Iterable, Any, Union, Tuple
from torch.nn import functional as F
from torch import Tensor, nn
import torch
import math
import numpy as np

tensor_map = Callable[[Any], Tensor]
tensor_maps = Iterable[tensor_map]
EncoderInput = NamedTuple('EncoderInput', [('encoder_input', Tensor), ('mask', Optional[Tensor])])
DecoderInput = NamedTuple('DecoderInput', [('encoder_output', Tensor), ('decoder_input', Tensor),
                                           ('encoder_mask', Optional[Tensor]), ('decoder_mask', Tensor)])


def ScaledDotProduct(queries: Tensor, keys: Tensor, values: Tensor,
                     mask: Optional[Tensor] = None) -> Tensor:
    b, _, dk = keys.shape
    weights = torch.bmm(queries, keys.transpose(2, 1)) / math.sqrt(dk)  # [B, M, N]
    if mask is not None:
        weights = weights.masked_fill(mask == 0, value=-float('inf'))
    weights = F.softmax(weights, dim=-1)  # [B, M, N]
    return torch.bmm(weights, values)


def MultiHeadAttentionFn(queries: Tensor, keys: Tensor, values: Tensor,
                         qts: tensor_maps, kts: tensor_maps, vts: tensor_maps, wo: tensor_map,
                         mask: Optional[Tensor] = None) -> Tensor:
    qs = [qt(queries) for qt in qts]
    ks = [kt(keys) for kt in kts]
    vs = [vt(values) for vt in vts]
    outputs = [ScaledDotProduct(qs[i], ks[i], vs[i], mask) for i in range(len(qs))]
    outputs = torch.cat(outputs, dim=-1)
    return wo(outputs)


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

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class EncoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int) -> None:
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.ffn = FFN(d_intermediate, d_model)
        self.ln_mha = nn.LayerNorm(normalized_shape=d_model)
        self.ln_ffn = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: EncoderInput) -> EncoderInput:
        mha_x = self.ln_mha(x.encoder_input + self.mha(x.encoder_input, x.encoder_input, x.encoder_input, x.mask))
        return EncoderInput(self.ln_ffn(mha_x + self.ffn(mha_x)), x.mask)


def Encoder(num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int) -> nn.Sequential:
    layers = [EncoderLayer(num_heads, d_model, d_k, d_v, d_intermediate) for _
              in range(num_layers)]
    return nn.Sequential(*layers)


def PE(b: int, n: int, d_inp: int, d_model: int, freq: int = 10000) -> Tensor:
    pe = torch.zeros(n, d_model).float()
    position = torch.arange(0, n).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_inp, 2).float() *
                         - (math.log(freq) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.repeat(b, 1, 1)


def Mask(size: Union[Tuple[int, int], Tuple[int, int, int]]) -> Tensor:
    mask = np.triu(np.ones(size), k=1)
    return torch.from_numpy(mask) == 0


class DecoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int) -> None:
        super(DecoderLayer, self).__init__()
        self.mask_mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.ffn = FFN(d_intermediate, d_model)
        self.ln_m_mha = nn.LayerNorm(normalized_shape=d_model)
        self.ln_mha = nn.LayerNorm(normalized_shape=d_model)
        self.ln_ffn = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: DecoderInput) -> DecoderInput:
        t = x.decoder_input.shape[1]
        m_mha_x = self.ln_m_mha(x.decoder_input +
                                self.mask_mha(x.decoder_input, x.decoder_input, x.decoder_input, x.decoder_mask))
        mha_x = self.ln_mha(m_mha_x + self.mha(m_mha_x, x.encoder_output, x.encoder_output,
                                               x.encoder_mask[:, :t, :]))
        return DecoderInput(encoder_mask=x.encoder_mask, encoder_output=x.encoder_output,
                            decoder_input=self.ln_ffn(mha_x + self.ffn(mha_x)), decoder_mask=x.decoder_mask)


def Decoder(num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int) -> nn.Sequential:
    return nn.Sequential(*[DecoderLayer(num_heads, d_model, d_k, d_v, d_intermediate) for _ in range(num_layers)])


class Transformer(nn.Module):
    def __init__(self, num_classes: int, output_embedder: Callable[[Any], Tensor],
                 encoder_layers: int = 6, num_heads: int = 8, decoder_layers: int = 6, d_model: int = 300,
                 d_intermediate: int = 128, device: str='cpu') -> None:
        self.device = device
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers=encoder_layers, num_heads=num_heads, d_model=d_model,
                               d_k=d_model // num_heads, d_v=d_model // num_heads,
                               d_intermediate=d_intermediate).to(self.device)
        self.decoder = Decoder(num_layers=decoder_layers, num_heads=num_heads, d_model=d_model,
                               d_k=d_model // num_heads, d_v=d_model // num_heads,
                               d_intermediate=d_intermediate).to(self.device)
        self.predictor = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=num_classes),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.output_embedder = output_embedder

    def forward(self, encoder_input: Tensor, decoder_input: Tensor, encoder_mask: Tensor,
                decoder_mask: Tensor) -> Tensor:
        b, n, dk = encoder_input.shape
        pe = PE(b, n, dk, dk).to(self.device)
        encoder_output = self.encoder(EncoderInput(encoder_input + pe, encoder_mask))
        decoder_output = self.decoder(DecoderInput(encoder_output=encoder_output.encoder_input,
                                                   encoder_mask=encoder_mask, decoder_input=decoder_input + pe,
                                                   decoder_mask=decoder_mask))
        return self.predictor(decoder_output.decoder_input)

    def infer(self, encoder_input: Tensor, encoder_mask: Tensor) -> Tensor:
        b, n, dk = encoder_input.shape
        pe = PE(b, n, dk, dk).to(self.device)
        encoder_output = self.encoder(EncoderInput(encoder_input + pe, encoder_mask)).encoder_input
        decoder_output = torch.ones(b, 1, dk).to(self.device)  # todo
        for t in range(n - 1):
            decoder_step = self.decoder(DecoderInput(encoder_output=encoder_output, encoder_mask=encoder_mask,
                                                     decoder_input=decoder_output,
                                                     decoder_mask=Mask((b, t + 1, t + 1)).to(self.device)))\
                .decoder_input
            prob_t = self.predictor(decoder_step[:, -1])
            class_t = prob_t.argmax(dim=-1)
            emb_t = self.output_embedder(class_t).unsqueeze(1)
            decoder_output = torch.cat([decoder_output, emb_t], dim=1)
        return decoder_output


def test(device: str):
    t = Transformer(5, lambda x: torch.rand(x.shape[0], 300), device=device)
    encoder_input = torch.rand(5, 3, 300).to(device)
    encoder_mask = torch.ones(5, 3, 3).to(device)
    decoder_input = torch.rand(5, 3, 300).to(device)
    decoder_mask = Mask((5, 3, 3)).to(device)
    a_v = t.infer(encoder_input, encoder_mask)
    b_v = t.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)
    import pdb
    pdb.set_trace()
