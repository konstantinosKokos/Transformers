from typing import NamedTuple, Optional, Callable, Iterable, Any, Union, Tuple, List, Sequence
from torch.nn import functional as F
from torch import nn
import torch
import math
import numpy as np

try:
    from src.utils import *
except ImportError:
    from Transformer.src.utils import *


class RecurrentEncoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, dropout: float) -> None:
        super(RecurrentEncoderLayer, self).__init__()
        self.dropout_rate = dropout
        self.mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.ln_mha = nn.LayerNorm(normalized_shape=d_model)
        self.ln_ffn = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: EncoderInput, ffn: tensor_map) -> EncoderInput:
        mha_x = F.dropout(
            self.mha(x.encoder_input, x.encoder_input, x.encoder_input, x.mask), p=self.dropout_rate) + x.encoder_input
        mha_x = self.ln_mha(mha_x)
        ffn_x = self.ln_ffn(F.dropout(ffn(mha_x), p=self.dropout_rate) + mha_x)
        return EncoderInput(encoder_input=ffn_x, mask=x.mask)


class RecurrentEncoder(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int,
                 dropout=0.1) -> None:
        super(RecurrentEncoder, self).__init__()
        self.shared_ffn = FFN(d_intermediate, d_model)
        self.layers = nn.ModuleList([RecurrentEncoderLayer(num_heads, d_model, d_k, d_v, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x: EncoderInput) -> EncoderInput:
        b, n, dk = x.encoder_input.shape
        for i, layer in enumerate(self.layers):
            x = layer(EncoderInput(encoder_input=x.encoder_input + PT(b, i, n, dk, dk, device=x.encoder_input.device),
                                   mask=x.mask), self.shared_ffn)
        return x


class RecurrentDecoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, dropout: float) -> None:
        super(RecurrentDecoderLayer, self).__init__()
        self.dropout_rate = dropout
        self.mask_mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.ln_m_mha = nn.LayerNorm(normalized_shape=d_model)
        self.ln_mha = nn.LayerNorm(normalized_shape=d_model)
        self.ln_ffn = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: DecoderInput, ffn: tensor_map) -> DecoderInput:
        t = x.decoder_input.shape[1]
        m_mha_x = self.mask_mha(x.decoder_input, x.decoder_input, x.decoder_input, x.decoder_mask)
        m_mha_x = F.dropout(m_mha_x, p=self.dropout_rate) + x.decoder_input
        m_mha_x = self.ln_m_mha(m_mha_x)
        mha_x = self.mha(m_mha_x, x.encoder_output, x.encoder_output, x.encoder_mask[:, :t, :])
        mha_x = F.dropout(mha_x, p=self.dropout_rate) + m_mha_x
        mha_x = self.ln_mha(mha_x)
        ffn_x = ffn(mha_x)
        ffn_x = F.dropout(ffn_x, p=self.dropout_rate) + mha_x
        ffn_x = self.ln_ffn(ffn_x)
        return DecoderInput(encoder_output=x.encoder_output,
                            decoder_input=ffn_x,
                            decoder_mask=x.decoder_mask,
                            encoder_mask=x.encoder_mask)


class RecurrentDecoder(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int,
                 dropout: float=0.1) -> None:
        super(RecurrentDecoder, self).__init__()
        self.shared_ffn = FFN(d_intermediate, d_model)
        self.layers = nn.ModuleList([RecurrentDecoderLayer(num_heads, d_model, d_k, d_v, dropout)
                                    for _ in range(num_layers)])

    def forward(self, x: DecoderInput) -> DecoderInput:
        b, n, dk = x.decoder_input.shape
        for i, layer in enumerate(self.layers):
            x = layer(DecoderInput(decoder_input=x.decoder_input + PT(b, i, n, dk, dk, device=x.decoder_input.device),
                                   encoder_output=x.encoder_output,
                                   decoder_mask=x.decoder_mask,
                                   encoder_mask=x.encoder_mask), self.shared_ffn)
        return x


class UniversalTransformer(nn.Module):
    def __init__(self, num_classes: int, output_embedder: tensor_map,
                 encoder_layers: int = 6, num_heads: int = 8, decoder_layers: int = 6, d_model: int = 300,
                 d_intermediate: int = 1024, dropout: float=0.1, device: str='cpu') -> None:
        self.device = device
        super(UniversalTransformer, self).__init__()
        self.encoder = RecurrentEncoder(num_layers=encoder_layers, num_heads=num_heads, d_model=d_model,
                                        d_k=d_model // num_heads, d_v=d_model // num_heads,
                                        dropout=dropout, d_intermediate=d_intermediate).to(self.device)
        self.decoder = RecurrentDecoder(num_layers=decoder_layers, num_heads=num_heads, d_model=d_model,
                                        d_k=d_model // num_heads, d_v=d_model // num_heads,
                                        dropout=dropout, d_intermediate=d_intermediate).to(self.device)
        self.predictor = nn.Linear(in_features=d_model, out_features=num_classes).to(self.device)
        self.output_embedder = output_embedder

    def forward(self, encoder_input: FloatTensor, decoder_input: FloatTensor, encoder_mask: LongTensor,
                decoder_mask: LongTensor) -> FloatTensor:
        self.train()

        encoder_output = self.encoder(EncoderInput(encoder_input=encoder_input,
                                                   mask=encoder_mask))
        decoder_output = self.decoder(DecoderInput(encoder_output=encoder_output.encoder_input,
                                                   encoder_mask=encoder_mask,
                                                   decoder_input=decoder_input,
                                                   decoder_mask=decoder_mask))
        return torch.log(sigsoftmax(self.predictor(decoder_output.decoder_input)))


def test(device: str):
    sl = 25
    nc = 1000

    embedder = torch.nn.Embedding(nc, 300).to(device)
    t = UniversalTransformer(12, embedder, device=device)
    encoder_input = torch.rand(128, sl, 300).to(device)
    encoder_mask = torch.ones(128, sl, sl).to(device)
    decoder_input = torch.rand(128, sl, 300).to(device)
    decoder_mask = Mask((128, sl, sl)).to(device)
    f_v = t.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)