from typing import NamedTuple, Optional, Callable, Iterable, Any, Union, Tuple, List, Sequence
from torch.nn import functional as F
from torch import nn
import torch
import math
import numpy as np

try:
    from src.utils import *
    from src.Transformer import EncoderLayer, DecoderLayer
except ImportError:
    from Transformer.src.utils import *
    from Transformer.src.Transformer import EncoderLayer, DecoderLayer


class RecurrentEncoder(nn.Module):
    def __init__(self, num_steps: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int,
                 dropout=0.1) -> None:
        super(RecurrentEncoder, self).__init__()
        self.layer = EncoderLayer(num_heads, d_model, d_k, d_v, d_intermediate, dropout)
        self.num_repeats = num_steps

    def forward(self, x: EncoderInput) -> EncoderInput:
        b, n, dk = x.encoder_input.shape
        for i in range(self.num_repeats):
            x = self.layer(EncoderInput(encoder_input=x.encoder_input + PT(b, i, n, dk, dk,
                                                                           device=x.encoder_input.device),
                                        mask=x.mask))
        return x


class RecurrentDecoder(nn.Module):
    def __init__(self, num_steps: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int,
                 dropout: float=0.1) -> None:
        super(RecurrentDecoder, self).__init__()
        self.layer = DecoderLayer(num_heads, d_model, d_k, d_v, d_intermediate, dropout)
        self.num_repeats = num_steps

    def forward(self, x: DecoderInput) -> DecoderInput:
        b, n, dk = x.decoder_input.shape
        for i in range(self.num_repeats):
            x = self.layer(DecoderInput(decoder_input=x.decoder_input + PT(b, i, n, dk, dk,
                                                                           device=x.decoder_input.device),
                                        encoder_output=x.encoder_output,
                                        decoder_mask=x.decoder_mask,
                                        encoder_mask=x.encoder_mask))
        return x


class UniversalTransformer(nn.Module):
    def __init__(self, num_classes: int, output_embedder: tensor_map,
                 encoder_layers: int = 6, num_heads: int = 8, decoder_layers: int = 6, d_model: int = 300,
                 d_intermediate: int = 1024, dropout: float=0.1, device: str='cpu') -> None:
        self.device = device
        super(UniversalTransformer, self).__init__()
        self.encoder = RecurrentEncoder(num_steps=encoder_layers, num_heads=num_heads, d_model=d_model,
                                        d_k=d_model // num_heads, d_v=d_model // num_heads,
                                        dropout=dropout, d_intermediate=d_intermediate).to(self.device)
        self.decoder = RecurrentDecoder(num_steps=decoder_layers, num_heads=num_heads, d_model=d_model,
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

    def infer(self, encoder_input: FloatTensor, encoder_mask: LongTensor, sos_symbol: int) -> FloatTensor:
        self.eval()

        b, n, dk = encoder_input.shape
        pe = PT(b, 0, n, dk, dk, device=self.device)
        encoder_output = self.encoder(EncoderInput(encoder_input + pe, encoder_mask)).encoder_input
        sos_symbols = (torch.ones(b, device=self.device) * sos_symbol).long()
        decoder_output = self.output_embedder(sos_symbols).unsqueeze(1) + pe[:, 0:1, :]
        output_probs = torch.Tensor().to(self.device)
        inferer = Inferer(self, encoder_output, encoder_mask, b)

        for t in range(n):
            prob_t = inferer(decoder_output, t)
            class_t = prob_t.argmax(dim=-1)
            emb_t = self.output_embedder(class_t).unsqueeze(1) + pe[:, t + 1:t + 2, :]
            decoder_output = torch.cat([decoder_output, emb_t], dim=1)
            output_probs = torch.cat([output_probs, prob_t.unsqueeze(1)], dim=1)
        return output_probs


class Inferer(object):
    def __init__(self, transformer: UniversalTransformer, encoder_output: FloatTensor, encoder_mask: FloatTensor,
                 b: int) -> None:
        self.transformer = transformer
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self.b = b

    def __call__(self, decoder_input: FloatTensor, t: int) -> FloatTensor:
        return self.transformer.infer_next(self.encoder_output, self.encoder_mask, decoder_input, t, self.b)


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