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
EncoderInput = NamedTuple('EncoderInput', [('encoder_input', FloatTensor), ('mask', Optional[LongTensor])])
DecoderInput = NamedTuple('DecoderInput', [('encoder_output', FloatTensor), ('decoder_input', FloatTensor),
                                           ('encoder_mask', Optional[LongTensor]), ('decoder_mask', LongTensor)])


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


def fuzz(y: LongTensor, num_classes: int) -> FloatTensor:
    return NotImplemented


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


class EncoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int) -> None:
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.ffn = FFN(d_intermediate, d_model)
        self.ln_mha = nn.LayerNorm(normalized_shape=d_model)
        self.ln_ffn = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: EncoderInput) -> EncoderInput:
        x = EncoderInput(encoder_input=F.dropout(x.encoder_input, p=0.1, training=self.training),
                         mask=x.mask)
        mha_x = self.ln_mha(x.encoder_input + self.mha(x.encoder_input, x.encoder_input, x.encoder_input, x.mask))
        return EncoderInput(encoder_input=self.ln_ffn(mha_x + self.ffn(mha_x)),
                            mask=x.mask)


def Encoder(num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int) -> nn.Sequential:
    layers = [EncoderLayer(num_heads, d_model, d_k, d_v, d_intermediate) for _
              in range(num_layers)]
    return nn.Sequential(*layers)


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
        x = DecoderInput(encoder_mask=x.encoder_mask,
                         encoder_output=x.encoder_output,
                         decoder_input=F.dropout(x.decoder_input, 0.1, training=self.training),
                         decoder_mask=x.decoder_mask)
        t = x.decoder_input.shape[1]
        m_mha_x = self.ln_m_mha(x.decoder_input +
                                self.mask_mha(x.decoder_input, x.decoder_input, x.decoder_input, x.decoder_mask))
        mha_x = self.ln_mha(m_mha_x + self.mha(m_mha_x, x.encoder_output, x.encoder_output,
                                               x.encoder_mask[:, :t, :]))
        return DecoderInput(encoder_mask=x.encoder_mask,
                            encoder_output=x.encoder_output,
                            decoder_input=self.ln_ffn(mha_x + self.ffn(mha_x)),
                            decoder_mask=x.decoder_mask)


def Decoder(num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int) -> nn.Sequential:
    return nn.Sequential(*[DecoderLayer(num_heads, d_model, d_k, d_v, d_intermediate) for _ in range(num_layers)])


class Transformer(nn.Module):
    def __init__(self, num_classes: int, output_embedder: tensor_map,
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
        self.predictor = nn.Linear(in_features=d_model, out_features=num_classes).to(self.device)
        self.output_embedder = output_embedder

    def forward(self, encoder_input: FloatTensor, decoder_input: FloatTensor, encoder_mask: LongTensor,
                decoder_mask: LongTensor) -> FloatTensor:
        self.train()

        b, n, dk = encoder_input.shape
        pe = PE(b, n, dk, dk, device=self.device)
        encoder_output = self.encoder(EncoderInput(encoder_input + pe, encoder_mask))
        decoder_output = self.decoder(DecoderInput(encoder_output=encoder_output.encoder_input,
                                                   encoder_mask=encoder_mask, decoder_input=decoder_input + pe,
                                                   decoder_mask=decoder_mask))
        return torch.log(sigsoftmax(self.predictor(decoder_output.decoder_input)))

    def infer(self, encoder_input: FloatTensor, encoder_mask: LongTensor, sos_symbol: int) -> FloatTensor:
        self.eval()

        b, n, dk = encoder_input.shape
        pe = PE(b, n, dk, dk, device=self.device)
        encoder_output = self.encoder(EncoderInput(encoder_input + pe, encoder_mask)).encoder_input
        sos_symbols = (torch.ones(b) * sos_symbol).long().to(self.device)
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

    def infer_next(self, encoder_output: FloatTensor, encoder_mask: LongTensor, decoder_output: FloatTensor,
                   t: int, b: int) -> FloatTensor:
        decoder_step = self.decoder(DecoderInput(encoder_output=encoder_output, encoder_mask=encoder_mask,
                                                 decoder_input=decoder_output,
                                                 decoder_mask=Mask((b, t + 1, t + 1)).to(self.device))) \
            .decoder_input
        prob_t = self.predictor(decoder_step[:, -1])
        return sigsoftmax(prob_t)  # b, num_classes

    def beam_search(self, encoder_input: FloatTensor, encoder_mask: LongTensor, sos_symbol: int, beam_width: int):
        self.eval()

        def forward_index(dim1: int, dim2: int) -> int:
            return dim1 * beam_width + dim2

        def backward_index(idx: int) -> Tuple[int, int]:
            return idx // beam_width, idx - (idx // beam_width) * beam_width

        with torch.no_grad():
            b, n, dk = encoder_input.shape
            pe = PE(b, n, dk, dk, device=self.device)
            encoder_output = self.encoder(EncoderInput(encoder_input + pe, encoder_mask)).encoder_input
            sos_symbols = (torch.ones(b, device=self.device) * sos_symbol).long()
            decoder_output = self.output_embedder(sos_symbols).unsqueeze(1) + pe[:, 0:1, :]
            inferer = Inferer(self, encoder_output, encoder_mask, b)

            outer_beam_paths = [torch.ones(b, 1).long().to(self.device) * sos_symbol
                                for _ in range(beam_width)]  # List of k [B, t] tensors
            outer_beam_scores = [torch.ones(b).to(self.device) for _ in range(beam_width)]  # List of k [B] tensors
            outer_beam_decoder_outputs = [decoder_output for _ in range(beam_width)]  # list of k [B, t, E] tensors

            for t in range(n-1):
                inner_beam_paths = torch.ones(b, beam_width**2, device=self.device, dtype=torch.long)
                inner_beam_scores = torch.zeros(b, beam_width**2, device=self.device)

                # iterate over outer beams
                for k_outer in range(beam_width):
                    prob_t = inferer(outer_beam_decoder_outputs[k_outer], t)  # B, num_classes
                    inner_beam_top_k = argmax_top_k(prob_t, k=beam_width)

                    # generate subsequent beams from current outer beam
                    for k_inner in range(beam_width):
                        # todo I am not masking the probabilities

                        # evaluate each generated beam
                        inner_beam_scores[:, forward_index(k_outer, k_inner)] = \
                            outer_beam_scores[k_outer] * inner_beam_top_k[k_inner][0]

                        inner_beam_paths[:, forward_index(k_outer, k_inner)] = inner_beam_top_k[k_inner][1]

                # select the best k inner beams
                outer_beam_top_k = argmax_top_k(inner_beam_scores, k=beam_width)
                # assign as new outer beam scores
                outer_beam_scores = [outer_beam_top_k[k_outer][0] for k_outer in range(beam_width)]

                # get the indices of the top_k in the k^2 matrix
                # and map each index to a pair of indices indexing the [k, k] space
                square_indices = [list(map(backward_index, x[1].tolist())) for x in outer_beam_top_k]

                new_outer_beam_paths = []
                for new_best in square_indices:
                    this_beam_path = torch.zeros(b, t+2, device=self.device, dtype=torch.long)
                    for k_outer, k_inner in new_best:
                        for s in range(b):
                            this_beam_path[s, :t+1] = outer_beam_paths[k_outer][s:s+1]
                            this_beam_path[s, t+1] = inner_beam_paths[s, forward_index(k_outer, k_inner)]
                    new_outer_beam_paths.append(this_beam_path)
                outer_beam_paths = new_outer_beam_paths

                # todo: I am re-embedding everything -> (NlogN)
                outer_beam_decoder_outputs = [self.output_embedder(x)+pe[:, :t+2] for x in outer_beam_paths]
        return outer_beam_paths, outer_beam_scores


class Inferer(object):
    def __init__(self, transformer: Transformer, encoder_output: FloatTensor, encoder_mask: FloatTensor,
                 b: int) -> None:
        self.transformer = transformer
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self.b = b

    def __call__(self, decoder_input: FloatTensor, t: int) -> FloatTensor:
        return self.transformer.infer_next(self.encoder_output, self.encoder_mask, decoder_input, t, self.b)


def test(device: str):
    embedder = torch.nn.Embedding(12, 300).to(device)
    t = Transformer(12, embedder, device=device)
    encoder_input = torch.rand(5, 3, 300).to(device)
    encoder_mask = torch.ones(5, 3, 3).to(device)
    decoder_input = torch.rand(5, 3, 300).to(device)
    decoder_mask = Mask((5, 3, 3)).to(device)
    paths, scores = t.beam_search(encoder_input, encoder_mask, 0, 3)
    f_v = t.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)
    i_v = t.infer(encoder_input, encoder_mask, 0)
