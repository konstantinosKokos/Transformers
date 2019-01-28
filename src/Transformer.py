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


class EncoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int, dropout: float) -> None:
        super(EncoderLayer, self).__init__()
        self.dropout_rate = dropout
        self.mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.ffn = FFN(d_intermediate, d_model)
        self.ln_mha = nn.LayerNorm(normalized_shape=d_model)
        self.ln_ffn = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: EncoderInput) -> EncoderInput:
        mha_x = self.mha(x.encoder_input, x.encoder_input, x.encoder_input, x.mask)
        mha_x = self.ln_mha(F.dropout(mha_x, p=self.dropout_rate) + x.encoder_input)
        ffn_x = self.ffn(mha_x)
        ffn_x = self.ln_ffn(F.dropout(ffn_x, p=self.dropout_rate) + mha_x)
        return EncoderInput(encoder_input=ffn_x, mask=x.mask)


def Encoder(num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int, dropout: float=0.1)\
        -> nn.Sequential:
    layers = [EncoderLayer(num_heads, d_model, d_k, d_v, d_intermediate, dropout) for _
              in range(num_layers)]
    return nn.Sequential(*layers)


class DecoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int, dropout: float) \
            -> None:
        super(DecoderLayer, self).__init__()
        self.dropout_rate = dropout
        self.mask_mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.ffn = FFN(d_intermediate, d_model)
        self.ln_m_mha = nn.LayerNorm(normalized_shape=d_model)
        self.ln_mha = nn.LayerNorm(normalized_shape=d_model)
        self.ln_ffn = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: DecoderInput) -> DecoderInput:
        t = x.decoder_input.shape[1]
        m_mha_x = self.mask_mha(x.decoder_input, x.decoder_input, x.decoder_input, x.decoder_mask)
        m_mha_x = self.ln_m_mha(F.dropout(m_mha_x, p=self.dropout_rate) + x.decoder_input)
        mha_x = self.mha(m_mha_x, x.encoder_output, x.encoder_output, x.encoder_mask[:, :t, :])
        mha_x = self.ln_mha(F.dropout(mha_x, p=self.dropout_rate) + m_mha_x)
        ffn_x = self.ffn(mha_x)
        ffn_x = self.ln_ffn(F.dropout(ffn_x, p=self.dropout_rate) + mha_x)
        return DecoderInput(encoder_output=x.encoder_output,
                            decoder_input=ffn_x,
                            decoder_mask=x.decoder_mask,
                            encoder_mask=x.encoder_mask)


def Decoder(num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int, dropout: float=0.1)\
        -> nn.Sequential:
    return nn.Sequential(*[DecoderLayer(num_heads, d_model, d_k, d_v, d_intermediate, dropout) for _ in range(num_layers)])


class Transformer(nn.Module):
    def __init__(self, num_classes: int, output_embedder: tensor_map,
                 encoder_layers: int = 6, num_heads: int = 8, decoder_layers: int = 6, d_model: int = 300,
                 d_intermediate: int = 128, dropout: float=0.1, device: str='cpu') -> None:
        self.device = device
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers=encoder_layers, num_heads=num_heads, d_model=d_model,
                               d_k=d_model // num_heads, d_v=d_model // num_heads,
                               d_intermediate=d_intermediate, dropout=dropout).to(self.device)
        self.decoder = Decoder(num_layers=decoder_layers, num_heads=num_heads, d_model=d_model,
                               d_k=d_model // num_heads, d_v=d_model // num_heads,
                               d_intermediate=d_intermediate, dropout=dropout).to(self.device)
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

            outer_beam_paths = torch.ones(beam_width, b, 1, device=self.device, dtype=torch.long)
            outer_beam_scores = torch.ones(beam_width, b, device=self.device)
            outer_beam_decoder_outputs = decoder_output.repeat(beam_width, 1, 1, 1)

            for t in range(n-1):
                inner_beam_paths = torch.ones(b, beam_width**2, device=self.device, dtype=torch.long)
                inner_beam_scores = torch.zeros(b, beam_width**2, device=self.device)

                # iterate over outer beams
                for k_outer in range(beam_width):
                    prob_t = inferer(outer_beam_decoder_outputs[k_outer], t)  # B, num_classes
                    inner_beam_top_k = argmax_top_k(prob_t, k=beam_width)

                    # generate subsequent beams from current outer beam
                    for k_inner in range(beam_width):
                        non_masked_sentences = encoder_mask[:, t+1, t+1] == 1
                        masked_sentences = encoder_mask[:, t+1, t+1] == 0

                        # evaluate each generated beam
                        inner_beam_scores[non_masked_sentences, forward_index(k_outer, k_inner)] = \
                            outer_beam_scores[k_outer][non_masked_sentences] *\
                            inner_beam_top_k[k_inner][0][non_masked_sentences]

                        inner_beam_scores[masked_sentences, forward_index(k_outer, k_inner)] = \
                            outer_beam_scores[k_outer][masked_sentences]

                        inner_beam_paths[:, forward_index(k_outer, k_inner)] = inner_beam_top_k[k_inner][1]

                # select the best k inner beams
                outer_beam_top_k = argmax_top_k(inner_beam_scores, k=beam_width)
                # assign as new outer beam scores
                outer_beam_scores = torch.cat([outer_beam_top_k[k_outer][0].unsqueeze(0)
                                               for k_outer in range(beam_width)], dim=0)

                # get the indices of the top_k in the k^2 matrix
                # and map each index to a pair of indices indexing the [k, k] space
                square_indices = [list(map(backward_index, x[1].tolist())) for x in outer_beam_top_k]

                new_outer_beam_paths = torch.zeros(beam_width, b, t+2, device=self.device, dtype=torch.long)
                for i, new_best in enumerate(square_indices):
                    this_beam_path = torch.zeros(b, t+2, device=self.device, dtype=torch.long)
                    for k_outer, k_inner in new_best:
                        for s in range(b):
                            this_beam_path[s, :t+1] = outer_beam_paths[k_outer][s:s+1]
                            this_beam_path[s, t+1] = inner_beam_paths[s, forward_index(k_outer, k_inner)]
                    new_outer_beam_paths[i] = this_beam_path
                    # new_outer_beam_paths.append(this_beam_path)
                outer_beam_paths = new_outer_beam_paths

                # todo: I am re-embedding everything -> (NlogN)
                outer_beam_decoder_outputs = torch.cat([(self.output_embedder(x)+pe[:, :t+2]).unsqueeze(0)
                                                        for x in outer_beam_paths],
                                                       dim=0)
        return outer_beam_paths, outer_beam_scores

    def vectorized_beam_search(self, encoder_input: FloatTensor, encoder_mask: LongTensor, sos_symbol: int,
                               beam_width: int):
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
            # tensor of shape B, 1, F
            decoder_output = self.output_embedder(sos_symbols).unsqueeze(1) + pe[:, 0:1, :]

            inferer = Inferer(self, encoder_output.repeat(beam_width, 1, 1),
                              encoder_mask.repeat(beam_width, 1, 1), b*beam_width)

            # tensor of shape K, B, 1
            outer_beam_paths = torch.ones(beam_width, b, 1, device=self.device, dtype=torch.long)
            # tensor of shape K, B
            outer_beam_scores = torch.ones(beam_width, b, device=self.device)
            # tensor of shape K * B, 1, F
            outer_beam_decoder_outputs = decoder_output.repeat(beam_width, 1, 1)

            for t in range(n - 1):
                # todo : exception for t == 0

                # tensor of shape K, B, N
                probs_t = inferer(outer_beam_decoder_outputs, t).view(beam_width, b, -1)

                # list of K lists of K tuples of float tensor of shape B, long tensor of shape B
                per_beam_top_k = [argmax_top_k(probs_t[i], k=beam_width) for i in range(beam_width)]

                # tensor of shape K, K, B
                per_beam_scores = torch.cat([torch.cat([x[0].unsqueeze(0) for x in y], dim=0).unsqueeze(0)
                                             for y in per_beam_top_k], dim=0)
                # tensor of shape K, K, B
                per_beam_paths = torch.cat([torch.cat([x[1].unsqueeze(0) for x in y], dim=0).unsqueeze(0)
                                           for y in per_beam_top_k], dim=0)

                # tensor of shape K, K, B
                masked_sentences = (encoder_mask[:, t+1, t+1] == 0).repeat(beam_width, beam_width, 1)
                per_beam_scores[masked_sentences] = 1

                # tensor of shape K, K, B containing the updated scores
                per_beam_scores = per_beam_scores * outer_beam_scores

                # tensor of shape K^2, B
                per_beam_scores = per_beam_scores.view(beam_width**2, b)
                # list of k tuples of float tensor of shape B, long tensor of shape B
                outer_beam_top_k = argmax_top_k(per_beam_scores.transpose(1, 0), k=beam_width)

                # tensor of shape K, B
                outer_beam_scores = torch.cat([x[0].unsqueeze(0) for x in outer_beam_top_k], dim=0)

                # tensor of shape K, B, t+2
                new_outer_beam_paths = torch.zeros(beam_width, b, t + 2, device=self.device, dtype=torch.long)
                new_outer_beam_decoder_outputs = torch.zeros(beam_width, b, t+2, dk,
                                                             device=self.device, dtype=torch.float)
                outer_beam_decoder_outputs = outer_beam_decoder_outputs.view(beam_width, b, t+1, dk)

                square_indices = [list(map(backward_index, x[1].tolist())) for x in outer_beam_top_k]

                # update the paths and embeddings
                for i, new_best in enumerate(square_indices):
                    for k_outer, k_inner in new_best:
                        for s in range(b):
                            new_outer_beam_paths[i, s, :t+1] = outer_beam_paths[k_outer][s:s + 1]
                            new_outer_beam_paths[i, s, t+1:t+2] = per_beam_paths[k_outer, k_inner, s]
                            new_outer_beam_decoder_outputs[i, s, :t + 1] = outer_beam_decoder_outputs[i, s]

                new_outer_beam_decoder_outputs[:, :, t+1] = \
                    self.output_embedder(new_outer_beam_paths[:, :, t+1]) + pe[:, t+1].repeat(beam_width, 1, 1)
                # new_outer_beam_decoder_outputs = self.output_embedder(new_outer_beam_paths) + pe[:, :t+2]

                outer_beam_paths = new_outer_beam_paths
                outer_beam_decoder_outputs = new_outer_beam_decoder_outputs.view(beam_width * b, t+2, dk)
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
    sl = 25
    nc = 1000

    embedder = torch.nn.Embedding(nc, 300).to(device)
    t = Transformer(12, embedder, device=device)
    encoder_input = torch.rand(128, sl, 300).to(device)
    encoder_mask = torch.ones(128, sl, sl).to(device)
    decoder_input = torch.rand(128, sl, 300).to(device)
    decoder_mask = Mask((128, sl, sl)).to(device)
    # paths, scores = t.vectorized_beam_search(encoder_input, encoder_mask, 0, 3)
    # p2, s2 = t.beam_search(encoder_input, encoder_mask, 0, 3)
    # print((paths - p2).sum())
    # print((scores - s2).sum())
    # import pdb
    # pdb.set_trace()
    f_v = t.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)
    i_v = t.infer(encoder_input, encoder_mask, 0)
