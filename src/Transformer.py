from typing import NamedTuple, Optional, Callable, Iterable, Any, Union, Tuple, List, Sequence
from torch.nn import functional as F
from torch import nn
import torch
import math
import numpy as np

try:
    from src.utils import *
    from src.BeamSearch import Beam
except ImportError:
    from Transformer.src.utils import *
    from Transformer.src.BeamSearch import Beam


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
        mha_x = F.dropout(mha_x, p=self.dropout_rate, training=self.training)
        mha_x = mha_x + x.encoder_input
        mha_x = self.ln_mha(mha_x)

        ffn_x = self.ffn(mha_x)
        ffn_x = F.dropout(ffn_x, p=self.dropout_rate, training=self.training)
        ffn_x = ffn_x + mha_x
        ffn_x = self.ln_ffn(ffn_x)
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
        m_mha_x = F.dropout(m_mha_x, p=self.dropout_rate, training=self.training)
        m_mha_x = m_mha_x + x.decoder_input
        m_mha_x = self.ln_m_mha(m_mha_x )

        mha_x = self.mha(m_mha_x, x.encoder_output, x.encoder_output, x.encoder_mask[:, :t, :])
        mha_x = F.dropout(mha_x, p=self.dropout_rate, training=self.training)
        mha_x = mha_x + m_mha_x
        mha_x = self.ln_mha(mha_x)

        ffn_x = self.ffn(mha_x)
        ffn_x = F.dropout(ffn_x, p=self.dropout_rate, training=self.training)
        ffn_x = ffn_x + mha_x
        ffn_x = self.ln_ffn(ffn_x)

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

        with torch.no_grad():
            b, n, dk = encoder_input.shape
            pe = PE(b, n, dk, dk, device=self.device)
            encoder_output = self.encoder(EncoderInput(encoder_input + pe, encoder_mask)).encoder_input
            sos_symbols = (torch.ones(b) * sos_symbol).long().to(self.device)
            decoder_output = self.output_embedder(sos_symbols).unsqueeze(1) + pe[:, 0:1, :]
            output_probs = torch.Tensor().to(self.device)
            inferer = infer_wrapper(self, encoder_output, encoder_mask, b)

            for t in range(n):
                prob_t = inferer(decoder_output, t)
                class_t = prob_t.argmax(dim=-1)
                emb_t = self.output_embedder(class_t).unsqueeze(1) + pe[:, t + 1:t + 2, :]
                decoder_output = torch.cat([decoder_output, emb_t], dim=1)
                output_probs = torch.cat([output_probs, prob_t.unsqueeze(1)], dim=1)

        return output_probs

    def infer_one(self, encoder_output: FloatTensor, encoder_mask: LongTensor, decoder_output: FloatTensor,
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
            sos_symbols = torch.ones(b, device=self.device, dtype=torch.long).fill_(sos_symbol)
            decoder_output = self.output_embedder(sos_symbols).unsqueeze(1) + pe[:, 0:1, :]
            inferer = infer_wrapper(self, encoder_output, encoder_mask, b)

            outer_beam_paths = torch.ones(beam_width, b, 1, device=self.device, dtype=torch.long)
            outer_beam_scores = torch.ones(beam_width, b, device=self.device)
            outer_beam_decoder_outputs = decoder_output.repeat(beam_width, 1, 1, 1)

            for t in range(n - 1):
                inner_beam_paths = torch.ones(b, beam_width ** 2, device=self.device, dtype=torch.long)
                inner_beam_scores = torch.cat([outer_beam_scores.transpose(1, 0) for _ in range(beam_width)], dim=1)

                # iterate over outer beams
                for k_outer in range(beam_width):
                    prob_t = inferer(outer_beam_decoder_outputs[k_outer], t)  # B, num_classes
                    inner_beam_top_k = argmax_top_k(prob_t, k=beam_width)

                    # generate subsequent beams from current outer beam
                    for k_inner in range(beam_width):
                        non_masked_sentences = encoder_mask[:, t + 1, t + 1] == 1

                        # evaluate each generated beam
                        inner_beam_scores[non_masked_sentences, forward_index(k_outer, k_inner)] *= \
                            inner_beam_top_k[k_inner][0][non_masked_sentences]

                        inner_beam_paths[:, forward_index(k_outer, k_inner)] = inner_beam_top_k[k_inner][1]

                # select the best k inner beams
                outer_beam_top_k = argmax_top_k(inner_beam_scores, k=beam_width)
                # assign as new outer beam scores
                outer_beam_scores = torch.cat([outer_beam_top_k[k_outer][0].unsqueeze(0)
                                               for k_outer in range(beam_width)], dim=0)

                # get the indices of the top_k in the k^2 matrix
                # and map each index to a pair of indices indexing the [k, k] space
                square_indices = [list(map(backward_index, x[1].tolist())) for x in outer_beam_top_k]

                new_outer_beam_paths = torch.zeros(beam_width, b, t + 2, device=self.device, dtype=torch.long)
                for i, new_best in enumerate(square_indices):
                    this_beam_path = torch.zeros(b, t + 2, device=self.device, dtype=torch.long)
                    for k_outer, k_inner in new_best:
                        for s in range(b):
                            this_beam_path[s, :t + 1] = outer_beam_paths[k_outer][s:s + 1]
                            this_beam_path[s, t + 1] = inner_beam_paths[s, forward_index(k_outer, k_inner)]
                    new_outer_beam_paths[i] = this_beam_path
                    # new_outer_beam_paths.append(this_beam_path)
                outer_beam_paths = new_outer_beam_paths

                # todo: I am re-embedding everything -> (NlogN)
                outer_beam_decoder_outputs = torch.cat([(self.output_embedder(x) + pe[:, :t + 2]).unsqueeze(0)
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

            inferer = infer_wrapper(self, encoder_output, encoder_mask, b)

            # first branching
            probs_0 = inferer(decoder_output, 0)
            outer_beam_scores, outer_beam_paths = argmax_top_k(probs_0, k=beam_width)
            outer_beam_decoder_outputs = torch.cat((decoder_output.repeat(beam_width, 1, 1),
                                                    self.output_embedder(outer_beam_paths).view(beam_width*b, 1, dk)),
                                                   dim=1)

            outer_beam_paths = outer_beam_paths.unsqueeze(-1)
            inferer = infer_wrapper(self, encoder_output.repeat(beam_width, 1, 1),
                                    encoder_mask.repeat(beam_width, 1, 1), b * beam_width)

            for t in range(1, n - 1):
                # tensor of shape K, B, N
                probs_t = inferer(outer_beam_decoder_outputs, t).view(beam_width, b, -1)

                # list of K lists of K tuples of float tensor of shape B, long tensor of shape B
                per_beam_top_k = [argmax_top_k(probs_t[i], k=beam_width) for i in range(beam_width)]

                # tensor of shape K, K, B (assert outer is
                per_beam_scores = torch.cat([x[0].unsqueeze(0) for x in per_beam_top_k])
                per_beam_paths = torch.cat([x[1].unsqueeze(0) for x in per_beam_top_k])

                # tensor of shape K, K, B
                masked_sentences = (encoder_mask[:, t + 1, t + 1] == 0).repeat(beam_width, beam_width, 1)
                per_beam_scores[masked_sentences] = 1

                # tensor of shape K, K, B containing the updated scores
                per_beam_scores = per_beam_scores * outer_beam_scores

                # tensor of shape K^2, B -> B, K^2
                per_beam_scores = per_beam_scores.view(beam_width ** 2, b).transpose(1, 0)
                # tensors of shape K, B
                outer_beam_scores, outer_beam_indices = argmax_top_k(per_beam_scores, k=beam_width)
                square_indices = [list(map(backward_index, x)) for x in outer_beam_indices.tolist()]

                # tensor of shape K, B, t+2, F
                new_outer_beam_decoder_outputs = torch.zeros(beam_width, b, t + 2, dk,
                                                             device=self.device, dtype=torch.float)

                # todo replace slicing and indexing ops with stack/cat
                # update the paths and embeddings
                new_outer_beam_paths = torch.tensor([], dtype=torch.long, device=self.device)
                for i, new_best in enumerate(square_indices):
                    this_beam_path = torch.tensor([], dtype=torch.long, device=self.device)
                    for s, (k_outer, k_inner) in enumerate(new_best):
                        this_sentence_history = outer_beam_paths[k_outer][s:s+1].squeeze(0)
                        this_sentence_future = per_beam_paths[k_outer, k_inner, s:s+1]
                        this_sentence_path = torch.cat((this_sentence_history, this_sentence_future))

                        # this_sentence_path = torch.cat((outer_beam_paths[k_outer][s],
                        #                                 per_beam_paths[k_outer, k_inner, s].unsqueeze(0)))
                        this_beam_path = torch.cat((this_beam_path, this_sentence_path.unsqueeze(0)), dim=0)

                    new_outer_beam_paths = torch.cat((new_outer_beam_paths, this_beam_path.unsqueeze(0)), dim=0)
                new_outer_beam_decoder_outputs[:, :, t + 1] = \
                    self.output_embedder(new_outer_beam_paths[:, :, t]) + pe[:, t + 1].repeat(beam_width, 1, 1)
                # new_outer_beam_decoder_outputs = self.output_embedder(new_outer_beam_paths) + pe[:, :t+2]

                outer_beam_paths = new_outer_beam_paths
                outer_beam_decoder_outputs = new_outer_beam_decoder_outputs.view(beam_width * b, t + 2, dk)
        return outer_beam_paths, outer_beam_scores


def test(device: str):
    sl = 25
    nc = 1000

    embedder = torch.nn.Embedding(nc, 300).to(device)
    t = Transformer(12, embedder, device=device)
    encoder_input = torch.rand(128, sl, 300).to(device)
    encoder_mask = torch.ones(128, sl, sl).to(device)
    decoder_input = torch.rand(128, sl, 300).to(device)
    decoder_mask = Mask((128, sl, sl)).to(device)
    beam = t.vectorized_beam_search(encoder_input[0:20], encoder_mask[0:20], 0, 3)
    # import pdb
    # pdb.set_trace()
    f_v = t.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)
    i_v = t.infer(encoder_input, encoder_mask, 0)
