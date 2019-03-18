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
        pt = PT(b, self.num_repeats, n, dk, dk, device=x.encoder_input.device)
        for i in range(self.num_repeats):
            x = self.layer(EncoderInput(encoder_input=x.encoder_input + pt[i],
                                        mask=x.mask[:, :n]))
        return x


class RecurrentDecoder(nn.Module):
    def __init__(self, num_steps: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int,
                 dropout: float=0.1) -> None:
        super(RecurrentDecoder, self).__init__()
        self.layer = DecoderLayer(num_heads, d_model, d_k, d_v, d_intermediate, dropout)
        self.num_repeats = num_steps

    def forward(self, x: DecoderInput) -> DecoderInput:
        b, n, dk = x.decoder_input.shape
        pt = PT(b, self.num_repeats, n, dk, dk, device=x.encoder_output.device)
        for i in range(self.num_repeats):
            x = self.layer(DecoderInput(decoder_input=x.decoder_input + pt[i],
                                        encoder_output=x.encoder_output,
                                        decoder_mask=x.decoder_mask,
                                        encoder_mask=x.encoder_mask))
        return x


class UniversalTransformer(nn.Module):
    def __init__(self, num_classes: int, encoder_layers: int=6, encoder_heads: int=8, decoder_heads: int=8,
                 decoder_layers: int=6, d_model: int=300, d_intermediate: int=1024, dropout: float=0.1,
                 device: str='cpu', activation: Callable[[FloatTensor], FloatTensor]=sigsoftmax,
                 reuse_embedding: bool=True) -> None:
        self.device = device
        super(UniversalTransformer, self).__init__()
        self.encoder = RecurrentEncoder(num_steps=encoder_layers, num_heads=encoder_heads, d_model=d_model,
                                        d_k=d_model // encoder_heads, d_v=d_model // encoder_heads,
                                        dropout=dropout, d_intermediate=d_intermediate).to(self.device)
        self.decoder = RecurrentDecoder(num_steps=decoder_layers, num_heads=decoder_heads, d_model=d_model,
                                        d_k=d_model // decoder_heads, d_v=d_model // decoder_heads,
                                        dropout=dropout, d_intermediate=d_intermediate).to(self.device)
        self.embedding_matrix = torch.nn.Parameter(torch.rand(num_classes, d_model, device=device) * 0.02)
        self.output_embedder = lambda x: torch.nn.functional.embedding(x, self.embedding_matrix, padding_idx=0,
                                                                       scale_grad_by_freq=True)
        if reuse_embedding:
            self.predictor = lambda x: x@(self.embedding_matrix.transpose(1, 0) + 1e-10)
        else:
            self.predictor = nn.Linear(in_features=d_model, out_features=num_classes).to(self.device)
        # self.output_embedder = output_embedder
        self.activation = activation

    def forward(self, encoder_input: FloatTensor, decoder_input: FloatTensor, encoder_mask: LongTensor,
                decoder_mask: LongTensor) -> FloatTensor:
        self.train()

        encoder_output = self.encoder(EncoderInput(encoder_input=encoder_input,
                                                   mask=encoder_mask))
        decoder_output = self.decoder(DecoderInput(encoder_output=encoder_output.encoder_input,
                                                   encoder_mask=encoder_mask,
                                                   decoder_input=decoder_input,
                                                   decoder_mask=decoder_mask))
        prediction = self.predictor(decoder_output.decoder_input)
        return torch.log(self.activation(prediction))

    def infer(self, encoder_input: FloatTensor, encoder_mask: LongTensor, sos_symbol: int) -> FloatTensor:
        self.eval()

        with torch.no_grad():
            b, n, dk = encoder_input.shape
            max_steps = encoder_mask.shape[1]
            encoder_output = self.encoder(EncoderInput(encoder_input, encoder_mask)).encoder_input
            sos_symbols = (torch.ones(b, device=self.device) * sos_symbol).long()
            decoder_output = self.output_embedder(sos_symbols).unsqueeze(1)
            output_probs = torch.Tensor().to(self.device)
            inferer = infer_wrapper(self, encoder_output, encoder_mask, b)
            decoder_mask = Mask((b, encoder_mask.shape[1], encoder_mask.shape[1])).to(self.device)

            for t in range(max_steps):
                prob_t = inferer(decoder_output, t, decoder_mask)
                class_t = prob_t.argmax(dim=-1)
                emb_t = self.output_embedder(class_t).unsqueeze(1)
                decoder_output = torch.cat([decoder_output, emb_t], dim=1)
                output_probs = torch.cat([output_probs, prob_t.unsqueeze(1)], dim=1)
        return output_probs

    def infer_one(self, encoder_output: FloatTensor, encoder_mask: LongTensor, decoder_output: FloatTensor,
                  t: int, b: int, decoder_mask: Optional[LongTensor]=None) -> FloatTensor:
        if decoder_mask is None:
            decoder_mask = Mask((b, t+1, t+1)).to(self.device)
        decoder_step = self.decoder(DecoderInput(encoder_output=encoder_output, encoder_mask=encoder_mask,
                                                 decoder_input=decoder_output,
                                                 decoder_mask=decoder_mask[:, :t+1, :t+1])).decoder_input

        prob_t = self.predictor(decoder_step[:, -1])
        return self.activation(prob_t)  # b, num_classes

    def vectorized_beam_search(self, encoder_input: FloatTensor, encoder_mask: LongTensor, sos_symbol: int,
                               beam_width: int):
        self.eval()

        def forward_index(dim1: int, dim2: int) -> int:
            return dim1 * beam_width + dim2

        def backward_index(idx: int) -> Tuple[int, int]:
            return idx // beam_width, idx - (idx // beam_width) * beam_width

        with torch.no_grad():
            b, n, dk = encoder_input.shape

            encoder_output = self.encoder(EncoderInput(encoder_input, encoder_mask)).encoder_input
            sos_symbols = (torch.ones(b, device=self.device) * sos_symbol).long()
            # tensor of shape B, 1, F
            decoder_output = self.output_embedder(sos_symbols).unsqueeze(1)

            decoder_mask = Mask((b, encoder_mask.shape[1], encoder_mask.shape[1])).to(self.device)

            # construct first inferer for single batch
            inferer = infer_wrapper(self, encoder_output, encoder_mask, b)

            # first branching
            # get first outer probabilities
            probs_0 = inferer(decoder_output, 0, decoder_mask)
            # pick best K of them
            outer_beam_scores, outer_beam_paths = argmax_top_k(probs_0, k=beam_width)
            # embed them, concatenate with sos symbols and reshape for batching
            outer_beam_decoder_outputs = torch.cat((decoder_output.repeat(beam_width, 1, 1),
                                                    self.output_embedder(outer_beam_paths).view(beam_width*b, 1, dk)),
                                                   dim=1)
            # expand the paths for later
            outer_beam_paths = outer_beam_paths.unsqueeze(-1)
            # construct a new inferer for batched beams
            inferer = infer_wrapper(self, encoder_output.repeat(beam_width, 1, 1),
                                    encoder_mask.repeat(beam_width, 1, 1), b * beam_width)

            decoder_mask = decoder_mask.repeat(beam_width, 1, 1)

            for t in range(1, n-1):
                # tensor of shape K, B, N
                probs_t = inferer(outer_beam_decoder_outputs, t, decoder_mask).view(beam_width, b, -1)

                # list of K tuples, each containing scores and indices
                per_beam_top_k = [argmax_top_k(probs_t[i], k=beam_width) for i in range(beam_width)]

                # tensor of shape K0, K1, B, where K0 indexes the source and K1 indexes the gen
                per_beam_scores = torch.cat([x[0].unsqueeze(0) for x in per_beam_top_k])
                per_beam_paths = torch.cat([x[1].unsqueeze(0) for x in per_beam_top_k])

                # tensor of shape K, K, B
                masked_sentences = (encoder_mask[:, t + 1, t + 1] == 0).repeat(beam_width, beam_width, 1)
                per_beam_scores[masked_sentences] = 0.

                outer_beam_scores = outer_beam_scores.unsqueeze(1).expand(beam_width, beam_width, b)
                per_beam_scores = per_beam_scores + outer_beam_scores

                # tensor of shape K^2, B -> B, K^2
                per_beam_scores = per_beam_scores.view(beam_width ** 2, b).transpose(1, 0)
                # tensors of shape K, B
                outer_beam_scores, outer_beam_indices = argmax_top_k(per_beam_scores, k=beam_width)

                square_indices = [list(map(backward_index, x)) for x in outer_beam_indices.tolist()]

                # tensor of shape K, B, t+2, F
                new_outer_beam_decoder_outputs = torch.zeros(beam_width, b, t + 2, dk,
                                                             device=self.device, dtype=torch.float)
                # update the paths and embeddings
                new_outer_beam_paths = torch.tensor([], dtype=torch.long, device=self.device)
                outer_beam_decoder_outputs = outer_beam_decoder_outputs.view(beam_width, b, t+1, dk)
                for i, new_best in enumerate(square_indices):
                    this_beam_path = torch.tensor([], dtype=torch.long, device=self.device)
                    for s, (k_outer, k_inner) in enumerate(new_best):
                        this_sentence_history = outer_beam_paths[k_outer][s:s+1].squeeze(0)
                        this_sentence_future = per_beam_paths[k_outer, k_inner, s:s+1]
                        this_sentence_path = torch.cat((this_sentence_history, this_sentence_future))
                        this_beam_path = torch.cat((this_beam_path, this_sentence_path.unsqueeze(0)), dim=0)
                        new_outer_beam_decoder_outputs[i, s, :t+1] = outer_beam_decoder_outputs[k_outer][s]

                    new_outer_beam_paths = torch.cat((new_outer_beam_paths, this_beam_path.unsqueeze(0)), dim=0)

                new_outer_beam_decoder_outputs[:, :, t + 1] = \
                    self.output_embedder(new_outer_beam_paths[:, :, -1])

                outer_beam_paths = new_outer_beam_paths
                outer_beam_decoder_outputs = new_outer_beam_decoder_outputs.view(beam_width * b, t + 2, dk)
        return outer_beam_paths, outer_beam_scores


def test(device: str):
    sl = 25
    nc = 1000

    t = UniversalTransformer(12, device=device)
    encoder_input = torch.rand(128, sl, 300).to(device)
    encoder_mask = torch.ones(128, sl * 2, sl).to(device)
    decoder_input = torch.rand(128, sl * 2, 300).to(device)
    decoder_mask = Mask((128, sl*2, sl*2)).to(device)
    paths, scores = t.vectorized_beam_search(encoder_input[:5], encoder_mask[:5], 0, 3)
    f_v = t.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)
    i_v = t.infer(encoder_input, encoder_mask, 0)
