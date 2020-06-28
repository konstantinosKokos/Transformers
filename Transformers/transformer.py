from torch.nn import Module
from Transformers.encoder import make_encoder
from Transformers.decoder import make_decoder
from Transformers.embedding import InvertibleEmbedder

from torch import Tensor, LongTensor, triu, ones, empty, cat

from typing import NoReturn


class Transformer(Module):
    def __init__(self, num_inp_classes: int, num_out_classes: int, encoder_heads: int, decoder_heads,
                 encoder_layers: int, decoder_layers: int, d_model: int, d_intermediate: int, dropout: float,
                 input_pad: int, output_pad: int) -> None:
        super(Transformer, self).__init__()
        self.inp_embedder = InvertibleEmbedder(num_embeddings=num_inp_classes, embedding_dim=d_model,
                                               padding_idx=input_pad, scale_by_sqrt=True)
        self.out_embedder = InvertibleEmbedder(num_embeddings=num_out_classes, embedding_dim=d_model,
                                               padding_idx=output_pad, scale_by_sqrt=True)
        self.encoder = make_encoder(num_layers=encoder_layers, num_heads=encoder_heads, d_model=d_model,
                                    d_k=d_model//encoder_heads, d_v=d_model//encoder_heads,
                                    d_intermediate=d_intermediate, dropout=dropout)
        self.decoder = make_decoder(num_layers=decoder_layers, num_heads_enc=encoder_heads, num_heads_dec=decoder_heads,
                                    d_encoder=d_model, d_decoder=d_model, d_atn_enc=d_model//encoder_heads,
                                    d_atn_dec=d_model//decoder_heads, d_v_enc=d_model//encoder_heads,
                                    d_v_dec=d_model//decoder_heads, d_interm=d_intermediate, dropout_rate=dropout)

    def forward(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError('Implicit forward not allowed.')

    def forward_train(self, encoder_idxes: LongTensor, decoder_idxes: LongTensor, encoder_mask: LongTensor,
                      decoder_mask: LongTensor) -> Tensor:
        self.train()

        encoder_embeddings = self.inp_embedder.embed(encoder_idxes)
        decoder_embeddings = self.out_embedder.embed(decoder_idxes)

        encoded = self.encoder((encoder_embeddings, encoder_mask))[0]
        decoded = self.decoder((encoded, encoder_mask, decoder_embeddings, decoder_mask))[2]
        return self.out_embedder.invert(decoded)

    def forward_greedy(self, encoder_idxes: LongTensor, encoder_mask: LongTensor, max_decode_length: int,
                       sos_id: int) -> LongTensor:
        self.eval()
        device = encoder_idxes.device

        b, s_in = encoder_idxes.shape
        s_out = max_decode_length

        encoder_embeddings = self.inp_embedder.embed(encoder_idxes)
        causal_mask = self._make_causal_mask(b, s_out).to(device)
        decoder_mask = self._make_encdec_mask(encoder_mask, s_out)

        encoded = self.encoder((encoder_embeddings, encoder_mask))

        output_symbols = (ones(b) * sos_id).long().to(device)
        decoder_input = (self.out_embedder.embed(output_symbols)).unsqueeze(1)
        output_symbols = output_symbols.unsqueeze(1)
        decoder_output = empty(b, s_out, encoder_embeddings.shape[2]).to(device)

        for t in range(s_out):
            _decoder_tuple_input = (encoded, decoder_mask,
                                    decoder_input, causal_mask[:, :t + 1, :t + 1])
            repr_t = self.decoder(_decoder_tuple_input)[2][:, -1]
            weights_t = self.out_embedder.invert(repr_t)
            class_t = weights_t.argmax(dim=-1)
            output_symbols = cat([output_symbols, class_t.unsqueeze(1)], dim=1)
            next_embedding = self.out_embedder.embed(class_t).unsqueeze(1)
            decoder_input = cat([decoder_input, next_embedding], dim=1)
            decoder_output[:, t] = repr_t
        return output_symbols

    def _make_encdec_mask(self, encoder_mask: LongTensor, max_dec_len: int) -> LongTensor:
        return encoder_mask[:, 0: 1, :].repeat(1, max_dec_len, 1)

    @staticmethod
    def _make_causal_mask(b: int, n: int) -> LongTensor:
        upper_triangular = triu(ones(b, n, n), diagonal=1)
        return ones(b, n, n) - upper_triangular
