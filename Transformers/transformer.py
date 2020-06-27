from torch.nn import Module
from Transformers.encoder import make_encoder
from Transformers.decoder import make_decoder
from Transformers.embedding import InvertibleEmbedder

from torch import Tensor, LongTensor

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
