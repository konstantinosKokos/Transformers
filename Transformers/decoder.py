from torch import Tensor, LongTensor
from torch.nn import Module, LayerNorm, ModuleList, Dropout

from Transformers.multihead_atn import MultiHeadAttention
from Transformers.ffn import FFN

from typing import Tuple, List


class Decoder(Module):
    def __init__(self, *modules: List['DecoderLayer']) -> None:
        super(Decoder, self).__init__()
        self.encoders = ModuleList(*modules)

    def forward(self, inp: Tuple[Tensor, LongTensor, Tensor, LongTensor]) \
            -> Tuple[Tensor, LongTensor, Tensor, LongTensor]:
        for encoder in self.encoders:
            inp = encoder(inp)
        return inp


class DecoderLayer(Module):
    """
     Implements a single bidirectional encoder layer.
     """
    def __init__(self, num_heads_enc: int, num_heads_dec: int, d_encoder: int, d_decoder: int,
                 d_atn_enc: int, d_atn_dec: int, d_v_enc: int, d_v_dec: int, d_interm: int, dropout_rate: float = 0.1):
        """
        :param num_heads_enc: The number of encoder-attentive heads.
        :param num_heads_dec: The number of self-attentive heads.
        :param d_encoder: The encoder dimensionality.
        :param d_decoder: The decoder dimensionality.
        :param d_atn_enc: The dimensionality of each encoder head.
        :param d_atn_dec: The dimensionality of each decoder head.
        :param d_v_enc: The dimensionality of each encoder transformation.
        :param d_v_dec: The dimensionality of each decoder transformation.
        :param d_interm: The dimensionality of the intermediate two-layer position-wise transformation.
        :param dropout_rate: The dropout rate applied through the layer.
        """
        super(DecoderLayer, self).__init__()
        self.dropout = Dropout(dropout_rate)
        self.mask_mha = MultiHeadAttention(num_heads=num_heads_dec, d_q_in=d_decoder, d_k_in=d_decoder,
                                           d_v_in=d_decoder, d_atn=d_atn_dec, d_v=d_v_dec,
                                           d_out=d_decoder, dropout_rate=dropout_rate)
        self.ln_masked_mha = LayerNorm(d_decoder)
        self.mha = MultiHeadAttention(num_heads=num_heads_enc, d_q_in=d_decoder, d_k_in=d_encoder,
                                      d_v_in=d_encoder, d_atn=d_atn_enc, d_v=d_v_enc,
                                      d_out=d_decoder, dropout_rate=dropout_rate)
        self.ln_mha = LayerNorm(d_decoder)
        self.ffn = FFN(d_model=d_decoder, d_ff=d_interm, dropout_rate=dropout_rate)
        self.ln_ffn = LayerNorm(d_decoder)

    def forward(self, inps: Tuple[Tensor, LongTensor, Tensor, LongTensor]) \
            -> Tuple[Tensor, LongTensor, Tensor, LongTensor]:
        """
        :param inps: A tuple containing tensors corresponding to (1) encoder_output: The 3-dimensional encoder's output,
         of shape (batch_size, enc_seq_len, enc_dim), (2) encoder_mask: The 3-dimensional encoder (padding) mask, of
         shape (batch_size, enc_seq_len, enc_seq_len), (3) decoder_input: The 3-dimensional output of the prior decoder
         (or embedding) layer of shape (batch_size, dec_seq_len, dec_dim), and (4) decoder_mask, the causal mask of
         shape (batch_size, dec_seq_len, enc_seq_len)
        :return: A tuple as in the input, with element (3) being the decoder layer's computation.
        """
        encoder_out, encoder_mask, decoder_in, decoder_mask = inps

        t = decoder_in.shape[1]

        x_drop = self.dropout(decoder_in)
        dec_atn = self.mask_mha(x_drop, x_drop, x_drop, decoder_mask)
        dec_atn = dec_atn + x_drop
        dec_atn = self.ln_masked_mha(dec_atn)

        enc_dec_atn = self.mha(dec_atn, encoder_out, encoder_out, encoder_mask[:, :t, :])
        enc_dec_atn = self.dropout(enc_dec_atn)
        enc_dec_atn = dec_atn + enc_dec_atn
        enc_dec_atn = self.ln_mha(enc_dec_atn)

        out = self.ffn(enc_dec_atn)
        out = self.dropout(out)
        out = out + enc_dec_atn
        out = self.ln_ffn(out)
        return encoder_out, encoder_mask, out, decoder_mask


def make_decoder(num_layers: int, num_heads_enc: int, num_heads_dec: int, d_encoder: int, d_decoder: int,
                 d_atn_enc: int, d_atn_dec: int, d_v_enc: int, d_v_dec: int, d_interm: int, dropout_rate: float = 0.1):
    """
            Constructs a chain of encoder layers as a single module.
    :param num_layers: The number of layers.
    :param num_heads_enc: The number of encoder-attentive heads per layer.
    :param num_heads_dec: The number of decoder-attentive heads per layer.
    :param d_encoder: The encoder dimensionality.
    :param d_decoder: The decoder dimensionality.
    :param d_atn_enc: The dimensionality of each encoder-attentive head.
    :param d_atn_dec: The dimensionality of each decoder-attentive head.
    :param d_v_enc: The dimensionality of each encoder value projection.
    :param d_v_dec: The dimensionality of each decoder value projection.
    :param d_interm: The intermediate dimensionality of the two-layer position-wise connection.
    :param dropout_rate: The dropout rate applied through the model.
    :return: A Decoder instance containing the chain of decoder layers.

    """
    return Decoder([DecoderLayer(num_heads_enc=num_heads_enc, num_heads_dec=num_heads_dec,
                                 d_encoder=d_encoder, d_decoder=d_decoder, d_atn_enc=d_atn_enc, d_atn_dec=d_atn_dec,
                                 d_v_enc=d_v_enc, d_v_dec=d_v_dec, d_interm=d_interm, dropout_rate=dropout_rate)
                    for _ in range(num_layers)])
