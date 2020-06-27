from torch import Tensor, LongTensor
from torch.nn import Module, LayerNorm, Dropout, ModuleList

from Transformers.multihead_atn import MultiHeadAttention
from Transformers.ffn import FFN

from typing import Tuple, List


class Encoder(Module):
    def __init__(self, *modules: List['EncoderLayer']) -> None:
        super(Encoder, self).__init__()
        self.encoders = ModuleList(*modules)

    def forward(self, inp: Tuple[Tensor, LongTensor]) -> Tuple[Tensor, LongTensor]:
        for encoder in self.encoders:
            inp = encoder(inp)
        return inp


class EncoderLayer(Module):
    """
    Implements a single bidirectional encoder layer.
    """
    def __init__(self, num_heads: int, d_model: int, d_atn: int, d_v: int, d_intermediate: int, dropout_rate: float) \
            -> None:
        """
        :param num_heads: The number of attention heads.
        :param d_model: The model dimensionality.
        :param d_atn: The dimensionality of each attention head (usually d_model//num_heads).
        :param d_v:The dimensionality of each value transformation (usually d_model//num_heads).
        :param d_intermediate: The intermediate dimensionality of the two-layer position-wise connection.
        :param dropout_rate: The dropout rate applied through the layer.
        """
        super(EncoderLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.mha = MultiHeadAttention(num_heads,
                                      d_q_in=d_model, d_k_in=d_model, d_v_in=d_model, d_atn=d_atn, d_v=d_v,
                                      d_out=d_model, dropout_rate=dropout_rate)
        self.ffn = FFN(d_model=d_model, d_ff=d_intermediate)
        self.ln_mha = LayerNorm(normalized_shape=d_model)
        self.ln_ffn = LayerNorm(normalized_shape=d_model)
        self.dropout = Dropout(dropout_rate)

    def forward(self, inps: Tuple[Tensor, LongTensor]) -> Tuple[Tensor, LongTensor]:
        """
        :param inps: A tuple consisting of tensor X of shape (batch_size, seq_len, model_dim) and a mask M of shape
        (batch_size, seq_len, seq_len) containing padding masks.
        :return: A tuple consisting of tensor Y of shape (batch_size, seq_len, model_dim) and the mask M.
        """
        encoder_input, encoder_mask = inps

        encoder_input = self.dropout(encoder_input)
        mha_x = self.mha(encoder_input, encoder_input, encoder_input, encoder_mask)
        mha_x = self.dropout(mha_x)
        mha_x = encoder_input + mha_x
        mha_x = self.ln_mha(mha_x)

        ffn_x = self.ffn(mha_x)
        ffn_x = self.dropout(ffn_x)
        ffn_x = ffn_x + mha_x
        ffn_x = self.ln_ffn(ffn_x)
        return ffn_x, encoder_mask


def make_encoder(num_layers: int, num_heads: int, d_model: int, d_k: int, d_v: int, d_intermediate: int,
                 dropout: float = 0.1) -> Encoder:
    """
        Constructs a chain of encoder layers as a single module.
    :param num_layers: The number of encoder layers.
    :param num_heads: The number of attention heads per layer.
    :param d_model: The model dimensionality.
    :param d_k: The dimensionality of each attention head (usually d_model//num_heads).
    :param d_v: The dimensionaltiy of each value transformation (usually d_model//num_heads).
    :param d_intermediate: The intermediate dimensionality of the two-layer position-wise connection.
    :param dropout: The dropout rate applied through the model.
    :return: An Encoder containing the chain of encoder layers.
    """
    return Encoder([EncoderLayer(num_heads, d_model, d_k, d_v, d_intermediate, dropout)
                    for _ in range(num_layers)])
