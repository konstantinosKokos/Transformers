from typing import Optional

import torch
from torch import Tensor, LongTensor
from torch.nn import Module, Linear, Dropout

from opt_einsum import contract


def multihead_atn_fn(queries: Tensor, keys: Tensor, values: Tensor,
                     mask: Optional[LongTensor] = None) -> Tensor:
    """
        Computes the masked scaled dot-product attention.

    :param queries: A 4-dimensional tensor of shape (batch_size, num_queries, dim, num_heads)
    :param keys: A 4-dimensional tensor of shape (batch_size, num_keys, dim, num_heads)
    :param values: A 4-dimensional tensor of shape (batch_size, num_keys, dim, num_heads)
    :param mask: A 3-dimensional tensor of shape (batch_size, num_queries, num_keys), where mask[i, j, k] should be 1
    if keys[i, k] is allowed to attend to queries[i, j] and 0 otherwise.
    :return: A 3-dimensional tensor of shape (batch_size, num_queries, dim * num_heads)
    """

    dk, num_heads = keys.shape[-2:]
    dividend = torch.sqrt(torch.tensor(dk, device=queries.device, dtype=torch.float, requires_grad=False))

    weights: Tensor = contract('bidh,bodh->bioh', queries, keys) / dividend
    if mask is not None:
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_heads)
        weights = weights.masked_fill_(mask == 0, value=-1e10)
    weights = weights.softmax(dim=-2)
    return contract('bioh,bodh->bidh', weights, values).flatten(-2)


class MultiHeadAttention(Module):
    """
        Neural module wrapping multihead scaled dot-product attention.
    """
    def __init__(self, num_heads: int, d_q_in: int, d_k_in: int, d_v_in: int,
                 d_atn: int, d_v: int, d_out: int, dropout_rate: float = 0.1) -> None:
        """
        :param num_heads: The number of heads.
        :param d_q_in: The dimensionality of the input queries.
        :param d_k_in: The dimensionality of the input keys.
        :param d_v_in: The dimensionality of the input values.
        :param d_atn: The dimensionality of each attention head.
        :param d_v: The dimensionality of the input values.
        :param d_out: The dimensionality of the output values.
        :param dropout_rate: Dropout rate, applied prior to the final linear projection.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.q_transformation = Linear(in_features=d_q_in, out_features=d_atn*num_heads, bias=False)
        self.k_transformation = Linear(in_features=d_k_in, out_features=d_atn*num_heads, bias=False)
        self.v_transformation = Linear(in_features=d_v_in, out_features=d_v*num_heads, bias=False)
        self.wo = Linear(in_features=num_heads * d_v, out_features=d_out, bias=False)
        self.dropout = Dropout(dropout_rate)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Optional[LongTensor] = None) -> Tensor:
        """
            Given query, key and value tensors, projects each to the appropriate dimensionality and computes the
            multihead scaled dot-product attention.
        :param queries: A 3-dimensional tensor of shape (batch_size, num_queries, query_dimensionality).
        :param keys: A 3-dimensional tensor of shape (batch_size, num_keys, key_dimensionality).
        :param values: A 3-dimensional tensor of shape (batch_size, num_keys, value_dimensionality).
        :param mask: A 3-dimensional tensor of shape (batch_size, num_queries, num_keys).
        :return: A 3-dimensional tensor of shape (batch_size, num_queries, output_dimensionality).
        """
        qs = self.q_transformation(queries).view(queries.shape[0], queries.shape[1], -1, self.num_heads)
        ks = self.k_transformation(keys).view(keys.shape[0], keys.shape[1], -1, self.num_heads)
        vs = self.v_transformation(values).view(values.shape[0], values.shape[1], -1, self.num_heads)
        mha = multihead_atn_fn(qs, ks, vs, mask)
        mha = self.dropout(mha)
        return self.wo(mha)

