from torch.nn import Module
import torch
from torch import LongTensor, Tensor
from torch.nn.functional import linear, embedding
from math import sqrt

from typing import NoReturn, Callable


def positional_encoding(b: int, n: int, d_model: int, freq: int = 10000, device: str = 'cpu') -> Tensor:
    pe = torch.zeros(n, d_model, device=device)
    position = torch.arange(0, n, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float) *
                         - (torch.log(torch.tensor(freq, dtype=torch.float, device=device)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.repeat(b, 1, 1)


def _standard_positional_encoding(x: Tensor) -> Tensor:
    return positional_encoding(x.shape[0], x.shape[1], x.shape[-1], device=str(x.device))


class InvertibleEmbedder(Module):
    """
        Implements an invertible embedder, as in Press & Wolf (2017).
    """
    def __init__(self, embedding_dim: int, num_embeddings: int, padding_idx: int, scale_by_sqrt: bool,
                 pos_enc: Callable[[Tensor], Tensor] = _standard_positional_encoding):
        """
        :param embedding_dim: The embedding dimensionality.
        :param num_embeddings: The number of classes.
        :param padding_idx: The padding index.
        :param scale_by_sqrt: Whether to divide the embeddings by the square root of the dimensionality.
        :param pos_enc: The positional encoding scheme used when embedding. Defaults to the standard transformer scheme.
        """

        super(InvertibleEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding_matrix = torch.nn.Parameter(data=torch.rand(num_embeddings, embedding_dim), requires_grad=True)
        self.padding_idx = padding_idx
        self.embedding_scale = sqrt(embedding_dim) if scale_by_sqrt else 1.

    def forward(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError('Implicit forward not allowed. Use either embed or invert.')

    def embed(self, idxes: LongTensor) -> Tensor:
        """
            Embeds an indexing tensor.
        :param idxes: A 2-dimensional tensor of shape (batch_size, seq_len) containing class indices.
        :return: A 3-dimensional tensor of shape (batch_size, seq_len, embedding_dim) containing class embeddings.
        """
        emb = embedding(idxes, self.embedding_matrix, self.padding_idx) * self.embedding_scale
        return emb + positional_encoding(idxes.shape[0], idxes.shape[1], self.embedding_dim, device=str(emb.device))

    def invert(self, weights: Tensor) -> Tensor:
        """
            Converts an embedding into class weights.
        :param weights: A 3-dimensional tensor of shape (batch_size, seq_len, embedding_dim) containing contextualized
        embeddings.
        :return: A 3-dimensional tensor of shape (batch_size, seq_len, num_classes) containing class weights.
        """
        return linear(weights, self.embedding_matrix)
