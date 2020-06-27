from torch.nn import Module, Linear, Dropout, functional
from torch import Tensor


class FFN(Module):
    """
        Implements a two-layer network, where the intermediate layer is activated by GELU.
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        """
        :param d_model: The input/output dimensionality.
        :param d_ff: The intermediate layer dimensionality.
        :param dropout_rate: The dropout rate, applied prior to the second projection.
        """
        super(FFN, self).__init__()
        self.linear_one = Linear(d_model, d_ff, bias=True)
        self.linear_two = Linear(d_ff, d_model, bias=True)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = functional.gelu(self.linear_one(x))
        x = self.dropout(x)
        return self.linear_two(x)
