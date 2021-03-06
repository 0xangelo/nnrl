"""Custom activation functions as neural network modules."""
import torch
from torch import nn


class Swish(nn.Module):
    r"""Swish activation function.

    Notes:
        Applies the mapping :math:`x \mapsto x \cdot \sigma(x)`,
        where :math:`sigma` is the sigmoid function.

    Reference:
        Eger, Steffen, Paul Youssef, and Iryna Gurevych.
        "Is it time to swish? Comparing deep learning activation functions
        across NLP tasks."
        arXiv preprint arXiv:1901.02671 (2019).
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # pylint:disable=arguments-differ,no-self-use,missing-function-docstring
        return value * value.sigmoid()
