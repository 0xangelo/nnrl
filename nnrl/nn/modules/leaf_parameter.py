# pylint:disable=missing-docstring
import torch
from torch import nn


class LeafParameter(nn.Module):
    """Holds a single paramater vector an expands it to match batch shape of inputs."""

    def __init__(self, in_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(in_features))

    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.bias.expand(inputs.shape[:-1] + (-1,))
