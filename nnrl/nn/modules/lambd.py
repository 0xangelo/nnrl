# pylint:disable=missing-docstring
from torch import nn


class Lambda(nn.Module):
    """Neural network module that stores and applies a function on inputs."""

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.func(inputs)
