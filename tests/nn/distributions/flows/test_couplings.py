import pytest
import torch

from nnrl.nn.distributions.flows.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
    PiecewiseRQSCouplingTransform,
)
from nnrl.nn.distributions.flows.masks import create_alternating_binary_mask
from nnrl.nn.networks import MLP, ResidualNet

PARITIES = (True, False)
IN_SIZES = (2, 3)
COUPLINGS = (
    AffineCouplingTransform,
    AdditiveCouplingTransform,
    PiecewiseRQSCouplingTransform,
)


@pytest.fixture(params=PARITIES, ids=(f"Parity({p})" for p in PARITIES))
def parity(request):
    return request.param


@pytest.fixture(params=IN_SIZES, ids=(f"InSize({i})" for i in IN_SIZES))
def mask(request, parity):
    return create_alternating_binary_mask(request.param, even=parity)


@pytest.fixture(params=(MLP, ResidualNet))
def transform_net_create_fn(request):
    return lambda i, o: request.param(i, o, 6)


@pytest.fixture(params=COUPLINGS)
def cls(request):
    return request.param


@pytest.mark.filterwarnings("ignore:Inputs to the softmax are not scaled::nnrl")
def test_creation(cls, mask, transform_net_create_fn):
    coupling = cls(mask, transform_net_create_fn)
    torch.jit.script(coupling)


@pytest.mark.filterwarnings("ignore:Inputs to the softmax are not scaled::nnrl")
def test_call(cls, mask, transform_net_create_fn):
    coupling = cls(mask, transform_net_create_fn)
    coupling = torch.jit.script(coupling)

    inputs = torch.randn(10, *mask.shape)
    params = {}
    out, logabsdet = coupling(inputs, params)

    latent, logdet = coupling(out, params, reverse=True)

    assert out.shape == inputs.shape
    assert latent.shape == inputs.shape
    assert logabsdet.shape == (10,)
    assert logdet.shape == (10,)
    assert torch.allclose(inputs, latent, atol=1e-5)
    assert torch.allclose(logabsdet, -logdet, atol=1e-5)
