import pytest
import torch

from nnrl.nn.distributions.flows import IAF, MAF


@pytest.fixture(params=(MAF, IAF))
def module(request):
    return request.param(4, parity=True)


@pytest.mark.skip(reason="Output is sometimes NaN. Needs investigation.")
def test_flow(module, torch_script):
    module = torch.jit.script(module) if torch_script else module

    inputs = torch.randn(1, 4, requires_grad=True)

    latent, logpz = module(inputs)

    assert latent.grad_fn is not None
    assert logpz.grad_fn is not None
    latent.sum().backward(retain_graph=True)
    assert inputs.grad is not None

    latent = latent.detach().requires_grad_()

    inputs_, logpx = module(latent, reverse=True)
    assert torch.allclose(inputs, inputs_)
    assert inputs_.grad_fn is not None
    assert logpx.grad_fn is not None
    inputs_.sum().backward()
    assert latent.grad is not None
