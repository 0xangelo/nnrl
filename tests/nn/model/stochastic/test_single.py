import pytest
import torch
from torch import Tensor
from torch.autograd import grad

from nnrl.nn import NormalParams
from nnrl.nn.model import StochasticModel
from nnrl.nn.model.stochastic import MLPModel, ResidualStochasticModel


@pytest.fixture(scope="module", params=(True, False), ids=lambda x: f"Residual({x})")
def residual(request) -> bool:
    return request.param


@pytest.fixture
def module_cls():
    return MLPModel


@pytest.fixture(params=(True, False), ids=lambda x: f"InputDependentScale({x})")
def input_dependent_scale(request) -> bool:
    return request.param


@pytest.fixture
def spec(module_cls, input_dependent_scale: bool):
    return module_cls.spec_cls(input_dependent_scale=input_dependent_scale)


@pytest.fixture
def module(module_cls, obs_space, action_space, spec, residual) -> StochasticModel:
    mod = module_cls(obs_space, action_space, spec)
    return ResidualStochasticModel(mod) if residual else mod


def test_init(module: StochasticModel):
    assert hasattr(module, "params")
    assert hasattr(module, "dist")


def test_forward(
    mocker, module: StochasticModel, obs: Tensor, action: Tensor, next_obs: Tensor
):
    # pylint:disable=too-many-arguments
    params_spy = mocker.spy(NormalParams, "forward")

    params = module(obs, action)
    assert "loc" in params
    assert "scale" in params
    assert params_spy.called

    loc, scale = params["loc"], params["scale"]
    assert loc.shape == next_obs.shape
    assert scale.shape == next_obs.shape
    assert loc.dtype == torch.float32
    assert scale.dtype == torch.float32

    params = set(module.parameters())
    for par in params:
        par.grad = None
    loc.mean().backward()
    assert any(p.grad is not None for p in params)

    for par in params:
        par.grad = None

    module(obs, action)["scale"].mean().backward()
    assert any(p.grad is not None for p in params)


def test_sample(
    module: StochasticModel, obs: Tensor, action: Tensor, next_obs: Tensor, rew: Tensor
):
    sampler = module.sample
    inputs = (module(obs, action),)

    samples, logp = sampler(*inputs)
    samples_, _ = sampler(*inputs)
    assert samples.shape == next_obs.shape
    assert samples.dtype == next_obs.dtype
    assert logp.shape == rew.shape
    assert logp.dtype == rew.dtype
    assert not torch.allclose(samples, samples_)


def test_rsample_gradient_propagation(
    module: StochasticModel, obs: Tensor, action: Tensor
):
    sampler = module.rsample
    obs.requires_grad_(True)
    action.requires_grad_(True)
    params = module(obs, action)

    sample, logp = sampler(params)
    assert obs.grad_fn is None
    assert action.grad_fn is None
    sample.sum().backward(retain_graph=True)
    assert obs.grad is not None
    assert action.grad is not None

    obs.grad, action.grad = None, None
    logp.sum().backward()
    assert obs.grad is not None
    assert action.grad is not None


def test_log_prob(
    module: StochasticModel, obs: Tensor, action: Tensor, next_obs: Tensor, rew: Tensor
):
    logp = module.log_prob(next_obs, module(obs, action))

    assert torch.is_tensor(logp)
    assert logp.shape == rew.shape

    logp.sum().backward()
    assert all(p.grad is not None for p in module.parameters())


def test_reproduce(
    module: StochasticModel, obs: Tensor, action: Tensor, next_obs: Tensor, rew: Tensor
):
    next_obs_, logp_ = module.reproduce(next_obs, module(obs, action))
    assert next_obs_.shape == next_obs.shape
    assert next_obs_.dtype == next_obs.dtype
    assert torch.allclose(next_obs_, next_obs, atol=1e-5)
    assert logp_.shape == rew.shape

    next_obs_.mean().backward()
    params = set(module.parameters())
    assert all(p.grad is not None for p in params)


def test_deterministic(
    module: StochasticModel, obs: Tensor, action: Tensor, rew: Tensor
):
    params = module(obs, action)

    obs1, logp1 = module.deterministic(params)
    assert torch.is_tensor(obs1)
    assert torch.is_tensor(logp1)
    assert obs1.shape == obs.shape
    assert logp1.shape == rew.shape
    assert obs1.dtype == obs.dtype
    assert logp1.dtype == rew.dtype

    obs2, logp2 = module.deterministic(params)
    assert torch.allclose(obs1, obs2)
    assert torch.allclose(logp1, logp2)

    assert obs1.grad_fn is not None
    obs1.sum().backward()
    assert any([p.grad is not None for p in module.parameters()])


def test_script(module: StochasticModel):
    torch.jit.script(module)


@pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/42459")
def test_script_model_ograd(module: StochasticModel, obs: Tensor, action: Tensor):
    model = torch.jit.script(module)
    obs = obs.clone().requires_grad_()

    rsample, _ = model.rsample(model(obs, action))
    (ograd,) = grad(rsample.mean(), [obs], create_graph=True)
    print(ograd)
    ograd.mean().backward()
    assert obs.grad is not None


@pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/42459")
def test_script_model_agrad(module: StochasticModel, obs: Tensor, action: Tensor):
    model = torch.jit.script(module)
    action = action.clone().requires_grad_()

    rsample, _ = model.rsample(model(obs, action))
    (agrad,) = grad(rsample.mean(), [action], create_graph=True)
    print(agrad)
    agrad.mean().backward()
    assert action.grad is not None
