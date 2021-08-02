import pytest
import torch
from gym.spaces import Box, Discrete
from torch import Tensor

from nnrl.nn.actor import MLPContinuousPolicy, MLPDiscretePolicy


@pytest.fixture(scope="module")
def base_cls():
    from nnrl.nn.actor import MLPStochasticPolicy

    return MLPStochasticPolicy


@pytest.fixture(scope="module")
def cont_cls():
    return MLPContinuousPolicy


@pytest.fixture(scope="module")
def disc_cls():
    return MLPDiscretePolicy


@pytest.fixture
def spec(base_cls):
    return base_cls.spec_cls()


@pytest.fixture(params=(True, False), ids=lambda x: f"InputDependentScale({x})")
def input_dependent_scale(request):
    return request.param


@pytest.fixture
def cont_policy(
    cont_cls, obs_space: Box, cont_space: Box, spec, input_dependent_scale: bool
) -> MLPContinuousPolicy:
    return cont_cls(obs_space, cont_space, spec, input_dependent_scale)


@pytest.fixture
def disc_policy(
    disc_cls, obs_space: Box, disc_space: Discrete, spec
) -> MLPDiscretePolicy:
    return disc_cls(obs_space, disc_space, spec)


def test_continuous_sample(
    cont_policy: MLPContinuousPolicy, obs: Tensor, cont_act: Tensor, rew: Tensor
):
    policy = cont_policy
    action = cont_act

    sampler = policy.rsample
    samples, logp = sampler(obs)
    samples_, _ = sampler(obs)
    assert samples.shape == action.shape
    assert samples.dtype == action.dtype
    assert logp.shape == rew.shape
    assert logp.dtype == rew.dtype
    assert not torch.allclose(samples, samples_)


def test_discrete_sample(
    disc_policy: MLPDiscretePolicy, obs: Tensor, disc_act: Tensor, rew: Tensor
):
    policy = disc_policy
    action = disc_act

    sampler = policy.sample
    samples, logp = sampler(obs)
    samples_, _ = sampler(obs)
    assert samples.shape == action.shape
    assert samples.dtype == action.dtype
    assert logp.shape == rew.shape
    assert logp.dtype == rew.dtype
    assert not torch.allclose(samples, samples_)


def test_continuous_params(
    cont_policy: MLPContinuousPolicy, obs: Tensor, cont_act: Tensor
):
    policy = cont_policy
    params = policy(obs)
    assert "loc" in params
    assert "scale" in params

    loc, scale = params["loc"], params["scale"]
    action = cont_act
    assert loc.shape == action.shape
    assert scale.shape == action.shape
    assert loc.dtype == torch.float32
    assert scale.dtype == torch.float32

    pi_params = set(policy.parameters())
    for par in pi_params:
        par.grad = None
    loc.mean().backward()
    assert any(p.grad is not None for p in pi_params)

    for par in pi_params:
        par.grad = None
    policy(obs)["scale"].mean().backward()
    assert any(p.grad is not None for p in pi_params)


def test_discrete_params(
    disc_policy: MLPDiscretePolicy, disc_space: Discrete, obs: Tensor
):
    policy = disc_policy

    params = policy(obs)
    assert "logits" in params
    logits = params["logits"]
    assert logits.shape[-1] == disc_space.n

    pi_params = set(policy.parameters())
    for par in pi_params:
        par.grad = None
    logits.mean().backward()
    assert any(p.grad is not None for p in pi_params)


def test_reproduce(
    cont_policy: MLPContinuousPolicy, obs: Tensor, cont_act: Tensor, rew: Tensor
):
    policy, acts = cont_policy, cont_act

    acts_, logp_ = policy.reproduce(obs, acts)
    assert acts_.shape == acts.shape
    assert acts_.dtype == acts.dtype
    assert torch.allclose(acts_, acts, atol=1e-5)
    assert logp_.shape == rew.shape

    acts_.mean().backward()
    pi_params = set(policy.parameters())
    assert all(p.grad is not None for p in pi_params)
