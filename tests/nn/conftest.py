import numpy as np
import pytest
import torch
from gym import spaces
from torch import Tensor

from nnrl.nn.actor import (
    DeterministicPolicy,
    MLPContinuousPolicy,
    MLPDeterministicPolicy,
)
from nnrl.nn.critic import ActionValueCritic
from nnrl.utils import fake_space_samples


@pytest.fixture(scope="module", params=((1,), (4,)), ids=("Obs1Dim", "Obs4Dim"))
def obs_space(request):
    return spaces.Box(-10, 10, shape=request.param)


@pytest.fixture(scope="module", params=((1,), (4,)), ids=("Act1Dim", "Act4Dim"))
def action_space(request):
    return spaces.Box(-1, 1, shape=request.param)


@pytest.fixture(
    params=(pytest.param(True, marks=pytest.mark.slow), False),
    ids=("TorchScript", "Eager"),
    scope="module",
)
def torch_script(request):
    return request.param


@pytest.fixture
def batch_size() -> int:
    return 32


@pytest.fixture
def obs(obs_space: spaces.Space, batch_size: int) -> Tensor:
    return fake_space_samples(obs_space, batch_size)


@pytest.fixture
def action(action_space: spaces.Space, batch_size: int) -> Tensor:
    return fake_space_samples(action_space, batch_size)


@pytest.fixture
def next_obs(obs_space: spaces.Space, batch_size: int) -> Tensor:
    return fake_space_samples(obs_space, batch_size)


@pytest.fixture
def rew(batch_size: int) -> Tensor:
    return torch.as_tensor(np.random.randn(batch_size).astype(np.float32))


@pytest.fixture
def deterministic_policies(obs_space, action_space):
    spec = MLPDeterministicPolicy.spec_cls(
        units=(32,), activation="ReLU", norm_beta=1.2
    )
    policy = MLPDeterministicPolicy(obs_space, action_space, spec)
    target_policy = DeterministicPolicy.add_gaussian_noise(policy, noise_stddev=0.3)
    return policy, target_policy


@pytest.fixture(params=(True, False), ids=(f"PiScaleDep({b})" for b in (True, False)))
def policy_input_scale(request):
    return request.param


@pytest.fixture
def stochastic_policy(obs_space, action_space, policy_input_scale):
    config = {"encoder": {"units": (32,)}}
    mlp_spec = MLPContinuousPolicy.spec_cls.from_dict(config)
    return MLPContinuousPolicy(
        obs_space, action_space, mlp_spec, input_dependent_scale=policy_input_scale
    )


@pytest.fixture(params=(1, 2), ids=(f"Critics({n})" for n in (1, 2)))
def action_critics(request, obs_space, action_space):
    config = {
        "encoder": {"units": [32]},
        "double_q": request.param == 2,
        "parallelize": False,
    }
    spec = ActionValueCritic.spec_cls.from_dict(config)

    act_critic = ActionValueCritic(obs_space, action_space, spec)
    return act_critic.q_values, act_critic.target_q_values
