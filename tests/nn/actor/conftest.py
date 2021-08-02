from typing import Union

import pytest
from gym.spaces import Box, Discrete
from torch import Tensor

from nnrl.utils import fake_space_samples

DISC_SPACES = (Discrete(2), Discrete(8))
CONT_SPACES = (Box(-1, 1, shape=(1,)), Box(-1, 1, shape=(3,)))
ACTION_SPACES = CONT_SPACES + DISC_SPACES


@pytest.fixture(params=DISC_SPACES, ids=(repr(a) for a in DISC_SPACES))
def disc_space(request) -> Discrete:
    return request.param


@pytest.fixture(params=CONT_SPACES, ids=(repr(a) for a in CONT_SPACES))
def cont_space(request) -> Box:
    return request.param


@pytest.fixture(params=ACTION_SPACES, ids=(repr(a) for a in ACTION_SPACES))
def action_space(request) -> Union[Box, Discrete]:
    return request.param


@pytest.fixture
def disc_act(disc_space: Discrete, batch_size: int) -> Tensor:
    return fake_space_samples(disc_space, batch_size)


@pytest.fixture
def cont_act(cont_space: Box, batch_size: int) -> Tensor:
    return fake_space_samples(cont_space, batch_size)


@pytest.fixture
def action(action_space: Union[Box, Discrete], batch_size: int) -> Tensor:
    return fake_space_samples(action_space, batch_size)
