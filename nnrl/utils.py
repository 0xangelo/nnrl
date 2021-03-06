"""PyTorch related utilities."""
from typing import Dict, Iterator, Union

import numpy as np
import torch
from gym import spaces
from torch import Tensor
from torch.autograd import grad


def flat_grad(
    outputs: Union[Tensor, Iterator[Tensor]], inputs: Iterator[Tensor], *args, **kwargs
) -> Tensor:
    """Compute gradients and return a flattened array."""
    params = list(inputs)
    grads = grad(outputs, params, *args, **kwargs)
    zeros = torch.zeros
    return torch.cat(
        [zeros(p.numel()) if g is None else g.flatten() for p, g in zip(params, grads)]
    )


def convert_to_tensor(arr, device: torch.device) -> Tensor:
    """Convert array-like object to tensor and cast it to appropriate device.

    Arguments:
        arr (array_like): object which can be converted using `np.asarray`
        device: device to cast the resulting tensor to

    Returns:
        The array converted to a `Tensor`.
    """
    if torch.is_tensor(arr):
        return arr.to(device)
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor.to(device)


class TensorDictDataset(torch.utils.data.Dataset):
    """Dataset wrapping a dict of tensors.

    Args:
        tensor_dict: dictionary mapping strings to tensors
    """

    def __init__(self, tensor_dict: Dict[str, Tensor]):
        super().__init__()
        batch_size = next(iter(tensor_dict.values())).size(0)
        assert all(tensor.size(0) == batch_size for tensor in tensor_dict.values())
        self.tensor_dict = tensor_dict

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {k: v[index] for k, v in self.tensor_dict.items()}

    def __len__(self) -> int:
        return next(iter(self.tensor_dict.values())).size(0)


def fake_space_samples(space: spaces.Space, batch_size: int) -> Tensor:
    """Create fake samples from a Gym space."""
    if isinstance(space, spaces.Box):
        shape = (batch_size,) + space.shape
        randn = np.random.randn(*shape)
        array = np.clip(randn, space.low, space.high).astype(space.dtype)
    elif isinstance(space, spaces.Discrete):
        randint = np.random.randint(space.n, size=batch_size)
        array = randint.astype(space.dtype)
    else:
        raise ValueError(f"Unsupported space type {type(space)}")
    return torch.as_tensor(array)
