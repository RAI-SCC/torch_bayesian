import torch
from torch import Tensor


def fixed_(tensor: Tensor, other: Tensor) -> Tensor:
    """Initialize tensor to be equal to other."""
    assert (
        tensor.size() == other.size()
    ), "Values must be provided for all tensor elements."
    with torch.no_grad():
        return tensor.put_(torch.arange(tensor.numel()), other)
