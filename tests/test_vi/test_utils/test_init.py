import torch
from torch.nn import Module, Parameter

from torch_bayesian.vi.utils import init


def test_fixed() -> None:
    """Test init.fixed_()."""
    param_shape = (5, 4)

    class ModuleDummy(Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_mean = Parameter(torch.empty(param_shape))

    dummy = ModuleDummy()

    iter1 = dummy.parameters()
    other1 = torch.randn(param_shape, requires_grad=False)
    assert not torch.allclose(dummy.weight_mean, other1)

    init.fixed_(dummy.weight_mean, other1)
    weight1 = iter1.__next__().clone()
    assert (dummy.weight_mean == other1).all()
    assert (weight1 == other1).all()
    assert weight1.requires_grad
