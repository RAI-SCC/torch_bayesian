from typing import Tuple

import torch
from torch import Tensor

from vi import VIBaseModule, VIModule
from vi.variational_distributions import VariationalDistribution


def test_normal_sample() -> None:
    """Test _normal_sample."""
    mean = torch.randn((3, 4))
    std = torch.zeros_like(mean)
    sample = VIModule._normal_sample(mean, std)
    assert sample.shape == mean.shape
    assert (sample == mean).all()

    std = torch.ones_like(mean)
    sample = VIModule._normal_sample(mean, std)
    assert not (sample == mean).all()


def test_expand_to_samples() -> None:
    """Test _expand_to_samples."""
    shape1 = (3, 4)
    sample1 = torch.randn(shape1)
    out1 = VIModule._expand_to_samples(sample1, samples=5)
    assert out1.shape == (5,) + shape1
    for s in out1:
        assert (s == sample1).all()

    shape2 = (5,)
    sample2 = torch.randn(shape2)
    out2 = VIModule._expand_to_samples(sample2, samples=1)
    assert out2.shape == (1,) + shape2
    for s in out2:
        assert (s == sample2).all()

    out3 = VIModule._expand_to_samples(None, samples=2)
    assert (out3 == torch.tensor([False, False])).all()


def test_no_forward_error() -> None:
    """Test that forward throws error if not implemented."""
    module = VIModule()
    try:
        module.forward(torch.randn((3, 4)))
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == 'Module [VIModule] is missing the required "forward" function'


def test_sampled_forward() -> None:
    """Test _sampled_forward."""

    class Test(VIModule):
        def __init__(self, ref: Tensor) -> None:
            super().__init__()
            self.ref = ref

        def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
            assert x.shape == self.ref.shape
            return x - self.ref, torch.tensor(False)

    shape1 = (3, 4)
    sample1 = torch.randn(shape1)
    test1 = Test(ref=sample1)
    assert (
        test1.sampled_forward(sample1, samples=5)[0] == torch.zeros((5,) + shape1)
    ).all()

    shape2 = (5,)
    sample2 = torch.randn(shape2)
    test2 = Test(ref=sample2)
    assert (
        test2.sampled_forward(sample2, samples=1)[0] == torch.zeros((1,) + shape2)
    ).all()


def test_name_maker() -> None:
    """Test VIBaseModule.variational_parameter_name."""
    assert VIBaseModule.variational_parameter_name("a", "b") == "_a_b"
    assert VIBaseModule.variational_parameter_name("vw", "xz") == "_vw_xz"


def test_vibasemodule() -> None:
    """Test VIBaseModule."""
    var_dict1 = dict(
        weight=(2, 3),
        bias=(3,),
    )
    var_params = ("mean", "std")
    default_params = (0.0, 0.3)

    class TestDistribution(VariationalDistribution):
        variational_parameters = var_params
        _default_variational_parameters = default_params

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            pass

        def log_prob(self, sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
            pass

    module = VIBaseModule(var_dict1, TestDistribution())

    for var in var_dict1:
        for param in var_params:
            param_name = module.variational_parameter_name(var, param)
            assert hasattr(module, param_name)
            if param != "mean":
                index = var_params.index(param)
                default = default_params[index]
                assert (getattr(module, param_name) == default).all()

    # Check that reset_mean randomizes the means
    weight_mean = module._weight_mean.clone()
    bias_mean = module._bias_mean.clone()

    module.reset_parameters()

    assert not (module._weight_mean == weight_mean).all()
    assert not (module._bias_mean == bias_mean).all()


def test_get_variational_parameters() -> None:
    """Test VIBaseModule.get_variational_parameters."""
    var_dict1 = dict(
        weight=(2, 3),
        bias=(3,),
    )
    var_params = ("mean", "std")
    default_params = (0.0, 0.3)

    class TestDistribution(VariationalDistribution):
        variational_parameters = var_params
        _default_variational_parameters = default_params

        def sample(self, mean: Tensor, std: Tensor) -> Tensor:
            pass

        def log_prob(self, sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
            pass

    module = VIBaseModule(var_dict1, TestDistribution())

    for variable in ("weight", "bias"):
        params_list = module.get_variational_parameters(variable)
        for param_name, param_value in zip(var_params, params_list):
            assert param_value.shape == var_dict1[variable]
            assert (
                param_value
                == getattr(
                    module, module.variational_parameter_name(variable, param_name)
                )
            ).all()
