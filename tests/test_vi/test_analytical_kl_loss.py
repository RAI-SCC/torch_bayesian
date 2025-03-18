import random
from math import log, pi

import pytest
import torch
from torch import Tensor

from torch_bayesian.vi import KullbackLeiblerModule
from torch_bayesian.vi.analytical_kl_loss import (
    NonBayesianDivergence,
    NormalNormalDivergence,
    UniformNormalDivergence,
)
from torch_bayesian.vi.utils import use_norm_constants


def test_klmodule() -> None:
    """Test KullbackLeiblerModule."""
    module = KullbackLeiblerModule()

    list_a = [torch.arange(27, 31)]
    list_b = [torch.arange(3)]
    reference = torch.cat([*list_a, *list_b])

    with pytest.raises(NotImplementedError):
        module(list_a, list_b)

    def dummy_forward(*args: Tensor) -> Tensor:
        return torch.cat(args)

    module.forward = dummy_forward

    out = module(list_a, list_b)
    assert torch.equal(out, reference)


def test_nonbayesian_klmodule() -> None:
    """Test NonBayesianDivergence."""
    sample_shape = [7, 10]
    prior_param_number = random.randint(1, 5)
    var_param_number = 1

    prior_params = [*torch.randn([prior_param_number, *sample_shape])]
    var_params = [*torch.randn([var_param_number, *sample_shape])]

    module = NonBayesianDivergence()
    out = module(prior_params, var_params)
    assert torch.equal(out, torch.tensor([0.0]))


@pytest.mark.parametrize("norm_constants", [(True,), (False,)])
def test_uniformnormal_klmodule(norm_constants: bool) -> None:
    """Test UniformNormalDivergence."""
    sample_shape = [7, 10]
    prior_param_number = 0
    var_param_number = 2

    prior_params = [*torch.randn([prior_param_number, *sample_shape])]
    var_params = [*torch.randn([var_param_number, *sample_shape])]

    use_norm_constants(norm_constants)
    module = UniformNormalDivergence()
    out = module(prior_params, var_params)
    if not norm_constants:
        reference = var_params[1].sum()
    else:
        reference = var_params[1] - 0.5 * (1 + log(2 * pi))
        reference = reference.sum()
    assert torch.equal(out, reference)


@pytest.mark.parametrize("norm_constants", [(True,), (False,)])
def test_normalnormal_klmodule(norm_constants: bool) -> None:
    """Test NormalNormalDivergence."""
    sample_shape = [7, 10]
    prior_param_number = 2
    var_param_number = 2

    prior_params = [*torch.randn([prior_param_number, *sample_shape])]
    var_params = [*torch.randn([var_param_number, *sample_shape])]

    use_norm_constants(norm_constants)
    module = NormalNormalDivergence()
    out = module(prior_params, var_params)

    reference = (prior_params[1] - var_params[1]) + (
        prior_params[0] - var_params[0]
    ).pow(2) * torch.exp(2 * var_params[1]) / (2 * torch.exp(2 * prior_params[1]))
    if norm_constants:
        reference = reference - 0.5
    assert torch.allclose(out, reference.sum())
