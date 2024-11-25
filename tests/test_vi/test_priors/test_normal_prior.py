from math import log

import torch
from torch.distributions import Normal
from torch.nn import Module, Parameter

from torch_bayesian.vi.priors import MeanFieldNormalPrior


def test_log_prob() -> None:
    """Test MeanFieldNormalPrior.log_prob."""
    mean = 0.0
    std = 1.0
    prior = MeanFieldNormalPrior()
    assert prior.std == 1.0
    ref_dist = Normal(mean, std)
    shape1 = (3, 4)
    sample = ref_dist.sample(shape1)
    ref1 = ref_dist.log_prob(sample)
    log_prob1 = prior.log_prob(sample)
    assert (torch.isclose(ref1, log_prob1)).all()

    mean = 0.7
    std = 0.3
    prior = MeanFieldNormalPrior(mean, std)
    assert prior.std == std
    ref_dist = Normal(mean, std)
    shape2 = (6,)
    sample = ref_dist.sample(shape2)
    ref2 = ref_dist.log_prob(sample)
    log_prob2 = prior.log_prob(sample)
    assert (torch.isclose(ref2, log_prob2)).all()


def test_normal_reset_parameters() -> None:
    """Test MeanFieldNormalPrior.reset_parameters()."""
    param_shape = (5, 4)

    class ModuleDummy(Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_mean = Parameter(torch.empty(param_shape))
            self.weight_log_std = Parameter(torch.empty(param_shape))

        @staticmethod
        def variational_parameter_name(variable: str, parameter: str) -> str:
            return f"{variable}_{parameter}"

    prior = MeanFieldNormalPrior(3.0, 2.0)
    dummy = ModuleDummy()

    iter1 = dummy.parameters()
    prior.reset_parameters(dummy, "weight")

    mean = iter1.__next__().clone()
    log_std = iter1.__next__().clone()

    assert (mean == 3.0).all()
    assert (log_std == log(2.0)).all()
