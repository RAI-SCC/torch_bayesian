from math import log

import torch
from torch.nn import Module, Parameter

from vi.priors import BasicQuietPrior


def test_log_prob() -> None:
    """Test BasicQuietPrior.log_prob()."""
    std_ratio1 = 0.5
    mean_mean1 = 0.1
    mean_std1 = 0.4
    prior1 = BasicQuietPrior(std_ratio1, mean_mean1, mean_std1)
    assert prior1.distribution_parameters == ("mean", "log_std")
    assert prior1._required_parameters == ("mean",)
    assert prior1._scaling_parameters == ("mean_mean", "mean_std")
    assert prior1._std_ratio == std_ratio1
    assert prior1.mean_mean == mean_mean1
    assert prior1.mean_std == mean_std1

    std_ratio2 = 1.0
    mean_mean2 = 0.0
    mean_std2 = 1.0
    prior2 = BasicQuietPrior()
    assert prior2._std_ratio == std_ratio2
    assert prior2.mean_mean == mean_mean2
    assert prior2.mean_std == mean_std2

    mean = torch.arange(0.1, 1.1, 0.1)
    sample1 = torch.zeros_like(mean)
    sample2 = mean.clone()
    ref1 = -0.5 - 0.5 * mean**2 - mean.log() - log(2 * torch.pi)
    ref2 = -0.5 * mean**2 - mean.log() - log(2 * torch.pi)

    log_prob1 = prior2.log_prob(sample1, mean)
    assert torch.allclose(log_prob1, ref1)

    log_prob2 = prior2.log_prob(sample2, mean)
    assert torch.allclose(log_prob2, ref2)


def test_reset_parameters() -> None:
    """Test BasicQuietPrior.reset_parameters()."""
    param_shape = (50, 40)

    class ModuleDummy(Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_mean = Parameter(torch.empty(param_shape))
            self.weight_log_std = Parameter(torch.empty(param_shape))

        @staticmethod
        def variational_parameter_name(variable: str, parameter: str) -> str:
            return f"{variable}_{parameter}"

    prior = BasicQuietPrior(0.5, -1.0, 5.0)
    dummy = ModuleDummy()
    mean0 = dummy.weight_mean.clone()

    iter1 = dummy.parameters()
    prior.reset_parameters(dummy, "weight")

    mean = iter1.__next__().clone()
    log_std = iter1.__next__().clone()

    assert not torch.allclose(mean0, mean)
    assert torch.allclose(mean.mean(), torch.tensor(-1.0), atol=1e-1)
    assert torch.allclose(mean.std(), torch.tensor(5.0), rtol=1e-1)
    assert (log_std == (mean.abs() / 2).log()).all()
