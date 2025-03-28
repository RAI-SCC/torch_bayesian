from math import log

import torch
from pytest import mark
from torch.distributions import StudentT

from torch_bayesian.vi.utils import use_norm_constants
from torch_bayesian.vi.variational_distributions import StudentTVarDist


@mark.parametrize("degrees_of_freedom", [1.0, 2.0, 3.0, 4.0])
def test_student_t_sample(degrees_of_freedom: float, device: torch.device) -> None:
    """Test _student_t_sample."""
    mean = torch.randn((3, 4), device=device)
    std = torch.zeros_like(mean, device=device)

    vardist = StudentTVarDist(degrees_of_freedom=degrees_of_freedom)

    sample = vardist._student_t_sample(mean, std)
    assert sample.shape == mean.shape
    assert torch.equal(sample, mean)
    assert sample.device == device

    std = torch.ones_like(mean, device=device)
    sample = vardist._student_t_sample(mean, std)
    assert not torch.equal(sample, mean)
    assert sample.device == device


@mark.parametrize("degrees_of_freedom", [1.0, 2.0, 3.0, 4.0])
def test_sample(degrees_of_freedom: float, device: torch.device) -> None:
    """Test StudentTVarDist.sample."""
    vardist = StudentTVarDist(degrees_of_freedom=degrees_of_freedom)
    mean = torch.randn((3, 4), device=device)
    log_scale = torch.full_like(mean, -float("inf"), device=device)
    sample = vardist.sample(mean, log_scale)
    assert sample.shape == mean.shape
    assert torch.equal(sample, mean)
    assert sample.device == device

    mean = torch.randn((6,), device=device)
    log_scale = torch.zeros_like(mean, device=device)
    sample = vardist.sample(mean, log_scale)
    assert not torch.equal(sample, mean)
    assert sample.device == device


@mark.parametrize(
    "norm_constants,degrees_of_freedom",
    [(True, 1.0), (False, 1.0), (True, 4.0), (False, 4.0)],
)
def test_log_prob(
    norm_constants: bool, degrees_of_freedom: float, device: torch.device
) -> None:
    """Test StudentTVarDist.log_prob."""
    vardist = StudentTVarDist(degrees_of_freedom=degrees_of_freedom)
    use_norm_constants(norm_constants)

    sample_shape = (3, 4)
    mean = torch.randn(sample_shape, device=device)
    log_scale = torch.randn(sample_shape, device=device)
    sample = vardist.sample(mean, log_scale)
    ref1 = StudentT(
        degrees_of_freedom * torch.ones_like(mean), loc=mean, scale=torch.exp(log_scale)
    ).log_prob(sample)
    if not norm_constants:
        norm_value = (
            0.5 * log(torch.pi * degrees_of_freedom)
            + torch.lgamma(torch.tensor(degrees_of_freedom / 2.0))
            - torch.lgamma(torch.tensor((degrees_of_freedom + 1.0) / 2.0))
        ).to(device=device)
        norm_const = torch.full_like(mean, norm_value.item(), device=device)
        ref1 += norm_const
    log_prob1 = vardist.log_prob(sample, mean, log_scale)
    assert torch.allclose(ref1, log_prob1, atol=3e-7)
    assert log_prob1.device == device

    sample_shape2 = (6,)
    mean = torch.randn(sample_shape2, device=device)
    log_scale = torch.zeros_like(mean, device=device)
    sample = vardist.sample(mean, log_scale)
    ref2 = StudentT(
        degrees_of_freedom * torch.ones_like(mean), loc=mean, scale=torch.exp(log_scale)
    ).log_prob(sample)
    if not norm_constants:
        norm_value = (
            0.5 * log(torch.pi * degrees_of_freedom)
            + torch.lgamma(torch.tensor(degrees_of_freedom / 2.0))
            - torch.lgamma(torch.tensor((degrees_of_freedom + 1.0) / 2.0))
        ).to(device=device)
        norm_const = torch.full_like(mean, norm_value.item(), device=device)
        ref2 += norm_const
    log_prob2 = vardist.log_prob(sample, mean, log_scale)
    assert torch.allclose(ref2, log_prob2, atol=3e-7)
    assert log_prob2.device == device
