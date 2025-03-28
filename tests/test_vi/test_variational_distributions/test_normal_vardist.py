import torch
from pytest import mark
from torch.distributions import Normal

from torch_bayesian.vi.utils import use_norm_constants
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist


def test_normal_sample(device: torch.device) -> None:
    """Test _normal_sample."""
    mean = torch.randn((3, 4), device=device)
    std = torch.zeros_like(mean, device=device)
    sample = MeanFieldNormalVarDist._normal_sample(mean, std)
    assert sample.shape == mean.shape
    assert torch.equal(sample, mean)
    assert sample.device == device

    std = torch.ones_like(mean)
    sample = MeanFieldNormalVarDist._normal_sample(mean, std)
    assert not torch.equal(sample, mean)
    assert sample.device == device


def test_sample(device: torch.device) -> None:
    """Test MeanFieldNormalVarDist.sample."""
    vardist = MeanFieldNormalVarDist()
    mean = torch.randn((3, 4), device=device)
    log_std = torch.full_like(mean, -float("inf"), device=device)
    sample = vardist.sample(mean, log_std)
    assert sample.shape == mean.shape
    assert torch.equal(sample, mean)
    assert sample.device == device

    mean = torch.randn((6,), device=device)
    log_std = torch.zeros_like(mean, device=device)
    sample = vardist.sample(mean, log_std)
    assert not torch.equal(sample, mean)
    assert sample.device == device


@mark.parametrize("norm_constants", [True, False])
def test_log_prob(norm_constants: bool, device: torch.device) -> None:
    """Test MeanFieldNormalVarDist.log_prob."""
    vardist = MeanFieldNormalVarDist()
    use_norm_constants(norm_constants)

    sample_shape = (3, 4)
    mean = torch.randn(sample_shape, device=device)
    log_std = torch.randn(sample_shape, device=device)
    sample = vardist.sample(mean, log_std)
    ref1 = Normal(mean, torch.exp(log_std)).log_prob(sample).to(device=device)
    if not norm_constants:
        norm_const = torch.full_like(mean, 2 * torch.pi, device=device).log() / 2
        ref1 += norm_const
    log_prob1 = vardist.log_prob(sample, mean, log_std)
    assert torch.allclose(ref1, log_prob1)
    assert log_prob1.device == device

    sample_shape2 = (6,)
    mean = torch.randn(sample_shape2, device=device)
    log_std = torch.zeros_like(mean, device=device)
    sample = vardist.sample(mean, log_std)
    ref2 = Normal(mean, torch.exp(log_std)).log_prob(sample).to(device=device)
    if not norm_constants:
        norm_const = torch.full_like(mean, 2 * torch.pi, device=device).log() / 2
        ref2 += norm_const
    log_prob2 = vardist.log_prob(sample, mean, log_std)
    assert torch.allclose(ref2, log_prob2)
    assert log_prob2.device == device
