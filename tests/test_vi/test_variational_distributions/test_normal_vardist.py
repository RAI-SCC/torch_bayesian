import torch
from torch.distributions import Normal

from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist


def test_normal_sample() -> None:
    """Test _normal_sample."""
    mean = torch.randn((3, 4))
    std = torch.zeros_like(mean)
    sample = MeanFieldNormalVarDist._normal_sample(mean, std)
    assert sample.shape == mean.shape
    assert (sample == mean).all()

    std = torch.ones_like(mean)
    sample = MeanFieldNormalVarDist._normal_sample(mean, std)
    assert not (sample == mean).all()


def test_sample() -> None:
    """Test MeanFieldNormalVarDist.sample."""
    vardist = MeanFieldNormalVarDist()
    mean = torch.randn((3, 4))
    log_std = torch.full_like(mean, -float("inf"))
    sample = vardist.sample(mean, log_std)
    assert sample.shape == mean.shape
    assert (sample == mean).all()

    mean = torch.randn((6,))
    log_std = torch.zeros_like(mean)
    sample = vardist.sample(mean, log_std)
    assert not (sample == mean).all()


def test_log_prob() -> None:
    """Test MeanFieldNormalVarDist.log_prob."""
    vardist = MeanFieldNormalVarDist()
    mean = torch.randn((3, 4))
    log_std = torch.randn((3, 4))
    sample = vardist.sample(mean, log_std)
    ref1 = Normal(mean, torch.exp(log_std)).log_prob(sample)
    log_prob1 = vardist.log_prob(sample, mean, log_std)
    assert (torch.isclose(ref1, log_prob1)).all()

    mean = torch.randn((6,))
    log_std = torch.zeros_like(mean)
    sample = vardist.sample(mean, log_std)
    ref2 = Normal(mean, torch.exp(log_std)).log_prob(sample)
    log_prob2 = vardist.log_prob(sample, mean, log_std)
    assert (torch.isclose(ref2, log_prob2)).all()
