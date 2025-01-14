import torch

from torch_bayesian.vi.variational_distributions import NonBayesian


def test_sample() -> None:
    """Test NonBayesian.sample()."""
    vardist = NonBayesian()
    mean = torch.randn((3, 4))
    sample = vardist.sample(mean)
    assert sample.shape == mean.shape
    assert torch.allclose(sample, mean)


def test_log_prob() -> None:
    """Test NonBayesian.log_prob()."""
    vardist = NonBayesian()
    mean = torch.randn((5, 3, 4))
    sample = vardist.sample(mean)
    ref1 = torch.zeros_like(mean)

    log_prob = vardist.log_prob(sample, mean)
    assert (ref1 == log_prob).all()
