import torch

from torch_bayesian.vi.variational_distributions import NonBayesian


def test_sample(device: torch.device) -> None:
    """Test NonBayesian.sample()."""
    vardist = NonBayesian()
    mean = torch.randn((3, 4), device=device)
    sample = vardist.sample(mean)
    assert sample.shape == mean.shape
    assert torch.allclose(sample, mean)
    assert sample.device == device


def test_log_prob(device: torch.device) -> None:
    """Test NonBayesian.log_prob()."""
    vardist = NonBayesian()
    mean = torch.randn((5, 3, 4), device=device)
    sample = vardist.sample(mean)
    ref1 = torch.zeros_like(mean, device=device)

    log_prob = vardist.log_prob(sample, mean)
    assert torch.allclose(ref1, log_prob)
    assert log_prob.device == device
