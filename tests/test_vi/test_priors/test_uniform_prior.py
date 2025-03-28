import torch

from torch_bayesian.vi.priors import UniformPrior


def test_log_prob(device: torch.device) -> None:
    """Test UniformPrior.log_prob()."""
    prior = UniformPrior()

    sample1 = torch.rand(5)
    sample2 = torch.rand([3, 6])

    out1 = prior.log_prob(sample1)
    out2 = prior.log_prob(sample2)
    assert out1 == 0.0
    assert out2 == 0.0
    assert out1.device == device
    assert out2.device == device
