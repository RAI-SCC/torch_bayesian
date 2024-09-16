import torch

from vi.priors import UniformPrior


def test_log_prob() -> None:
    """Test UniformPrior.log_prob()."""
    prior = UniformPrior()

    sample1 = torch.rand(5)
    sample2 = torch.rand([3, 6])

    assert prior.log_prob(sample1) == 0.0
    assert prior.log_prob(sample2) == 0.0
