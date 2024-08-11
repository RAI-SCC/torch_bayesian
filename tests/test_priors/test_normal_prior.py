import torch
from torch.distributions import Normal

from vi.priors import MeanFieldNormalPrior


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
