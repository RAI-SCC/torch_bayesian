import torch
from torch.distributions import Normal

from torch_bayesian.vi.predictive_distributions import (
    MeanFieldNormalPredictiveDistribution,
)


def test_normal_predictive_distribution() -> None:
    """Test MeanFieldNormalPredictiveDistribution."""
    predictive_dist = MeanFieldNormalPredictiveDistribution()

    samples = torch.randn((7, 5, 3))
    reference = torch.randn((5, 3))
    target_mean = samples.mean(dim=0)
    target_std = samples.std(dim=0)
    target_log_prob = Normal(target_mean, target_std).log_prob(reference)

    test_mean, test_std = predictive_dist.predictive_parameters_from_samples(samples)
    assert torch.allclose(test_mean, target_mean)
    assert torch.allclose(test_std, target_std)

    test_log_prob = predictive_dist.log_prob_from_parameters(
        reference, (target_mean, target_std)
    )
    assert torch.allclose(test_log_prob, target_log_prob)

    end2end_log_prob = predictive_dist.log_prob_from_samples(reference, samples)
    assert torch.allclose(end2end_log_prob, test_log_prob)
