import torch
from pytest import mark
from torch.distributions import Normal

from torch_bayesian.vi.predictive_distributions import (
    MeanFieldNormalPredictiveDistribution,
)
from torch_bayesian.vi.utils import use_norm_constants


@mark.parametrize("norm_constants", [True, False])
def test_normal_predictive_distribution(
    norm_constants: bool, device: torch.device
) -> None:
    """Test MeanFieldNormalPredictiveDistribution."""
    predictive_dist = MeanFieldNormalPredictiveDistribution()
    use_norm_constants(norm_constants)

    nr_samples = 50
    sample_shape = (5, 3)
    samples = torch.randn((nr_samples, *sample_shape), device=device)
    reference = torch.randn(sample_shape, device=device)
    target_mean = samples.mean(dim=0)
    target_std = samples.std(dim=0)
    target_log_prob = Normal(target_mean, target_std).log_prob(reference)
    if not norm_constants:
        norm_const = torch.full_like(target_mean, 2 * torch.pi, device=device).log() / 2
        target_log_prob += norm_const

    test_mean, test_std = predictive_dist.predictive_parameters_from_samples(samples)
    assert torch.allclose(test_mean, target_mean)
    assert torch.allclose(test_std, target_std)
    assert test_mean.device == device
    assert test_std.device == device

    test_log_prob = predictive_dist.log_prob_from_parameters(
        reference, (target_mean, target_std)
    )
    assert torch.allclose(test_log_prob, target_log_prob, atol=1e-7)
    assert test_log_prob.device == device

    end2end_log_prob = predictive_dist.log_prob_from_samples(reference, samples)
    assert torch.allclose(end2end_log_prob, test_log_prob)
    assert end2end_log_prob.device == device
