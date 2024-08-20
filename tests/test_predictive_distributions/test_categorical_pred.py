import torch
from torch.distributions import Categorical

from vi.predictive_distributions import CategoricalPredictiveDistribution


def test_categorical_predictive_distribution() -> None:
    """Test CategoricalPredictiveDistribution."""
    predictive_dist = CategoricalPredictiveDistribution()
    predictive_prob = CategoricalPredictiveDistribution(input_type="probs")

    samples = 4
    batch = 3
    categories = 5
    probs = torch.rand((samples, batch, categories))
    target = torch.randint(0, categories, (batch,))

    p1 = predictive_dist.predictive_parameters_from_samples(probs.log())
    p2 = predictive_prob.predictive_parameters_from_samples(probs)
    assert p1.shape == (batch, categories)
    assert p2.shape == (batch, categories)
    assert torch.allclose(p1, p2)

    ref_dist1 = Categorical(probs=probs)
    assert torch.allclose(p1, ref_dist1.probs.mean(dim=0))

    ref_dist2 = Categorical(probs=ref_dist1.probs.mean(dim=0))
    target_log_prob = ref_dist2.log_prob(target)

    log_prob1 = predictive_dist.log_prob_from_parameters(target, p1)
    log_prob2 = predictive_dist.log_prob_from_parameters(target, p2)
    assert log_prob1.shape == (batch,)
    assert torch.allclose(log_prob1, target_log_prob)
    assert torch.allclose(log_prob2, target_log_prob)
