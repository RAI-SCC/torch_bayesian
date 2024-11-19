from typing import Tuple

import torch
from torch import Tensor
from torch_kmeans import KMeans
from vi.utils import silhouette_scores
from .base import PredictiveDistribution


class MixtureGaussianPredictiveDistribution(PredictiveDistribution):
    """Predictive distribution assuming uncorrelated, normal distributed forecasts."""

    predictive_parameters = ("prob", "mean0", "mean1", "std0", "std1")

    @staticmethod
    def predictive_parameters_from_samples(samples: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Calculate mean and standard deviation of samples."""
        prob = torch.empty((samples.shape[1], 1))
        mean0 = torch.empty((samples.shape[1], 1))
        mean1 = torch.empty((samples.shape[1], 1))
        std0 = torch.empty((samples.shape[1], 1))
        std1 = torch.empty((samples.shape[1], 1))
        for bs in range(samples.shape[1]):
            model = KMeans(n_clusters=2, verbose=False)
            subsamples = torch.unsqueeze(samples[:, bs, :], 0)
            labels = model.fit_predict(subsamples)
            group0 = subsamples[labels == 0]
            group1 = subsamples[labels == 1]
            p = group0.shape[0] / (group0.shape[0] + group1.shape[0])
            m0 = group0.mean()
            s0 = torch.sqrt(torch.mean(group0**2, dim=0) - m0**2)
            m1 = group1.mean()
            s1 = torch.sqrt(torch.mean(group1**2, dim=0) - m1**2)
            prob[bs, 0] = p
            mean0[bs, 0] = m0
            mean1[bs, 0] = m1
            std0[bs, 0] = s0
            std1[bs, 0] = s1
        return prob, mean0, mean1, std0, std1

    @staticmethod
    def log_prob_from_parameters(
        reference: Tensor, parameters: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], eps: float = 1e-10
    ) -> Tensor:
        """Calculate log probability of reference given mean and standard deviation."""
        prob, mean0, mean1, std0, std1 = parameters
        var0 = (std0+eps) ** 2
        var1 = (std1+eps) ** 2
        like_0 = torch.log(prob) - 0.5 * ((reference - mean0) ** 2 / var0 + torch.log(2 * torch.pi * var0))
        like_1 = torch.log(1 - prob) - 0.5 * ((reference - mean1) ** 2 / var1 + torch.log(2 * torch.pi * var1))
        likelihood = torch.logsumexp(torch.cat((like_0, like_1), dim=1), 1)
        #if torch.any(torch.isnan(likelihood)):
        #    print(std1)
        return likelihood
