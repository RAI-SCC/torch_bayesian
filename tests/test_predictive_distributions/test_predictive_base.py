from typing import Tuple

import torch
from torch import Tensor

from vi.predictive_distributions import PredictiveDistribution


def test_parameter_checking() -> None:
    """Test enforcement or required parameters for subclasses of PredictiveDistribution."""

    # predictive_parameters assertion
    class Test1(PredictiveDistribution):
        pass

    try:
        Test1()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define predictive_parameters"

    # predictive_parameters_from_samples assertion
    class Test2(PredictiveDistribution):
        predictive_parameters = ("mean", "std")

    try:
        Test2()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define predictive_parameters_from_samples"

    # log_prob_from_parameters assertion
    class Test3(PredictiveDistribution):
        predictive_parameters = ("mean", "std")

        def predictive_parameters_from_samples(
            self, samples: Tensor
        ) -> Tuple[Tensor, Tensor]:
            return samples, samples

    try:
        Test3()
        raise AssertionError
    except NotImplementedError as e:
        assert str(e) == "Subclasses must define log_prob_from_parameters"

    # Test correct init
    class Test4(PredictiveDistribution):
        predictive_parameters = ("mean", "std")

        def predictive_parameters_from_samples(
            self, samples: Tensor
        ) -> Tuple[Tensor, Tensor]:
            return samples, samples

        def log_prob_from_parameters(
            self, reference: Tensor, parameters: Tuple[Tensor, Tensor]
        ) -> Tensor:
            return reference

    _ = Test4()


def test_log_prob_from_samples() -> None:
    """Test PredictiveDistribution.log_prob_from_samples."""

    class Test(PredictiveDistribution):
        predictive_parameters = ("mean",)

        def predictive_parameters_from_samples(self, samples: Tensor) -> Tensor:
            return samples.sum(dim=0)

        def log_prob_from_parameters(
            self, reference: Tensor, parameters: Tensor
        ) -> Tensor:
            return reference + parameters

    test = Test()
    samples = torch.randn((5, 3, 4))
    reference = torch.randn((3, 4))
    target = samples.sum(dim=0) + reference

    out = test.log_prob_from_samples(reference, samples)
    assert (target == out).all()
