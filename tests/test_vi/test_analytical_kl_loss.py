import random
from math import log, pi
from typing import Optional, Type, Union

import pytest
import torch
from torch import Tensor

from torch_bayesian.vi import (
    AnalyticalKullbackLeiblerLoss,
    KullbackLeiblerLoss,
    KullbackLeiblerModule,
    VILinear,
    VIModule,
    VISequential,
)
from torch_bayesian.vi.analytical_kl_loss import (
    NonBayesianDivergence,
    NormalNormalDivergence,
    UniformNormalDivergence,
)
from torch_bayesian.vi.predictive_distributions import (
    MeanFieldNormalPredictiveDistribution,
)
from torch_bayesian.vi.priors import MeanFieldNormalPrior, Prior, UniformPrior
from torch_bayesian.vi.utils import use_norm_constants
from torch_bayesian.vi.variational_distributions import (
    MeanFieldNormalVarDist,
    NonBayesian,
    VariationalDistribution,
)


def test_klmodule() -> None:
    """Test KullbackLeiblerModule."""
    module = KullbackLeiblerModule()

    list_a = [torch.arange(27, 31)]
    list_b = [torch.arange(3)]
    reference = torch.cat([*list_a, *list_b])

    with pytest.raises(NotImplementedError):
        module(list_a, list_b)

    def dummy_forward(*args: Tensor) -> Tensor:
        return torch.cat(args)

    module.forward = dummy_forward

    out = module(list_a, list_b)
    assert torch.equal(out, reference)


def test_nonbayesian_klmodule() -> None:
    """Test NonBayesianDivergence."""
    sample_shape = [7, 10]
    prior_param_number = random.randint(1, 5)
    var_param_number = 1

    prior_params = [*torch.randn([prior_param_number, *sample_shape])]
    var_params = [*torch.randn([var_param_number, *sample_shape])]

    module = NonBayesianDivergence()
    out = module(prior_params, var_params)
    assert torch.equal(out, torch.tensor([0.0]))


@pytest.mark.parametrize("norm_constants", [(True,), (False,)])
def test_uniformnormal_klmodule(norm_constants: bool) -> None:
    """Test UniformNormalDivergence."""
    sample_shape = [7, 10]
    prior_param_number = 0
    var_param_number = 2

    prior_params = [*torch.randn([prior_param_number, *sample_shape])]
    var_params = [*torch.randn([var_param_number, *sample_shape])]

    use_norm_constants(norm_constants)
    module = UniformNormalDivergence()
    out = module(prior_params, var_params)
    if not norm_constants:
        reference = var_params[1].sum()
    else:
        reference = var_params[1] - 0.5 * (1 + log(2 * pi))
        reference = reference.sum()
    assert torch.equal(out, reference)


@pytest.mark.parametrize("norm_constants", [(True,), (False,)])
def test_normalnormal_klmodule(norm_constants: bool) -> None:
    """Test NormalNormalDivergence."""
    sample_shape = [7, 10]
    prior_param_number = 2
    var_param_number = 2

    prior_params = [*torch.randn([prior_param_number, *sample_shape])]
    var_params = [*torch.randn([var_param_number, *sample_shape])]

    use_norm_constants(norm_constants)
    module = NormalNormalDivergence()
    out = module(prior_params, var_params)

    reference = (prior_params[1] - var_params[1]) + (
        prior_params[0] - var_params[0]
    ).pow(2) * torch.exp(2 * var_params[1]) / (2 * torch.exp(2 * prior_params[1]))
    if norm_constants:
        reference = reference - 0.5
    assert torch.allclose(out, reference.sum())


class DummyPrior(Prior):
    """Dummy prior for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.distribution_parameters = ()

    def log_prob(self, *args: Tensor) -> Tensor:
        """Return dummy log probability."""
        return torch.zeros(1)


class DummyVarDist(VariationalDistribution):
    """Dummy variational distribution for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.variational_parameters = ()
        self._default_variational_parameters = ()

    def sample(self) -> Tensor:
        """Return dummy sample."""
        return torch.zeros(1)

    def log_prob(self, *args: Tensor) -> Tensor:
        """Return dummy log probability."""
        return torch.zeros(1)


@pytest.mark.parametrize(
    "prior,var_dist,target",
    [
        (MeanFieldNormalPrior, MeanFieldNormalVarDist, NormalNormalDivergence),
        (UniformPrior, MeanFieldNormalVarDist, UniformNormalDivergence),
        (MeanFieldNormalPrior, NonBayesian, NonBayesianDivergence),
        (UniformPrior, NonBayesian, NonBayesianDivergence),
        (DummyPrior, MeanFieldNormalVarDist, "fail"),
        (MeanFieldNormalPrior, DummyVarDist, "fail"),
    ],
)
def test_detect_divergence(
    prior: Type[Prior],
    var_dist: Type[VariationalDistribution],
    target: Union[str, Type[KullbackLeiblerModule]],
) -> None:
    """Test AnalyticalKullbackLeiblerLoss._detect_divergence()."""
    if target == "fail":
        with pytest.raises(
            NotImplementedError,
            match=f"Analytical loss is not implemented for {prior.__name__} and {var_dist.__name__}.",
        ):
            AnalyticalKullbackLeiblerLoss._detect_divergence(prior(), var_dist())
    elif not isinstance(target, str):
        assert isinstance(
            AnalyticalKullbackLeiblerLoss._detect_divergence(prior(), var_dist()),
            target,
        )
    else:
        raise ValueError("Invalid target specification.")


class DummyMLP(VIModule):
    """Dummy MLP for testing."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        prior: Prior = MeanFieldNormalPrior(),
        var_dist: VariationalDistribution = MeanFieldNormalVarDist(),
        alt_prior: Optional[Prior] = None,
        alt_vardist: Optional[VariationalDistribution] = None,
    ) -> None:
        super().__init__()
        alt_prior = alt_prior or prior
        alt_var_dist = alt_vardist or var_dist

        self.layers = VISequential(
            VILinear(
                in_features,
                hidden_features,
                variational_distribution=var_dist,
                prior=prior,
            ),
            VILinear(
                hidden_features,
                out_features,
                variational_distribution=alt_var_dist,
                prior=alt_prior,
            ),
        )

    def forward(self, input_: Tensor) -> Tensor:
        """Make forward pass."""
        return self.layers(input_)


def test_prior_matching() -> None:
    """Test AnalyticalKullbackLeiblerLoss.prior_matching()."""
    prior = MeanFieldNormalPrior()
    var_dist = MeanFieldNormalVarDist()

    f_in = 8
    f_hidden = 160
    f_out = 10

    batch_size = 100
    samples = 100

    model = DummyMLP(f_in, f_hidden, f_out, prior=prior, var_dist=var_dist)
    criterion = AnalyticalKullbackLeiblerLoss(
        model, MeanFieldNormalPredictiveDistribution(), samples
    )
    ref_criterion = KullbackLeiblerLoss(
        MeanFieldNormalPredictiveDistribution(), samples, track=True
    )

    sample = torch.rand([batch_size, f_in])
    target = torch.rand([batch_size, f_out])

    model.return_log_probs()

    out = model(sample, samples=samples)

    analytical_prior_matching = criterion.prior_matching()
    ref_criterion(out, target)
    ref_prior_matching = ref_criterion.log["prior_matching"]  # type: ignore [index]
    print(ref_prior_matching[0] / analytical_prior_matching.item())
