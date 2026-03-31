import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Tuple
from warnings import warn

from torch import Tensor
from torch.nn import init
from torch.nn.common_types import _tensor_list_t

from ..utils import PostInitCallMeta

if TYPE_CHECKING:
    from ..base import VIModule  # pragma: no cover


def _init_constant(
    parameter: Tensor, default: float, fan_in: int, is_log: bool, eps: float = 1e-5
) -> None:
    scale = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    if is_log:
        init.constant_(parameter, default + math.log(scale + eps))
    else:
        init.constant_(parameter, scale * default)


def _init_uniform(parameter: Tensor, fan_in: int) -> None:
    if parameter.dim() < 2:
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(parameter, -bound, bound)
    else:
        init.kaiming_uniform_(parameter, a=math.sqrt(5))


class Distribution(metaclass=PostInitCallMeta):
    r"""
    Base class for distributions.

    Distributions are implemented as subclasses of this. However, not all distributions
    can function or make sense in all available roles. Furthermore, each role has
    distinct requirements, which can be implemented independently.

    Available roles are prior, variational distribution, and predictive distribution.
    To represent this there is a mix-in interface for each of these roles:
    :class:`~Prior`, :class:`~VariationalDistribution`, and
    :class:`~PredictiveDistribution`. Each custom distribution must use at least one of
    these interfaces.

    To simplify detection of correct usage this class has the three attributes
    :attr:`~self.is_prior`, :attr:`~self.is_variational_distribution`, and
    :attr:`~self.is_predictive_distribution`, which are `False` by default. When
    implementing custom distributions the flags for the intended usages must be set to
    `True`. This also implements initialization time checking that provides feedback if
    any required components for the flagged uses is missing.
    """

    def __post_init__(self) -> None:
        r"""Ensure distribution is set up for specified modes."""
        if not (
            isinstance(self, Prior)
            or isinstance(self, VariationalDistribution)
            or isinstance(self, PredictiveDistribution)
        ):
            raise TypeError(
                "A Distribution must use at least one of Prior, VariationalDistribution"
                ", or PredictiveDistribution as interface."
            )

    @property
    def primary_parameter(self) -> str:
        """The distribution parameter that is closest to a non-Bayesian weight."""
        return self.distribution_parameters[0]

    @property
    @abstractmethod
    def distribution_parameters(self) -> Tuple[str, ...]:
        r"""
        The names of the parameters characterizing the distribution.

        The first element should be the name of the primary paramater, i.e., the
        parameter that is closest to a non-Bayesian weight - typically a mean or loc.
        Any parameters that begin with "log_" will be rescaled as logarithmic parameters.
        """
        ...

    @abstractmethod
    def log_prob(self, sample: Tensor, parameters: _tensor_list_t) -> Tensor:
        r"""
        Calculate the log probability of a sample given the distribution parameters.

        Parameters
        ----------
        sample: Tensor
            A Tensor of samples for which to calculate the log probability.
        parameters: Tensor | Tuple[Tensor, ...]
            One Tensor for each entry of :attr:`~self.distribution_parameters` in the
            same order. These must be broadcastable to the shape of `sample`. If
            additional parameters are needed for hierarchical priors they should appear
            at the end of parameters in the order specified by `_required_parameters`.

        Returns
        -------
        Tensor
            The log probability of the samples given the distribution parameters. This
            Tensor has the same shape as `sample`.
        """
        ...

    def match_parameters(
        self, distribution_parameters: Tuple[str, ...]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        r"""
        Compare distribution parameters to another set of parameters.

        Typically, this is used to compare to the distribution parameters of another
        :class:`~torch_blue.vi.distributions.Distributions`.

        Parameters
        ----------
        distribution_parameters : Tuple[str]
            Tuple of parameter names to compare.

        Returns
        -------
        Tuple[Dict[str, int], Dict[str, int]]
            The first dictionary maps the names of the shared parameters to their index
            in :attr:`~self.variational_parameters`. The second dictionary maps the
            names of parameters exclusive to :attr:`~self.variational_parameters` to
            their index.
        """
        shared_params = {}
        diff_params = {}

        for i, var_param in enumerate(self.distribution_parameters):
            if var_param in distribution_parameters:
                shared_params[var_param] = i
            else:
                diff_params[var_param] = i

        return shared_params, diff_params


class Prior(Distribution):
    r"""
    Interface for a :class:`~Distribution` usable as prior.

    A prior specifies knowledge about the parameter distribution before training. In
    Bayesian training, weights are generally drawn towards the prior, unless they take
    an important role. Mathematically, this prior pull can take the same role as weight
    decay.

    Each prior must name the :attr:`~self.distribution_parameters` that define it as
    well as the way to calculate the log likelihood of a weight configuration in the
    :meth:`~self.log_prob` method.

    Generally, each name in :attr:`~self.distribution_parameters` should also be an
    attribute of the class storing that parameter. This is necessary since the prior
    typically needs to be rescaled based on the layer width. By default, each parameter
    is assumed to require scaling, but in certain cases like the shape parameter of a
    Gamma distribution, this might not be the case. It that case the subset of scaling
    parameters must be specified in :attr:`~self._scaling_parameters`. Non-scaling
    parameters are technically not required to be a class attribute.

    Furthermore, parameters might only assume positive values. These should be stored as
    logarithm of their true value, mapping them to the whole real line. Their parameter
    name should begin with the prefix "log\_", which is automatically detected and
    handled during rescaling.

    To enable the ``prior_initialization`` functionality, the class must implement the
    :meth:`~self.reset_parameters_to_prior` method. Which initializes the parameters for
    one random variable of a model, whose variational parameters are supported by the
    prior, to the prior values.

    Parameters
    ----------
    _required_parameters: Tuple[str, ...], default: ()
        External parameters besides a sample needed to calculate :meth:`~log_prob`.
    _scaling_parameters: Tuple[str, ...], default: :attr:`~distribution_parameters`
        Parameters that need to be rescaled for prior rescaling.
    """

    _rescaled: bool = False
    _required_parameters: Tuple[str, ...] = ()
    _scaling_parameters: Tuple[str, ...]

    def __post_init__(self) -> None:
        r"""Ensure instance has required attributes to operate as prior."""
        if not hasattr(self, "_scaling_parameters"):
            self._scaling_parameters = self.distribution_parameters
        for parameter in self._scaling_parameters:
            assert hasattr(self, parameter), (
                f"Module [{type(self).__name__}] is missing exposed "
                f"scaling parameter [{parameter}]"
            )

        super().__post_init__()

    def get_parameters(self) -> _tensor_list_t:
        r"""
        Get a tuple of the values of the distribution parameters.

        The default assumes that all parameters are stored as attributes under name name
        specified in :attr:`~Distribution.distribution_parameters`. Otherwise, this
        method must be overwritten.

        Returns
        -------
        Tensor | Tuple[Tensor, ...]
            The values of the distribution parameters in the order of
            :attr:`~Distribution.distribution_parameters`.
        """
        return tuple(getattr(self, name) for name in self.distribution_parameters)

    def prior_log_prob(
        self, sample: Tensor, hyperparameters: _tensor_list_t = ()
    ) -> Tensor:
        r"""
        Compute the log probability of sample based on the distribution parameters.

        Function to calculate the log likelihood of a weight configuration under this
        prior. This is a wrapper around :meth:`~Distribution.log_prob` automating the
        inclusion of the prior parameters via :meth:`~get_parameters`, and allowing
        customization.

        Parameters
        ----------
        sample: Tensor
            A Tensor of values to calculate the log probability for.
        hyperparameters: Tensor | Tuple[Tensor, ...]
            External parameters that might be needed hierarchical priors.

        Returns
        -------
            The log probability of the samples given the distribution parameters. This
            Tensor has the same shape as `sample`.
        """
        parameters = self.get_parameters()
        return self.log_prob(sample, (*parameters, *hyperparameters))

    def kaiming_rescale(self, fan_in: int, eps: float = 1e-5) -> None:
        r"""
        Rescale the prior based on layer width, for normalization.

        Parameters from :attr:`~self._scaling_parameters` are scaled linearly based on
        the square root of the layer width, unless their name begins with "log\_", in
        which case they are scaled such that their exponential scales in the same way.

        Parameters
        ----------
        fan_in : int
            The relevant layer width.
        eps: float, default: 1e-5
            Epsilon for numerical stability.

        Returns
        -------
        None
        """
        if self._rescaled:
            warn(
                f"{type(self).__name__} has already been rescaled. Ignoring rescaling."
            )
            pass
        else:
            self._rescaled = True
            scale = 1 / math.sqrt(3 * fan_in) if fan_in > 0 else 0

            for parameter in self._scaling_parameters:
                param = getattr(self, parameter)
                if parameter.startswith("log"):
                    setattr(self, parameter, param + math.log(scale + eps))
                else:
                    setattr(self, parameter, param * scale)

    def reset_parameters_to_prior(self, module: "VIModule", variable: str) -> None:
        r"""
        Initialize the parameters of a VIModule according to the prior distribution.

        To enable the ``prior_initialization`` functionality, the class must implement
        this method. It initializes the parameters for one random variable of a
        :class:`~torch_blue.vi.VIModule`, whose variational parameters are
        supported by the prior, to the prior values. To that end the name of the random
        variable to initialize is passed to the method.

        This method is called separately for each submodule and therefore does not have
        to consider any further submodules. It is also called separately for each
        random variable but should manage all variational parameters the prior can
        provide. By convention, a prior whose distribution parameters are a true subset
        of the variational parameters initializes the parameters it can handle, e.g. a
        :class:`~.MeanFieldNormalPrior` can can be used to initialize parameters for any
        variational distribution that uses a `mean` and `log_std`.

        Parameters
        ----------
        module: VIModule
            The module containing the parameters to reset.
        variable: str
            The name of the random variable to reset as given by
            :attr:`variational_parameters` of the associated
            :class:`~torch_blue.vi.distributions.Distribution`.

        Returns
        -------
        None
        """
        warn(
            f'Module [{type(self).__name__}] is missing the "reset_parameters_to_prior" method'
            f" and therefore does not support prior initialization."
        )


class VariationalDistribution(Distribution):
    r"""
    Interface for a :class:`~Distribution` usable as variational distribution.

    A variational distribution specifies the parametrization used to fit the true weight
    distribution, i.e., the weight posterior.

    Each variational distribution must name the :attr:`~Distribution.distribution_parameters`
    that will be optimized during training as well as a default for each parameter in
    :attr:`~self._default_variational_parameters`. It is important to note that the
    first parameter is assumed to be a form of mean or mode of the distribution that
    might be used as the weight in a non-Bayesian version of the network. While a
    default value for it must be given, it will usually be ignored in initialization in
    favor of initializing it similar to non-Bayesian weights.

    Additionally, a :meth:`~self.sample` method must be defined that accepts one Tensor
    for each variational parameter and returns a sample from the specified
    distributions. Finally, the way to calculate the log likelihood of a weight
    configuration in the :meth:`~self.variational_log_prob` method is required.

    Attributes
    ----------
    distribution_parameters : Tuple[str, ...]
        The names of the variational parameters that characterize the distribution.
        These are fit during training.
    _default_variational_parameters : Tuple[float, ...]
        Default initialization values for the variational parameters. If the parameter
        is "mean", "mode" or "loc", it is initialized analogously to non-Bayesian
        weights and the default is ignored.
    """

    @property
    @abstractmethod
    def _default_variational_parameters(self) -> Tuple[float, ...]:
        r"""
        The default values of the parameters characterizing the distribution.

        The first element should be the name of the primary paramater, i.e., the
        parameter that is closest to a non-Bayesian weight - typically a mean or loc.
        Any parameters that begin with "log_" will be rescaled as logarithmic parameters.
        """
        ...

    def __post_init__(self) -> None:
        r"""Ensure instance has required attributes to operate as variational distribution."""
        assert len(self.distribution_parameters) == len(
            self._default_variational_parameters
        ), "Each variational parameter must be assigned a default value"

        super().__post_init__()

    def reset_variational_parameters(
        self,
        module: "VIModule",
        variable: str,
        fan_in: int,
        kaiming_scaling: bool = True,
    ) -> None:
        r"""
        Reset the variational parameters of module.

        Parameters equivalent to non-Bayesian weights (currently "mean", "mode", or
        "loc") are reset accordingly using Kaiming uniform initialization based on
        `fan_in` (cf. :meth:`torch.init._calculate_fan_in_and_fan_out`). Other
        parameters are initialized to the fixed values specified by class defaults,
        i.e., :attr:`_default_variational_parameters`.
        If `kaiming_scaling` is ``True`` , the defaults are scaled with
        `scale` * `default`. Any parameter beginning with "log" is assumed to be in log
        space and scaled with `default` + log(`scale`). The scale is 1 / sqrt(`fan_in`)
        for vectors and 1 / sqrt(3 * `fan_in`) for matrices.

        Parameters
        ----------
        module: VIModule
            Module to reset parameters.
        variable: str
            Name of the variable to reset.
        fan_in: int
            Size of the input parameter map.
        kaiming_scaling: bool, default: True
            Whether th scale all parameters according to input map size.
        """
        for parameter, default in zip(
            self.distribution_parameters, self._default_variational_parameters
        ):
            parameter_name = module.variational_parameter_name(variable, parameter)
            param = getattr(module, parameter_name)

            if parameter in ["mean", "mode", "loc"]:
                _init_uniform(param, fan_in)
            elif not kaiming_scaling:
                init.constant_(param, default)
            else:
                is_log = parameter.startswith("log")
                _init_constant(param, default, fan_in, is_log)

    def variational_log_prob(
        self, sample: Tensor, parameters: _tensor_list_t
    ) -> Tensor:
        r"""
        Compute the log probability of `sample`.

        This method is used to calculate the log probability of a weight configuration
        under this distribution. It accepts one Tensor containing a weight configuration
        and a tuple with one Tensor for each variational parameter in the order
        specified in :attr:`~Distribution.distribution_parameters` and returns the
        log probability of the weight configuration. All input Tensors must have the
        same shape.

        Parameters
        ----------
        sample: Tensor
            A Tensor of values to calculate the log probability for.
        parameters: Tensor | Tuple[Tensor, ...]
            The predicitve parameters in the same order as specified by
            :attr:`~Distribution.distribution_parameters`.

        Returns
        -------
        Tensor
            The log probability of the sample under the distribution specified by the
            provided parameters.
        """
        return self.log_prob(sample, parameters)

    @abstractmethod
    def sample(self, parameters: _tensor_list_t) -> Tensor:
        r"""
        Draw a differentiable sample from the distribution.

        This method is used to sample the weigh matrices in the forward pass.
        It accepts atuple with one Tensor for each variational parameter in the order
        specified in :attr:`~Distribution.distribution_parameters` and returns a sample
        from the distribution of the same shape. All input Tensors must have the same
        shape.

        Parameters
        ----------
        parameters: Tensor | Tuple[Tensor, ...]
            A tuple of distribution parameters as specified in
            :attr:`~Distribution.distribution_parameters`.

        Returns
        -------
        Tensor
            A differentiable sample from the distribution specified by the provided
            parameters.
        """
        ...


class PredictiveDistribution(Distribution):
    r"""
    Interface for a :class:`~Distribution` usable as predictive distribution.

    A predictive distribution is the assumed distribution of the model outputs. Its
    parameters should be derivable from sufficient samples for the same prediction.
    Each distribution must define which parameters are used to represent a prediction.
    For example, regression might use a predictive mean and standard deviation, while
        classification might use a probability for each class.

    Furthermore, the distribution must be able to assign a probability to each possible
    prediction given the expected prediction. This is required for loss calculation.
        Typically, it is enough for subclasses to define :meth:`~log_prob_from_parameters`
    and :meth:`~predictive_parameters_from_samples`, which the class automatically uses
    to first calculate the predictive parameters from the provided samples and then the
    log likelihood of those samples from the parameters. In case this detour does not
    work :meth:`~log_prob_from_samples` can be overwritten. However,
    :meth:`~predictive_parameters_from_samples` should still be defined to allow
    extracting predictions.
    """

    def __post_init__(self) -> None:
        r"""Ensure instance has all required attributes to operate as predictive distribution."""
        super().__post_init__()

    @abstractmethod
    def predictive_parameters_from_samples(self, sample: Tensor) -> Tuple[Tensor, ...]:
        r"""
        Calculate the parametric prediction from a set of samples.

        Predictions take the form of distribution parameters, like mean and standard
        deviation for a Gaussian distribution. This method should estimate these from
        a set of samples.

        Parameters
        ----------
        sample: Tensor
            A Tensor of batched samples of shape (S, \*), where S is the number of
            samples.

        Returns
        -------
            A tuple of Tensors, one for each distribution parameter in the same order as
            specified :attr:`~Distribution.distribution_parameters`. Shape: (\*,).
        """
        ...

    def log_prob_from_parameters(
        self, reference: Tensor, parameters: _tensor_list_t
    ) -> Tensor:
        r"""
        Calculate the log probability of reference form the predictive parameters.

        Parameters
        ----------
        reference: Tensor
            The ground truth label as Tensor of the same shape as each Tensor in
            `parameters`.
        parameters: Tensor | Tuple[Tensor, ...]
            The predicitve parameters in the same order as specified by
            :attr:`~Distribution.distribution_parameters` and returned by
            :meth:`~predictive_parameters_from_samples`.

        Returns
        -------
            The log probability of the reference under the predicted distribution.
        """
        return self.log_prob(reference, parameters)

    def log_prob_from_samples(self, reference: Tensor, samples: Tensor) -> Tensor:
        r"""
        Calculate the log probability for reference given a set of samples.

        Usually combines :meth:`~predictive_parameters_from_samples` and
        :meth:`~log_prob_from_parameters`, but can be redefined, if needed.

        Parameters
        ----------
        reference : Tensor
            Expected prediction as Tensor of shape (\*)
        samples : Tensor
            Model prediction as Tensor of shape (S, \*), where S is the number of samples.

        Returns
        -------
        Tensor
            The log probability of the reference under the predicted distribution.
            Shape: (1,).
        """
        params = self.predictive_parameters_from_samples(samples)
        return self.log_prob_from_parameters(reference, params)
