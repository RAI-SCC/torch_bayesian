import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.utils.hooks as hooks
from torch import Tensor
from torch.nn import Module, Parameter, init
from torch.nn.modules.module import (
    _global_backward_hooks,
    _global_backward_pre_hooks,
    _global_forward_hooks,
    _global_forward_hooks_always_called,
    _global_forward_pre_hooks,
    _WrappedHook,
)

from .priors import Prior
from .utils import PostInitCallMeta
from .variational_distributions import VarDist


def _forward_unimplemented(self: Module, *input_: Optional[Tensor]) -> Tuple[Tensor]:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
        For VIModules all inputs must be Tensors.
    """
    raise NotImplementedError(
        f'Module [{type(self).__name__}] is missing the required "forward" function'
    )


class VIModule(Module, metaclass=PostInitCallMeta):
    """Base class for Modules using Variational Inference."""

    forward: Callable[..., Union[Tensor, Tuple[Tensor, ...]]] = _forward_unimplemented
    _return_log_prob: bool = True
    # this is set to False during the first forward pass by the outermost module for each submodule and True for itself
    # that way submodules automatically call forward and the outermost calls sampled_forward instead
    _has_sampling_responsibility: bool

    @staticmethod
    def _expand_to_samples(input_: Optional[Tensor], samples: int) -> Tensor:
        if input_ is None:
            input_ = torch.tensor(False)
        return input_.expand(samples, *input_.shape)

    def sampled_forward(
        self, *input_: Optional[Tensor], samples: int = 10
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Forward pass of the module evaluating multiple weight samples.

        Parameters
        ----------
         input_: Tensor
            Any number of input Tensors
        samples : int
            Number of weight samples to evaluate

        Returns
        -------
        Union[Tensor, Tuple[Tensor, ...]]
            One or multiple Tensors
        """
        expanded = [self._expand_to_samples(x, samples=samples) for x in input_]
        return torch.vmap(self.forward, randomness="different")(*expanded)

    def return_log_prob(self, mode: bool = True) -> None:
        """
        Set whether the module returns log probabilities.

        Log probabilities are required for most standard losses.

        Parameters
        ----------
        mode : bool
            Whether to enable (`True`) or disable (`False`) returning of log probs.
        """
        for module in self.modules():
            if isinstance(module, VIModule):
                module._return_log_prob = mode

    def _set_sampling_responsibility(self) -> None:
        for module in self.modules():
            if isinstance(module, VIModule):
                module._has_sampling_responsibility = False
        self._has_sampling_responsibility = True

    def __post_init__(self) -> None:
        """
        After __init__ set sampling responsibility.

        Since higher level modules overwrite _has_sampling_responsibility for lower ones only the top level class will have it set to True, making it use sampled_forward by default.
        """
        self._set_sampling_responsibility()

    # Copied from pytorch 2.4, basically untested since assumed working
    # Change: choose between sampled_forward and forward base on _has_sampling_responsibility
    def _slow_forward(self, *input_: Any, **kwargs: Any) -> Any:  # pragma: no cover
        tracing_state = torch._C._get_tracing_state()
        forward_call = (
            self.sampled_forward if self._has_sampling_responsibility else self.forward
        )
        if not tracing_state or isinstance(forward_call, torch._C.ScriptMethod):
            return forward_call(*input_, **kwargs)
        recording_scopes = torch.jit._trace._trace_module_map is not None
        if recording_scopes:
            # type ignore was added because at this point one knows that
            # torch.jit._trace._trace_module_map is not Optional and has type Dict[Any, Any]
            name = (
                torch.jit._trace._trace_module_map[self]
                if self in torch.jit._trace._trace_module_map
                else None
            )  # type: ignore[index, operator] # noqa: B950
            if name:
                tracing_state.push_scope(name)
            else:
                recording_scopes = False
        try:
            result = forward_call(*input_, **kwargs)
        finally:
            if recording_scopes:
                tracing_state.pop_scope()
        return result

    # Copied from pytorch 2.4, basically untested since assumed working
    # Change: choose between sampled_forward and forward base on _has_sampling_responsibility
    def _call_impl(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        if torch._C._get_tracing_state():
            forward_call = self._slow_forward
        elif self._has_sampling_responsibility:
            forward_call = self.sampled_forward
        else:
            forward_call = self.forward
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (
            self._backward_hooks
            or self._backward_pre_hooks
            or self._forward_hooks
            or self._forward_pre_hooks
            or _global_backward_hooks
            or _global_backward_pre_hooks
            or _global_forward_hooks
            or _global_forward_pre_hooks
        ):
            return forward_call(*args, **kwargs)

        try:
            result = None
            called_always_called_hooks = set()

            full_backward_hooks, non_full_backward_hooks = [], []
            backward_pre_hooks = []
            if self._backward_pre_hooks or _global_backward_pre_hooks:
                backward_pre_hooks = self._get_backward_pre_hooks()

            if self._backward_hooks or _global_backward_hooks:
                full_backward_hooks, non_full_backward_hooks = (
                    self._get_backward_hooks()
                )

            if _global_forward_pre_hooks or self._forward_pre_hooks:
                for hook_id, hook in (
                    *_global_forward_pre_hooks.items(),
                    *self._forward_pre_hooks.items(),
                ):
                    if hook_id in self._forward_pre_hooks_with_kwargs:
                        args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
                        if args_kwargs_result is not None:
                            if (
                                isinstance(args_kwargs_result, tuple)
                                and len(args_kwargs_result) == 2
                            ):
                                args, kwargs = args_kwargs_result
                            else:
                                raise RuntimeError(
                                    "forward pre-hook must return None or a tuple "
                                    f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                                )
                    else:
                        args_result = hook(self, args)
                        if args_result is not None:
                            if not isinstance(args_result, tuple):
                                args_result = (args_result,)
                            args = args_result

            bw_hook = None
            if full_backward_hooks or backward_pre_hooks:
                bw_hook = hooks.BackwardHook(
                    self, full_backward_hooks, backward_pre_hooks
                )
                args = bw_hook.setup_input_hook(args)

            result = forward_call(*args, **kwargs)
            if _global_forward_hooks or self._forward_hooks:
                for hook_id, hook in (
                    *_global_forward_hooks.items(),
                    *self._forward_hooks.items(),
                ):
                    # mark that always called hook is run
                    if (
                        hook_id in self._forward_hooks_always_called
                        or hook_id in _global_forward_hooks_always_called
                    ):
                        called_always_called_hooks.add(hook_id)

                    if hook_id in self._forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, result)
                    else:
                        hook_result = hook(self, args, result)

                    if hook_result is not None:
                        result = hook_result

            if bw_hook:
                if not isinstance(result, (torch.Tensor, tuple)):
                    warnings.warn(
                        "For backward hooks to be called,"
                        " module output should be a Tensor or a tuple of Tensors"
                        f" but received {type(result)}"
                    )
                result = bw_hook.setup_output_hook(result)

            # Handle the non-full backward hooks
            if non_full_backward_hooks:
                var = result
                while not isinstance(var, torch.Tensor):
                    if isinstance(var, dict):
                        var = next(
                            v for v in var.values() if isinstance(v, torch.Tensor)
                        )
                    else:
                        var = var[0]
                grad_fn = var.grad_fn
                if grad_fn is not None:
                    for hook in non_full_backward_hooks:
                        grad_fn.register_hook(_WrappedHook(hook, self))
                    self._maybe_warn_non_full_backward_hook(args, result, grad_fn)

            return result

        except Exception:
            # run always called hooks if they have not already been run
            # For now only forward hooks have the always_call option but perhaps
            # this functionality should be added to full backward hooks as well.
            for hook_id, hook in _global_forward_hooks.items():
                if (
                    hook_id in _global_forward_hooks_always_called
                    and hook_id not in called_always_called_hooks
                ):  # type: ignore[possibly-undefined]
                    try:
                        hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn(
                            "global module forward hook with ``always_call=True`` raised an exception "
                            f"that was silenced as another error was raised in forward: {str(e)}"
                        )
                        continue

            for hook_id, hook in self._forward_hooks.items():
                if (
                    hook_id in self._forward_hooks_always_called
                    and hook_id not in called_always_called_hooks
                ):  # type: ignore[possibly-undefined]
                    try:
                        if hook_id in self._forward_hooks_with_kwargs:
                            hook_result = hook(self, args, kwargs, result)  # type: ignore[possibly-undefined]
                        else:
                            hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn(
                            "module forward hook with ``always_call=True`` raised an exception "
                            f"that was silenced as another error was raised in forward: {str(e)}"
                        )
                        continue
            # raise exception raised in try block
            raise


class VIBaseModule(VIModule):
    """Base class for VIModules that draw weights from a variational distribution."""

    random_variables: Tuple[str, ...] = ("weight", "bias")

    def __init__(
        self,
        variable_shapes: Dict[str, Tuple[int, ...]],
        variational_distribution: Union[VarDist, List[VarDist]],
        prior: Union[Prior, List[Prior]],
        rescale_prior: bool = False,
        prior_initialization: bool = False,
        return_log_prob: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(variational_distribution, List):
            assert (
                len(variational_distribution) == len(self.random_variables)
            ), "Provide either exactly one variational distribution or exactly one for each random variable"
            self.variational_distribution = variational_distribution
        else:
            self.variational_distribution = [variational_distribution] * len(
                self.random_variables
            )

        if isinstance(prior, List):
            assert (
                len(prior) == len(self.random_variables)
            ), "Provide either exactly one prior distribution or exactly one for each random variable"
            self.prior = prior
        else:
            self.prior = [prior] * len(self.random_variables)

        if rescale_prior:
            shape_dummy = torch.zeros(variable_shapes[self.random_variables[0]])
            fan_in, _ = init._calculate_fan_in_and_fan_out(shape_dummy)
            for prior in self.prior:
                prior.kaiming_rescale(fan_in)

        self._rescale_prior = rescale_prior
        self._prior_init = prior_initialization
        self._return_log_prob = return_log_prob

        for variable, vardist in zip(
            self.random_variables, self.variational_distribution
        ):
            assert variable in variable_shapes, f"shape of {variable} is missing"
            shape = variable_shapes[variable]
            for variational_parameter in vardist.variational_parameters:
                parameter_name = self.variational_parameter_name(
                    variable, variational_parameter
                )
                setattr(
                    self,
                    parameter_name,
                    Parameter(torch.empty(shape, **factory_kwargs)),
                )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset or initialize the parameters of the Module."""
        weight_name = self.variational_parameter_name(
            self.random_variables[0],
            self.variational_distribution[0].variational_parameters[0],
        )
        fan_in, _ = init._calculate_fan_in_and_fan_out(getattr(self, weight_name))
        for variable, vardist, prior in zip(
            self.random_variables, self.variational_distribution, self.prior
        ):
            vardist.reset_parameters(self, variable, fan_in)
            if self._prior_init:
                prior.reset_parameters(self, variable)

    #    def reset_mean(self) -> None:
    #        """Reset the means of random variables similar to non-Bayesian Networks."""
    #        for variable in self.random_variables:
    #            parameter_name = self.variational_parameter_name(variable, "mean")
    #            if variable == "bias" and hasattr(self, parameter_name):
    #                weight_mean = self.variational_parameter_name("weight", "mean")
    #                assert hasattr(
    #                    self, weight_mean
    #                ), "Standard initialization of bias requires weight"
    #                fan_in, _ = init._calculate_fan_in_and_fan_out(
    #                    getattr(self, weight_mean)
    #                )
    #                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #                init.uniform_(getattr(self, parameter_name), -bound, bound)
    #            elif hasattr(self, parameter_name):
    #                init.kaiming_uniform_(getattr(self, parameter_name), a=math.sqrt(5))

    @staticmethod
    def variational_parameter_name(variable: str, variational_parameter: str) -> str:
        """Obtain the attribute name of the variational parameter for the specified variable."""
        spec = ["", variable, variational_parameter]
        return "_".join(spec)

    def get_variational_parameters(self, variable: str) -> List[Tensor]:
        """Obtain all variational parameters for the specified variable."""
        vardist = self.variational_distribution[self.random_variables.index(variable)]
        return [
            getattr(self, self.variational_parameter_name(variable, param))
            for param in vardist.variational_parameters
        ]

    def get_log_probs(self, sampled_params: Iterable[Tensor]) -> Tuple[Tensor, Tensor]:
        """Get prior and variational log prob of the sampled parameters."""
        variational_log_prob = 0.0
        prior_log_prob = 0.0
        for sample, variable, vardist, prior in zip(
            sampled_params,
            self.random_variables,
            self.variational_distribution,
            self.prior,
        ):
            variational_parameters = self.get_variational_parameters(variable)
            variational_log_prob = (
                variational_log_prob
                + vardist.log_prob(sample, *variational_parameters).sum()
            )

            prior_params = [
                getattr(self, self.variational_parameter_name(variable, param))
                for param in prior._required_parameters
            ]
            prior_log_prob = (
                prior_log_prob + prior.log_prob(sample, *prior_params).sum()
            )
        return prior_log_prob, variational_log_prob

    def sample_variables(self) -> List[Tensor]:
        """Draw one sample from the variational distribution of each random variable."""
        params = []
        for variable, vardist in zip(
            self.random_variables, self.variational_distribution
        ):
            variational_parameters = self.get_variational_parameters(variable)
            params.append(vardist.sample(*variational_parameters))
        return params
