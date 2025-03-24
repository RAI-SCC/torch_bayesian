"""Definition of convolutional layers."""

from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple

from .base import VIBaseModule
from .priors import MeanFieldNormalPrior
from .utils.common_types import VIReturn, _prior_any_t, _vardist_any_t, _VIkwargs
from .variational_distributions import MeanFieldNormalVarDist


class _VIConvNd(VIBaseModule):
    """Private class."""

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[Tensor]}

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:  # type: ignore[empty-body]
        ...  # pragma: no cover

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        variational_distribution: _vardist_any_t,
        prior: _prior_any_t,
        rescale_prior: bool = False,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        vikwargs: _VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )

        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if transposed:
            weight_shape = (in_channels, out_channels // groups, *kernel_size)
        else:
            weight_shape = (out_channels, in_channels // groups, *kernel_size)

        bias_shape = (out_channels,)
        variable_shapes = dict(
            weight=weight_shape,
            bias=bias_shape,
        )

        if bias:
            self.random_variables = ("weight", "bias")
        else:
            self.random_variables = ("weight",)

        super().__init__(variable_shapes=variable_shapes, **vikwargs)

    def __setstate__(self, state: Any) -> None:
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


class VIConv1d(_VIConvNd):
    """Equivalent of nn.Conv1d with variational inference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
        prior: _prior_any_t = MeanFieldNormalPrior(),
        rescale_prior: bool = False,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        vikwargs: _VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **vikwargs,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Optional[Tensor] = None
    ) -> Tensor:
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input_,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> VIReturn[Tensor]:
        """
        Forward computation.

        Applies a 1D convolution to a tensor of shape [N, C_in, W_in]. Where
        N is the batch size, C are channels, and W the width of the input.

        Parameters
        ----------
        input_: Tensor
            Input tensor of shape [N, C_in, W_in].

        Returns
        -------
        output, log_probs if return_log_probs else output

        output: Tensor
            Output tensor of shape [N, C_out, W_out].
            Auto-sampling will add a sample dimension at the start for the overall output.
        log_probs: Tensor
            Tensor of shape (2,) containing the total prior and variational log
            probability (in that order) of the sampled weights and biases.
            Only returned if return_log_probs.
        """
        params = self.sample_variables()

        output = self._conv_forward(input_, *params)

        if self._return_log_probs:
            log_probs = self.get_log_probs(params)
            return output, log_probs
        else:
            return output


class VIConv2d(_VIConvNd):
    """
    Equivalent of nn.Conv2d with variational inference.

    Called with the same arguments as nn.VIConv2d, but accepts additional arguments.
    This module's random variables are
    ("weight", "bias") if bias == True
    ("weight", )       if bias == False

    Additional Parameters
    ---------------------
    variational_distribution: Union[VarDist, List[VarDist]]
        Variational distribution which specifies the assumed weight distribution. A list of
        distributions may be provided to specify different choices for each random variable.
        Default: MeanFieldNormalVarDist()
    prior: Union[Prior, List[Prior]]
        Prior distribution which specifies the previous knowledge about the weight distribution.
        A list of distributions may be provided to specify different choices for each random
        variable. Default: MeanFieldNormalPrior()
    rescale_prior: bool
        If True prior._scaling_parameters are scaled with the sqrt of the layer width.
        This may be necessary to maintain normalization for wide layers. Default: False
    prior_initialization: bool
        If True parameters are initialized according to the prior. If False parameters are
        initialized similar to non-Bayesian networks. Default: False
    return_log_probs: bool
        If True the model forward pass returns the log probability of the sampled weight.
        This is required for the standard loss calculation. Default: True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
        prior: _prior_any_t = MeanFieldNormalPrior(),
        rescale_prior: bool = False,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        vikwargs: _VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **vikwargs,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Optional[Tensor] = None
    ) -> Tensor:
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input_,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> VIReturn[Tensor]:
        """
        Forward computation.

        Applies a 2D convolution to a tensor of shape [N, C_in, H_in, W_in]. Where
        N is the batch size, C are channels, and [H, W] are height and width of the input.

        Parameters
        ----------
        input_: Tensor
            Input tensor of shape [N, C_in, H_in, W_in].

        Returns
        -------
        output, log_probs if return_log_probs else output

        output: Tensor
            Output tensor of shape [N, C_out, H_out, W_out].
            Auto-sampling will add a sample dimension at the start for the overall output.
        log_probs: Tensor
            Tensor of shape (2,) containing the total prior and variational log
            probability (in that order) of the sampled weights and biases.
            Only returned if return_log_probs.
        """
        params = self.sample_variables()

        output = self._conv_forward(input_, *params)

        if self._return_log_probs:
            log_probs = self.get_log_probs(params)
            return output, log_probs
        else:
            return output


class VIConv3d(_VIConvNd):
    """
    Equivalent of nn.Conv3d with variational inference.

    Called with the same arguments as nn.VIConv3d, but accepts additional arguments.
    This module's random variables are
    ("weight", "bias") if bias == True
    ("weight", )       if bias == False

    Parameters
    ----------
    variational_distribution: Union[VarDist, List[VarDist]]
        Variational distribution which specifies the assumed weight distribution. A list of
        distributions may be provided to specify different choices for each random variable.
        Default: MeanFieldNormalVarDist()
    prior: Union[Prior, List[Prior]]
        Prior distribution which specifies the previous knowledge about the weight distribution.
        A list of distributions may be provided to specify different choices for each random
        variable. Default: MeanFieldNormalPrior()
    rescale_prior: bool
        If True prior._scaling_parameters are scaled with the sqrt of the layer width.
        This may be necessary to maintain normalization for wide layers. Default: False
    prior_initialization: bool
        If True parameters are initialized according to the prior. If False parameters are
        initialized similar to non-Bayesian networks. Default: False
    return_log_probs: bool
        If True the model forward pass returns the log probability of the sampled weight.
        This is required for the standard loss calculation. Default: True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
        prior: _prior_any_t = MeanFieldNormalPrior(),
        rescale_prior: bool = False,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        vikwargs: _VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **vikwargs,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Optional[Tensor] = None
    ) -> Tensor:
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input_,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> VIReturn[Tensor]:
        """
        Forward computation.

        Applies a 3D convolution to a tensor of shape [N, C_in, D_in, H_in, W_in]. Where
        N is the batch size, C are channels, and [D, H, W] are depth, height and width of the input.

        Parameters
        ----------
        input_: Tensor
            Input tensor of shape [N, C_in, D_in, H_in, W_in].

        Returns
        -------
        output, log_probs if return_log_probs else output

        output: Tensor
            Output tensor of shape [N, C_out, D_out, H_out, W_out].
            Auto-sampling will add a sample dimension at the start for the overall output.
        log_probs: Tensor
            Tensor of shape (2,) containing the total prior and variational log
            probability (in that order) of the sampled weights and biases.
            Only returned if return_log_probs.
        """
        params = self.sample_variables()

        output = self._conv_forward(input_, *params)

        if self._return_log_probs:
            log_probs = self.get_log_probs(params)
            return output, log_probs
        else:
            return output
