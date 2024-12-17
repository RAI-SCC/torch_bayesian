from typing import TYPE_CHECKING, Optional, Tuple, TypedDict, TypeVar, Union

import torch
from torch import Tensor
from torch.nn.common_types import _scalar_or_tuple_any_t
from typing_extensions import TypeAlias

if TYPE_CHECKING:  # pragma: no cover
    from ..priors import Prior
    from ..variational_distributions import VariationalDistribution

_prior_any_t: TypeAlias = _scalar_or_tuple_any_t["Prior"]
_vardist_any_t: TypeAlias = _scalar_or_tuple_any_t["VariationalDistribution"]

T = TypeVar("T")
_log_prob_return_format = Tuple[T, Tensor]
VIReturn = Union[T, _log_prob_return_format[T]]


class _VIkwargs(TypedDict):
    variational_distribution: _vardist_any_t
    prior: _prior_any_t
    rescale_prior: bool
    kaiming_initialization: bool
    prior_initialization: bool
    return_log_probs: bool
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]
