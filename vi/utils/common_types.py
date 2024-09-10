from typing import TYPE_CHECKING, Tuple, TypeVar

from torch import Tensor
from torch.nn.common_types import _scalar_or_tuple_any_t

if TYPE_CHECKING:
    pass

_prior_any_t = _scalar_or_tuple_any_t["Prior"]
_vardist_any_t = _scalar_or_tuple_any_t["VariationalDistribution"]

T = TypeVar("T")
_log_prob_return_format = Tuple[T, Tensor]
