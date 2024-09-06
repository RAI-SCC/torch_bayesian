from torch import Tensor
from torch.nn.common_types import _tensor_list_t

from .common_types import _log_prob_return_format


def to_log_prob_return_format(
    output: _tensor_list_t, prior_log_prob: Tensor, variational_log_prob: Tensor
) -> _log_prob_return_format[_tensor_list_t]:
    """
    Arranges model output and log probabilities into the standard return format.

    Parameters
    ----------
    output : Union[Tensor, List[Tensor]]
        The model output as one or multiple Tensors.
    prior_log_prob : Tensor
        The log probability of the module weights in the prior distribution.
    variational_log_prob : Tensor
        The log probability of the module weights in the variational distribution.

    Returns
    -------
    Union[Tensor, List[Tensor]]
        The model output as one or multiple Tensors.
    Tuple[Tensor, Tensor]
        A Tuple: (prior_log_prob, variational_log_prob).
    """
    return output, (prior_log_prob, variational_log_prob)
