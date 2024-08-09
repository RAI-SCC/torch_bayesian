from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module


def _forward_unimplemented(
    self: Module, *input_: Optional[Tensor]
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError(
        f'Module [{type(self).__name__}] is missing the required "forward" function'
    )


class VIModule(Module):
    """Base class for Modules using Variational Inference."""

    forward: Callable[..., Tuple[Tensor, Optional[Dict[str, Tensor]]]] = (
        _forward_unimplemented
    )

    @staticmethod
    def _expand_to_samples(input_: Optional[Tensor], samples: int) -> Tensor:
        if input_ is None:
            input_ = torch.tensor(False)
        return input_.expand(samples, *input_.shape)

    def sampled_forward(
        self, *input_: Optional[Tensor], samples: int = 10
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Forward pass of the module evaluating multiple weight samples.

        Parameters
        ----------
         : Tensor
            Any number of input Tensors
        samples : int
            Number of weight samples to evaluate

        Returns
        -------
        Tensor
            All model outputs as Tensor with the sample dimension first
        Optional[Dict[str, Tensor]]
            Dictionary containing the sampled weights and associated variational parameters (need for some losses)
        """
        expanded = [self._expand_to_samples(x, samples=samples) for x in input_]
        return torch.vmap(self.forward)(*expanded)

    @staticmethod
    def _normal_sample(mean: Tensor, std: Tensor) -> Tensor:
        base_sample = torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
        sample = std * base_sample + mean
        return sample
