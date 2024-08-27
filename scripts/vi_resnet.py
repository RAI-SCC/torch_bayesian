from typing import Tuple, Union

import torch
import torch.utils.data

# from torchvision.transforms import Compose
import vi
from vi import VIModule
from vi.priors import MeanFieldNormalPrior, Prior
from vi.variational_distributions import MeanFieldNormalVarDist, VariationalDistribution

# MODEL
# Define neural network by subclassing PyTorch's nn.Module.
# Save to a separate Python module file `model.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.


class VIResNetBlock(VIModule):
    """
    A single VIResNet block.

    Attributes
    ----------
    features : torch.nn.container.Sequential
        both convolutional layers in one VISequential
    dowsampling: Optional(vi.Conv2d)
        downsampling layer in case stride > 1


    Methods
    -------
    __init__()
        The constructor defining the network's architecture.
    forward()
        The forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        variational_distribution: VariationalDistribution = MeanFieldNormalVarDist(),
        prior: Prior = MeanFieldNormalPrior(),
    ) -> None:
        """
        Initialize VIResNet block.

        Parameters
        ----------
        in_features : int
            The input feature dimension.
        out_features : int
            The output feature dimension.
        stride : int
            The stride of the convolutional layers.
        variational_distribution : VariationalDistribution
            The variational distribution used in all the VI components.
        prior: Prior
            The prior used in all the VI components.
        """
        super().__init__()

        self.features = vi.VISequential(
            vi.VIConv2d(
                in_features,
                out_features,
                kernel_size=3,
                stride=stride,
                padding=1,
                prior=prior,
                variational_distribution=variational_distribution,
            ),
            torch.nn.ReLU(inplace=True),
            vi.VIConv2d(
                out_features,
                out_features,
                kernel_size=3,
                stride=1,
                padding=1,
                prior=prior,
                variational_distribution=variational_distribution,
            ),
        )

        self.downsampling = None
        if stride > 1:
            self.downsampling = vi.VIConv2d(
                in_features,
                out_features,
                kernel_size=1,
                stride=stride,
                prior=prior,
                variational_distribution=variational_distribution,
            )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Do forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input data.
        """
        out = self.features(x)
        if self._return_log_prob:
            if self.downsampling is not None:
                down_x = self.downsampling(x)
                out_value = torch.relu(out[0] + down_x[0])
                out_priorlogprob = out[-2] + down_x[-2]
                out_varlogprob = out[-1] + down_x[-1]
                return out_value, out_priorlogprob, out_varlogprob
            else:
                out_value = torch.relu(out[0] + x)
                return out_value, out[-2], out[-1]
        else:
            if self.downsampling is not None:
                out = out + self.downsampling(x)
            else:
                out = out + x

            out = torch.relu(out)
            return out


class VIResNet20(VIModule):
    """
    A VI version of ResNet20.

    Attributes
    ----------
    features : torch.nn.container.Sequential
        all layers in one VISequential


    Methods
    -------
    __init__()
        The constructor defining the network's architecture.
    forward()
        The forward pass.
    """

    def __init__(
        self,
        num_classes: int = 10,
        variational_distribution: VariationalDistribution = MeanFieldNormalVarDist(),
        prior: Prior = MeanFieldNormalPrior(),
    ) -> None:
        """
        Initialize VIResNet block.

        Parameters
        ----------
        num_classes : int
            The number of classes in the underlying classification problem.
        variational_distribution : VariationalDistribution
            The variational distribution used in all the VI components.
        prior: Prior
            The prior used in all the VI components.
        """
        super().__init__()

        self.features = vi.VISequential(
            vi.VIConv2d(
                3,
                16,
                kernel_size=3,
                padding=1,
                prior=prior,
                variational_distribution=variational_distribution,
            ),
            torch.nn.ReLU(inplace=True),
            VIResNetBlock(
                16, 16, variational_distribution=variational_distribution, prior=prior
            ),
            VIResNetBlock(
                16, 16, variational_distribution=variational_distribution, prior=prior
            ),
            VIResNetBlock(
                16, 16, variational_distribution=variational_distribution, prior=prior
            ),
            VIResNetBlock(
                16,
                32,
                stride=2,
                variational_distribution=variational_distribution,
                prior=prior,
            ),
            VIResNetBlock(
                32, 32, variational_distribution=variational_distribution, prior=prior
            ),
            VIResNetBlock(
                32, 32, variational_distribution=variational_distribution, prior=prior
            ),
            VIResNetBlock(
                32,
                64,
                stride=2,
                variational_distribution=variational_distribution,
                prior=prior,
            ),
            VIResNetBlock(
                64, 64, variational_distribution=variational_distribution, prior=prior
            ),
            VIResNetBlock(
                64, 64, variational_distribution=variational_distribution, prior=prior
            ),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(start_dim=-3),
            vi.VILinear(64, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Do forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input data.
        """
        out = self.features(x)
        return out
