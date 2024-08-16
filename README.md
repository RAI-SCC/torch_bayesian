# vi - Easy Variational Inference

This package provides a simple way for non-expert users to implement and
train Bayesian Neural Networks (BNNs) with Variational Inference (VI).
To make this as easy as possible most components mirror component from
[pytorch](https://pytorch.org/docs/stable/index.html). If you would import
any class from `torch.nn` you can either add the prefix `VI` and import it
from this package instead, e.g. `nn.Linear`  &rarr; `vi.VILinear`, or just
use it as is with modules provided here, e.g. `nn.ReLU`.

For the simplest approach three changes need to be made:
1. All custom modules containing `VIModules` need to be `VIModules`

    For more details see the explanation on [Auto-Sampling](#auto-sampling)

2. The loss function must be switched

    When in doubt, use `vi.KullbackLeiblerLoss` with `MeanFieldNormalPredictiveDistribution` for regression tasks
    or `CategoricalPredictiveDistribution` for classfication. More details [here](#the-predictive-distribution).

3. For some losses `VIModules` must return the probability of their weights

    This means fully flexible models must be able to dynamically handle
    working with two more or less outputs. If a model is just a sequence
    `nn.Module`s and `VIModule`s (including residual connections) this can
    be handled with `VISequential` and `VIResidualConnection`.

For a (potentially familar) example see `scripts/pytorch_tutorial.py`, which
contains a copy of the pytorch [Quickstart tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
modified to train a BNN with variational inference.

### Variational Inference

### The Predictive Distribution

### Auto-Sampling

### The Prior

### The Variational Distribution


#### The Documentation grind tracker

- [ ] README
- [ ] vi.base
- [ ] conv
- [x] kl_loss
- [x] linear
- [x] sequential


- variation_distributions

    - [ ] base
    - [ ] normal


- priors

    - [ ] base
    - [ ] normal


- predictive_distributions

  - [ ] base
  - [ ] categorical
  - [ ] normal
