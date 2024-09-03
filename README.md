# vi - Easy Variational Inference

This package provides a simple way for non-expert users to implement and
train Bayesian Neural Networks (BNNs) with Variational Inference (VI).
To make this as easy as possible most components mirror components from
[pytorch](https://pytorch.org/docs/stable/index.html).

- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Level 1](#level-1)
  - [Level 2](#level-2)
  - [Level 3](#level-3)

## Installation

We heavily recommend installing ``vi`` in a dedicated `Python3.12+` [virtual environment](https://docs.python.org/3/library/venv.html).
You can install ``vi`` directly from the GitHub repository via:

```console
$ pip install git+https://github.com/RAI-SCC/vi
```

Alternatively, you can install ``vi`` locally. To achieve this, there are two steps you need to follow:

1. Clone the repository

```console
$ git clone https://github.com/RAI-SCC/vi
```

2. Install the code locally

```console
$ pip install -e .
```

To get the development dependencies, run:

```console
$ pip install -e ."[dev]"
```

If you want to run scripts from the script folder, run:

```console
$ pip install -e ."[scripts]"
```


## Quickstart

This Quickstart guide assumes basic familiarity with [pytorch](https://pytorch.org/docs/stable/index.html)
and knowledge of how to implement the intended model in it.
Three levels are introduced:
- [Level 1](#level-1): Simple sequential layer stacks
- [Level 2](#level-2): Customizing Bayesian assumptions and VI kwargs
- [Level 3](#level-3): Non-sequential models and log probabilities

### Level 1

Many parts of a neural network remain completely unchanged when turning it into a BNN.
Indeed, only `Module`s containing `nn.Parameter`s, need to be changed. Therefore, if a
pytorch model fulfills two requirements it can be transferred almost unchanged:

1. All pytorch `Module`s containing parameters have equivalents in this package (table below).
2. The model can be expressed purly as a sequential application of a list of layers, i.e.
with `nn.Sequential`.

| pytorch     | vi replacement |
|-------------|----------------|
| `nn.Linear` | `VILinear`     |
| `nn.Conv1d` | `VIConv1d`     |
| `nn.Conv2d` | `VIConv2d`     |
| `nn.Conv3d` | `VIConv3d`     |

Given these two conditions, inherit the module from `vi.VIModule` instead of `nn.Module`
and use `vi.VISequential` instead of `nn.Sequential`. Then replace all layers
containing parameters as shown in the table above. For basic usage initialize these
modules with the same arguments as their pytorch equivalent. For advanced usage see
[Quickstart: Level 2](#level-2). Many other layers can be included as-is. In particular
activation functions, pooling, padding, and normalization (even dropout, though they
should not be necessary since the prior acts as regularization). Currently not supported
are recurrent, Transformer and transposed convolution layers.

Additionally, the loss must be replaced. To start out use `vi.KullbackLeiblerLoss`,
which requires a `PredictiveDistribution` and the size of the training dataset (this is
important for balancing of assumptions and data, see
[Variational Inference](#variational-inference)).
Choose your `PredictiveDistribution` from the table below based on the loss you would
use in pytorch.

| pytorch               | vi replacement (import from `vi.predictive_distributions`) |
|-----------------------|------------------------------------------------------------|
| `nn.MSELoss`          | `MeanFieldNormalPredicitveDistribution`                    |
| `nn.CrossEntropyLoss` | `CategoricalPredicitveDistribution`                        |

> **Note:** Reasons for the requirement to use `VISequential` (and how to overcome it)
> are described in [Quickstart: Level 3](#level-3). However, adding residual connections
> from the start to the end of a block of layers can also be achieved using
> `VIResidualConnection`, which acts the same as `VISequential`, but adds the input to
> the output.

### Level 2

While the interface of `VIModule`s is kept intentionally similar to pytorch, there are
additional arguments that customize the Bayesian assumptions that all provided layers
accept and custom modules should generally accept and pass on to submodules:
- variational_distribution (`VariationalDistribution`): defines the weight distribution
and variational parameters (more details [here](#the-variational-distribution)). The
default `MeanFieldNormalVarDist` assumes normal distributed, uncorrelated weights
described by a mean and a standard deviation. While there are currently no alternatives
the initial value of the standard deviation can be customized here.
- prior (`Prior`): defines the assumptions on the weight distribution and acts as
regularizer (more details [here](#the-prior)). The default `MeanFieldNormalPrior`
assumes normal distributed, uncorrelated weights with mean 0 and standard deviation 1
  (also known as an uninformative or standard normal prior). Mean and standard deviation
can be adapted here. Particularly reducing the standard deviation may help convergence
at the risk of an overconfident model. Other available priors:
  - `BasicQuietPrior`: a prior that correlates mean and standard deviation to
  disincentivize noisy weights
- rescale_prior (`bool`): Experimental. Scales the prior similar to Kaiming-initialization
may help with convergence, but may lead to overconfidence. Current research.
- prior_initialization (`bool`): Experimental. Initialize parameters from the prior
instead of according to standard non-Bayesian methods. May lead to much faster
convergence, but can cause the issues Kaiming-initialization counteracts unless
rescale_prior is also set to True. Current research.
- return_log_prob (`bool`): This is the topic of [Quickstart: Level 3](#level-3).

### Level 3



If you would import any class from `torch.nn` you can either add the prefix `VI` and import it
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
- [x] conv
- [x] kl_loss
- [x] linear
- [x] sequential


- variation_distributions

    - [ ] base
    - [ ] normal


- priors

    - [ ] base
    - [ ] normal
    - [ ] quiet


- predictive_distributions

  - [x] base
  - [ ] categorical
  - [ ] normal

### ToDo

- Check if log params should be set to `-inf` if `fan_in` is 0
- BasicQuietPrior might need an eps to avoid infinity for mean = 0
