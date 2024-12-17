from typing import Dict, Optional, Set, Tuple

import torch
from torch import Tensor

try:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
except ImportError:
    _has_matplotlib = False
else:
    _has_matplotlib = True


def derive_variables_and_parameters(
    state_dict: Dict[str, Tensor], is_vi: bool = True, strict: bool = False
) -> Tuple[Set[str], Optional[Set[str]]]:
    """Derive the used random variables and variational parameters of a VIModule state_dict."""
    if is_vi:
        variables_set = set()
        parameter_set = set()
        for key in state_dict:
            try:
                variable, parameter = _variable_and_parameter_from_key(key)
            except ValueError:
                continue

            variables_set.add(variable)
            parameter_set.add(parameter)
    else:
        variables_set = set([key.split(".")[-1] for key in state_dict])
        parameter_set = None

    if not strict:
        variables_set = _filter_overlapping_keys(variables_set)
        parameter_set = (
            _filter_overlapping_keys(parameter_set)
            if parameter_set is not None
            else None
        )

    return variables_set, parameter_set


def _variable_and_parameter_from_key(key: str) -> Tuple[str, str]:
    split = key.split(".")[-1].split("_")
    if len(split) < 3:
        raise ValueError(f"Invalid key {key} is not from a VIModule")
    elif split[-2] == "log":
        variable = "_".join(split[1:-2])
        parameter = "_".join(split[-2:])
    else:
        variable = "_".join(split[1:-1])
        parameter = "_".join(split[-1:])
    return variable, parameter


def _filter_overlapping_keys(keys: Set[str]) -> Set[str]:
    """
    Filter keys that end with the name of other keys.

    Used to treat e.g. both "in_proj_weight" and "weight" as the same category.
    """
    filtered_keys = set()
    for item in keys:
        contains_other = any([(ref != item) and item.endswith(ref) for ref in keys])
        if not contains_other:
            filtered_keys.add(item)
    return filtered_keys


def aggregate_by_variable_and_parameter(
    state_dict: Dict[str, Tensor], is_vi: bool = True, strict: bool = False
) -> Dict[str, Dict[str, Tensor]]:
    """Convert state dict to values sorted by variable and parameter."""
    variables_set, parameter_set = derive_variables_and_parameters(
        state_dict, is_vi, strict
    )
    if parameter_set is None:
        parameter_set = {"mean"}

    aggregated = {}
    for variable in variables_set:
        variable_dict = {}
        for parameter in parameter_set:
            variable_dict[parameter] = torch.empty(0)
        aggregated[variable] = variable_dict

    for key in state_dict:
        if is_vi:
            try:
                variable, parameter = _variable_and_parameter_from_key(key)
            except ValueError:
                continue
        else:
            variable = key.split(".")[-1]
            parameter = "mean"

        if not strict:
            for var in variables_set:
                if variable.endswith(var):
                    variable = var
                    break

        aggregated[variable][parameter] = torch.cat(
            [aggregated[variable][parameter], state_dict[key].flatten()]
        )
    return aggregated


def plot_weights(
    weight_dict: Dict[str, Dict[str, Tensor]],
    plot_vars: Tuple[str, ...] = ("weight", "bias"),
    xy_params: Tuple[str, str] = ("log_std", "mean"),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot weight distribution in parameter space."""
    if not _has_matplotlib:
        raise ImportError("plot_weights requires optional dependency matplotlib")

    mpl.rcParams["font.size"] = 20
    plt.rcParams["figure.dpi"] = 100

    labels = [
        param if not param.startswith("log_") else param[4:] for param in xy_params
    ]

    num_plots = len(plot_vars)
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 5, 5))
    plt.subplots_adjust(left=0.2)

    for var, ax in zip(plot_vars, axes):
        xdata = _prepare_data(weight_dict[var], xy_params[0])
        ydata = _prepare_data(weight_dict[var], xy_params[1])
        ax.plot(xdata, ydata, "o", markersize=1, label=var.title())

        ax.set_xlabel(" ".join([var.title(), labels[0]]))
        ax.set_ylabel(" ".join([var.title(), labels[1]]))

    fig.tight_layout()
    return fig, axes


def _prepare_data(var_dict: Dict[str, Tensor], param: str) -> Tensor:
    if param.startswith("log"):
        return torch.exp(var_dict[param])
    else:
        return var_dict[param]


if __name__ == "__main__":
    from vi import VILinear

    model = VILinear(16, 17)
    weight_dict = aggregate_by_variable_and_parameter(model.state_dict())
    fig, ax = plot_weights(weight_dict)
    plt.show()
