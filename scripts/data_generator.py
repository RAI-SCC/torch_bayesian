import torch
import numpy as np

from torch.utils.data import Dataset
from torch.distributions import Normal, Uniform, StudentT, Gamma, Bernoulli


def data_generator(x_lims, dist, width, size, dof=5):

    x_true = torch.linspace(np.min(x_lims), np.max(x_lims), size)
    y_true = ( x_true**3 +5*x_true**2  )/100

    if dist == "normal":
        errors = Normal(0, width)
        y_true = y_true + errors.sample(torch.Size([len(x_true)]))
    elif dist == "uniform":
        errors = Uniform(-width/2, width*1/2)
        y_true = y_true + errors.sample(torch.Size([len(x_true)]))
    elif dist == "student_t":
        errors = StudentT(dof, 0, width)
        y_true = y_true + errors.sample(torch.Size([len(x_true)]))
    elif dist == "gamma":
        mean = torch.abs(y_true)
        alpha = 2.0
        beta = 0.5
        errors = Gamma(alpha, beta)
        y_true = y_true + errors.sample(torch.Size([len(x_true)]))
    elif dist == "mixture_gaussian":
        p = Bernoulli(0.5).sample(torch.Size([len(x_true)]))

        normal_1 = Normal(-1, width*1/2)
        normal_2 = Normal(7, width*1/2)
        y_true = y_true + p*normal_1.sample(torch.Size([len(x_true)])) + (1-p)*normal_2.sample(torch.Size([len(x_true)]))

    else:
        raise SystemExit('Unknown distribution')

    return x_true, y_true, errors


class CustomDataset(Dataset):
    """
    Custom dataset class for data generation.

    Attributes
    ----------
    x_true: numpy array
        True values for x.
    y_true: numpy array
        True values for y.

    Methods
    -------
    __len__()
        get the length of dataset
    __getitem__(size, data_interval)
        get the subset of data for a specific size and x interval
    """

    def __init__(self, x_true, y_true):
        self.x_true = x_true
        self.y_true = y_true

    def __len__(self):
        return len(self.x_true)

    def __getitem__(self, size, data_interval):
        interval = np.where((self.x_true >= data_interval[0]) & (self.x_true <= data_interval[1]))[0]
        rng = np.random.default_rng(seed=42)
        data_inds = rng.integers(low=0, high=interval.size - 1, size=size)
        noise_perturbations = rng.normal(loc=0, scale=0.1, size=len(data_inds))
        x_data = self.x_true[interval][data_inds]
        y_data = self.y_true[interval][data_inds] + noise_perturbations

        #x_data = torch.from_numpy(x_data).float()
        #y_data = torch.from_numpy(y_data).float().unsqueeze(1)
        return x_data, y_data
