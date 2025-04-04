import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
def create_mlp_dataset(data, input_steps=24, output_steps=6):
    """
    Creates dataset from a 1D array with sliding windows.
    """
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i: i + input_steps])
        y.append(data[i + input_steps: i + input_steps + output_steps])
    return np.array(X), np.array(y)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # Expect X to be (samples, 24) already.
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

