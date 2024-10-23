from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import vi
from vi import VIModule
from vi.predictive_distributions import CategoricalPredictiveDistribution

import polars as pl

class TimeseriesDataset(Dataset):
    def __init__(self, raw, input_length, output_length):
        self.raw = raw
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.raw) - (self.input_length + self.output_length) + 1

    def __getitem__(self, index):
        return (self.raw[index:index+self.input_length],
                self.raw[index+self.input_length:index+self.input_length+self.output_length])

class AlternativeTimeseriesDataset(Dataset):
    def __init__(self, raw, input_length, output_length):
        self.raw = raw
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return int(len(self.raw) / (self.input_length + self.output_length))

    def __getitem__(self, index):
        start = index * (self.input_length + self.output_length)
        return (self.raw[start:start+self.input_length],
                self.raw[start+self.input_length:start+self.input_length+self.output_length])

def torch_tutorial_MCD() -> None:
    input_length = 150
    output_length = 50
    batch_size = 32

    df = pl.read_csv("data/ENTSOEEnergyLoads/de.csv",
        dtypes={"start": pl.Datetime, "end": pl.Datetime, "load": pl.Float32},
    )
    x = df["load"]
    x = x.fill_null(strategy="backward")
    normalized_x = (x - x.mean()) / x.std()
    x_tensor = normalized_x.to_torch()
    data_train, data_test = x_tensor[: int(len(x) * 0.7)], x_tensor[int(len(x) * 0.7):]
    dataset_train = TimeseriesDataset(data_train, input_length, output_length)
    dataset_test = TimeseriesDataset(data_test, input_length, output_length)

    # Create data loaders.
    train_dataloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)

    for x, y in test_dataloader:
        print(f"Shape of X [N, H]: {x.shape}")
        print(f"Shape of y [N, F]: {y.shape} {y.dtype}")
        break

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Define model
    class NeuralNetwork(vi.VIModule):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = vi.VISequential(
                vi.VILinear(150, 150),
                nn.ReLU(),
                vi.VILinear(150, 100),
                nn.ReLU(),
                vi.VILinear(100, 50),
            )

        def forward(self, x_: Tensor) -> Tensor:
            x_ = self.flatten(x_)
            logits = self.linear_relu_stack(x_)
            return logits

    model = NeuralNetwork().to(device)
    model.return_log_probs(False)
    print(model)

    predictive_distribution = CategoricalPredictiveDistribution()
    loss_fn = vi.MeanSquaredErrorLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(
        dataloader: DataLoader,
        model: VIModule,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        size = len(dataloader.dataset)
        model.train()
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader: DataLoader, model: VIModule, loss_fn: Callable) -> None:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                samples = model(x)
                test_loss += loss_fn(samples, y).item()

        test_loss /= num_batches
        print(
            f"Test Error: Avg loss: {test_loss:>8f} \n"
        )

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == "__main__":
    torch_tutorial_MCD()
