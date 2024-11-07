import torch
from torch import Tensor, nn
from torch.nn import MSELoss
from entsoe_data_load import TimeseriesDataset
from torch.utils.data import DataLoader
import polars as pl
from typing import Callable
import vi
from vi import VIModule
from vi.variational_distributions import MeanFieldNormalVarDist
from mean_std_plot import sigma_weight_plot
from random_sample_plot import plot_random_samples

def torch_tutorial_MCD(input_length, hidden1, hidden2, output_length, batch_size, epochs) -> None:

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
        def __init__(self, variational_distribution=MeanFieldNormalVarDist()) -> None:
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = vi.VISequential(
                vi.VILinear(input_length, hidden1, variational_distribution=variational_distribution),
                nn.ReLU(),
                vi.VILinear(hidden1, hidden2, variational_distribution=variational_distribution),
                nn.ReLU(),
                vi.VILinear(hidden2, output_length, variational_distribution=variational_distribution),
            )

        def forward(self, x_: Tensor) -> Tensor:
            x_ = self.flatten(x_)
            logits = self.linear_relu_stack(x_)
            return logits

    model = NeuralNetwork(variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)
    print(model)

    #predictive_distribution = MeanFieldNormalPredictiveDistribution()
    #loss_fn = vi.KullbackLeiblerLoss(
    #    predictive_distribution, dataset_size=len(dataset_train)
    #)
    loss_fn = vi.MeanSquaredErrorLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

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

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "log_std" in name:
                base = name.removesuffix('log_std')
                weight_layer_name = base + "mean"
                weight_param = dict(model.named_parameters())[weight_layer_name]
                sigma_weight_plot(weight_param, param, base)

    plot_random_samples(model, test_dataloader)
