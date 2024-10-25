from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import vi
from vi import VIModule
from vi.predictive_distributions import CategoricalPredictiveDistribution

import polars as pl
import matplotlib.pyplot as plt

kit_green = (0, 150/255, 130/255)
kit_blue = (70/255, 100/255, 170/255)
kit_red = (162/255, 34/255, 35/255)
kit_green_50 = (0.5, 0.7941, 0.7549)
kit_blue_50 = (0.6373, 0.6961, 0.8333)
kit_red_50 = (0.8176, 0.5667, 0.5686)

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

def plot_samples(model: VIModule, dataloader: DataLoader):
    num_batches = len(dataloader)
    random_batch = int(torch.randint(low = 0, high= num_batches-1, size = (1,)))
    model.eval()
    with torch.no_grad():
        n = 0
        for x, y  in dataloader:
            if n < random_batch:
                n += 1
            else:
                break

        samples = model(x)
        mean_samples = samples.mean(dim=0)
        std_samples = torch.std(samples, dim=0)
        num_samples = y.shape[0]
        random_sample =int(torch.randint(low=0, high=num_samples - 1, size=(1,)))
        mean = mean_samples[random_sample]
        std = std_samples[random_sample]
        real = y[random_sample]
        in_mod = x[random_sample]
        input_length = in_mod.shape[0]
        output_length = real.shape[0]

    plt.clf()
    plt.plot(in_mod.numpy(), color="blue", label="inputs")
    # plot outputs and ground truth behind input sequence
    plt.plot(
        range(input_length, input_length + output_length),
        mean.numpy(),
        color="orange",
        label="outputs",
    )
    plt.plot(
        range(input_length, input_length + output_length),
        real.numpy(),
        color="green",
        label="ground truth",
    )
    plt.legend()
    plt.xticks(range(0, input_length + output_length, 2))
    plt.fill_between(
        range(input_length, input_length + output_length),
        (torch.add(mean, std, alpha=1)).numpy(),
        (torch.add(mean, std, alpha=-1)).numpy(),
        color="orange",
        alpha=0.1,
    )
    file_name = 'res.png'
    plt.savefig(file_name)
    # plt.plot(range(input_length, input_length + output_length), (torch.add(v_outputs[i, :], voutput_std[i, :], alpha=1)).numpy(), color='orange', alpha=.3, label="upper conf bound")
    # plt.plot(range(input_length, input_length + output_length), (torch.add(v_outputs[i, :], voutput_std[i, :], alpha=-1)).numpy(), color='orange', alpha=.3, label="lower conf bound")
    #plt.show()
    # time.sleep(5)



def sigma_weight_plot(weights, sigmas, base_name):
    # Creates scatter plot for a layer's sigma values and weights
    final_weights = (torch.flatten(weights)).tolist()
    final_sigma = (torch.abs(torch.flatten(sigmas))).tolist()

    x1 = final_sigma

    y1 = final_weights

    plt.figure(figsize=(10, 8))
    plt.scatter(x1, y1, c=kit_blue_50,
                linewidths=2,
                marker="s",
                edgecolor=kit_blue,
                s=50)
    # plt.xlim(0, 1)
    # plt.ylim(-0.5, 0.5)

    plt.xlabel("Sigma Values", fontsize=18)
    plt.ylabel("Weights", fontsize=18)

    file_name = base_name + '-sigma-values.png'
    plt.savefig(file_name)
    #plt.show()

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

    epochs = 25

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

    plot_samples(model, test_dataloader)

if __name__ == "__main__":
    torch_tutorial_MCD()
