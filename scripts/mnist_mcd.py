from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import vi
from vi import VIModule
import torch.nn.functional as F
from vi.variational_distributions import MeanFieldNormalVarDist
from vi.linear import sampling
all_sampling_time = []
epoch_sampling_time = []
last_len = 0

def mnist_mcd() -> None:
    """Reimplement the pytorch quickstart tutorial with BNNs."""
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 256

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for x, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
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
                vi.VILinear(28 * 28, 512, variational_distribution=variational_distribution),
                nn.ReLU(),
                vi.VILinear(512, 512, variational_distribution=variational_distribution),
                nn.ReLU(),
                vi.VILinear(512, 10, variational_distribution=variational_distribution),
            )

        def forward(self, x_: Tensor) -> Tensor:
            x_ = self.flatten(x_)
            logits = self.linear_relu_stack(x_)
            return logits


    model = NeuralNetwork(variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)
    print(model)

    loss_fn =  F.cross_entropy
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    def train(
        dataloader: DataLoader,
        model: VIModule,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        sample_num
    ) -> None:
        size = len(dataloader.dataset)
        model.train()
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x, samples = sample_num)
            mean_model_output = pred.mean(dim=0)
            probs = F.softmax(mean_model_output, dim = 1)

            loss = loss_fn(probs, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            global epoch_sampling_time
            global last_len
            new_sampling = sampling[last_len:]
            last_len = len(sampling)
            batch_sampling_time = torch.tensor(new_sampling)

            if epoch_sampling_time == None:
                epoch_sampling_time = batch_sampling_time
            else:
                epoch_sampling_time = torch.cat((epoch_sampling_time, batch_sampling_time))

    def test(dataloader: DataLoader, model: VIModule, loss_fn: Callable, sample_num) -> None:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0.0, 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x, samples = sample_num)
                mean_model_output = pred.mean(dim=0)
                probs = F.softmax(mean_model_output, dim=1)

                test_loss += loss_fn(probs, y).item()

                correct += (probs.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    epochs = 5
    sample_num = 1
    global all_sampling_time
    global epoch_sampling_time
    all_sampling_time = None

    for t in range(epochs):
        epoch_sampling_time = None
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, sample_num)
        test(test_dataloader, model, loss_fn, sample_num)
        print(torch.mean(epoch_sampling_time))
        print(torch.std(epoch_sampling_time))
        print(torch.sum(epoch_sampling_time))
        print(epoch_sampling_time.size())
        if all_sampling_time == None:
            all_sampling_time = epoch_sampling_time
        else:
            all_sampling_time = torch.cat((all_sampling_time, epoch_sampling_time))
        sample_num *= 10
    print("Done!")
    print(torch.mean(all_sampling_time))
    print(torch.std(all_sampling_time))
    print(torch.sum(all_sampling_time))


if __name__ == "__main__":
    mnist_mcd()
