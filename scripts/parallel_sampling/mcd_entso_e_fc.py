# Sample parallel implementation of a training loop with the entso_e dataset and a fully connected model architecture.
import torch
from torch import Tensor, nn
import torch_bayesian.vi as vi
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from torch.utils.data import DataLoader
from typing import Callable
import polars as pl
from entsoe_data_load import TimeseriesDataset
train_loss_list = []
test_loss_list = []


class NeuralNetwork(vi.VIModule):
    def __init__(self, input_length, hidden1, hidden2, output_length,
                 variational_distribution=MeanFieldNormalVarDist()) -> None:
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


def train(
        dataloader: DataLoader,
        model: vi.VIModule,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        sample_num,
        train_loss_list,
        device,
):
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Get predictions
        pred = model(x, samples=sample_num)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    train_loss_list.append(loss.item())
    return model


def test(dataloader: DataLoader,
         model: vi.VIModule,
         loss_fn: Callable,
         sample_num,
         test_loss_list,
         device
         ):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            samples = model(x, samples=sample_num)
            test_loss += loss_fn(samples, y).item()

    test_loss /= num_batches

    test_loss_list.append(test_loss)

    print(
        f"Test Error: Avg loss: {test_loss:>8f} \n"
    )

    return


if __name__ == "__main__":
    input_length = 50
    output_length = 10
    hidden1 = 40
    hidden2 = 20
    batch_size = 32
    epochs = 10
    random_seed = 42
    all_sample_num = 64
    print(all_sample_num)
    # mp.set_start_method("fork", force=True)
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

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.device(device)
    model = NeuralNetwork(input_length, hidden1, hidden2, output_length,
                          variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)

    print(f"Using {device} device")
    print(model)
    loss_fn = vi.MeanSquaredErrorLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    sample_num = all_sample_num

    torch.manual_seed(random_seed)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        model = train(train_dataloader, model, loss_fn, optimizer, sample_num, train_loss_list,
                      device)
        test(test_dataloader, model, loss_fn, sample_num, test_loss_list, device)

