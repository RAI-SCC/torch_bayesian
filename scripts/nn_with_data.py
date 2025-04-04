from typing import Callable
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
import os
from vi import VIModule
from vi.priors import MeanFieldNormalPrior
from vi.variational_distributions import MeanFieldNormalVarDist
import matplotlib.pyplot as plt
import time
from entsoe_data_generator import create_mlp_dataset, TimeSeriesDataset

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bnn_with_normal() -> None:
    """Reimplement the pytorch quickstart tutorial with BNNs."""
    # Download training data from open datasets.
    df = pd.read_csv('/home/peihsuan/Desktop/vi/vi/scripts/hourly_average_load.csv', parse_dates=['timestamp'])
    df = df[df['timestamp'].dt.year < 2020]
    series = df['avg_load'].valuesseries = df['avg_load'].values

    # Normalize the data (recommended for neural networks)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # Create input-output pairs (24 hr input, 6 hr output)
    X, y = create_mlp_dataset(series_scaled, input_steps=24, output_steps=6)

    # Split dataset: use last 7 days as test/validation (adjust as needed)
    test_size = 24 * 7  # 7 days, note: number of samples in dataset not hours
    x_train, x_val, x_test = X[:-2*test_size], X[-2*test_size:-test_size], X[-test_size:]
    y_train, y_val, y_test = y[:-2*test_size], y[-2*test_size:-test_size], y[-test_size:]

    # Create PyTorch Datasets and DataLoaders
    batch_size = 32
    #train_dataset = TimeSeriesDataset(X_train, y_train)
    #val_dataset = TimeSeriesDataset(X_val, y_val)

    train_data = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float())
    val_data = TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).float())
    test_data = TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).float())
    batch_size = 16
    # Create data loaders.
    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

    for x, y in test_dataloader:
        #x.unsqueeze_(-1)
        #y.unsqueeze_(-1)
        print(f"Shape of X: {x.shape}")
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
    class NeuralNetwork(nn.Module):
        def __init__(self, prior=MeanFieldNormalPrior(), prior_initialization=True, rescale_prior=True,
                     variational_distribution=MeanFieldNormalVarDist()) -> None:
            super().__init__()

            self.linear_relu_stack = nn.Sequential(
                nn.Linear(24, 10),
                #nn.ReLU(),
                #vi.VILinear(32, 32, prior=prior,
                #            prior_initialization=prior_initialization, rescale_prior=rescale_prior,
                #            variational_distribution=variational_distribution),
                nn.ReLU(),
                nn.Linear(10, 6),
            )

        def forward(self, x_: Tensor) -> Tensor:
            return self.linear_relu_stack(x_)

    model = NeuralNetwork()
    print(model)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train(
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        size = len(dataloader.dataset)
        model.train()
        loss_hist = []
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss_hist.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return loss_hist

    def validate(
        dataloader: DataLoader,
        model: VIModule,
        loss_fn: Callable,
    ) -> None:
        val_loss_hist = []
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)
            val_loss_hist.append(loss.item())
            val_loss = sum(val_loss_hist)/len(val_loss_hist)

        return val_loss

    def test(dataloader: DataLoader, model: VIModule, loss_fn: Callable) -> None:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        loss_hist = []
        #crpss = []

        test_loss, mse = 0.0, 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                loss_hist.append(test_loss)

        test_loss /= num_batches

        print(
            f"Test Error: Avg loss: {test_loss:>8f} "#, CRPS: {crps:>8f} \n"
        )
        return test_loss

    start = time.time()
    epochs = 50
    loss_hist = []
    val_loss = []
    count = 0
    min_val_loss = 1000.0
    best_state = model.state_dict()
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        #train(train_dataloader, model, loss_fn, optimizer)
        #loss_hist.extend(test(test_dataloader, model, loss_fn))
        loss_hist.extend(train(train_dataloader, model, loss_fn, optimizer))
        val_loss = validate(val_dataloader, model, loss_fn)


        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_state = model.state_dict()
            count = 0
        else:
            count += 1
        if count >= 3:
            break
    model.load_state_dict(best_state)
    test_loss = test(test_dataloader, model, loss_fn)
    print(f"test loss: {test_loss}")


    plt.plot(loss_hist)
    plt.yscale('log')
    plt.title("Loss")
    plt.show()
    y_pred = model(torch.tensor(x_test).float())
    plot_ind = 1
    y_pred = y_pred[plot_ind,:].unsqueeze_(0)
    y_test = torch.tensor(y_test[plot_ind,:]).unsqueeze_(0)
    x_test = torch.tensor(x_test[plot_ind,:]).unsqueeze_(0)
    y_pred_inv = torch.tensor(scaler.inverse_transform(y_pred.detach().numpy()))
    x_test_inv = torch.tensor(scaler.inverse_transform(x_test))
    y_test_inv = torch.tensor(scaler.inverse_transform(y_test.detach().numpy()))

    plt.plot(range(30),  np.append(x_test_inv,y_test_inv.squeeze()), marker='o', label='training')
    plt.plot(range(24,30),y_test_inv.squeeze(), marker='o', c="orange", label='prediction')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot an example: first sample from the validation set
    plt.figure(figsize=(8, 5))
    plt.plot(range(30),  np.append(x_test_inv,y_test_inv.squeeze()), marker='o', label='Actual')
    #plt.plot(range(6), y_test_inv.squeeze(), marker='o', label='Actual')
    plt.plot(range(24,30), y_pred_inv.squeeze(), marker='o', label='Predicted')
    plt.xlabel("Hour (future)")
    plt.ylabel("Load")
    plt.title("MLP Prediction vs Actual (First Validation Sample)")
    plt.legend()
    plt.grid(True)
    plt.legend(prop={'size': 18})
    plt.show()
    plt.plot(range(6),  y_test_inv.squeeze(), marker='o', label='Actual')
    #plt.plot(range(6), y_test_inv.squeeze(), marker='o', label='Actual')
    plt.plot(range(6), y_pred_inv.squeeze(), marker='o', label='Predicted')
    plt.ylim((33000, 51000))
    plt.xlabel("Hour (future)")
    plt.ylabel("Load")
    plt.title("MLP Prediction vs Actual (First Validation Sample)")
    plt.legend()
    plt.grid(True)
    plt.legend(prop={'size': 18})
    plt.show()
    end = time.time()
    print("Done! time consumed:", end - start)
    return x_train, y_train, x_test, y_pred


if __name__ == "__main__":
    # Set seeds for reproducibility
    set_all_seeds(297)
    x_train, y_train, x_test, y_pred = bnn_with_normal()
    #torch.save(x_train, "x_train.pt")
    #torch.save(y_train, "y_train.pt")
    #torch.save(x_test, "x_test.pt")
    #torch.save(y_pred, "y_pred.pt")
