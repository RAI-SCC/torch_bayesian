from typing import Callable
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from properscoring import crps_quadrature, crps_gaussian
from scipy.stats import ecdf
from statsmodels.distributions.empirical_distribution import ECDF
import os
import vi
from vi import VIModule
from vi.priors import UniformPrior, MeanFieldNormalPrior
from vi.variational_distributions import MeanFieldNormalVarDist
from vi.predictive_distributions import MixtureGaussianEqualStdPredictiveDistribution, MeanFieldNormalPredictiveDistribution, StudentTwithDOFPredictiveDistribution, SkewNormalPredictiveDistribution
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
    val_size = 24*7
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
    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

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
    class NeuralNetwork(vi.VIModule):
        def __init__(self, prior=MeanFieldNormalPrior(), prior_initialization=True, rescale_prior=True,
                     variational_distribution=MeanFieldNormalVarDist()) -> None:
            super().__init__()

            self.linear_relu_stack = vi.VISequential(
                vi.VILinear(24, 10, prior=prior,
                            prior_initialization=prior_initialization, rescale_prior=rescale_prior,
                            variational_distribution=variational_distribution),
                #nn.ReLU(),
                #vi.VILinear(10, 10, prior=prior,
                #            prior_initialization=prior_initialization, rescale_prior=rescale_prior,
                #            variational_distribution=variational_distribution),
                nn.ReLU(),
                vi.VILinear(10, 6, prior=prior,
                            prior_initialization=prior_initialization, rescale_prior=rescale_prior,
                            variational_distribution=variational_distribution),
            )

        def forward(self, x_: Tensor) -> Tensor:
            return self.linear_relu_stack(x_)

    model = NeuralNetwork(prior=MeanFieldNormalPrior(), variational_distribution=MeanFieldNormalVarDist()).to(device)
    model.return_log_probs()
    #plt.scatter(x_train.detach().numpy(),y_train.detach().numpy())
    #plt.title("data", fontsize = 20)
    #plt.xlabel("x", fontsize = 15)
    #plt.ylabel("y", fontsize = 15)
    #plt.show()
    #plt.hist(y_train.detach().numpy())
    #plt.title("distribution of y", fontsize = 30)
    #plt.show()
    print(model)

    predictive_distribution = MeanFieldNormalPredictiveDistribution()
    loss_fn = vi.KullbackLeiblerLoss(
        predictive_distribution, dataset_size=len(train_data)#, heat=0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train(
        dataloader: DataLoader,
        model: VIModule,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        size = len(dataloader.dataset)
        model.train()
        loss_hist = []
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            #x.unsqueeze_(-1)
            #y.unsqueeze_(-1)
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            #print(x)
            pred = model(x, samples=10)
            #print(pred)
            #if batch==10:
            #    plt.hist(pred[0][:,4,:].detach().numpy(), bins=20)
            #    plt.title("sampling of y", fontsize=30)
            #    plt.show()

            #print(pred)

            loss = loss_fn(pred, y)
            loss_hist.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            #for param in model.parameters():
            #    print(param.names)
            #    print(param)
                #break
            #exit()

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
        mse = 0.0
        size = len(dataloader.dataset)
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            #x.unsqueeze_(-1)
            #y.unsqueeze_(-1)
            x, y = x.to(device), y.to(device)
            # Compute prediction error
            pred = model(x, samples=10)
            loss = loss_fn(pred, y)
            val_loss_hist.append(loss.item())
            mse += torch.sum((pred[0].mean(dim=0) - y) ** 2) / dataloader.batch_size

        mse /= size
        val_loss = sum(val_loss_hist) / len(val_loss_hist)
        print(
            f"Validate Error: \n MSE: {mse:>8f}, Avg loss: {val_loss:>8f} \n"
        )

        return val_loss

    def test(dataloader: DataLoader, model: VIModule, loss_fn: Callable) -> None:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        loss_hist = []
        crpss = []
        mse_values = []

        test_loss, mse = 0.0, 0.0
        with torch.no_grad():
            for x, y in dataloader:
                #x.unsqueeze_(-1)
                #y.unsqueeze_(-1)
                x, y = x.to(device), y.to(device)
                pred = model(x, samples=100)
                test_loss += loss_fn(pred, y).item()
                loss_hist.append(test_loss)
                #print(loss_fn(pred, y).item())
                #y_inv = scaler.inverse_transform(y)
                #print(pred[0].shape)
                #y_pred_inv = scaler.inverse_transform(pred[0])
                #mse += torch.sum((y_pred_inv.mean(dim=0) - y_inv)**2)/dataloader.batch_size

                for i in range(batch_size):
                    #print(y[i].shape, pred[0][:,i,:].flatten().shape)
                    #print(y[i])
                    #print(pred[0][:,i,:])

                    y_inv = scaler.inverse_transform(y[i].reshape(1, -1))
                    y_pred_inv = scaler.inverse_transform(pred[0][:,i,:])
                    #crps_value = crps_quadrature(y_inv, ECDF(y_pred_inv), tol=1e1).item()
                    mse_value = np.mean((y_pred_inv.mean() - y_inv)**2)
                    crps_value = 0.0
                    #mse_value = 0.0
                    for j in range(y_inv.shape[1]):

                        crps_value += crps_gaussian(y_inv[:,j], mu=y_pred_inv[:,j].mean(), sig=y_pred_inv[:,j].std()).item()
                        #crps_quadrature(y_inv[:,j], ECDF(y_pred_inv[:,j]), tol=1e3).item()
                    #crps_value = crps_quadrature(y_inv, ECDF(y_pred_inv))
                    #mse_value += ((y_pred_inv.mean()-y_inv)**2).mean()
                    #crps_value = crps_gaussian(y[i], pred[0][:, i, :].mean(dim=0),  pred[0][:, i, :].std(dim=0))
                    #crps_value = crps_value/y_inv.shape[1]
                    crpss.append(crps_value)
                    mse_values.append(mse_value)
                    #print(crps_value.item())

        test_loss /= num_batches
        mse /= size
        crps = torch.tensor(sum(crpss) / size)
        mse_value = sum(mse_values) / size

        print(
            f"Test Error: \n MSE: {mse_value:>8f}, Avg loss: {test_loss:>8f} , CRPS: {crps:>8f} \n"
        )
        return test_loss

    start = time.time()
    epochs = 100
    loss_hist = []
    val_loss = []
    count = 0
    min_test_loss = 1000.0
    min_val_loss = 1000.0
    checkpoint = {
        'epoch': 0,
        'state_dict': model.state_dict(),
        # You can add more info, e.g., optimizer state, loss, etc.
    }
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        #train(train_dataloader, model, loss_fn, optimizer)
        #loss_hist.extend(test(test_dataloader, model, loss_fn))
        loss_hist.extend(train(train_dataloader, model, loss_fn, optimizer))
        val_loss = validate(val_dataloader, model, loss_fn)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            checkpoint = {
                'epoch': t,
                'state_dict': model.state_dict(),
                # You can add more info, e.g., optimizer state, loss, etc.
            }
            count = 0
        else:
            count += 1
        if count >= 3:
            #print(t)
            break
    best_state = checkpoint['state_dict']
    best_epoch = checkpoint['epoch']
    print(best_epoch)
    model.load_state_dict(best_state)
    test_loss = test(test_dataloader, model, loss_fn)
    print(f"test loss: {test_loss}")




    #plt.plot(np.log(np.exp(loss_hist)+1))
    plt.plot(loss_hist)
    #plt.yscale('log')
    plt.title("Loss")
    plt.show()
    y_pred = model(torch.tensor(x_test).float(), samples=100)[0]
    plot_ind = 1
    y_pred = y_pred[:,plot_ind,:]
    y_test = torch.tensor(y_test[plot_ind,:]).unsqueeze_(0)
    x_test = torch.tensor(x_test[plot_ind, :]).unsqueeze_(0)

    y_pred_inv = torch.tensor(scaler.inverse_transform(y_pred.detach().numpy()))
    x_test_inv = torch.tensor(scaler.inverse_transform(x_test.detach().numpy()))
    y_test_inv = torch.tensor(scaler.inverse_transform(y_test.detach().numpy()))

    # Plot an example: first sample from the validation set
    plt.plot(range(6), y_test_inv.squeeze(), marker='o', label='Actual')
    plt.plot(range(6), y_pred_inv.mean(dim=0), marker='o', label='Predicted')
    y_pred_lower = y_pred_inv.quantile(0.025, dim=0)
    y_pred_upper = y_pred_inv.quantile(0.975, dim=0)
    plt.plot(range(6), y_pred_lower, c='#bcbd22')
    plt.plot(range(6), y_pred_upper, c='#bcbd22')
    plt.ylim((33000,51000))
    plt.fill_between(range(6), y_pred_lower, y_pred_upper, alpha=0.4, color='#bcbd22')
    plt.xlabel("Hour (future)")
    plt.ylabel("Load")
    plt.title("MLP Prediction vs Actual (First Validation Sample)")
    plt.legend()
    plt.grid(True)
    plt.legend(prop={'size': 18})
    plt.show()
    # Plot an example: first sample from the validation set
    plt.plot(range(30), np.append(x_test_inv, y_test_inv.squeeze()), marker='o', label='training')
    #plt.plot(range(6), y_test_inv.squeeze(), marker='o', label='Actual')
    plt.plot(range(24, 30), y_pred_inv.mean(dim=0), marker='o', label='Predicted')
    y_pred_lower = y_pred_inv.quantile(0.025, dim=0)
    y_pred_upper = y_pred_inv.quantile(0.975, dim=0)
    plt.plot(range(24, 30), y_pred_lower, c='#bcbd22')
    plt.plot(range(24, 30), y_pred_upper, c='#bcbd22')
    plt.fill_between(range(24, 30), y_pred_lower, y_pred_upper, alpha=0.4, color='#bcbd22')
    plt.xlabel("Hour")
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
    set_all_seeds(608)
    x_train, y_train, x_test, y_pred = bnn_with_normal()
    #torch.save(x_train, "x_train.pt")
    #torch.save(y_train, "y_train.pt")
    #torch.save(x_test, "x_test.pt")
    #torch.save(y_pred, "y_pred.pt")


