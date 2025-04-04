from typing import Callable
import numpy as np
import torch
import statistics
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from properscoring import crps_quadrature, crps_gaussian
from scipy.stats import ecdf
from statsmodels.distributions.empirical_distribution import ECDF
import os
import vi
from vi import VIModule
from vi.priors import UniformPrior, MeanFieldNormalPrior, BasicQuietPrior
from vi.variational_distributions import MeanFieldNormalVarDist
from vi.predictive_distributions import StudentTPredictiveDistribution, MixtureGaussianEqualStdPredictiveDistribution, MeanFieldNormalPredictiveDistribution, StudentTwithDOFPredictiveDistribution, SkewNormalPredictiveDistribution, MixtureGaussianPredictiveDistribution
from data_generator import CustomDataset, data_generator
from statistics import mean
import matplotlib.pyplot as plt
import time

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bnn_with_normal() -> None:
    """Reimplement the pytorch quickstart tutorial with BNNs."""
    # Download training data from open datasets.
    x_true, y_true, error_dist = data_generator(x_lims=[-5, 5], dist="gamma", size=100000)

    dataset = CustomDataset(x_true, y_true)
    x_train, y_train = dataset.__getitem__(30000, data_interval=[-4.5, 4.5])
    train_data = TensorDataset(x_train, y_train)

    x_val, y_val = dataset.__getitem__(100, data_interval=[-4.5, 4.5])
    val_data = TensorDataset(x_val, y_val)

    x_test, y_test = dataset.__getitem__(1000, data_interval=[-5, 5])
    test_data = TensorDataset(x_test, y_test)

    batch_size = 16

    # Create data loaders.
    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

    for x, y in test_dataloader:
        x.unsqueeze_(-1)
        y.unsqueeze_(-1)
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
                vi.VILinear(1, 100, prior=prior,
                            prior_initialization=prior_initialization, rescale_prior=rescale_prior,
                            variational_distribution=variational_distribution),
                #nn.ReLU(),
                #vi.VILinear(32, 32, prior=prior,
                #            prior_initialization=prior_initialization, rescale_prior=rescale_prior,
                #            variational_distribution=variational_distribution),
                nn.ReLU(),
                vi.VILinear(100, 1, prior=prior,
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

    predictive_distribution = SkewNormalPredictiveDistribution()
    loss_fn = vi.KullbackLeiblerLoss(
        predictive_distribution, dataset_size=len(train_data)#, heat=0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
            x.unsqueeze_(-1)
            y.unsqueeze_(-1)
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
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x.unsqueeze_(-1)
            y.unsqueeze_(-1)
            x, y = x.to(device), y.to(device)
            # Compute prediction error
            pred = model(x, samples=10)
            loss = loss_fn(pred, y)
            val_loss_hist.append(loss.item())

        return val_loss_hist

    def test(dataloader: DataLoader, model: VIModule, loss_fn: Callable) -> None:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        loss_hist = []
        crpss = []

        test_loss, mse = 0.0, 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x.unsqueeze_(-1)
                y.unsqueeze_(-1)
                x, y = x.to(device), y.to(device)
                pred = model(x, samples=100)
                test_loss += loss_fn(pred, y).item()
                loss_hist.append(test_loss)
                #print(loss_fn(pred, y).item())
                mse += torch.sum((pred[0].mean(dim=0) - y)**2)/dataloader.batch_size

                for i in range(batch_size):
                    #print(y[i].shape, pred[0][:,i,:].flatten().shape)
                    #print(y[i])
                    #print(pred[0][:,i,:])

                    crps_value = crps_quadrature(y[i], ECDF(pred[0][:,i,:].flatten()), tol=1e1)
                    #crps_value = crps_gaussian(y[i], pred[0][:, i, :].mean(dim=0),  pred[0][:, i, :].std(dim=0))
                    crpss.append(crps_value.item())
                    #print(crps_value.item())

        test_loss /= num_batches
        mse /= size
        crps = mean(crpss)

        print(
            f"Test Error: \n MSE: {mse:>8f}, Avg loss: {test_loss:>8f} , CRPS: {crps:>8f} \n"
        )
        return test_loss

    start = time.time()
    epochs = 50
    loss_hist = []
    val_loss = []
    count = 0
    min_test_loss = 1000.0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        #train(train_dataloader, model, loss_fn, optimizer)
        #loss_hist.extend(test(test_dataloader, model, loss_fn))
        loss_hist.extend(train(train_dataloader, model, loss_fn, optimizer))
        val_loss = validate(val_dataloader, model, loss_fn)
        test_loss = test(test_dataloader, model, loss_fn)


        if test_loss < min_test_loss-1:
            min_test_loss = test_loss
            count = 0
        else:
            count += 1
        if count >= 5:
            break



    #plt.plot(np.log(np.exp(loss_hist)+1))
    plt.plot(loss_hist)
    plt.yscale('log')
    plt.title("Loss")
    plt.show()
    y_pred = model(x_test.unsqueeze_(-1), samples=100)
    x_test = torch.squeeze(x_test)
    x_test_sorted, indices = torch.sort(x_test)

    x_test_sorted = x_test_sorted.detach().numpy()
    y_pred_lower = np.squeeze(np.squeeze(y_pred[0].quantile(0.025, dim=0))[indices].detach().numpy())

    #y_pred_lower = np.squeeze(y_pred_lower.detach().numpy())
    y_pred_upper = np.squeeze(np.squeeze(y_pred[0].quantile(0.975, dim=0))[indices].detach().numpy())
    plt.hist(y_pred[0][:,0,0].detach().numpy())
    plt.show()
    plt.scatter(x_train.detach().numpy(), y_train.detach().numpy(), s=3, color='#1f77b4', alpha=0.1)

    plt.plot(x_test_sorted, y_pred_lower, c='#bcbd22')
    plt.plot(x_test_sorted, y_pred_upper, c='#bcbd22')
    plt.fill_between(x_test_sorted, y_pred_lower, y_pred_upper, alpha=0.4, color='#bcbd22')

    plt.scatter(x_test.detach().numpy(), y_pred[0].mean(dim=0).detach().numpy(), s=5, c='#d62728')
    plt.scatter([], [], color='#1f77b4', s=20, label="training data")
    plt.plot([],[], label="prediction", color='#d62728')
    plt.plot([], [], label="confidence interval", color='#bcbd22')

    plt.legend(prop={'size': 18})
    plt.show()
    end = time.time()
    print("Done! time consumed:", end - start)
    return x_train, y_train, x_test, y_pred


if __name__ == "__main__":
    # Set seeds for reproducibility
    set_all_seeds(1)
    x_train, y_train, x_test, y_pred = bnn_with_normal()
    #torch.save(x_train, "x_train.pt")
    #torch.save(y_train, "y_train.pt")
    #torch.save(x_test, "x_test.pt")
    #torch.save(y_pred, "y_pred.pt")
