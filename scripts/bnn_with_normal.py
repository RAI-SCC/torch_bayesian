from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from properscoring import crps_quadrature, crps_gaussian
from scipy.stats import ecdf
from statsmodels.distributions.empirical_distribution import ECDF
import os
import vi
from vi import VIModule
from vi.priors import LaplacePrior, UniformPrior, MeanFieldNormalPrior, BasicQuietPrior, MixtureGaussianPrior
from vi.variational_distributions import MeanFieldNormalVarDist, MixtureGaussianVarDist
from vi.predictive_distributions import MeanFieldNormalPredictiveDistribution, StudentTwithDOFPredictiveDistribution, SkewNormalPredictiveDistribution
from data_generator import CustomDataset, data_generator
from statistics import mean


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bnn_with_normal() -> None:
    """Reimplement the pytorch quickstart tutorial with BNNs."""
    # Download training data from open datasets.
    x_true, y_true, error_dist = data_generator(x_lims=[-20, 20], dist="student_t",  width=1.5, size=200000)

    dataset = CustomDataset(x_true, y_true)
    x_train, y_train = dataset.__getitem__(10000, data_interval=[-18, 18])
    train_data = TensorDataset(x_train, y_train)

    x_test, y_test = dataset.__getitem__(1000, data_interval=[-20, 20])
    test_data = TensorDataset(x_test, y_test)

    batch_size = 8

    # Create data loaders.
    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
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
                vi.VILinear(1, 2, prior=prior,
                            prior_initialization=prior_initialization, rescale_prior=rescale_prior,
                            variational_distribution=variational_distribution),
                nn.ReLU(),
                vi.VILinear(2, 2, prior=prior,
                            prior_initialization=prior_initialization, rescale_prior=rescale_prior,
                            variational_distribution=variational_distribution),
                nn.ReLU(),
                vi.VILinear(2, 1, prior=prior,
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

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return loss_hist

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
            f"Test Error: \n MSE: {mse:>8f}, CRPS: {crps:>8f}, Avg loss: {test_loss:>8f} \n"
        )
        #return loss_hist,

    epochs = 40
    loss_hist = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        #train(train_dataloader, model, loss_fn, optimizer)
        #loss_hist.extend(test(test_dataloader, model, loss_fn))
        loss_hist.extend(train(train_dataloader, model, loss_fn, optimizer))
        test(test_dataloader, model, loss_fn)

    #plt.plot(np.log(np.exp(loss_hist)+1))
    #plt.plot(loss_hist)
    #plt.yscale('log')
    #plt.title("Loss")
    #plt.show()
    #plt.scatter(x_train.detach().numpy(), y_train.detach().numpy(), s=7)
    y_pred = model(x_test.unsqueeze_(-1))

    #plt.scatter(x_test.detach().numpy(), y_pred[0].quantile(0.025, dim=0).detach().numpy(), alpha=0.5, s=3, c="y")
    #plt.scatter(x_test.detach().numpy(), y_pred[0].quantile(0.975, dim=0).detach().numpy(), alpha=0.5, s=3, c="y")
    #plt.scatter(x_test.detach().numpy(), y_pred[0].mean(dim=0).detach().numpy(), s=5, c="r")
    #plt.plot([],[], label="prediction", color="r")
    #plt.plot([], [], label="confidence interval", color="y")
    #plt.legend(prop={'size': 15})
    #plt.show()
    print("Done!")


if __name__ == "__main__":
    # Set seeds for reproducibility
    set_all_seeds(1)
    bnn_with_normal()
