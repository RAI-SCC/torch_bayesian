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
from mpi4py import MPI
import numpy as np
sampling_state = None
train_loss_list = []
test_loss_list = []
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F


# Define model
class NeuralNetwork(vi.VIModule):
    def __init__(self, input_length, hidden1, hidden2, output_length, variational_distribution=MeanFieldNormalVarDist()) -> None:
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
def check_variable_consistency(value, want_difference):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Gather all values at root process
    all_values = comm.gather(value, root=0)
    # Check consistency on the root process
    if rank == 0:
        all_values_tensor = torch.stack(all_values, dim=0)
        if want_difference:
            if (all_values_tensor[0]==all_values_tensor).all():
                raise ValueError(f"All samples are the same: {all_values}")
        else:
            if not (all_values_tensor[0]==all_values_tensor).all():
                raise ValueError(f"Inconsistent values across processes: {all_values}")

def test_same_data_load(x, y):
    # Check whether same data is loaded in all processes
    check_variable_consistency(x, False)
    check_variable_consistency(y, False)
    return

def test_different_samples(samples):
    # Check if different results are sampled in different processes
    check_variable_consistency(samples, True)
    return

def test_same_current_model(model):
    # Check if the current model is the same in all processes
    for name, param in model.named_parameters():
        check_variable_consistency(param, False)
    return


def train(
    dataloader: DataLoader,
    model: VIModule,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    sample_num,
    enable_tests,
    train_loss_list,
    device,
    isClassification
):
    # Communication variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    global sampling_state # Randomness switch
    model.train()

    for batch, (x, y) in enumerate(dataloader):

        if enable_tests:
            test_same_data_load(x, y)

        x, y = x.to(device), y.to(device)

        #Switch to process specific randomness
        regular_state = torch.get_rng_state()
        if sampling_state == None:
            torch.manual_seed(rank)
        else:
            torch.set_rng_state(sampling_state)

        if enable_tests:
            test_same_current_model(model)

        # Get predictions
        pred = model(x, samples = sample_num)

        if enable_tests:
            test_different_samples(pred)

        #Switch to general randomness
        sampling_state = torch.get_rng_state()
        torch.set_rng_state(regular_state)
        if isClassification:
            mean_model_output = pred.mean(dim=0)
            probs = F.softmax(mean_model_output, dim=1)
            loss = loss_fn(probs, y)
        else:
            loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                # Convert gradients to a NumPy array for MPI communication
                grad_numpy = param.grad.data.numpy()

                # Create an array to hold the averaged gradients
                grad_global = np.zeros_like(grad_numpy)

                # Perform all_reduce to sum gradients across all processes
                comm.Allreduce(grad_numpy, grad_global, op=MPI.SUM)

                # Average the gradients
                grad_global /= world_size

                # Copy the averaged gradients back to the parameter
                param.grad.data = torch.tensor(grad_global, dtype=param.grad.data.dtype)

        optimizer.step()
        optimizer.zero_grad()

    train_loss_list.append(loss.item())
    return model, train_loss_list

def test(dataloader: DataLoader,
    model: VIModule,
    loss_fn: Callable,
    sample_num,
    enable_tests,
    test_loss_list,
    device,
    isClassification
):
    # Communication variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    correct = 0.0
    global sampling_state # Randomness switch

    with torch.no_grad():
        for x, y in dataloader:

            if enable_tests:
                test_same_data_load(x, y)

            x, y = x.to(device), y.to(device)

            regular_state = torch.get_rng_state()
            if sampling_state == None:
                torch.manual_seed(rank)
            else:
                torch.set_rng_state(sampling_state)

            if enable_tests:
                test_same_current_model(model)

            samples = model(x, samples = sample_num)

            if enable_tests:
                test_different_samples(samples)

            sampling_state = torch.get_rng_state()
            torch.set_rng_state(regular_state)

            samples_numpy = samples.numpy()

            # Create an array to hold the averaged loss values
            samples_global = np.zeros_like(samples_numpy)
            comm.Reduce(samples_numpy, samples_global, op=MPI.SUM, root=0)
            if rank == 0:
                samples_global /= world_size

                # Copy the averaged loss values
                if isClassification:
                    mean_model_output = torch.tensor(samples_global, dtype=samples.dtype).mean(dim=0)
                    samples = F.softmax(mean_model_output, dim=1)
                    correct += (samples.argmax(1) == y).type(torch.float).sum().item()
                else:
                    samples = torch.tensor(samples_global, dtype=samples.dtype)

                test_loss += loss_fn(samples, y).item()
    if rank == 0:
        test_loss /= num_batches
        correct /= len(dataloader.dataset)
        test_loss_list.append(test_loss)
        if isClassification:
            print(
                f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
            )
        else:
            print(
                f"Test Error: Avg loss: {test_loss:>8f} \n"
            )
    return test_loss_list

def random_plot(dataloader: DataLoader, model: VIModule, sample_num, enable_tests, device) -> None:
    global sampling_state
    # Communication variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    num_batches = len(dataloader)
    random_batch = int(torch.randint(low=0, high=num_batches - 1, size=(1,)))
    model.eval()
    with torch.no_grad():
        n = 0
        for x, y in dataloader:
            if n < random_batch:
                n += 1
            else:
                break
        if enable_tests:
            test_same_data_load(x, y)
        x, y = x.to(device), y.to(device)
        regular_state = torch.get_rng_state()
        if sampling_state == None:
            torch.manual_seed(rank)
        else:
            torch.set_rng_state(sampling_state)
        if enable_tests:
            test_same_current_model(model)
        samples = model(x, samples=sample_num)
        if enable_tests:
            test_different_samples(samples)
        sampling_state = torch.get_rng_state()
        torch.set_rng_state(regular_state)

        samples_sq = torch.mean(samples,0)
        samples_numpy = samples_sq.numpy()

        # Create an array to hold the averaged gradients
        samples_global = np.zeros_like(samples_numpy)
        comm.Allreduce(samples_numpy, samples_global, op=MPI.SUM)

        mean_samples = samples_global/world_size
        squared_diff = (samples_numpy-mean_samples) ** 2
        squared_diff_global = np.zeros_like(squared_diff)
        comm.Allreduce(squared_diff, squared_diff_global, op=MPI.SUM)
        variance = squared_diff_global / world_size
        std_samples = np.sqrt(variance)

        num_samples = y.shape[0]
        random_sample = int(torch.randint(low=0, high=num_samples - 1, size=(1,)))
        mean = mean_samples[random_sample]
        std = std_samples[random_sample]
        real = y[random_sample]
        in_mod = x[random_sample]
        input_length = in_mod.shape[0]
        output_length = real.shape[0]

    if rank == 0:
        plt.clf()
        plt.plot(in_mod.numpy(), color="blue", label="inputs")
        # plot outputs and ground truth behind input sequence
        plt.plot(
            range(input_length, input_length + output_length),
            mean,
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
            (torch.add(torch.from_numpy(mean), torch.from_numpy(std), alpha=1)).numpy(),
            (torch.add(torch.from_numpy(mean), torch.from_numpy(std), alpha=-1)).numpy(),
            color="orange",
            alpha=0.1,
        )
        file_name = 'random_sample.png'
        plt.savefig(file_name)

def parallel_mcd(input_length, hidden1, hidden2, output_length, batch_size, epochs, all_sample_num) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    enable_tests = True

    global sampling_state
    global train_loss_list
    global test_loss_list
    train_loss_list = []
    test_loss_list = []
    sampling_state = None

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

    model = NeuralNetwork(input_length, hidden1, hidden2, output_length,variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)
    if rank==0:
        print(f"Using {device} device")
        print(model)
    loss_fn = vi.MeanSquaredErrorLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    sample_num = int(all_sample_num / comm.size)
    for t in range(epochs):
        if rank==0:
            print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, sample_num, enable_tests, train_loss_list, device, isClassification=False)
        test(test_dataloader, model, loss_fn, sample_num, enable_tests, test_loss_list, device, isClassification=False)

    random_plot(test_dataloader, model, sample_num, enable_tests, device)

    if rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "log_std" in name:
                    # print(name)
                    base = name.removesuffix('log_std')
                    weight_layer_name = base + "mean"
                    weight_param = dict(model.named_parameters())[weight_layer_name]
                    sigma_weight_plot(weight_param, param, base)
        plt.clf()
        plt.plot(train_loss_list,
                 color="orange",
                 label="train loss", )
        plt.plot(test_loss_list,
                 color="blue",
                 label="test loss", )
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel("MSE Loss", fontsize=18)
        plt.legend()
        plt.savefig("loss_curve.png")

        print("Done!")
def mnist_parallel_mcd(hidden1=512, hidden2=256, batch_size=256, epochs=5, all_sample_num=50) -> None:
    # Download training data from open datasets.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    enable_tests = True
    global sampling_state
    global train_loss_list
    global test_loss_list
    train_loss_list = []
    test_loss_list = []
    sampling_state = None

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

    #batch_size = 256
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = NeuralNetwork(28*28, hidden1, hidden2, 10, variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)
    if rank==0:
        print(model)

    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)
    sample_num = int(all_sample_num / comm.size)
    for t in range(epochs):
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, sample_num, enable_tests, train_loss_list, device,
              isClassification=True)
        test(test_dataloader, model, loss_fn, sample_num, enable_tests, test_loss_list, device, isClassification=True)


    if rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "log_std" in name:
                    # print(name)
                    base = name.removesuffix('log_std')
                    weight_layer_name = base + "mean"
                    weight_param = dict(model.named_parameters())[weight_layer_name]
                    sigma_weight_plot(weight_param, param, base)
        plt.clf()
        plt.plot(train_loss_list,
                 color="orange",
                 label="train loss", )
        plt.plot(test_loss_list,
                 color="blue",
                 label="test loss", )
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel("MSE Loss", fontsize=18)
        plt.legend()
        plt.savefig("loss_curve.png")

        print("Done!")
