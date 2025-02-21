# Sample parallel implementation of a training loop with the entso_e dataset and a fully connected model architecture.
import torch
from torch import Tensor, nn
import torch_bayesian.vi as vi
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from torch.utils.data import DataLoader
from typing import Callable
import torch.distributed as dist
import torch.nn.functional as F
import os
import torch.multiprocessing as mp
from torchvision.transforms import ToTensor
from torchvision import datasets
sampling_state = None
train_loss_list = []
test_loss_list = []


class NeuralNetwork(vi.VIModule):
    def __init__(self, in_channel, hidden1, hidden2, output_length,
                 variational_distribution=MeanFieldNormalVarDist()) -> None:
        super().__init__()
        self.conv_stack = vi.VISequential(
            vi.VIConv2d(in_channels=in_channel, out_channels=hidden1, kernel_size=3, stride=1, padding=1,
                        variational_distribution=variational_distribution),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            vi.VIConv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=3, stride=1, padding=1,
                        variational_distribution=variational_distribution),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            vi.VILinear(hidden2*7*7, output_length, variational_distribution=variational_distribution),
        )

    def forward(self, x_: Tensor) -> Tensor:
        logits = self.conv_stack(x_)
        return logits

def setup(rank, world_size):
    """
        Initialize the distributed process group.
        """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("gloo", rank=rank, world_size=world_size) # Change to "nccl" for GPUs

def cleanup():
    """
    Clean up the process group.
    """
    dist.destroy_process_group()


def train(
        dataloader: DataLoader,
        model: vi.VIModule,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        sample_num,
        train_loss_list,
        rank,
        world_size,
        device,
):
    # Communication variables
    global sampling_state  # Randomness switch
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        # Switch to process specific randomness
        regular_state = torch.get_rng_state()
        if sampling_state == None:
            torch.manual_seed(rank)
        else:
            torch.set_rng_state(sampling_state)

        # Get predictions
        pred = model(x, samples=sample_num)

        # Switch to general randomness
        sampling_state = torch.get_rng_state()
        torch.set_rng_state(regular_state)

        mean_model_output = pred.mean(dim=0)
        probs = F.softmax(mean_model_output, dim=1)
        loss = loss_fn(probs, y)
        # Backpropagation
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                grad_global = param.grad.data
                dist.all_reduce(grad_global, op=dist.ReduceOp.SUM)

                # Average the gradients
                grad_global /= world_size

                # Copy the averaged gradients back to the parameter
                param.grad.data = torch.tensor(grad_global, dtype=param.grad.data.dtype)

        optimizer.step()
        optimizer.zero_grad()

    train_loss_list.append(loss.item())
    return model

def test(dataloader: DataLoader,
         model: vi.VIModule,
         loss_fn: Callable,
         sample_num,
         test_loss_list,
         rank,
         world_size,
         device
         ):

    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    correct = 0.0
    global sampling_state  # Randomness switch

    with torch.no_grad():
        for x, y in dataloader:

            x, y = x.to(device), y.to(device)

            regular_state = torch.get_rng_state()
            if sampling_state == None:
                torch.manual_seed(rank)
            else:
                torch.set_rng_state(sampling_state)

            samples = model(x, samples=sample_num)


            sampling_state = torch.get_rng_state()
            torch.set_rng_state(regular_state)


            samples_global = samples
            dist.all_reduce(samples_global, op=dist.ReduceOp.SUM)

            if rank == 0:
                samples_global /= world_size
                mean_model_output = torch.tensor(samples_global, dtype=samples.dtype).mean(dim=0)
                samples = F.softmax(mean_model_output, dim=1)
                correct += (samples.argmax(1) == y).type(torch.float).sum().item()
                test_loss += loss_fn(samples, y).item()

    if rank == 0:
        test_loss /= num_batches
        correct /= len(dataloader.dataset)
        test_loss_list.append(test_loss)

        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    return


def distributed(rank, world_size, parameters):
    setup(rank, world_size)
    (train_dataloader, test_dataloader, model, loss_fn, optimizer, sample_num, train_loss_list, test_loss_list,
     random_seed, epochs, device) = parameters
    # Do stuff here
    torch.manual_seed(random_seed)
    for t in range(epochs):
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        model = train(train_dataloader, model, loss_fn, optimizer, sample_num, train_loss_list, rank, world_size,device)
        test(test_dataloader, model, loss_fn, sample_num, test_loss_list,rank, world_size, device)

    cleanup()

if __name__ == "__main__":
    # Hyper-parameters
    world_size = 4  # Set the number of processes
    in_channel = 1
    output_length = 10
    hidden1 = 8
    hidden2 = 16
    batch_size = 64
    epochs = 10
    random_seed = 42
    all_sample_num = 64
    lr = 1e-3
    #mp.set_start_method("fork", force=True)
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = NeuralNetwork(in_channel, hidden1, hidden2, output_length,
                          variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)

    print(f"Using {device} device")
    print(model)
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    sample_num = int(all_sample_num / world_size)
    parameters = (train_dataloader, test_dataloader, model, loss_fn, optimizer, sample_num, train_loss_list, test_loss_list, random_seed, epochs, device)


    mp.spawn(distributed, args=(world_size,parameters), nprocs=world_size, join=True)

