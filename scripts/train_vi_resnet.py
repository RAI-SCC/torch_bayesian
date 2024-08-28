import os

import torch
import torch.distributed as dist
from DDP_VIAlexNet import (
    compute_accuracy_ddp,
    get_dataloaders_cifar10_ddp,
    train_model_ddp,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from vi_resnet import VIResNet20
from VIAlexNet import get_transforms_cifar10

from vi.priors import BasicQuietPrior
from vi.variational_distributions import MeanFieldNormalVarDist

# TRAINING
# Save to a separate Python module file `utils_train.py` to import the functions from
# into your main script and run the training as a batch job later on.


def main_ddp() -> None:
    """Distributed data-parallel training of ResNet20 on the CIFAR-10 dataset."""
    ## world_size = int(os.getenv("...")  # Get overall number of processes from SLURM environment variable.
    # world_size = int(os.environ["WORLD_SIZE"])
    ## rank = int(os.getenv("...")  # Get individual process ID from SLURM environment variable.
    # rank = int(os.environ["RANK"])
    # print(f"Rank, world size, device count: {rank}, {world_size}, {torch.cuda.device_count()}")

    rank = int(os.getenv("SLURM_PROCID") or "")  # Get individual process ID.
    world_size = int(
        os.getenv("SLURM_NTASKS") or ""
    )  # Get overall number of processes.
    slurm_localid = int(os.getenv("SLURM_LOCALID") or "")

    # Initialize GPUs and dataloaders
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(slurm_localid)

    # Initialize DistributedDataParallel.
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method="env://"
    )

    if dist.is_initialized():
        print(
            f"Rank {rank}/{world_size}: Process group initialized with torch rank {torch.distributed.get_rank()} and torch world size {torch.distributed.get_world_size()}."
        )

    dist.new_group(list(range(world_size)))

    if rank == 0:
        ## Check if distributed package available.
        print("Backend: ", dist.get_backend())
        ## Check if NCCL backend available.

    # On each host with N GPUs, spawn up N processes, while ensuring that
    # each process individually works on a single GPU from 0 to N-1.

    batch_size = 32  # Set batch size.
    num_epochs = 100  # Set number of epochs to be trained.
    data_root = "data/cifar10"  # Path to data dir
    learning_rate = 1e-3
    # momentum = 0.65
    betas = (0.65, 0.999)
    weight_decay = 0.0
    start_std = 0.001
    # prior_std = 5 * start_std
    prior_std_ratio = 0.4
    prior_mean_std = 10 * start_std / prior_std_ratio
    # Get transforms for data preprocessing to make smaller CIFAR-10 images work with AlexNet using helper function from task 1.
    train_transforms, test_transforms = get_transforms_cifar10()

    # Get distributed dataloaders for training and validation data on all ranks.
    train_loader, valid_loader = get_dataloaders_cifar10_ddp(
        batch_size=batch_size,
        data_root=data_root,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
    )

    # Create AlexNet model with 10 classes for CIFAR-10 and move it to GPU.
    vardist = MeanFieldNormalVarDist(initial_std=start_std)
    # prior = MeanFieldNormalPrior(std=prior_std)
    prior = BasicQuietPrior(std_ratio=prior_std_ratio, mean_std=prior_mean_std)
    model = VIResNet20(num_classes=10, variational_distribution=vardist, prior=prior)
    model.to(device)
    model.return_log_prob()
    # Wrap model with DDP.
    ddp_model = nn.parallel.DistributedDataParallel(
        model, device_ids=[slurm_localid], output_device=slurm_localid
    )

    # Set up stochastic gradient descent optimizer from torch.optim package.
    # Use parameters of DDP model here!
    # optimizer = torch.optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(
        ddp_model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay
    )

    # Train DDP model.
    train_model_ddp(
        model=ddp_model,
        num_epochs=num_epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        num_samples=5,
    )

    # Test final model on root.
    if dist.get_rank() == 0:
        test_dataset = CIFAR10(
            root=data_root,
            train=False,
            transform=test_transforms,
        )  # Get dataset for test data.
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )  # Get dataloader for test data.
        # Compute accuracy on test data.
        test_acc = compute_accuracy_ddp(ddp_model, test_loader, samples=5)
        ## Print test accuracy.
        print(f"Test Accuracy: {test_acc:.2f}")

    ## Destroy process group.
    dist.destroy_process_group()


if __name__ == "__main__":
    main_ddp()
