import os
import time
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from VIAlexNet import (
    VIAlexNet,
    get_transforms_cifar10,
    make_train_validation_split,
)

import vi
from vi.predictive_distributions import CategoricalPredictiveDistribution
from vi.priors import BasicQuietPrior
from vi.variational_distributions import MeanFieldNormalVarDist

# DATA
# Save to a separate Python module file `utils_data.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.


def get_dataloaders_cifar10_ddp(
    batch_size: int,
    data_root: str = "data",
    validation_fraction: float = 0.1,
    train_transforms: Optional[Callable] = None,
    test_transforms: Optional[Callable] = None,
    seed: int = 123,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get distributed CIFAR-10 dataloaders for training and validation in a DDP setting.

    Parameters
    ----------
    batch_size : int
        The batch size.
    data_root : str
        The path to the data directory.
    validation_fraction : float
        The fraction of training samples used for validation.
    train_transforms : Callable
        The transform applied to the training data.
    test_transforms : Callable
        The transform applied to the testing data (inference).
    seed : int
        Seed for train-validation split.

    Returns
    -------
    DataLoader
        The training dataloader.
    DataLoader
        The validation dataloader.
    """
    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = None
    if (
        dist.get_rank() == 0
    ):  # Only root shall download dataset if data is not already there.
        train_dataset = CIFAR10(
            root=data_root, train=True, transform=train_transforms, download=True
        )

    dist.barrier()  # Barrier

    if (
        dist.get_rank() != 0
    ):  # Other ranks must not download dataset at the same time in parallel.
        train_dataset = CIFAR10(root=data_root, train=True, transform=train_transforms)

    valid_dataset = CIFAR10(root=data_root, train=True, transform=test_transforms)

    ## PERFORM INDEX-BASED TRAIN-VALIDATION SPLIT OF ORIGINAL TRAINING DATA.
    ## train_indices, valid_indices = ...  # Extract train and validation indices using helper function from task 1.
    train_indices, valid_indices = make_train_validation_split(
        train_dataset, seed, validation_fraction
    )

    # Split into training and validation dataset according to specified validation fraction.
    train_dataset = Subset(train_dataset, train_indices)
    valid_dataset = Subset(valid_dataset, valid_indices)

    # Sampler that restricts data loading to a subset of the dataset.
    # Especially useful in conjunction with DistributedDataParallel.
    # Each process can pass a DistributedSampler instance as a DataLoader sampler,
    # and load a subset of the original dataset that is exclusive to it.

    # Get samplers.
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True,
    )

    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True,
    )

    # Get dataloaders.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=train_sampler,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=valid_sampler,
    )

    return train_loader, valid_loader


# EVALUATION
# Save to a separate Python module file `utils_eval.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.


def get_right_ddp(
    model: nn.Module, data_loader: DataLoader, samples: int = 10
) -> Tuple[Tensor, Tensor]:
    """
    Compute the number of correctly predicted samples and the overall number of samples in a given dataset.

    This function is needed to compute the accuracy over multiple processors in a distributed data-parallel setting.

    Parameters
    ----------
    model : nn.Module
        The model.
    data_loader : DataLoader
        The dataloader.
    samples : int
        The number of samples to compute accuracy from.

    Returns
    -------
    Tensor
        The number of correctly predicted samples.
    Tensor
        The overall number of samples in the dataset.
    """
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = torch.tensor(features).cuda()
            targets = torch.tensor(targets).cuda()
            targets = targets.float().cuda()
            # CALCULATE PREDICTIONS OF CURRENT MODEL ON FEATURES OF INPUT DATA.
            logits = model(features, samples=samples)[0]
            probs = F.softmax(logits, dim=-1)
            outputs = probs.mean(dim=0)
            ## Determine class with highest score.
            prediction = outputs.argmax(dim=1)
            ## Compare predictions to actual labels to determine number of correctly predicted samples.
            correct_pred += prediction.eq(targets).sum().item()
            ## Determine overall number of samples.
            num_examples += targets.size(0)

            ## Determine class with highest score.
            # _, predicted_labels = torch.max(logits, 1)  # Get class with highest score.
            ## Update overall number of samples.
            ## Compare predictions to actual labels to determine number of correctly predicted samples.

    correct_pred = torch.tensor([correct_pred]).cuda()
    num_examples = torch.tensor([num_examples]).cuda()
    return correct_pred, num_examples


def compute_accuracy_ddp(
    model: nn.Module, data_loader: DataLoader, samples: int = 10
) -> float:
    """
    Compute the accuracy of the model's predictions on given labeled data.

    Parameters
    ----------
    model : nn.Module
        The model.
    data_loader : DataLoader
        The dataloader.
    samples : int
        The number of samples to compute accuracy from.

    Returns
    -------
    float
        The model's accuracy on the given dataset in percent.
    """
    correct_pred, num_examples = get_right_ddp(model, data_loader, samples=samples)
    return correct_pred.item() / num_examples.item() * 100


# TRAINING
# Save to a separate Python module file `utils_train.py` to import the functions from
# into your main script and run the training as a batch job later on.


def train_model_ddp(
    model: nn.Module,
    num_epochs: int,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_samples: int = 10,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train the model in distributed data-parallel fashion.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    num_epochs : int
        The number of epochs to train.
    train_loader : DataLoader
        The training dataloader.
    valid_loader : DataLoader
        The validation dataloader.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    num_samples : int
        Number of samples per forward pass. Default: 10

    Returns
    -------
    List[float]
        The epoch-wise loss history.
    List[float]
        The epoch-wise training accuracy history.
    List[float]
        The epoch-wise validation accuracy history.
    """
    start = time.perf_counter()
    rank = dist.get_rank()  # Get local process ID (= rank).
    world_size = dist.get_world_size()  # Get overall number of processes.

    loss_history: List[float] = []
    train_acc_history: List[float] = []
    valid_acc_history: List[float] = []  # Initialize history lists.

    predictive_distribution = CategoricalPredictiveDistribution()
    loss_fn = vi.KullbackLeiblerLoss(
        predictive_distribution, dataset_size=len(train_loader.dataset)
    )

    loss = torch.empty(1)
    epoch_average_loss = 0.0

    # Actual training starts here.
    for epoch in range(num_epochs):  # Loop over epochs.
        train_loader.sampler.set_epoch(
            epoch
        )  # Set current epoch for distributed dataloader.
        ## Set model to training mode.
        model.train()

        for batch_idx, (features, targets) in enumerate(
            train_loader
        ):  # Loop over mini batches.
            # Convert dataset to GPU device.
            features = features.cuda()
            targets = targets.cuda()

            with torch.autocast(device_type="cuda"):
                ## Zero out gradients.
                optimizer.zero_grad()
                # FORWARD & BACKWARD PASS
                logits = model(features, samples=num_samples)
                loss = loss_fn(*logits, targets)
                try:
                    assert not torch.isinf(loss).item()
                    assert not torch.isnan(loss).item()
                except AssertionError as err:
                    print("Logit NaNs: ", torch.any(torch.isnan(logits[0])))
                    print("Logit Infs: ", torch.any(torch.isinf(logits[0])))
                    print("Loss: ", loss.item())
                    raise err

            torch.cuda.empty_cache()
            ## Calculate gradients of loss w.r.t. model parameters in backward pass.
            loss.backward()
            ## Perform single optimization step to update model parameters via optimizer.
            #
            optimizer.step()
            #
            # LOGGING
            ## Calculate effective mini-batch loss as process-averaged mini-mini-batch loss.
            ## Sum up mini-mini-batch losses from all processes and divide by number of processes.
            ## Use collective communication functions from `torch.distributed` package.
            # Note that `torch.distributed` collective communication functions will only
            # work with `torch` tensors, i.e., floats, ints, etc. must be converted before!
            ## Append globally averaged loss of this epoch to history list.
            dist.all_reduce(loss)

            if rank == 0:
                loss /= world_size
                loss_history.append(
                    loss.item()
                )  # Append globally averaged loss of this epoch to history list.
                epoch_average_loss += loss.item()

                print(
                    f"Epoch: {epoch + 1:03d}/{num_epochs:03d} "
                    f"| Batch {(batch_idx+1):04d}/{len(train_loader):04d} "
                    f"| Averaged Loss: {loss.item():.4f}"
                )

        epoch_average_loss /= len(train_loader)

        # Validation starts here.

        ## Set model to evaluation mode.
        model.eval()
        with torch.no_grad():  # Disable gradient calculation.
            # Validate model in data-parallel fashion.
            # Determine number of correctly classified samples and overall number
            # of samples in training and validation dataset.
            #
            right_train, num_train = get_right_ddp(
                model, train_loader, samples=num_samples
            )
            right_valid, num_valid = get_right_ddp(
                model, valid_loader, samples=num_samples
            )
            #
            ## Sum up number of correctly classified samples in training dataset,
            dist.all_reduce(right_train)
            ## overall number of considered samples in training dataset,
            dist.all_reduce(num_train)
            ## number of correctly classified samples in validation dataset,
            dist.all_reduce(right_valid)
            ## and overall number of samples in validation dataset over all processes.
            dist.all_reduce(num_valid)
            ## Use collective communication functions from `torch.distributed` package.
            #
            # Note that `torch.distributed` collective communication functions will only
            # work with torch tensors, i.e., floats, ints, etc. must be converted before!
            # From these values, calculate overall training + validation accuracy.
            #
            train_acc = right_train.item() / num_train.item() * 100
            valid_acc = right_valid.item() / num_valid.item() * 100
            ## Append accuracy values to corresponding history lists.

            if rank == 0:
                print(
                    f"Epoch: {epoch + 1:03d}/{num_epochs:03d} "
                    f"| Avg Loss: {epoch_average_loss:.4f}"
                    f"| Train: {train_acc :.2f}% "
                    f"| Validation: {valid_acc :.2f}%"
                )

        elapsed = (time.perf_counter() - start) / 60  # Measure training time per epoch.
        elapsed_sync = torch.Tensor([elapsed]).cuda()
        dist.all_reduce(elapsed_sync)
        elapsed = elapsed_sync.item()
        elapsed /= world_size

        if rank == 0:
            print(f"Time elapsed: {elapsed} min")

    # Stop timer and calculate training time elapsed after epoch.
    elapsed = time.perf_counter() - start
    elapsed_sync = torch.tensor([elapsed]).cuda()
    ## Calculate average training time elapsed after each epoch over all processes,
    ## i.e., sum up times from all processes and divide by overall number of processes.
    ## Use collective communication functions from torch.distributed package.
    # Note that torch.distributed collective communication functions will only
    # work with torch tensors, i.e., floats, ints, etc. must be converted before!
    dist.all_reduce(elapsed_sync)
    elapsed = elapsed_sync.item()
    elapsed /= world_size * 60

    if rank == 0:
        ## Print process-averaged training time after each epoch.
        print(f"Total time elapsed: {elapsed} min")
        torch.save(loss_history, f"loss_{world_size}_gpu.pt")
        torch.save(train_acc_history, f"train_acc_{world_size}_gpu.pt")
        torch.save(valid_acc_history, f"valid_acc_{world_size}_gpu.pt")

    return loss_history, train_acc_history, valid_acc_history


def main_ddp() -> None:
    """Distributed data-parallel training of AlexNet on the CIFAR-10 dataset."""
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
            f"Rank {rank}/{world_size}: Process group initialized with torch rank {dist.get_rank()} and torch world size {dist.get_world_size()}."
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
    betas = (0.65, 0.999)
    # momentum = 0.5
    weight_decay = 0.0
    start_std = 0.00025
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
    model = VIAlexNet(
        num_classes=10, dropout=0.0, variational_distribution=vardist, prior=prior
    )
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
        ddp_model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
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
