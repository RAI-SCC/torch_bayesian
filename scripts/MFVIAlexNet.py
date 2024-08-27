# import random
# import os
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision

# from torchvision.transforms import Compose
import vi
from vi import VIModule
from vi.predictive_distributions import CategoricalPredictiveDistribution
from vi.variational_distributions import MeanFieldNormalVarDist, VariationalDistribution

# MODEL
# Define neural network by subclassing PyTorch's nn.Module.
# Save to a separate Python module file `model.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.


class MFVIAlexNet(VIModule):
    """
    AlexNet architecture.

    Attributes
    ----------
    features : torch.nn.container.Sequential
        everything in AlexNet, all in one VISequential

    Methods
    -------
    __init__()
        The constructor defining the network's architecture.
    forward()
        The forward pass.
    """

    # Initialize neural network layers in __init__.
    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.5,
        variational_distribution: VariationalDistribution = MeanFieldNormalVarDist(),
    ) -> None:
        """
        Initialize AlexNet architecture.

        Parameters
        ----------
        num_classes : int
            The number of classes in the underlying classification problem.
        dropout : float
            The dropout probability.
        """
        super().__init__()
        self.features = vi.VISequential(
            # AlexNet has 8 layers: 5 convolutional layers, some followed by max-pooling (see figure),
            # and 3 fully connected layers. In this model, we use nn.ReLU between our layers.
            # nn.Sequential is an ordered container of modules.
            # The data is passed through all the modules in the same order as defined.
            # You can use sequential containers to put together a quick network.
            #
            # IMPLEMENT FEATURE-EXTRACTOR PART OF ALEXNET HERE!
            # 1st convolutional layer (+ max-pooling)
            vi.VIConv2d(
                3,
                64,
                kernel_size=11,
                stride=4,
                padding=2,
                variational_distribution=variational_distribution,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            ## 2nd convolutional layer (+ max-pooling)
            vi.VIConv2d(
                64,
                192,
                kernel_size=5,
                padding=2,
                variational_distribution=variational_distribution,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            ## 3rd + 4th convolutional layer
            vi.VIConv2d(
                192,
                384,
                kernel_size=3,
                padding=1,
                variational_distribution=variational_distribution,
            ),
            torch.nn.ReLU(inplace=True),
            vi.VIConv2d(
                384,
                256,
                kernel_size=3,
                padding=1,
                variational_distribution=variational_distribution,
            ),
            torch.nn.ReLU(inplace=True),
            ## 5th convolutional layer (+ max-pooling)
            vi.VIConv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                variational_distribution=variational_distribution,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            # Average pooling to downscale possibly larger input images.
            torch.nn.AdaptiveAvgPool2d((6, 6)),
            torch.nn.Flatten(1),
            # IMPLEMENT FULLY CONNECTED PART HERE!
            # 6th, 7th + 8th fully connected layer
            # The linear layer is a module that applies a linear transformation
            # on the input using its stored weights and biases.
            # 6th fully connected layer (+ dropout)
            torch.nn.Dropout(p=dropout),
            vi.VILinear(
                256 * 6 * 6, 4096, variational_distribution=variational_distribution
            ),
            torch.nn.ReLU(inplace=True),
            ## 7th fully connected layer (+ dropout)
            torch.nn.Dropout(p=dropout),
            vi.VILinear(4096, 4096, variational_distribution=variational_distribution),
            torch.nn.ReLU(inplace=True),
            # 8th (output) layer
            vi.VILinear(
                4096, num_classes, variational_distribution=variational_distribution
            ),
        )

    # Forward pass: Implement operations on the input data, i.e., apply model to input x.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Do forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The model's output.
        """
        x = self.features(x)
        return x


# DATA
# Save to a separate Python module file `utils_data.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.


def get_transforms_cifar10() -> (
    Tuple[torchvision.transforms.Compose, torchvision.transforms.Compose]
):
    """
    Get transforms applied to CIFAR-10 data for AlexNet training and inference.

    Returns
    -------
    torchvision.transforms.Compose
        The transforms applied to CIFAR-10 for training AlexNet.
    torchvision.transforms.Compose
        The transforms applied to CIFAR-10 to run inference with AlexNet.
    """
    # Transforms applied to training data (randomness to make network more robust against overfitting)
    train_transforms = (
        torchvision.transforms.Compose(  # Compose several transforms together.
            [
                torchvision.transforms.Resize(
                    (70, 70)
                ),  # Upsample CIFAR-10 images to make them work with AlexNet.
                torchvision.transforms.RandomCrop(
                    (64, 64)
                ),  # Randomly crop image to make NN more robust against overfitting.
                torchvision.transforms.ToTensor(),  # Convert image into torch tensor.
                torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1,1] via (image-mean)/std.
            ]
        )
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((70, 70)),
            torchvision.transforms.CenterCrop((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return train_transforms, test_transforms


def make_train_validation_split(
    train_dataset: torchvision.datasets.CIFAR10,
    seed: int = 123,
    validation_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split original CIFAR-10 training data into train and validation sets.

    Parameters
    ----------
    train_dataset : torchvision.datasets.CIFAR10
        The original CIFAR-10 training dataset.
    seed : int
        The seed used to split the data.
    validation_fraction : float
        The fraction of samples used for validation.

    Returns
    -------
    numpy.ndarray
        The sample indices for the training dataset.
    numpy.ndarray
        The sample indices for the validation dataset.
    """
    num_samples = len(
        train_dataset
    )  # Get overall number of samples in original training data.
    rng = np.random.default_rng(
        seed=seed
    )  # Set same seed over all ranks for consistent train-test split.
    idx = np.arange(0, num_samples)  # Construct array of all indices.
    rng.shuffle(idx)  # Shuffle them.
    num_validate = int(
        validation_fraction * num_samples
    )  # Determine number of validation samples from validation split.
    return (
        idx[num_validate:],
        idx[0:num_validate],
    )  # Extract and return train and validation indices.


def get_dataloaders_cifar10(
    batch_size: int,
    data_root: str = "data",
    validation_fraction: float = 0.1,
    train_transforms: Optional[Callable] = None,
    test_transforms: Optional[Callable] = None,
    seed: int = 123,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Get dataloaders for training, validation, and testing on the CIFAR-10 dataset.

    Parameters
    ----------
    batch_size : int
        The mini-batch size.
    data_root : str
        The path to the dataset.
    validation_fraction : float
        The fraction of the original training data used for validation.
    train_transforms : Callable[[Any], Any]
        The transform applied to the training data.
    test_transforms : Callable[[Any], Any]
        The transform applied to the validation/testing data (inference).
    seed : int
        The seed for the validation-train split.

    Returns
    -------
    torch.utils.data.DataLoader
        The training dataloader.
    torch.utils.data.DataLoader
        The validation dataloader.
    torch.utils.data.DataLoader
        The testing dataloader.
    """
    if train_transforms is None:
        train_transforms = torchvision.transforms.ToTensor()

    if test_transforms is None:
        test_transforms = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=train_transforms, download=True
    )

    valid_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=test_transforms
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, transform=test_transforms
    )

    # Perform index-based train-validation split of original training data.
    train_indices, valid_indices = make_train_validation_split(
        train_dataset, seed, validation_fraction
    )  # Get train and validation indices.

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


# EVALUATION
# Save to a separate Python module file `utils_eval.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.


def compute_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    samples: int = 10,
) -> float:
    """
    Compute the accuracy of the model's predictions on given labeled data.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    data_loader : torch.utils.data.DataLoader
        The dataloader.
    device : torch.device
        The device to use.

    Returns
    -------
    float
        The model's accuracy on the given dataset in percent.
    """
    with torch.no_grad():  # Disable gradient calculation to reduce memory consumption.
        # Initialize number of correctly predicted samples + overall number of samples.
        correct_pred, num_examples = (
            0,
            0,
        )  # Initialize number of correctly predicted and overall samples, respectively.

        for i, (features, targets) in enumerate(data_loader):
            # CONVERT DATASET TO USED DEVICE.
            ## features = ...
            features = features.to(device)
            ## targets = ...
            targets = targets.to(device)
            #
            # CALCULATE PREDICTIONS OF CURRENT MODEL ON FEATURES OF INPUT DATA.
            ## logits = ...
            logits = model(features, samples=samples)[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # print(probs.shape)
            outputs = probs.mean(dim=0)
            # print(outputs.sum(dim=1))
            ## Determine class with highest score.
            prediction = outputs.argmax(dim=1)
            # print(prediction)
            # print(targets)
            ## Compare predictions to actual labels to determine number of correctly predicted samples.
            correct_pred += prediction.eq(targets).sum().item()
            ## Determine overall number of samples.
            num_examples += targets.size(0)

        # CALCULATE AND RETURN ACCURACY AS PERCENTAGE OF CORRECTLY PREDICTED SAMPLES.
        accuracy = correct_pred / num_examples * 100
        ## return ...
        return accuracy


# TRAINING
# Save to a separate Python module file `utils_train.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.
# from utils_eval import compute_accuracy


def train_model(
    model: torch.nn.Module,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logging_interval: int = 50,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    num_samples: int = 10,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train your model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    num_epochs : int
        The number of epochs to train
    train_loader : torch.utils.data.DataLoader
        The training dataloader.
    valid_loader : torch.utils.data.DataLoader
        The validation dataloader.
    test_loader : torch.utils.data.DataLoader
        The testing dataloader.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    device : torch.device
        The device to train on.
    logging_interval : int
        The logging interval.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        An optional learning rate scheduler.

    Returns
    -------
    List[float]
        The loss history.
    List[float]
        The training accuracy history.
    List[float]
        The validation accuracy history.
    """
    ## start = ... # Start timer to measure training time.
    start = time.time()
    # Initialize history lists for loss, training accuracy, and validation accuracy.
    loss_history: List[float] = []
    train_acc_history: List[float] = []
    valid_acc_history: List[float] = []

    predictive_distribution = CategoricalPredictiveDistribution()
    loss_fn = vi.KullbackLeiblerLoss(
        predictive_distribution, dataset_size=len(train_loader.dataset)
    )

    # ACTUAL TRAINING STARTS HERE.
    for epoch in range(num_epochs):  # Loop over epochs.
        # IMPLEMENT TRAINING LOOP HERE.
        #
        ## Set model to training mode.
        #  Thus, layers like dropout which behave differently on train and
        #  test procedures know what is going on and can behave accordingly.
        model.train()

        for batch_idx, (features, targets) in enumerate(
            train_loader
        ):  # Loop over mini batches.
            # CONVERT DATASET TO USED DEVICE.
            ## features = ...  # Move features to used device.
            features = features.to(device)
            ## targets = ...  # Move targets to used device.
            targets = targets.to(device)
            #
            # FORWARD & BACKWARD PASS
            ## logits = ...  # Get predictions of model with current parameters.
            logits = model(features, samples=num_samples)
            ## loss = ...  # Calculate cross-entropy loss on current mini-batch.
            # print(logits[0][0,0,:])
            loss = loss_fn(*logits, targets)
            assert not torch.isinf(loss).item()
            # print(loss.item())
            ## Zero out gradients.
            optimizer.zero_grad()
            ## Calculate gradients of loss w.r.t. model parameters in backward pass.
            loss.backward()
            ## Perform single optimization step to update model parameters via optimizer.
            #
            optimizer.step()

            # LOGGING
            ## Append loss to history list.
            if not batch_idx % logging_interval:
                print(
                    f"Epoch: {epoch + 1:03d}/{num_epochs:03d} "
                    f"| Batch {(batch_idx+1):04d}/{len(train_loader):04d} "
                    f"| Loss: {loss:.4f}"
                )

        # VALIDATION STARTS HERE.
        #
        ## Set model to evaluation mode.
        model.eval()

        with (
            torch.no_grad()
        ):  # Disable gradient calculation to reduce memory consumption.
            # COMPUTE ACCURACY OF CURRENT MODEL PREDICTIONS ON TRAINING + VALIDATION DATASETS.
            ## train_acc = compute_accuracy(...)  # Compute accuracy on training data.
            train_acc = compute_accuracy(
                model, train_loader, device, samples=num_samples
            )
            ## valid_acc = compute_accuracy(...)  # Compute accuracy on validation data.
            valid_acc = compute_accuracy(
                model, valid_loader, device, samples=num_samples
            )
            print(
                f"Epoch: {epoch + 1:03d}/{num_epochs:03d} "
                f"| Train: {train_acc :.2f}% "
                f"| Validation: {valid_acc :.2f}%"
            )

            ## APPEND ACCURACY VALUES TO CORRESPONDING HISTORY LISTS.
            train_acc_history.append(train_acc)
            valid_acc_history.append(valid_acc)

        ## elapsed = ...  # Stop timer and calculate training time elapsed after epoch.
        elapsed = time.time() - start
        ## Print training time elapsed after epoch.
        print(f"Elapsed time: {elapsed:.2f}s")

        if scheduler is not None:  # Adapt learning rate.
            scheduler.step(valid_acc_history[-1])

    ## elapsed = ...  # Stop timer and calculate total training time.
    elapsed = time.time() - start
    ## Print overall training time.
    print(f"Total training time: {elapsed:.2f}s")

    # FINAL TESTING STARTS HERE.
    #
    ## test_acc = compute_accuracy(...)  # Compute accuracy on test data.
    test_acc = compute_accuracy(model, test_loader, device, samples=num_samples)
    ## Print test accuracy.
    print(f"Test accuracy: {test_acc:.2f}%")

    ## Return history lists for loss, training accuracy, and validation accuracy.
    return loss_history, train_acc_history, valid_acc_history


if __name__ == "__main__":
    # DATASET
    # Include into your main script to be executed when running as a batch job later on.

    # Transforms on your data allow you to take it from its source state and transform it into ready-for-training data.
    # Get transforms applied to CIFAR-10 data for training and inference.
    ## train_transforms, test_transforms = ...

    batch_size = 256  # Set mini-batch size hyperparameter.
    data_root = "data/cifar"  # Path to data dir.
    rand_seed = 123

    # GET PYTORCH DATALOADERS FOR TRAINING, TESTING, AND VALIDATION DATASET.
    ## train_loader, valid_loader, test_loader = get_dataloaders_cifar10(...)
    train_transforms, test_transforms = get_transforms_cifar10()
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=batch_size,
        data_root=data_root,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        seed=rand_seed,
    )

    # Check loaded dataset.
    for images, labels in train_loader:
        print("Image batch dimensions:", images.shape)
        print("Image label dimensions:", labels.shape)
        print("Class labels of 10 examples:", labels[:10])
        break

    # SETTINGS
    # Include into your main script to be executed when running as a batch job later on.

    num_epochs = 20  # Number of epochs
    lr = 1e-1  # Learning rate
    start_std = 1 / 1000

    # Get device used for training, e.g., check via torch.cuda.is_available().
    ## device = ...
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## Print used device.
    print("Device:", device)

    ## model = ...  # Build an instance of AlexNet with 10 classes for CIFAR-10 and convert it to the used device.
    model = MFVIAlexNet(
        num_classes=10,
        variational_distribution=MeanFieldNormalVarDist(initial_std=start_std),
    )
    model.to(device)
    model.return_log_prob()
    ## Print model.
    # print(model)

    # Set up an SGD optimizer from the `torch.optim` package.
    # Use a momentum of 0.9 and a learning rate of 0.1.
    ## optimizer = ...
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Set up a LR scheduler.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, mode="max", verbose=True
    )

    start_acc = compute_accuracy(model, valid_loader, device, samples=5)
    print("Start Accuracy:", start_acc)

    # TRAIN MODEL.
    ## loss_history, train_acc_history, valid_acc_history = train_model(...)
    loss_history, train_acc_history, valid_acc_history = train_model(
        model=model,
        num_epochs=num_epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        logging_interval=10,
        num_samples=5,
    )

    # Save history lists for loss, training accuracy, and validation accuracy.S
    torch.save(loss_history, "loss.pt")
    torch.save(train_acc_history, "train_acc.pt")
    torch.save(valid_acc_history, "valid_acc.pt")
