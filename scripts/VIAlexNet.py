import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

import vi
from vi import VIModule
from vi.predictive_distributions import CategoricalPredictiveDistribution
from vi.priors import MeanFieldNormalPrior, Prior
from vi.variational_distributions import MeanFieldNormalVarDist, VariationalDistribution

# MODEL
# Define neural network by subclassing vi.VIModule.
# Save to a separate Python module file `model.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.


class VIAlexNet(VIModule):
    """
    AlexNet architecture.

    Attributes
    ----------
    features : VISequential
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
        prior: Prior = MeanFieldNormalPrior(),
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
                prior=prior,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ## 2nd convolutional layer (+ max-pooling)
            vi.VIConv2d(
                64,
                192,
                kernel_size=5,
                padding=2,
                variational_distribution=variational_distribution,
                prior=prior,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ## 3rd + 4th convolutional layer
            vi.VIConv2d(
                192,
                384,
                kernel_size=3,
                padding=1,
                variational_distribution=variational_distribution,
                prior=prior,
            ),
            nn.ReLU(inplace=True),
            vi.VIConv2d(
                384,
                256,
                kernel_size=3,
                padding=1,
                variational_distribution=variational_distribution,
                prior=prior,
            ),
            nn.ReLU(inplace=True),
            ## 5th convolutional layer (+ max-pooling)
            vi.VIConv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                variational_distribution=variational_distribution,
                prior=prior,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Average pooling to downscale possibly larger input images.
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(1),
            # IMPLEMENT FULLY CONNECTED PART HERE!
            # 6th, 7th + 8th fully connected layer
            # The linear layer is a module that applies a linear transformation
            # on the input using its stored weights and biases.
            # 6th fully connected layer (+ dropout)
            nn.Dropout(p=dropout),
            vi.VILinear(
                256 * 6 * 6,
                4096,
                variational_distribution=variational_distribution,
                prior=prior,
            ),
            nn.ReLU(inplace=True),
            ## 7th fully connected layer (+ dropout)
            nn.Dropout(p=dropout),
            vi.VILinear(
                4096,
                4096,
                variational_distribution=variational_distribution,
                prior=prior,
            ),
            nn.ReLU(inplace=True),
            # 8th (output) layer
            vi.VILinear(
                4096,
                num_classes,
                variational_distribution=variational_distribution,
                prior=prior,
            ),
        )

    # Forward pass: Implement operations on the input data, i.e., apply model to input x.
    def forward(self, x: Tensor) -> Tensor:
        """
        Do forward pass.

        Parameters
        ----------
        x : Tensor
            The input data.

        Returns
        -------
        Tensor
            The model's output.
        """
        x = self.features(x)

        return x


# DATA
# Save to a separate Python module file `utils_data.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.


def get_transforms_cifar10() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms applied to CIFAR-10 data for AlexNet training and inference.

    Returns
    -------
    transforms.Compose
        The transforms applied to CIFAR-10 for training AlexNet.
    transforms.Compose
        The transforms applied to CIFAR-10 to run inference with AlexNet.
    """
    # Transforms applied to training data (randomness to make network more robust against overfitting)
    train_transforms = transforms.Compose(  # Compose several transforms together.
        [
            # Upsample CIFAR-10 images to make them work with AlexNet.
            transforms.Resize((70, 70)),
            # Randomly crop image to make NN more robust against overfitting.
            transforms.RandomCrop((64, 64)),
            # Convert image into torch tensor.
            transforms.ToTensor(),
            # Normalize to [-1,1] via (image-mean)/std.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((70, 70)),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return train_transforms, test_transforms


def make_train_validation_split(
    train_dataset: CIFAR10,
    seed: int = 123,
    validation_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split original CIFAR-10 training data into train and validation sets.

    Parameters
    ----------
    train_dataset : CIFAR10
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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    train_transforms : Callable
        The transform applied to the training data.
    test_transforms : Callable
        The transform applied to the validation/testing data (inference).
    seed : int
        The seed for the validation-train split.

    Returns
    -------
    DataLoader
        The training dataloader.
    DataLoader
        The validation dataloader.
    DataLoader
        The testing dataloader.
    """
    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = CIFAR10(
        root=data_root, train=True, transform=train_transforms, download=True
    )
    valid_dataset = CIFAR10(root=data_root, train=True, transform=test_transforms)
    test_dataset = CIFAR10(root=data_root, train=False, transform=test_transforms)

    # Perform index-based train-validation split of original training data.
    train_indices, valid_indices = make_train_validation_split(
        train_dataset, seed, validation_fraction
    )  # Get train and validation indices.

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=train_sampler,
    )

    test_loader = DataLoader(
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
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    samples: int = 10,
) -> float:
    """
    Compute the accuracy of the model's predictions on given labeled data.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    data_loader : DataLoader
        The dataloader.
    device : torch.device
        The device to use.
    samples : int
        The number of samples to compute accuracy from.

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
            features = features.to(device)
            targets = targets.to(device)
            #
            # CALCULATE PREDICTIONS OF CURRENT MODEL ON FEATURES OF INPUT DATA.
            logits = model(features, samples=samples)[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            outputs = probs.mean(dim=0)
            ## Determine class with highest score.
            prediction = outputs.argmax(dim=1)
            ## Compare predictions to actual labels to determine number of correctly predicted samples.
            correct_pred += prediction.eq(targets).sum().item()
            ## Determine overall number of samples.
            num_examples += targets.size(0)

        # CALCULATE AND RETURN ACCURACY AS PERCENTAGE OF CORRECTLY PREDICTED SAMPLES.
        accuracy = correct_pred / num_examples * 100
        return accuracy


# TRAINING
# Save to a separate Python module file `utils_train.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.
# from utils_eval import compute_accuracy


def train_model(
    model: nn.Module,
    num_epochs: int,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
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
    model : nn.Module
        The model to train.
    num_epochs : int
        The number of epochs to train
    train_loader : DataLoader
        The training dataloader.
    valid_loader : DataLoader
        The validation dataloader.
    test_loader : DataLoader
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
            features = features.to(device)
            targets = targets.to(device)
            #
            # FORWARD & BACKWARD PASS
            logits = model(features, samples=num_samples)
            loss = loss_fn(*logits, targets)
            assert not torch.isinf(loss).item()
            ## Zero out gradients.
            optimizer.zero_grad()
            ## Calculate gradients of loss w.r.t. model parameters in backward pass.
            loss.backward()
            ## Perform single optimization step to update model parameters via optimizer.
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
            # Compute accuracy on training data.
            train_acc = compute_accuracy(
                model, train_loader, device, samples=num_samples
            )
            # Compute accuracy on validation data.
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

        # Stop timer and calculate training time elapsed after epoch.
        elapsed = time.time() - start
        ## Print training time elapsed after epoch.
        print(f"Elapsed time: {elapsed:.2f}s")

        if scheduler is not None:  # Adapt learning rate.
            scheduler.step(valid_acc_history[-1])

    # Stop timer and calculate total training time.
    elapsed = time.time() - start
    ## Print overall training time.
    print(f"Total training time: {elapsed:.2f}s")

    # FINAL TESTING STARTS HERE.
    #
    # Compute accuracy on test data.
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
    train_transforms, test_transforms = get_transforms_cifar10()

    batch_size = 256  # Set mini-batch size hyperparameter.
    data_root = "data/cifar"  # Path to data dir.
    rand_seed = 123

    # GET PYTORCH DATALOADERS FOR TRAINING, TESTING, AND VALIDATION DATASET.
    ## train_loader, valid_loader, test_loader = get_dataloaders_cifar10(...)
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## Print used device.
    print("Device:", device)

    # Build an instance of AlexNet with 10 classes for CIFAR-10 and convert it to the used device.
    model = VIAlexNet(
        num_classes=10,
        variational_distribution=MeanFieldNormalVarDist(initial_std=start_std),
    )
    model.to(device)
    model.return_log_prob()
    ## Print model.
    # print(model)

    # Set up an SGD optimizer from the `torch.optim` package.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Set up a LR scheduler.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, mode="max", verbose=True
    )

    start_acc = compute_accuracy(model, valid_loader, device, samples=5)
    print("Start Accuracy:", start_acc)

    # TRAIN MODEL.
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

    # Save history lists for loss, training accuracy, and validation accuracy.
    torch.save(loss_history, "loss.pt")
    torch.save(train_acc_history, "train_acc.pt")
    torch.save(valid_acc_history, "valid_acc.pt")
