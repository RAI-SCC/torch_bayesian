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
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
sampling_state = None
train_loss_list = []
test_loss_list = []
from timing_utils import cuda_time_function, print_cuda_timing_summary

# Define a subclass of ImageFolder that filters classes
class SubsetImageFolder(ImageFolder):
    def __init__(self, root, classes_to_keep, transform=None):
        super().__init__(root, transform=transform)
        # Build class to index map for selected classes
        self.classes = classes_to_keep
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # Filter samples
        filtered_samples = []
        for path, _ in self.samples:
            class_name = os.path.basename(os.path.dirname(path))
            if class_name in self.class_to_idx:
                filtered_samples.append((path, self.class_to_idx[class_name]))
        
        self.samples = filtered_samples
        self.targets = [s[1] for s in self.samples]

class IMAGENETCNN(vi.VIModule):
    def __init__(self, variational_distribution=MeanFieldNormalVarDist()):
        super().__init__()

        # Convolutional Block 1
        self.conv1 = vi.VIConv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3,
                        variational_distribution=variational_distribution)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    

        # Convolutional Block 2
        self.conv2 = vi.VIConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,
                        variational_distribution=variational_distribution)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3 = vi.VIConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,
                        variational_distribution=variational_distribution)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 28 → 14

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, 128, 1, 1]
        self.fc = vi.VILinear(128, 1000, variational_distribution=variational_distribution)  


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten to [B, 128]
        x = self.fc(x)
        return x


def setup(rank, world_size):
    # Initialize distributed backend
    dist.init_process_group(
        backend="nccl",  # Use NCCL for CUDA
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

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
    print(len(dataloader))
    counter = 0
    for batch, (x, y) in enumerate(dataloader):
        counter += 1
        if counter%100 == 0:
            print(counter)
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
        #probs = F.softmax(mean_model_output, dim=1)
        loss = loss_fn(mean_model_output, y)
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
                test_loss += loss_fn(mean_model_output, y).item()

    if rank == 0:
        test_loss /= num_batches
        correct /= len(dataloader.dataset)
        test_loss_list.append(test_loss)

        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    return
   

if __name__ == "__main__":
    # Hyper-parameters
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    #torch.cuda.set_device(local_rank)
    set_device = "cuda:" + str(local_rank)
    torch.device(set_device)
    
    batch_size = 128
    epochs = 3
    random_seed = 42
    all_sample_num = 16
    print(all_sample_num)
    lr = 1e-3

    imagenet_train_path = "/hkfs/home/dataset/datasets/imagenet-2012/original/imagenet-raw/ILSVRC/Data/CLS-LOC/train"
    imagenet_val_path = "/hkfs/home/dataset/datasets/imagenet-2012/original/imagenet-raw/ILSVRC/Data/CLS-LOC/val"


    # Get the first 100 class folder names (sorted alphabetically for consistency)
    all_classes = sorted(os.listdir(imagenet_train_path))
    subset_classes = all_classes[:100]


    transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
    
    # Create the training and validation datasets
    training_data = SubsetImageFolder(imagenet_train_path, subset_classes, transform=transform)
    test_data = SubsetImageFolder(imagenet_val_path, subset_classes, transform=transform)


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
    model = IMAGENETCNN(variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)

    print(f"Using {device} device")
    print(model)
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    sample_num = int(all_sample_num / world_size)
    print(sample_num)
    
    setup(rank, world_size)
    
    # Do stuff here
    torch.manual_seed(random_seed)
    for t in range(epochs):
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        model = train(train_dataloader, model, loss_fn, optimizer, sample_num, train_loss_list, rank, world_size,device)
        test(test_dataloader, model, loss_fn, sample_num, test_loss_list,rank, world_size, device)

    cleanup()
    print_cuda_timing_summary()

