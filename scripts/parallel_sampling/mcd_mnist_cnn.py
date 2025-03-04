# Sample parallel implementation of a training loop with the entso_e dataset and a fully connected model architecture.
import torch
from torch import Tensor, nn
import torch_bayesian.vi as vi
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from torch.utils.data import DataLoader
from typing import Callable
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision import datasets
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
            vi.VILinear(hidden2 * 7 * 7, output_length, variational_distribution=variational_distribution),
        )

    def forward(self, x_: Tensor) -> Tensor:
        logits = self.conv_stack(x_)
        return logits



def train(
        dataloader: DataLoader,
        model: vi.VIModule,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        sample_num,
        train_loss_list,
        device,
):

    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)


        # Get predictions
        pred = model(x, samples=sample_num)
        mean_model_output = pred.mean(dim=0)
        probs = F.softmax(mean_model_output, dim=1)
        loss = loss_fn(probs, y)
        # Backpropagation
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    train_loss_list.append(loss.item())
    return model


def test(dataloader: DataLoader,
         model: vi.VIModule,
         loss_fn: Callable,
         sample_num,
         test_loss_list,
         device
         ):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            samples = model(x, samples=sample_num)
            mean_model_output = torch.tensor(samples, dtype=samples.dtype).mean(dim=0)
            samples = F.softmax(mean_model_output, dim=1)
            correct += (samples.argmax(1) == y).type(torch.float).sum().item()
            test_loss += loss_fn(samples, y).item()

    test_loss /= num_batches
    correct /= len(dataloader.dataset)
    test_loss_list.append(test_loss)

    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    return


if __name__ == "__main__":
    # Hyper-parameters
    in_channel = 1
    output_length = 10
    hidden1 = 8
    hidden2 = 16
    batch_size = 64
    epochs = 5
    random_seed = 42
    all_sample_num = 4
    lr = 1e-3
    # mp.set_start_method("fork", force=True)
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

    sample_num = all_sample_num


    # Do stuff here
    torch.manual_seed(random_seed)
    for t in range(epochs):

        print(f"Epoch {t + 1}\n-------------------------------")
        model = train(train_dataloader, model, loss_fn, optimizer, sample_num, train_loss_list,
                      device)
        test(test_dataloader, model, loss_fn, sample_num, test_loss_list, device)

