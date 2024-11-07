import matplotlib.pyplot as plt
import torch
from vi import VIModule
from torch.utils.data import DataLoader

def plot_random_samples(model: VIModule, dataloader: DataLoader):
    num_batches = len(dataloader)
    random_batch = int(torch.randint(low = 0, high= num_batches-1, size = (1,)))
    model.eval()
    with torch.no_grad():
        n = 0
        for x, y  in dataloader:
            if n < random_batch:
                n += 1
            else:
                break

        samples = model(x)
        mean_samples = samples.mean(dim=0)
        std_samples = torch.std(samples, dim=0)
        num_samples = y.shape[0]
        random_sample =int(torch.randint(low=0, high=num_samples - 1, size=(1,)))
        mean = mean_samples[random_sample]
        std = std_samples[random_sample]
        real = y[random_sample]
        in_mod = x[random_sample]
        input_length = in_mod.shape[0]
        output_length = real.shape[0]

    plt.clf()
    plt.plot(in_mod.numpy(), color="blue", label="inputs")
    # plot outputs and ground truth behind input sequence
    plt.plot(
        range(input_length, input_length + output_length),
        mean.numpy(),
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
        (torch.add(mean, std, alpha=1)).numpy(),
        (torch.add(mean, std, alpha=-1)).numpy(),
        color="orange",
        alpha=0.1,
    )
    file_name = 'random_sample.png'
    plt.savefig(file_name)