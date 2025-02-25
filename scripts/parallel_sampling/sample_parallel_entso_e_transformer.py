# Sample parallel implementation of a training loop with the entso_e dataset and a fully connected model architecture.
import torch
from torch import Tensor, nn
import torch_bayesian.vi as vi
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from torch.utils.data import DataLoader
from typing import Callable
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os
import torch.multiprocessing as mp
import polars as pl
from entsoe_data_load import TimeseriesDatasetUnsqueeze
sampling_state = None
train_loss_list = []
test_loss_list = []
HISTORY_WINDOW = 50
PREDICTION_WINDOW = 10


class TimeSeriesTransformer(vi.VIModule):
    def __init__(self, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, variational_distribution):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = vi.VITransformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            variational_distribution= variational_distribution
        )
        self.encoder_projection = vi.VILinear(1, embed_dim, variational_distribution=variational_distribution)
        self.decoder_projection = vi.VILinear(1, embed_dim, variational_distribution=variational_distribution)
        self.fc_out = vi.VILinear(embed_dim, 1, variational_distribution=variational_distribution)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.encoder_projection(src)
        tgt_emb = self.decoder_projection(tgt)
        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        return self.fc_out(output)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)
    return mask

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
        future_input = y[:, :-1, :]
        future_output = y[:, 1:, :]

        src_mask = generate_square_subsequent_mask(HISTORY_WINDOW).to(torch.float32)
        tgt_mask = generate_square_subsequent_mask(future_input.size(1)).to(torch.float32)

        output = model(x, future_input, src_mask, tgt_mask, samples = sample_num)

        # Switch to general randomness
        sampling_state = torch.get_rng_state()
        torch.set_rng_state(regular_state)

        loss = loss_fn(output, future_output)
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
    global sampling_state  # Randomness switch

    with torch.no_grad():
        for x, y in dataloader:

            x, y = x.to(device), y.to(device)

            regular_state = torch.get_rng_state()
            if sampling_state == None:
                torch.manual_seed(rank)
            else:
                torch.set_rng_state(sampling_state)

            src_mask = generate_square_subsequent_mask(HISTORY_WINDOW).to(torch.float32)
            tgt_input = torch.zeros((1, PREDICTION_WINDOW, 1))  # Start with zeros or last known value
            tgt_mask = generate_square_subsequent_mask(PREDICTION_WINDOW).to(torch.float32)

            samples = model(x, tgt_input, src_mask, tgt_mask, samples=sample_num)


            sampling_state = torch.get_rng_state()
            torch.set_rng_state(regular_state)


            samples_global = samples
            dist.all_reduce(samples_global, op=dist.ReduceOp.SUM)

            if rank == 0:
                samples_global /= world_size
                samples = torch.tensor(samples_global, dtype=samples.dtype)
                test_loss += loss_fn(samples, y).item()

    if rank == 0:
        test_loss /= num_batches

        test_loss_list.append(test_loss)

        print(
            f"Test Error: Avg loss: {test_loss:>8f} \n"
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
    input_length = HISTORY_WINDOW
    output_length = PREDICTION_WINDOW
    epochs = 10
    batch_size = 32
    random_seed = 42
    all_sample_num = 64
    EMBED_DIM = 128
    NHEAD = 4
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 256
    #mp.set_start_method("fork", force=True)
    df = pl.read_csv("data/ENTSOEEnergyLoads/de.csv",
                     dtypes={"start": pl.Datetime, "end": pl.Datetime, "load": pl.Float32},
                     )
    x = df["load"]
    x = x.fill_null(strategy="backward")
    normalized_x = (x - x.mean()) / x.std()
    x_tensor = normalized_x.to_torch()
    data_train, data_test = x_tensor[: int(len(x) * 0.7)], x_tensor[int(len(x) * 0.7):]
    dataset_train = TimeseriesDatasetUnsqueeze(data_train, input_length, output_length)
    dataset_test = TimeseriesDatasetUnsqueeze(data_test, input_length, output_length)

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
    model = TimeSeriesTransformer(EMBED_DIM, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD,
                          variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)

    print(f"Using {device} device")
    print(model)
    loss_fn = vi.MeanSquaredErrorLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    sample_num = int(all_sample_num / world_size)
    parameters = (train_dataloader, test_dataloader, model, loss_fn, optimizer, sample_num, train_loss_list, test_loss_list, random_seed, epochs, device)


    mp.spawn(distributed, args=(world_size,parameters), nprocs=world_size, join=True)

