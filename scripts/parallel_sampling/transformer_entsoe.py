import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import polars as pl
import os
import torch.distributed as dist
from entsoe_data_load import TimeseriesDataset
import torch_bayesian.vi as vi
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from timing_utils import cuda_time_function, print_cuda_timing_summary

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

rank = int(os.environ["SLURM_PROCID"])
local_rank = int(os.environ["SLURM_LOCALID"])
world_size = int(os.environ["SLURM_NTASKS"])
set_device = "cuda:" + str(local_rank)
torch.device(set_device)
sampling_state = None
batch_size = 32
seq_length = 50
forecast_horizon = 10
df = pl.read_csv("data/ENTSOEEnergyLoads/de.csv",
                     dtypes={"start": pl.Datetime, "end": pl.Datetime, "load": pl.Float32},
                     )
x = df["load"]
x = x.fill_null(strategy="backward")
normalized_x = (x - x.mean()) / x.std()
x_tensor = normalized_x.to_torch()
data_train, data_test = x_tensor[: int(len(x) * 0.7)], x_tensor[int(len(x) * 0.7):]
dataset_train = TimeseriesDataset(data_train, seq_length, forecast_horizon)
dataset_test = TimeseriesDataset(data_test, seq_length, forecast_horizon)
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


# Transformer Model for Time Series Forecasting
class TransformerTimeSeries(vi.VIModule):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, variational_distribution):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, model_dim))
        self.transformer = vi.VITransformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, variational_distribution=variational_distribution)
        self.fc = vi.VILinear(model_dim, 1, variational_distribution=variational_distribution)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        src, tgt = src.permute(1, 0, 2), tgt.permute(1, 0, 2)
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        output = output.permute(1, 0, 2)  # Convert back to (batch, seq_len, feature)
        return self.fc(output).squeeze(-1)


# Helper function to generate masks
def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask


# Model parameters
input_dim = 1
model_dim = 64
num_heads = 4
num_layers = 2
random_seed = 42
all_sample_num = 1
model = TransformerTimeSeries(input_dim, model_dim, num_heads, num_layers, variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
model.return_log_probs(False)
if rank==0:
    print(f"Using {device} device")
    print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = loss_fn = vi.MeanSquaredErrorLoss()
sample_num = int(all_sample_num / world_size)
    
setup(rank, world_size)

torch.manual_seed(random_seed)


def train_model(model, train_loader,sampling_state, epochs=5):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Switch to process specific randomness
            regular_state = torch.get_rng_state()
            if sampling_state == None:
                torch.manual_seed(rank)
            else:
                torch.set_rng_state(sampling_state)

            src = batch_x.unsqueeze(-1).to(device)
            tgt = batch_y.unsqueeze(-1)
            tgt_input = (torch.cat((torch.zeros_like(tgt[:, :1, :]), tgt[:, :-1, :]), dim=1)).to(device)  # Teacher forcing
            tgt_mask = (generate_square_subsequent_mask(tgt_input.size(1))).to(device)

            output = model(src, tgt_input, tgt_mask=tgt_mask, samples = sample_num)
            # Switch to general randomness
            sampling_state = torch.get_rng_state()
            torch.set_rng_state(regular_state)
            loss = criterion(output, batch_y)
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
            epoch_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.6f}")
    return sampling_state


sampling_state = train_model(model, train_dataloader, sampling_state)


def test_model(model, test_loader, sampling_state):
    model.eval()
    test_loss = 0.0
    predictions, actuals = [], []
    num_batches = len(test_loader)
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            regular_state = torch.get_rng_state()
            if sampling_state == None:
                torch.manual_seed(rank)
            else:
                torch.set_rng_state(sampling_state)
            src = batch_x.unsqueeze(-1).to(device)
            tgt_input = (torch.zeros(batch_x.size(0), forecast_horizon, 1)).to(device)  # Start with zeros

            for t in range(forecast_horizon):  # Autoregressive decoding
                tgt_mask = (generate_square_subsequent_mask(tgt_input.size(1))).to(device)
                output = model(src, tgt_input, tgt_mask=tgt_mask, samples = sample_num)
                #next_pred = output[:, -1:].unsqueeze(-1)
                #tgt_input = torch.cat((tgt_input, next_pred), dim=1)

            #predictions.append(tgt_input[:, 1:].squeeze())
            sampling_state = torch.get_rng_state()
            torch.set_rng_state(regular_state)
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            if rank == 0:
                output /= world_size
                test_loss += loss_fn(output, batch_y).item()
    if rank == 0:
        test_loss /= num_batches

        print(
            f"Test Error: Avg loss: {test_loss:>8f} \n"
        )

   



test_model(model, test_dataloader, sampling_state)
cleanup()
print_cuda_timing_summary()
