import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import polars as pl
from entsoe_data_load import TimeseriesDataset
import torch_bayesian.vi as vi
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist

batch_size = 32
seq_length = 50
forecast_horizon = 10
sample_num = 4
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
        self.embedding = nn.Linear(input_dim, model_dim).to(device)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, model_dim)).to(device)
        self.transformer = vi.VITransformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, variational_distribution=variational_distribution).to(device)
        self.fc = vi.VILinear(model_dim, 1, variational_distribution=variational_distribution).to(device)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_mask = tgt_mask.to(device)
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
model = TransformerTimeSeries(input_dim, model_dim, num_heads, num_layers, variational_distribution=MeanFieldNormalVarDist(initial_std=1.))
model.return_log_probs(False)
print(f"Using {device} device")
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = loss_fn = vi.MeanSquaredErrorLoss()


def train_model(model, train_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            src = batch_x.unsqueeze(-1)
            tgt = batch_y.unsqueeze(-1)
            tgt_input = torch.cat((torch.zeros_like(tgt[:, :1, :]), tgt[:, :-1, :]), dim=1)  # Teacher forcing
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))

            output = model(src, tgt_input, tgt_mask=tgt_mask, samples = sample_num)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.6f}")


train_model(model, train_dataloader)


def test_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            src = batch_x.unsqueeze(-1)
            tgt_input = torch.zeros(batch_x.size(0), forecast_horizon, 1)  # Start with zeros

            for t in range(forecast_horizon):  # Autoregressive decoding
                tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))
                output = model(src, tgt_input, tgt_mask=tgt_mask, samples = sample_num)
                next_pred = output[:, -1:].unsqueeze(-1)
                tgt_input = torch.cat((tgt_input, next_pred), dim=1)

            predictions.append(tgt_input[:, 1:].squeeze())

            test_loss += loss_fn(predictions, batch_y).item()
    test_loss /= len(test_dataloader)

    print(f"Test Error: Avg loss: {test_loss:>8f} \n")



test_model(model, test_dataloader)
