import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=4, num_layers=1):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        encoded = self.encoder(self.input_projection(x))
        return self.head(encoded[:, -1, :]).squeeze(-1)


def train_loop(model, X, y, epochs=20, lr=1e-3, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    n_samples = X_tensor.shape[0]

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            predictions = model(X_tensor[indices])
            loss = loss_fn(predictions, y_tensor[indices])
            loss.backward()
            optimizer.step()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X, dtype=torch.float32))
    return predictions.cpu().numpy()
