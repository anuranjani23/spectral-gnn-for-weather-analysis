import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool
from torch_geometric.data import Batch

class SpectralGNN(nn.Module):
    """
    Spectral Graph Neural Network for weather prediction
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                num_layers=3, K=3, dropout=0.1):
        super(SpectralGNN, self).__init__()
        
        self.dropout = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.conv_layers = nn.ModuleList([
            ChebConv(hidden_channels, hidden_channels, K=K) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        for i, conv in enumerate(self.conv_layers):
            identity = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity
            x = self.layer_norms[i](x)
        return self.output_proj(x)

class WeatherGNN(nn.Module):
    """
    Weather prediction model using Spectral GNN with temporal encoding
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 history_steps=3, forecast_steps=1, K=3, num_layers=3):
        super(WeatherGNN, self).__init__()
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps

        self.time_encoder = nn.Linear(1, hidden_channels)
        self.gnn = SpectralGNN(in_channels + hidden_channels, hidden_channels, 
                               hidden_channels, num_layers=num_layers, K=K)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels * forecast_steps)
        )

    def forward(self, history_batches):
        """
        Args:
            history_batches: List of `Batch` objects of length `history_steps`.
                             Each Batch contains node features, edge_index, and batch assignment.

        Returns:
            predictions: Tensor of shape [batch_size, forecast_steps, out_channels]
        """
        gnn_outputs = []

        for t, graph_batch in enumerate(history_batches):
            time_encoding = torch.ones(graph_batch.x.size(0), 1, device=graph_batch.x.device) * (t / self.history_steps)
            time_features = self.time_encoder(time_encoding)
            augmented_x = torch.cat([graph_batch.x, time_features], dim=1)

            node_features = self.gnn(augmented_x, graph_batch.edge_index, graph_batch.batch)
            graph_embedding = global_mean_pool(node_features, graph_batch.batch)  # [batch_size, hidden_channels]
            gnn_outputs.append(graph_embedding)

        gnn_sequence = torch.stack(gnn_outputs, dim=1)  # [batch_size, history_steps, hidden_dim]
        lstm_out, _ = self.lstm(gnn_sequence)
        final_embedding = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        predictions = self.mlp(final_embedding)
        predictions = predictions.view(predictions.size(0), self.forecast_steps, -1)

        return predictions

if __name__ == "__main__":
    # Dummy test
    from torch_geometric.data import Data, Batch

    in_channels = 4
    hidden_channels = 64
    out_channels = 4
    history_steps = 3
    forecast_steps = 1
    num_nodes = 100
    batch_size = 2

    model = WeatherGNN(in_channels, hidden_channels, out_channels, history_steps, forecast_steps)

    # Create dummy batches for history steps
    history_batches = []
    for _ in range(history_steps):
        graphs = []
        for _ in range(batch_size):
            x = torch.randn(num_nodes, in_channels)
            edge_index = torch.randint(0, num_nodes, (2, 300))
            graphs.append(Data(x=x, edge_index=edge_index))
        batch = Batch.from_data_list(graphs)
        history_batches.append(batch)

    out = model(history_batches)
    print(f"Output shape: {out.shape}")  # [batch_size, forecast_steps, out_channels]
