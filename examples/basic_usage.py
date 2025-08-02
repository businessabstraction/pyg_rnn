import torch
from torch_geometric.data import HeteroData
from pyg_rnn.nn import RNNConv

# Create example data
data = HeteroData()
data['event'].x = torch.randn(20, 8)
data['event'].seq_id = torch.tensor([0]*10 + [1]*10)
data['event'].time = torch.arange(20)

# Initialize model
model = RNNConv(
    rnn_cls=torch.nn.GRU,
    edge_index=data['event'].edge_index,
    event_time=data['event'].time,
    in_channels=8,
    hidden_channels=16
)

# Forward pass
output, hidden = model(data)

print("Output shape:", output.shape)
print("Hidden shape:", hidden.shape)
