# pyg-rnnconv

A PyTorch Geometric RNN-based Convolution layer for temporal graph processing.

## Installation

```bash
pip install pyg-rnnconv
```
## Example usage
```python
from pyg_rnn import RNNConv
import torch
from torch_geometric.data import Data

conv = RNNConv(torch.nn.GRU, edge_index=edge_index, event_time=event_time, in_channels=16, hidden_channels=32)

x = torch.randn((num_events, 16))
output = conv(x)
print(output.shape)
```

## Why pyg-rnn
This Module allows creation of RNN layers within a PyG model without going through a padded tensor. Itt works or makes swence for heterogeneous graphs only, and assumes that the model has:
- A tensor of Events, which has a time atribute, **sorted by time ascending**
- A tensor of Entities, without time attribute
- A one-to-many relationships between Entities and Events
Creating a padded tensor can be very prohibitive for a dataset with big lead and long tail For example, it you have 1 million groups of events, where most groups have 3-5 events (long take), but some have about 1000, you will need to create a padded tensor $1000000*1,000*n$, where $n$ is the number of properties within each event. That can be 100 times more space than needed 