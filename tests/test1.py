import torch
import pytest
from torch_geometric.data import HeteroData
from pyg_rnn.nn import RNNConv

@pytest.fixture
def hetero_example():
    data = HeteroData()

    # Entities
    num_entities = 4
    entity_dim = 8
    data['entity'].x = torch.randn(num_entities, entity_dim)

    # Events
    num_events = 10
    event_dim = 8
    data['event'].x = torch.randn(num_events, event_dim)
    data['event'].time = torch.arange(num_events, dtype=torch.float)

    # Define Entity → Event edges
    src = torch.randint(0, num_entities, (num_events,))
    dst = torch.arange(num_events)
    data['entity', 'has_event', 'event'].edge_index = torch.stack([src, dst])

    # Event sequence IDs (for batching sequences)
    # data['event'].seq_id = torch.randint(0, 3, (num_events,))

    return data

def test_rnnconv_forward(data):
    conv = RNNConv(torch.nn.GRU, edge_index=data['entity', 'has_event', 'event'].edge_index,
                   event_time=data['event'].time, in_channels=8, hidden_channels=16)
    out, hidden = conv(
        data['event'].x, 
        edge_index=data['entity', 'has_event', 'event'].edge_index,
        event_time=data['event'].time
    )

    assert out.shape == (10, 16), "Output tensor shape mismatch."
    # assert hidden.shape[1] == data['event'].seq_id.unique().size(0), \
    #     "Hidden tensor sequence batch size mismatch."
    
data = hetero_example()
test_rnnconv_forward(data)
