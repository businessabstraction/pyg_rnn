import torch
import pytest
from torch_geometric.data import HeteroData
from pyg_rnn import RNNConv

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
    print("Edge index:", data['entity', 'has_event', 'event'].edge_index.shape)

    # Event sequence IDs (for batching sequences)
    # data['event'].seq_id = torch.randint(0, 3, (num_events,))

    return data

def test_rnnconv_forward(hetero_example):
    conv = RNNConv(torch.nn.GRU, edge_index=hetero_example['entity', 'has_event', 'event'].edge_index,
                   event_time=hetero_example['event'].time, in_channels=8, hidden_channels=16)
    out = conv(
        hetero_example['event'].x
    )

    assert out.shape == (10, 16), "Output tensor shape mismatch."
    # assert hidden.shape[1] == hetero_example['event'].seq_id.unique().size(0), \
    #     "Hidden tensor sequence batch size mismatch."
    

