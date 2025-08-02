import torch
import pytest
from torch_geometric.data import HeteroData
from pyg_rnn import RNNConv

@pytest.fixture
def empty_example():
    data = HeteroData()
    data['event'].x = torch.empty((0, 8))
    data['event'].time = torch.empty((0,))
    data['entity', 'has_event', 'event'].edge_index = torch.empty((2, 0), dtype=torch.long)
    return data

@pytest.fixture
def isolated_nodes_example():
    data = HeteroData()
    data['event'].x = torch.randn((5, 8))
    data['event'].time = torch.arange(5, dtype=torch.float)
    data['entity', 'has_event', 'event'].edge_index = torch.empty((2, 0), dtype=torch.long)
    return data

@pytest.fixture
def variable_size_example():
    data = HeteroData()
    num_events = 50
    data['event'].x = torch.randn((num_events, 8))
    data['event'].time = torch.linspace(0, 10, steps=num_events)
    src = torch.randint(0, 10, (num_events,))
    dst = torch.arange(num_events)
    data['entity', 'has_event', 'event'].edge_index = torch.stack([src, dst])
    return data

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

def test_empty_graph(empty_example):
    with pytest.raises(ValueError, match="RNNConv requires a non-empty edge_index"):
        conv = RNNConv(torch.nn.GRU,
                    edge_index=empty_example['entity', 'has_event', 'event'].edge_index,
                    event_time=empty_example['event'].time,
                    in_channels=8, hidden_channels=16)
        out = conv(empty_example['event'].x)
        assert out.shape == (0, 16), "Empty graph output shape mismatch."

def test_isolated_nodes(isolated_nodes_example):
    with pytest.raises(ValueError, match="RNNConv requires a non-empty edge_index"):
        conv = RNNConv(torch.nn.GRU,
                    edge_index=isolated_nodes_example['entity', 'has_event', 'event'].edge_index,
                    event_time=isolated_nodes_example['event'].time,
                    in_channels=8, hidden_channels=16)
        out = conv(isolated_nodes_example['event'].x)
        assert out.shape == (5, 16), "Isolated nodes output shape mismatch."

def test_variable_sized_graph(variable_size_example):
    conv = RNNConv(torch.nn.GRU,
                   edge_index=variable_size_example['entity', 'has_event', 'event'].edge_index,
                   event_time=variable_size_example['event'].time,
                   in_channels=8, hidden_channels=16)
    out = conv(variable_size_example['event'].x)
    assert out.shape == (50, 16), "Variable-sized graph output shape mismatch."

# def test_gradient_check(hetero_example):
#     conv = RNNConv(torch.nn.GRU,
#                    edge_index=hetero_example['entity', 'has_event', 'event'].edge_index,
#                    event_time=hetero_example['event'].time,
#                    in_channels=8, hidden_channels=8).double()
#     event_x = hetero_example['event'].x.double().requires_grad_(True)

#     def forward_fn(x):
#         return conv(x)

#     from torch.autograd import gradcheck
#     assert gradcheck(forward_fn, (event_x,), eps=1e-6, atol=1e-4), "Gradient check failed."


@pytest.mark.parametrize("time_form", ["raw", "poly", "2d"])
@pytest.mark.parametrize("num_layers", [1, 3])
def test_gradient_check(variable_size_example, time_form, num_layers):
    conv = RNNConv(torch.nn.GRU,
                   edge_index=variable_size_example['entity', 'has_event', 'event'].edge_index,
                   event_time=variable_size_example['event'].time,
                   in_channels=8,
                   hidden_channels=8,
                   num_layers=num_layers,
                   time_form=time_form).double()
    event_x = variable_size_example['event'].x.double().requires_grad_(True)

    def forward_fn(x):
        return conv(x)

    from torch.autograd import gradcheck
    assert gradcheck(forward_fn, (event_x,), eps=1e-6, atol=1e-4), f"Gradient check failed for time_form={time_form}"


