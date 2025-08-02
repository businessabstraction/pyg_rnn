# pyg_rnn/nn/rnn_conv.py
from __future__ import annotations

from typing import Type

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter

from pyg_rnn.utils.pack import sequence_ptr_from_edge_index
from collections import defaultdict


class RNNConv(MessagePassing):
    r"""Applies a PyTorch RNN (e.g., GRU, LSTM, RNN) to variable-length sequences of node-associated events within a PyTorch Geometric graph structure.

    Parameters
    ----------
    rnn_cls : Type[nn.RNNBase]
        Constructor for a PyTorch RNN module, e.g., `nn.GRU`, `nn.LSTM`, or `nn.RNN`.
    edge_index : torch.LongTensor [2, num_edges]
        Tensor representing edges from entity nodes (sources) to event nodes (targets). Shape: `[2, num_edges]`.
    event_time : torch.Tensor [num_events]
        Tensor containing timestamps associated with each event node. Events must be pre-sorted chronologically.
    in_channels : int
        Dimension of input feature vectors for event nodes.
    hidden_channels : int, optional
        Hidden dimension used within the RNN. Defaults to `in_channels` if not provided.
    num_layers : int, optional
        Number of stacked RNN layers (default: `1`).
    edge_reverse : bool, optional
        If `True`, reverses edge direction in `edge_index` (default: `False`).
    bidirectional : bool, optional
        If `True`, utilizes a bidirectional RNN (default: `False`).
    time_mode : {'once', 'each', 'never'}, optional
        Specifies when temporal features are injected relative to RNN layers. Currently supports `'once'` (default) and `'never'`.
    time_form : {'raw', '2d', 'poly'}, optional
        Method to encode temporal deltas:
        - `'raw'`: Uses Δt directly.
        - `'2d'`: Projects Δt via a linear layer from 1→2 dimensions.
        - `'poly'`: Concatenates `[Δt, Δt², log(Δt)]`.
        Default is `'raw'`.
    each_proc : nn.Module, optional
        Custom module applied to events before RNN processing. Overrides `time_form` if provided (not implemented yet).
    **rnn_kwargs : dict
        Additional keyword arguments forwarded directly to the underlying RNN constructor.

    Attributes
    ----------
    out_channels : int
        Output feature dimension, determined by `hidden_channels` and RNN directionality.
    rnn : nn.RNNBase
        The underlying PyTorch RNN instance.
    encode2d : nn.Linear or None
        Linear transformation layer used if `time_form='2d'`; otherwise, None.
    sos_vector : nn.Parameter
        Learnable vector used as a temporal encoding placeholder for events without preceding timestamps.

    """

    def __init__(
        self,
        rnn_cls: Type[nn.RNNBase],
        edge_index: Tensor,
        event_time: Tensor, 
        in_channels: int,
        hidden_channels: int = None,
        num_layers: int = 1,
        edge_reverse: bool = False,
        bidirectional: bool = False,
        time_mode: str = "once",
        time_form: str = "raw",
        each_proc: nn.Module | None = None,
        **rnn_kwargs,
    ):
        """Initializes internal components, validates parameters, and prepares tensors for RNN processing.

        Raises
        ------
        ValueError
            If provided `edge_index` tensor is empty or incorrectly shaped.
        AssertionError
            If `time_mode`, `time_form`, or other parameters are not supported or invalid.
        """
        super().__init__(aggr="add", node_dim=0)  # aggregation unused but required
        assert each_proc is None or isinstance(each_proc, nn.Module), "each_proc must be a nn.Module"
        assert each_proc is None, "each_proc is not implemented yet"
        assert time_mode in ["once", "each", "never"], "time_mode must be one of ['once', 'each', 'never']"
        assert time_mode != "each", "time_mode='each' is not implemented yet"
        assert time_form in ["raw", "2d", "sin", "poly"], "time_form must be one of ['raw', '2d', 'sin', 'poly']"
        assert time_form != "sin", "time_form='sin' is not implemented yet"
        assert not bidirectional or rnn_cls in [nn.LSTM, nn.GRU], "bidirectional=True is only supported for LSTM and GRU"
        assert not bidirectional, "bidirectional=True is not implemented yet"
        assert edge_index.shape[0] == 2, "edge_index must be a (num_edges, 2) long tensor"
        if edge_index.numel() == 0:
            raise ValueError("RNNConv requires a non-empty edge_index, but received an empty tensor.")

        print(f"Creating RNNConv with {rnn_cls.__name__} and edge_index shape {edge_index.shape} and event_time shape {event_time.shape}")

        self.input_dim = in_channels
        self.time_dim = 0 if time_mode == "never" else {'raw': 1, '2d': 2, 'poly': 3}[time_form]
        self.hidden_dim = hidden_channels if hidden_channels is not None else self.input_dim + self.time_channels
        self.time_mode = time_mode
        self.time_form = time_form
        self.each_proc = each_proc
        self.edge_reverse = edge_reverse
        self.bidirectional = bidirectional

        self.time_dim = {'raw': 1, '2d': 2, 'poly': 3}[time_form]

        self.sos_vector = nn.Parameter(torch.zeros(self.time_dim), requires_grad=True)
        self.time_data = (torch.empty_like(event_time)).float()

        if time_mode != 'each':
            self.rnn: nn.RNNBase = rnn_cls(
                    input_size=self.input_dim + self.time_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    batch_first=True,  # we’ll pack anyway; batch_first simplifies unpacking
                    **rnn_kwargs,
                )
        print(self.rnn)
        self.out_channels = hidden_channels * (2 if bidirectional else 1)

        self.edge_index = edge_index.flip(0) if edge_reverse else edge_index
        assert torch.max(self.edge_index[1]) < event_time.shape[0], f"Edge index max {torch.max(self.edge_index[1])} contains out-of-bounds indices for event_time < {event_time.shape[0]}"

        run_ids = torch.arange(event_time.shape[0], dtype=torch.int64)
        nested_by_entity = defaultdict(list)
        for event_num in range(event_time.shape[0]):
            present_event_time = event_time[event_num]
            entity_ind_tensor = self.edge_index[0][self.edge_index[1] == event_num]
            entity_ind_list = entity_ind_tensor.tolist()
            assert len(entity_ind_list) == 1, f"Event {event_num} has {len(entity_ind_list)} Entities, expected 1"
            entity_list  = nested_by_entity[entity_ind_list[0]]
            prev_event_time = entity_list[-1] if entity_list else None
            entity_list.append(event_num)
            time_delta = present_event_time - prev_event_time if prev_event_time is not None else torch.nan
            self.time_data[event_num] = float(time_delta) 

        self.nanmask = torch.isnan(self.time_data)
        nested_tensors = [torch.tensor(v, dtype=torch.int64) for v in nested_by_entity.values()]
        events_packed = torch.nn.utils.rnn.pack_sequence(nested_tensors, enforce_sorted=False)
        self.perm = events_packed.data
        self.batch_sizes = events_packed.batch_sizes
        

        self.encode2d = torch.nn.Linear(1, 2) if time_form == "2d" else None

        self.reset_parameters()



    # --------------------------------------------------------------------- #
    #  Public API                                                           #
    # --------------------------------------------------------------------- #
    def reset_parameters(self) -> None:
        """Initializes or resets parameters of the internal RNN and temporal encoding layers."""
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        if self.encode2d is not None:
            nn.init.xavier_uniform_(self.encode2d.weight)
            nn.init.zeros_(self.encode2d.bias)

    def forward(
        self,
        x: Tensor,                # [num_events, in_channels]
    ) -> Tensor:                  # [num_events, out_channels]
        """Executes a forward pass of the RNN convolution, incorporating temporal features.

        Parameters
        ----------
        x : torch.Tensor [num_events, in_channels]
            Input tensor containing event node features.

        Returns
        -------
        torch.Tensor [num_events, out_channels]
            Tensor containing the processed event node features after applying the RNN-based convolution.

        Notes
        -----
        This method internally manages dtype consistency to ensure compatibility with PyTorch autograd mechanisms.
        """
        # 1. Build sequence pointers sorted by dst timestamp
        current_dtype = next(self.parameters()).dtype
        the_time__data = self.time_data.to(dtype=current_dtype)  # [num_events]

        if self.time_form == 'raw':
            time_delta = the_time__data.unsqueeze(-1)  # [num_events, 1]
        elif self.time_form == 'poly':
            safe_time_data = torch.clamp(the_time__data, min=1e-3)
            time_delta = torch.stack((safe_time_data, safe_time_data ** 2, torch.log(safe_time_data)), dim=-1)
        elif self.time_form == '2d':
            time_delta = self.encode2d(the_time__data.unsqueeze(-1))  # [num_events, 2]
        time_delta[self.nanmask] = self.sos_vector

        #time_delta = time_delta if x.dtype == torch.float else time_delta.to(dtype=x.dtype)

        add_time = torch.cat((x, time_delta), dim=-1)

        event_prepacked = add_time[self.perm]

        run_packed = torch.nn.utils.rnn.PackedSequence(event_prepacked, batch_sizes=self.batch_sizes.cpu().long())  # [1, num_runs, RUN_POST_COMP]

        run_grued_packed = (self.rnn(run_packed))[0]  # [1, num_runs, RUN_POST_COMP]

        run_grued = torch.empty_like(run_grued_packed.data)

        #run_grued[self.perm] = run_grued_packed.data
        run_grued = scatter(run_grued_packed.data, self.perm, dim=0, dim_size=x.size(0), reduce='sum')


        return run_grued

