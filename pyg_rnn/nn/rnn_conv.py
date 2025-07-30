# pyg_rnn/nn/rnn_conv.py
from __future__ import annotations

from typing import Type

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from ..utils.pack import sequence_ptr_from_edge_index  # helper described below
from collections import defaultdict


class RNNConv(MessagePassing):
    r"""Applies any *PyTorch* RNN to variable-length, per-node event sequences.

    Parameters
    ----------
    rnn_cls : Type[nn.RNNBase]
        A constructor for an RNN class (`nn.GRU`, `nn.LSTM`, `nn.RNN`, …).
    edge_index : PyG (num_edges, 2) long tensor, representing edges from Entities to Events.
    events: Tensor. Provided for .time only. Not to be saved as a parameter, as the actual 
                Events tensor will be have the same number of entries, but with different values
    time_channel: the parameter name specifies the time of event. Default 'time'.
    in_channels : int
        Size of the input event features  *x*.
    hidden_channels : int
        Hidden size used inside the RNN. Default None, which means the same as `in_channels`.
    num_layers : int, optional
        Number of stacked RNN layers (default: 1).
    bidirectional : bool, optional
        If *True*, use a bidirectional RNN (default: *False*).
    time_mode : str, optional
        Whether to inject time once before multy-layer RNN, after each layer, or never. ['once', 'each', 'never'] (Default 'once')
    time_form: str ['raw', '2d', 'sin', 'poly'], optional. Default is 'raw'. 'poly' means concat [Δt, Δt², log Δt] . '2d' means a Linear([1,2]) layer applied to Δt.
    each_proc: Module, optional
        A module that is applied to each event either before RNN, or before each RNN layer. if defined, overwrites time_form. (Default None)
    **rnn_kwargs
        Extra keyword arguments forwarded verbatim to `rnn_cls`.
    """

    def __init__(
        self,
        rnn_cls: Type[nn.RNNBase],
        edge_index: Tensor,
        events: Tensor, 
        in_channels: int,
        hidden_channels: int = None,
        time_channel: str = "time",
        num_layers: int = 1,
        bidirectional: bool = False,
        time_mode: str = "once",
        time_form: str = "raw",
        each_proc: nn.Module | None = None,
        **rnn_kwargs,
    ):
        super().__init__(aggr="add", node_dim=0)  # aggregation unused but required
        assert each_proc is None or isinstance(each_proc, nn.Module), "each_proc must be a nn.Module"
        assert each_proc is None, "each_proc is not implemented yet"
        assert time_mode in ["once", "each", "never"], "time_mode must be one of ['once', 'each', 'never']"
        assert time_mode != "each", "time_mode='each' is not implemented yet"
        assert time_form in ["raw", "2d", "sin", "poly"], "time_form must be one of ['raw', '2d', 'sin', 'poly']"
        assert time_form != "sin", "time_form='sin' is not implemented yet"
        assert not bidirectional or rnn_cls in [nn.LSTM, nn.GRU], "bidirectional=True is only supported for LSTM and GRU"
        assert not bidirectional, "bidirectional=True is not implemented yet"
        assert edge_index.shape[1] == 2, "edge_index must be a (num_edges, 2) long tensor"

        self.input_dim = in_channels
        self.time_channels = 0 if time_mode == "never" else {'raw': 1, '2d': 2, 'poly': 3}[time_form]
        self.hidden_dim = hidden_channels if hidden_channels is not None else self.input_dim + self.time_channels
        self.time_mode = time_mode
        self.time_form = time_form
        self.each_proc = each_proc
        self.bidirectional = bidirectional

        if time_mode != 'each':
            self.rnn: nn.RNNBase = rnn_cls(
                    input_size=self.input_dim + self.time_channels,
                    hidden_size=self.hidden_dim,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    batch_first=True,  # we’ll pack anyway; batch_first simplifies unpacking
                    **rnn_kwargs,
                )

        self.out_channels = hidden_channels * (2 if bidirectional else 1)

        self.edge_index = edge_index

        run_ids = torch.arange(events.num_nodes, dtype=torch.int64)
        nested_by_entity = defaultdict(list)
        for event_num in range(events.shape[0]):
            entity_ind_tensor = edge_index[0][edge_index[1] == event_num]
        horse_ind_list = horse_ind_tensor.tolist()
        assert len(horse_ind_list) == 1, f"Run {run_num} has {len(horse_ind_list)} Horses, expected 1"
        horse_list  = nested_by_horse[horse_ind_list[0]]
        horse_list.append(run_num)
    print(f"Packed {len(nested_by_horse)} Horses with {sum(len(v) for v in nested_by_horse.values())} Runs") if report else None
    nested_tensors = [torch.tensor(v, dtype=torch.int64) for v in nested_by_horse.values()]
    run_packed = torch.nn.utils.rnn.pack_sequence(nested_tensors, enforce_sorted=False)  
    sub_data['Run', 'packedhorse', 'Run'].edge_index = torch.stack([run_ids, run_packed.data], dim=0)
    sub_data['Run', 'packedhorse', 'Run'].batch_sizes = run_packed.batch_sizes
   

        self.reset_parameters()



    # --------------------------------------------------------------------- #
    #  Public API                                                           #
    # --------------------------------------------------------------------- #
    def reset_parameters(self) -> None:
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: Tensor,                # [num_events, in_channels]
        edge_index: Tensor,       # [2, num_edges]  (src → dst, *strictly* earlier ➜ later)
        num_nodes: int | None = None,
    ) -> Tensor:                  # [num_events, out_channels]
        # 1. Build sequence pointers sorted by dst timestamp
        seq_ptr = sequence_ptr_from_edge_index(edge_index, x.size(0), num_nodes)

        # 2. Reorder events into (batch, time) layout expected by pack()
        x_seq = x[seq_ptr.event_idx]                   # [total_time, in_channels]
        lengths = seq_ptr.lengths.cpu()                # List[int]

        # 3. Pack, run RNN, unpack
        packed: PackedSequence = pack_padded_sequence(
            x_seq, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        y_seq, _ = self.rnn(packed)                    # y_seq.data  → [total_time, H]
        y, _ = pad_packed_sequence(
            PackedSequence(y_seq.data, packed.batch_sizes),
            batch_first=True,
        )                                              # [batch, max_len, H]

        # 4. Map back to original event order
        out = torch.empty(
            (x.size(0), self.out_channels), device=x.device, dtype=y.dtype
        )
        out[seq_ptr.event_idx] = y.view(-1, self.out_channels)

        return out

    # optional convenience so layer behaves like a PyG “conv”
    def message(self, x_j: Tensor) -> Tensor:  # unused
        return x_j
