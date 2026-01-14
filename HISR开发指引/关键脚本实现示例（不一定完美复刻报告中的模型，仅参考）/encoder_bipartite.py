"""hisr.model.encoder_bipartite

Bipartite graph encoder Φ_θ for HISR.

Faithfulness to HISR构建报告V5:
- We model bucket-local operator A_{e,b} as a bipartite graph G_{e,b}
  between key nodes U_b and counter nodes V_{e,b}.
- The encoder performs T layers of *alternating message passing*:
  Counter->Key (C2K) then Key->Counter (K2C).
- Encoder outputs a disentangled representation:
  (z_c, z_v) = Φ_θ(...), where
    z_c: environment-invariant representation used for decoding,
    z_v: environment-specific representation (not directly used in the decoder).

Implementation notes
--------------------
- We avoid requiring third-party packages (e.g., torch_scatter) by using
  `index_add_` for aggregation.
- We provide both a pooled bucket representation (z_c/z_v) and per-key
  representations. The decoder may use z_c and optionally per-key invariant
  embeddings for node-level scoring.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from hisr.data.local_operator import BucketGraph


@dataclass
class EncoderOutput:
    z_c: torch.Tensor  # (d_z,)
    z_v: torch.Tensor  # (d_z,)
    h_key_c: torch.Tensor  # (num_keys, d_h)
    h_key_v: torch.Tensor  # (num_keys, d_h)
    h_ctr: torch.Tensor  # (num_counters, d_h)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BipartiteGNNEncoder(nn.Module):
    """Φ_θ: bipartite GNN encoder with disentanglement heads."""

    def __init__(
        self,
        *,
        d_node: int = 128,
        d_msg: int = 128,
        d_z: int = 128,
        num_layers: int = 3,
        use_gru: bool = True,
        dropout: float = 0.1,
        max_counter_id_buckets: int = 1 << 20,
        max_key_id_buckets: int = 1 << 20,
    ) -> None:
        super().__init__()
        self.d_node = int(d_node)
        self.d_z = int(d_z)
        self.num_layers = int(num_layers)
        self.use_gru = bool(use_gru)

        # Embeddings for structural ids (counter ids are large; use hashing mod a table)
        self.counter_emb = nn.Embedding(max_counter_id_buckets, d_node)
        self.key_emb = nn.Embedding(max_key_id_buckets, d_node)

        # Initial projection for counter values
        self.counter_val_proj = nn.Sequential(
            nn.Linear(2, d_node),
            nn.ReLU(inplace=True),
        )

        # Message MLPs
        self.msg_c2k = MLP(d_node, d_msg, d_node, dropout=dropout)
        self.msg_k2c = MLP(d_node, d_msg, d_node, dropout=dropout)

        # Update functions
        if self.use_gru:
            self.upd_key = nn.GRUCell(d_node, d_node)
            self.upd_ctr = nn.GRUCell(d_node, d_node)
        else:
            self.upd_key = MLP(2 * d_node, d_msg, d_node, dropout=dropout)
            self.upd_ctr = MLP(2 * d_node, d_msg, d_node, dropout=dropout)

        self.norm_key = nn.LayerNorm(d_node)
        self.norm_ctr = nn.LayerNorm(d_node)
        self.drop = nn.Dropout(dropout)

        # Disentanglement projections
        self.proj_key_c = MLP(d_node, d_msg, d_node, dropout=dropout)
        self.proj_key_v = MLP(d_node, d_msg, d_node, dropout=dropout)

        self.pool_to_zc = MLP(d_node, d_msg, d_z, dropout=dropout)
        self.pool_to_zv = MLP(d_node, d_msg, d_z, dropout=dropout)

    @staticmethod
    def _hash_ids(ids: torch.Tensor, buckets: int) -> torch.Tensor:
        # ids: (...,) int64/long
        # Use multiplicative hashing for stability.
        # (Knuth's multiplicative method)
        x = ids.to(torch.int64)
        x = (x * 2654435761) & 0xFFFFFFFFFFFFFFFF
        return (x % buckets).to(torch.long)

    def forward(
        self,
        graph: BucketGraph,
        *,
        key_ids: Optional[torch.Tensor] = None,
        counter_ids_global: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        """Encode a bucket graph.

        Parameters
        ----------
        graph:
            BucketGraph extracted by hisr.data.local_operator.
        key_ids:
            Optional stable integer ids for keys in this bucket (num_keys,).
            If None, we use range(num_keys).
        counter_ids_global:
            Optional global counter ids (num_counters,). If None, we use
            range(num_counters). For UCL/CM sketch, providing global ids helps
            preserve cross-bucket structural consistency.
        """
        device = graph.y.device
        num_keys = graph.num_keys
        num_ctrs = graph.num_counters

        if key_ids is None:
            key_ids = torch.arange(num_keys, device=device, dtype=torch.long)
        if counter_ids_global is None:
            if graph.counter_ids_global is not None:
                counter_ids_global = graph.counter_ids_global.to(device)
            else:
                counter_ids_global = torch.arange(num_ctrs, device=device, dtype=torch.long)

        # Initial node states
        key_ids_h = self._hash_ids(key_ids, self.key_emb.num_embeddings)
        ctr_ids_h = self._hash_ids(counter_ids_global, self.counter_emb.num_embeddings)

        h_key = self.key_emb(key_ids_h)

        y = graph.y
        # counter scalar features: [y, log(1+y)]
        ctr_feat = torch.stack([y, torch.log1p(torch.clamp_min(y, 0.0))], dim=-1)
        h_ctr = self.counter_emb(ctr_ids_h) + self.counter_val_proj(ctr_feat)

        # Edge list
        src_k = graph.edge_index[0]  # (E,)
        dst_c = graph.edge_index[1]  # (E,)

        for _ in range(self.num_layers):
            # ---------------------------
            # Counter -> Key (C2K)
            # ---------------------------
            msg = self.msg_c2k(h_ctr[dst_c])  # (E, d)
            agg = torch.zeros((num_keys, self.d_node), device=device, dtype=h_key.dtype)
            agg.index_add_(0, src_k, msg)
            # normalize by degree
            deg = torch.zeros((num_keys,), device=device, dtype=h_key.dtype)
            deg.index_add_(0, src_k, torch.ones_like(src_k, dtype=h_key.dtype))
            agg = agg / (deg.unsqueeze(-1).clamp_min(1.0))

            if self.use_gru:
                h_key = self.upd_key(self.drop(agg), h_key)
            else:
                h_key = self.upd_key(torch.cat([h_key, self.drop(agg)], dim=-1))
            h_key = self.norm_key(h_key)

            # ---------------------------
            # Key -> Counter (K2C)
            # ---------------------------
            msg2 = self.msg_k2c(h_key[src_k])
            agg2 = torch.zeros((num_ctrs, self.d_node), device=device, dtype=h_ctr.dtype)
            agg2.index_add_(0, dst_c, msg2)
            deg2 = torch.zeros((num_ctrs,), device=device, dtype=h_ctr.dtype)
            deg2.index_add_(0, dst_c, torch.ones_like(dst_c, dtype=h_ctr.dtype))
            agg2 = agg2 / (deg2.unsqueeze(-1).clamp_min(1.0))

            if self.use_gru:
                h_ctr = self.upd_ctr(self.drop(agg2), h_ctr)
            else:
                h_ctr = self.upd_ctr(torch.cat([h_ctr, self.drop(agg2)], dim=-1))
            h_ctr = self.norm_ctr(h_ctr)

        # Disentanglement heads
        h_key_c = self.proj_key_c(h_key)
        h_key_v = self.proj_key_v(h_key)

        # Pool to bucket-level representations
        if num_keys > 0:
            pooled_c = h_key_c.mean(dim=0)
            pooled_v = h_key_v.mean(dim=0)
        else:
            pooled_c = torch.zeros((self.d_node,), device=device, dtype=h_key.dtype)
            pooled_v = torch.zeros((self.d_node,), device=device, dtype=h_key.dtype)

        z_c = self.pool_to_zc(pooled_c)
        z_v = self.pool_to_zv(pooled_v)

        return EncoderOutput(z_c=z_c, z_v=z_v, h_key_c=h_key_c, h_key_v=h_key_v, h_ctr=h_ctr)
