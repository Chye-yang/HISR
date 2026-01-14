"""hisr.data.local_operator

Bucket-local operators and bipartite graph extraction for HISR.

Given a sketch view (Count-Min style) with depth d and width w, we have a global
counter vector y in R^{d*w}. For a candidate key set, the global measurement
operator A is a sparse binary matrix: each key connects to exactly d counters
(one per row).

HISR (HISR构建报告V5) operates bucket-wise. For each environment e and bucket b,
we extract:
- bucket-local observations: y_{e,b}
- bucket-local operator: A_{e,b}, represented as a sparse edge list of a
  bipartite graph G_{e,b}=(U_b, V_{e,b}, E_{e,b})

Important implementation note
----------------------------
If we only include counters touched by keys in bucket b, these counters may also
contain contributions from keys in other buckets. HISR's self-supervised losses
(as in Eq.(\ref{eq:hisr_base_risk})) treat this as structured noise. For more
accurate bucket-wise residuals, you can optionally run a residual subtraction
schedule (not implemented here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class BucketGraph:
    """A bucket-local bipartite graph instance."""

    # Observed counter values for local counter nodes (|V|,)
    y: torch.Tensor
    # Edge index in COO format: shape (2, E)
    # edge_index[0] = key_node_id in [0,|U|-1]
    # edge_index[1] = counter_node_id in [0,|V|-1]
    edge_index: torch.Tensor
    # Optional edge weights (E,)
    edge_weight: Optional[torch.Tensor]

    num_keys: int
    num_counters: int

    # Useful for debugging / analysis
    counter_ids_global: Optional[torch.Tensor] = None
    key_ids: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "BucketGraph":
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.to(device)
        if self.counter_ids_global is not None:
            self.counter_ids_global = self.counter_ids_global.to(device)
        if self.key_ids is not None:
            self.key_ids = self.key_ids.to(device)
        return self


def flatten_counters(cm_matrix: np.ndarray) -> np.ndarray:
    """Flatten a (depth,width) counter matrix into (depth*width,)"""
    if cm_matrix.ndim != 2:
        raise ValueError(f"cm_matrix must be 2D, got shape={cm_matrix.shape}")
    return cm_matrix.reshape(-1)


def extract_bucket_bipartite(
    *,
    cm_sketch,
    keys_in_bucket: Sequence[bytes],
    device: Optional[torch.device] = None,
    counter_value_dtype: torch.dtype = torch.float32,
    include_edge_weight: bool = False,
) -> BucketGraph:
    """Extract bucket-local bipartite graph from a CM-sketch-like object.

    Requirements on `cm_sketch`:
    - attributes: depth:int, width:int, matrix:np.ndarray(shape=(depth,width))
    - method: hash(key:bytes, col:int)->int

    Returns
    -------
    BucketGraph with:
    - y: local counter observations
    - edge_index: (2,E) edges from keys to counters
    """

    depth = int(cm_sketch.depth)
    width = int(cm_sketch.width)
    mat = np.asarray(cm_sketch.matrix)
    if mat.shape != (depth, width):
        raise ValueError(f"Unexpected matrix shape {mat.shape}, expected {(depth,width)}")

    # Build global counter indices touched by this bucket
    # Each key contributes exactly `depth` edges.
    key_count = len(keys_in_bucket)
    if key_count == 0:
        # Empty bucket: return a dummy graph
        y = torch.zeros((0,), dtype=counter_value_dtype)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return BucketGraph(
            y=y,
            edge_index=edge_index,
            edge_weight=None,
            num_keys=0,
            num_counters=0,
            counter_ids_global=torch.zeros((0,), dtype=torch.long),
            key_ids=torch.zeros((0,), dtype=torch.long),
        )

    global_ctr_ids: List[int] = []
    src_key: List[int] = []
    dst_ctr: List[int] = []

    for j, k in enumerate(keys_in_bucket):
        for i in range(depth):
            pos = int(cm_sketch.hash(k, i))
            gid = i * width + pos
            global_ctr_ids.append(gid)
            src_key.append(j)
            dst_ctr.append(gid)

    # Deduplicate global counter ids to form local counter node set V_{e,b}
    uniq_global = np.unique(np.array(global_ctr_ids, dtype=np.int64))
    # Mapping global->local
    gid_to_lid = {int(g): idx for idx, g in enumerate(uniq_global.tolist())}
    dst_ctr_local = [gid_to_lid[int(g)] for g in dst_ctr]

    # Build local y from global counters
    flat = flatten_counters(mat)
    y_local = flat[uniq_global]

    edge_index = torch.tensor([src_key, dst_ctr_local], dtype=torch.long)
    y = torch.tensor(y_local, dtype=counter_value_dtype)

    edge_weight = None
    if include_edge_weight:
        edge_weight = torch.ones((edge_index.shape[1],), dtype=counter_value_dtype)

    bg = BucketGraph(
        y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_keys=key_count,
        num_counters=len(uniq_global),
        counter_ids_global=torch.tensor(uniq_global, dtype=torch.long),
    )

    if device is not None:
        bg.to(device)
    return bg


def predict_counters_from_x(
    x_hat: torch.Tensor,
    graph: BucketGraph,
) -> torch.Tensor:
    """Compute \hat{y} = A x_hat for a bucket graph via scatter-add.

    x_hat: (num_keys,)
    returns y_hat: (num_counters,)
    """
    if x_hat.ndim != 1:
        raise ValueError(f"x_hat must be 1D, got {x_hat.shape}")
    if x_hat.shape[0] != graph.num_keys:
        raise ValueError(f"x_hat len mismatch: {x_hat.shape[0]} vs num_keys={graph.num_keys}")

    src = graph.edge_index[0]
    dst = graph.edge_index[1]

    msg = x_hat[src]
    if graph.edge_weight is not None:
        msg = msg * graph.edge_weight

    y_hat = torch.zeros((graph.num_counters,), device=x_hat.device, dtype=x_hat.dtype)
    y_hat.index_add_(0, dst, msg)
    return y_hat


def sparse_incidence_torch(
    graph: BucketGraph,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a torch sparse COO incidence matrix A_{e,b}.

    Shape: (num_counters, num_keys).

    This is mainly for debugging or for alternative loss implementations.
    """
    # COO indices: (row=counter, col=key)
    rows = graph.edge_index[1]
    cols = graph.edge_index[0]
    idx = torch.stack([rows, cols], dim=0)
    if graph.edge_weight is None:
        vals = torch.ones((idx.shape[1],), dtype=dtype, device=idx.device)
    else:
        vals = graph.edge_weight.to(dtype)

    return torch.sparse_coo_tensor(
        idx,
        vals,
        size=(graph.num_counters, graph.num_keys),
        device=idx.device,
        dtype=dtype,
    ).coalesce()
