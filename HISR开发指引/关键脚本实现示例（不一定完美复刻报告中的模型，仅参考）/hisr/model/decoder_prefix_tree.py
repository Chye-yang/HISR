"""hisr.model.decoder_prefix_tree

Hierarchical prefix-tree decoder D_θ for HISR.

Faithful mapping to HISR构建报告V5 (Sec. 3.3.3):
- We represent bucket-local unknown frequencies x_{e,b} as a mass distribution
  over keys in a logical bucket.
- We decode by *mass-conserving hierarchical splitting*:
    c^{(0)}(root) -> groups -> ... -> leaves
  where split ratios are produced by a neural scorer and normalized with softmax.
- This ensures the conservation constraint automatically:
    c(parent) = sum_{children} c(child)
  up to numeric precision.

The hierarchy (partitions P^(ℓ)(b)) is constructed from *prefix granularity*:
Coarse (/16) -> Medium (/24) -> Fine (/32 or key-level).

This module contains:
- PrefixTreeSpec: bucket-local hierarchy spec.
- PrefixTreeDecoder: split-ratio parameterization (Eq. (hisr_split) and (hisr_split_mass)).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from hisr.data.prefix import PrefixHierarchy


@dataclass
class TreeNode:
    node_id: int
    level: int  # 0=root, 1=/16, 2=/24, 3=leaf(keys)
    prefix_id: Optional[int]
    key_indices: List[int]  # indices of keys (within bucket) covered by this node


@dataclass
class PrefixTreeSpec:
    """A bucket-local prefix hierarchy specification."""

    num_keys: int
    # nodes grouped by level
    nodes_by_level: List[List[TreeNode]]
    # adjacency
    children: Dict[int, List[int]]
    parent: Dict[int, int]
    # map each key index -> leaf node id
    leaf_of_key: List[int]

    @property
    def depth(self) -> int:
        return len(self.nodes_by_level) - 1  # excluding root?

    @property
    def root(self) -> TreeNode:
        return self.nodes_by_level[0][0]


def build_prefix_tree(
    keys: Sequence[bytes],
    hierarchy: PrefixHierarchy,
    *,
    coarse_level: int = 0,
    medium_level: int = 1,
) -> PrefixTreeSpec:
    """Build a 3-level prefix tree (root -> coarse -> medium -> keys).

    Parameters
    ----------
    keys:
        Keys in this logical bucket (order defines output x_hat ordering).
    hierarchy:
        PrefixHierarchy providing prefix ids.
    coarse_level:
        Index into hierarchy.levels_bits for coarse prefix (default: /16).
    medium_level:
        Index into hierarchy.levels_bits for medium prefix (default: /24).

    Returns
    -------
    PrefixTreeSpec.

    Notes
    -----
    - Leaves are individual keys (level=3). This matches the "fine" resolution
      target in the user's setup (/32).
    - You can change hierarchy.levels_bits or levels to adapt to your dataset.
    """

    num_keys = len(keys)

    # Root node covers all keys.
    next_id = 0
    root = TreeNode(node_id=next_id, level=0, prefix_id=None, key_indices=list(range(num_keys)))
    next_id += 1

    # Group by coarse prefix
    bits_c = hierarchy.levels_bits[coarse_level]
    coarse_groups: Dict[int, List[int]] = {}
    for i, k in enumerate(keys):
        pid = hierarchy.prefix_id(k, bits_c)
        coarse_groups.setdefault(pid, []).append(i)

    coarse_nodes: List[TreeNode] = []
    children: Dict[int, List[int]] = {root.node_id: []}
    parent: Dict[int, int] = {}

    for pid, idxs in sorted(coarse_groups.items(), key=lambda kv: kv[0]):
        n = TreeNode(node_id=next_id, level=1, prefix_id=int(pid), key_indices=idxs)
        next_id += 1
        coarse_nodes.append(n)
        children[root.node_id].append(n.node_id)
        parent[n.node_id] = root.node_id
        children[n.node_id] = []

    # Group by medium prefix inside each coarse
    bits_m = hierarchy.levels_bits[medium_level]
    medium_nodes: List[TreeNode] = []

    for cn in coarse_nodes:
        med_groups: Dict[int, List[int]] = {}
        for i in cn.key_indices:
            pid = hierarchy.prefix_id(keys[i], bits_m)
            med_groups.setdefault(pid, []).append(i)
        for pid, idxs in sorted(med_groups.items(), key=lambda kv: kv[0]):
            n = TreeNode(node_id=next_id, level=2, prefix_id=int(pid), key_indices=idxs)
            next_id += 1
            medium_nodes.append(n)
            children[cn.node_id].append(n.node_id)
            parent[n.node_id] = cn.node_id
            children[n.node_id] = []

    # Leaves: individual keys
    leaf_nodes: List[TreeNode] = []
    leaf_of_key: List[int] = [-1 for _ in range(num_keys)]

    # We attach each key as a leaf under its /24 node.
    # To find parent /24 node: build a lookup from key index -> medium node.
    key_to_med: Dict[int, int] = {}
    for mn in medium_nodes:
        for i in mn.key_indices:
            key_to_med[i] = mn.node_id

    for i in range(num_keys):
        ln = TreeNode(node_id=next_id, level=3, prefix_id=None, key_indices=[i])
        next_id += 1
        leaf_nodes.append(ln)
        leaf_of_key[i] = ln.node_id
        p = key_to_med[i]
        children[p].append(ln.node_id)
        parent[ln.node_id] = p
        children[ln.node_id] = []

    nodes_by_level = [[root], coarse_nodes, medium_nodes, leaf_nodes]
    return PrefixTreeSpec(
        num_keys=num_keys,
        nodes_by_level=nodes_by_level,
        children=children,
        parent=parent,
        leaf_of_key=leaf_of_key,
    )


class PrefixTreeDecoder(nn.Module):
    """Mass-conserving hierarchical decoder D_θ.

    Inputs
    ------
    z_c: (d_z,) environment-invariant bucket representation.
    tree: PrefixTreeSpec describing partitions and parent/child relations.

    Outputs
    -------
    x_hat: (num_keys,) predicted key frequencies within the bucket.
    masses_by_level: dict(level -> tensor of node masses in that level order)

    Implementation
    --------------
    - Root mass is predicted by a positive head: softplus(w^T z_c).
    - Each child node receives a score computed by a shared scorer g_θ.
      For a parent u with children v in child(u):
          α_{u->v} = softmax(score(v))
          c(v) = α_{u->v} * c(u)
    """

    def __init__(
        self,
        *,
        d_z: int = 128,
        d_hidden: int = 128,
        dropout: float = 0.1,
        max_prefix_buckets: int = 1 << 20,
        max_level: int = 8,
    ) -> None:
        super().__init__()
        self.d_z = int(d_z)
        self.prefix_emb = nn.Embedding(max_prefix_buckets, d_hidden)
        self.level_emb = nn.Embedding(max_level, d_hidden)

        self.root_head = nn.Sequential(
            nn.Linear(d_z, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

        # Child scorer: scalar logit for each child.
        # Input: [z_c, child_prefix_emb, level_emb, log(child_size+1)]
        self.child_scorer = nn.Sequential(
            nn.Linear(d_z + d_hidden + d_hidden + 1, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    @staticmethod
    def _hash_prefix(prefix_id: int, buckets: int) -> int:
        # Deterministic hashing for large prefix ids
        x = int(prefix_id) & 0xFFFFFFFFFFFFFFFF
        x = (x * 11400714819323198485) & 0xFFFFFFFFFFFFFFFF
        return int(x % buckets)

    def _node_feature(self, node: TreeNode) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (prefix_emb, level_emb, size_scalar) for a node."""
        # prefix embedding: only meaningful for internal prefix nodes (level 1/2).
        if node.prefix_id is None:
            pid_h = 0
        else:
            pid_h = self._hash_prefix(node.prefix_id, self.prefix_emb.num_embeddings)
        pe = self.prefix_emb(torch.tensor(pid_h, device=self.prefix_emb.weight.device))
        le = self.level_emb(torch.tensor(min(node.level, self.level_emb.num_embeddings - 1), device=pe.device))
        size = torch.tensor([float(len(node.key_indices))], device=pe.device)
        size = torch.log1p(size)
        return pe, le, size

    def forward(
        self,
        z_c: torch.Tensor,
        tree: PrefixTreeSpec,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        if z_c.ndim != 1:
            raise ValueError(f"z_c must be 1D (d_z,), got {z_c.shape}")
        device = z_c.device

        # Store masses per node_id
        mass: Dict[int, torch.Tensor] = {}

        # Root mass c(root)
        c_root = torch.nn.functional.softplus(self.root_head(z_c)).squeeze(-1)
        mass[tree.root.node_id] = c_root

        # Level-wise arrays for hierarchical loss/debugging
        masses_by_level: Dict[int, torch.Tensor] = {0: torch.stack([c_root], dim=0)}

        # Traverse levels 0->1->2->3
        for lvl in range(0, len(tree.nodes_by_level) - 1):
            next_lvl = lvl + 1
            lvl_nodes = tree.nodes_by_level[lvl]

            # We will fill next level masses in the declared order
            next_masses: List[torch.Tensor] = []

            # We'll compute masses for children of each node
            for parent_node in lvl_nodes:
                parent_mass = mass[parent_node.node_id]
                child_ids = tree.children.get(parent_node.node_id, [])
                if not child_ids:
                    continue

                # Build logits for each child
                logits: List[torch.Tensor] = []
                for cid in child_ids:
                    child_node = _node_by_id(tree, cid)
                    pe, le, size = self._node_feature(child_node)
                    feat = torch.cat([z_c, pe.to(device), le.to(device), size.to(device)], dim=0)
                    logit = self.child_scorer(feat).squeeze(-1)
                    logits.append(logit)

                logit_vec = torch.stack(logits, dim=0)
                alpha = torch.softmax(logit_vec, dim=0)

                # Assign masses to children
                for a, cid in zip(alpha, child_ids):
                    mass[cid] = a * parent_mass

            # Collect masses for next level in the order of tree.nodes_by_level[next_lvl]
            for node in tree.nodes_by_level[next_lvl]:
                next_masses.append(mass[node.node_id])
            masses_by_level[next_lvl] = torch.stack(next_masses, dim=0)

        # Leaves are level=3 nodes, each corresponds to exactly one key index.
        x_hat = torch.zeros((tree.num_keys,), device=device)
        for key_idx, leaf_id in enumerate(tree.leaf_of_key):
            x_hat[key_idx] = mass[leaf_id]

        return x_hat, masses_by_level


def _node_by_id(tree: PrefixTreeSpec, node_id: int) -> TreeNode:
    # Linear scan is fine for typical bucket sizes (L up to a few thousand).
    # If you need larger L, build an id->node dict.
    for lvl_nodes in tree.nodes_by_level:
        for n in lvl_nodes:
            if n.node_id == node_id:
                return n
    raise KeyError(f"node_id {node_id} not found")
