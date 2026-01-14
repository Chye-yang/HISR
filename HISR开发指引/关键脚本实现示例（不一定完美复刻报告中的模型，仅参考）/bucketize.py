"""hisr.data.bucketize

Logical bucketing for HISR.

In HISR (HISR构建报告V5), we keep UCL-style "logical buckets" as the scalability
interface: we decode keys bucket-by-bucket with shared parameters.

This module partitions a *candidate key list* into B buckets of (approximately)
fixed length L, and provides stable key<->(bucket,offset) mappings.

Design principles
-----------------
- Deterministic partitioning: allow reproducible experiments.
- Pluggable strategies: simple sorted chunking, hash-based, or prefix-aware.
- Keep raw keys untouched (bytes); store derived integer ids only for indexing.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from .prefix import PrefixHierarchy


BucketStrategy = Literal["sorted", "hash", "prefix"]


def _key_to_int_for_sort(key: bytes, byteorder: str = "little") -> int:
    return int.from_bytes(key, byteorder=byteorder, signed=False)


@dataclass
class BucketIndex:
    """Indexing structure for logical buckets."""

    bucket_len: int
    buckets: List[List[bytes]]
    key_to_pos: Dict[bytes, Tuple[int, int]]  # key -> (bucket_id, offset)

    @property
    def num_buckets(self) -> int:
        return len(self.buckets)

    def bucket_keys(self, b: int) -> List[bytes]:
        return self.buckets[b]

    def locate(self, key: bytes) -> Tuple[int, int]:
        return self.key_to_pos[key]


def build_buckets(
    keys: Sequence[bytes],
    bucket_len: int,
    strategy: BucketStrategy = "sorted",
    *,
    key_sort_byteorder: str = "little",
    hash_seed: int = 0,
    prefix_hierarchy: Optional[PrefixHierarchy] = None,
    prefix_level: int = 0,
) -> BucketIndex:
    """Partition keys into logical buckets.

    Parameters
    ----------
    keys:
        Candidate keys to be decoded by HISR.
    bucket_len:
        Target number of keys per bucket (L in the report).
    strategy:
        - "sorted": sort by integer value then chunk.
        - "hash": stable hashing then bucket assignment.
        - "prefix": group by prefix at `prefix_level`, then chunk within each group.
    """

    if bucket_len <= 0:
        raise ValueError("bucket_len must be positive")

    keys_list = list(keys)

    if strategy == "sorted":
        keys_list.sort(key=lambda k: _key_to_int_for_sort(k, key_sort_byteorder))

    elif strategy == "hash":
        # Stable assignment: use python's hash but seeded through a simple xor.
        # NOTE: python's hash is salted per-process; we implement our own.
        def h(b: bytes) -> int:
            x = 1469598103934665603 ^ hash_seed  # FNV offset basis
            for ch in b:
                x ^= ch
                x *= 1099511628211
                x &= 0xFFFFFFFFFFFFFFFF
            return x

        keys_list.sort(key=h)

    elif strategy == "prefix":
        if prefix_hierarchy is None:
            raise ValueError("prefix strategy requires prefix_hierarchy")
        # Sort by (prefix, key) so same prefix groups are contiguous.
        bits = prefix_hierarchy.levels_bits[prefix_level]
        keys_list.sort(
            key=lambda k: (
                prefix_hierarchy.prefix_id(k, bits),
                _key_to_int_for_sort(k, key_sort_byteorder),
            )
        )

    else:
        raise ValueError(f"Unknown strategy={strategy}")

    buckets: List[List[bytes]] = []
    for i in range(0, len(keys_list), bucket_len):
        buckets.append(keys_list[i : i + bucket_len])

    key_to_pos: Dict[bytes, Tuple[int, int]] = {}
    for b, ks in enumerate(buckets):
        for j, k in enumerate(ks):
            key_to_pos[k] = (b, j)

    return BucketIndex(bucket_len=bucket_len, buckets=buckets, key_to_pos=key_to_pos)


def bucketize_subset(
    bucket_index: BucketIndex,
    keys_subset: Iterable[bytes],
) -> Dict[int, List[bytes]]:
    """Group a subset of keys into buckets using an existing BucketIndex."""
    out: Dict[int, List[bytes]] = {}
    for k in keys_subset:
        b, _ = bucket_index.locate(k)
        out.setdefault(b, []).append(k)
    return out
