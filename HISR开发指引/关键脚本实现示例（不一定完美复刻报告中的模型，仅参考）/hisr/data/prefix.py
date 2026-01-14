"""hisr.data.prefix

This module implements HISR's *hierarchical key-space representation*.

In HISR, "scale" corresponds to *prefix granularity* over the key space.
When keys are IPv4 addresses, this aligns with subnet prefixes (/16, /24, /32).
When keys are opaque 64-bit identifiers (as in many UCL-sketch datasets), we
still can define a hierarchy via *bit prefixes* of the integer key.

Key design goals (faithful to HISR构建报告V5):
- Provide a consistent hierarchy builder that can generate L1/L2/L3 candidates.
- Provide a ground-truth aggregator for prefix-level metrics.
- Avoid entangling with training code: this is purely data transformation.

Note:
- UCL-sketch stores keys as raw bytes and hashes them directly. We therefore
  preserve the raw bytes as the canonical key, and only derive prefix ids for
  hierarchical grouping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple


KeyMode = Literal[
    "u64",  # treat key as unsigned integer of arbitrary byte-length
    "ipv4_src",  # treat first 4 bytes as IPv4
    "ipv4_dst",  # treat last 4 bytes as IPv4 (for 8B keys)
]


def _to_uint(key: bytes, byteorder: str = "little") -> int:
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(f"key must be bytes, got {type(key)}")
    return int.from_bytes(bytes(key), byteorder=byteorder, signed=False)


def _to_ipv4_int(key: bytes, which: Literal["src", "dst"], byteorder: str = "big") -> int:
    """Extract a 32-bit IPv4 integer from key bytes.

    - `which='src'`: use first 4 bytes
    - `which='dst'`: use last 4 bytes

    We default to big-endian for IPv4 to match standard dotted-quad order.
    If your dataset stores IPv4 in little-endian, set `byteorder='little'`.
    """
    b = bytes(key)
    if len(b) < 4:
        raise ValueError(f"Need >=4 bytes to parse IPv4, got len={len(b)}")
    if which == "src":
        part = b[:4]
    else:
        part = b[-4:]
    return int.from_bytes(part, byteorder=byteorder, signed=False)


def ipv4_to_str(ip: int) -> str:
    ip = int(ip) & 0xFFFFFFFF
    return ".".join(str((ip >> (8 * k)) & 0xFF) for k in [3, 2, 1, 0])


@dataclass(frozen=True)
class PrefixSpec:
    """A single prefix spec.

    For IPv4, `bits` is among {16,24,32}. For u64 keys, choose any increasing
    sequence up to key_bitlen.
    """

    bits: int


@dataclass
class PrefixHierarchy:
    """Build a prefix hierarchy over keys.

    Parameters
    ----------
    key_mode:
        How to interpret raw bytes as an integer domain.
    levels_bits:
        A list of prefix bit-lengths, from coarse to fine.
        - IPv4 default: [16,24,32]
        - u64 default: [16,24,32] (treat as bit prefixes of u64); feel free to
          change to [16,32,48,64] etc.
    key_byteorder:
        Byteorder for `u64` conversion.
    ipv4_byteorder:
        Byteorder for ipv4 conversion.

    Outputs
    -------
    prefix_id:
        An integer identifier representing the prefix at a given level.
        For a domain with total bits `D`, the prefix of length p is:
            prefix = x >> (D - p)
        where x is the integer key.

    Note
    ----
    We do NOT stringify prefixes inside core logic; the caller can format.
    """

    key_mode: KeyMode = "u64"
    levels_bits: Sequence[int] = (16, 24, 32)
    key_byteorder: str = "little"
    ipv4_byteorder: str = "big"

    def __post_init__(self) -> None:
        lb = list(self.levels_bits)
        if sorted(lb) != lb:
            raise ValueError(f"levels_bits must be increasing, got {lb}")
        if any(b <= 0 for b in lb):
            raise ValueError(f"levels_bits must be positive, got {lb}")

    def domain_bits(self, key: bytes) -> int:
        if self.key_mode == "u64":
            return 8 * len(key)
        # ipv4
        return 32

    def key_to_int(self, key: bytes) -> int:
        if self.key_mode == "u64":
            return _to_uint(key, byteorder=self.key_byteorder)
        if self.key_mode == "ipv4_src":
            return _to_ipv4_int(key, which="src", byteorder=self.ipv4_byteorder)
        if self.key_mode == "ipv4_dst":
            return _to_ipv4_int(key, which="dst", byteorder=self.ipv4_byteorder)
        raise ValueError(f"Unknown key_mode={self.key_mode}")

    def prefix_id(self, key: bytes, bits: int) -> int:
        D = self.domain_bits(key)
        if bits > D:
            raise ValueError(f"prefix bits ({bits}) > domain bits ({D})")
        x = self.key_to_int(key)
        return x >> (D - bits)

    def key_prefixes(self, key: bytes) -> Tuple[int, ...]:
        return tuple(self.prefix_id(key, b) for b in self.levels_bits)

    def prefixes_at_level(self, keys: Iterable[bytes], level: int) -> List[int]:
        if level < 0 or level >= len(self.levels_bits):
            raise IndexError("level out of range")
        b = self.levels_bits[level]
        out = []
        for k in keys:
            out.append(self.prefix_id(k, b))
        return out


def aggregate_ground_truth_by_prefix(
    gt_key_freq: Mapping[bytes, int],
    hierarchy: PrefixHierarchy,
    level: int,
) -> Dict[int, int]:
    """Aggregate per-key ground truth counts into prefix-level totals."""
    prefix_counts: Dict[int, int] = {}
    bits = hierarchy.levels_bits[level]
    for k, v in gt_key_freq.items():
        pid = hierarchy.prefix_id(k, bits)
        prefix_counts[pid] = prefix_counts.get(pid, 0) + int(v)
    return prefix_counts


def select_topk_prefixes(prefix_counts: Mapping[int, int], k: int) -> List[int]:
    """Return prefix ids of top-k heavy prefixes by count."""
    if k <= 0:
        return []
    items = sorted(prefix_counts.items(), key=lambda kv: kv[1], reverse=True)
    return [pid for pid, _ in items[:k]]


def filter_keys_by_prefix(
    keys: Iterable[bytes],
    hierarchy: PrefixHierarchy,
    level: int,
    allowed_prefixes: Sequence[int],
) -> List[bytes]:
    """Keep keys whose prefix at `level` is in allowed set."""
    allow = set(int(x) for x in allowed_prefixes)
    bits = hierarchy.levels_bits[level]
    out: List[bytes] = []
    for k in keys:
        if hierarchy.prefix_id(k, bits) in allow:
            out.append(k)
    return out


def describe_prefix(
    prefix_id: int,
    bits: int,
    domain_bits: int = 32,
    assume_ipv4: bool = True,
) -> str:
    """Human-readable prefix description.

    - If assume_ipv4 and domain_bits==32: interpret prefix_id as top bits of IPv4.
      For /p: the prefix corresponds to (prefix_id << (32-p)).
    - Otherwise: represent as bitstring prefix.
    """
    if assume_ipv4 and domain_bits == 32:
        ip = int(prefix_id) << (32 - bits)
        return f"{ipv4_to_str(ip)}/{bits}"
    # generic
    return f"{prefix_id:0{bits}b}*"  # prefix bits then wildcard
