#!/usr/bin/env python3
"""scripts/train_eval_hisr.py

Unified training + evaluation entrypoint for
Hierarchical Invariant Sketch Resolution (HISR).

This script is designed to be *compatible with the UCL-sketch工程风格*:
- It reuses UCL's dataset loader (load_data.readTraces).
- It can reuse UCL's sketch component (UCL_sketch.UCLSketch) to ensure the
  same candidate key set (flowKeys + evictKeys) as UCL baselines.
- It supports multi-environment training using different CM hash seeds
  (environment variable s), and optional key-space phases (π) built on prefix
  granularity (coarse->medium->fine).

It is *faithful to HISR构建报告V5*:
- Environment definition e=(τ,s,π): time window τ (snapshot index), seed s
  (hash view), phase π (key-space stage / prefix focus).
- Objective structure: self-supervised risk + IRM penalty + disentanglement
  regularizers (placeholders can be extended).
- Bucket-local operator extracted as bipartite graph edge list.
- Decoder uses mass-conserving hierarchical splitting over prefix tree.

Notes
-----
This script is intentionally verbose and modular to support research iteration.
It is not a "toy" example; however, you may still need to adapt paths and
hyperparameters to your environment.

"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ucl_refs.load_data import readTraces
from ucl_refs.hash_function import GenHashSeed
from ucl_refs.ucl_sketch import UCLSketch
from ucl_refs.mertrics import (
    average_absolute_error,
    average_relative_error,
    weighted_mean_relative_difference,
    entropy_absolute_error,
)

from hisr.data.bucketize import BucketIndex, build_buckets
from hisr.data.local_operator import BucketGraph, extract_bucket_bipartite, predict_counters_from_x
from hisr.data.prefix import PrefixHierarchy, aggregate_ground_truth_by_prefix, select_topk_prefixes, filter_keys_by_prefix
from hisr.model.encoder_bipartite import BipartiteGNNEncoder
from hisr.model.decoder_prefix_tree import PrefixTreeDecoder, build_prefix_tree


# -------------------------
# Utilities
# -------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patch_cm_seeds(ucl_sketch: UCLSketch, seed_offset: int) -> None:
    """Change CM hash seeds while keeping HeavyFilter/BloomFilter behavior fixed."""
    depth = ucl_sketch.cm.depth
    ucl_sketch.cm.h = [GenHashSeed(i + seed_offset * 9973) for i in range(depth)]
    ucl_sketch.cm.s = [GenHashSeed(i + seed_offset * 9973) for i in range(depth)]
    ucl_sketch.cm.n = [GenHashSeed(i + seed_offset * 9973) for i in range(depth)]


@dataclass
class EnvView:
    """A sketch view corresponding to environment seed s."""

    s: int
    sketch: UCLSketch
    samples: np.ndarray  # (T, depth, width)


@dataclass
class StageSpec:
    """Key-space phase π (prefix granularity focus)."""

    name: str
    # keys used in this stage (subset of CM keys)
    keys: List[bytes]
    bucket_index: BucketIndex


@dataclass
class HISRConfig:
    # dataset
    data: str
    data_path: str
    key_size: int
    break_number: int

    # sketch params (must match UCL config style)
    slot_num: int
    width: int
    depth: int
    bf_width: int
    bf_hash: int

    # sampling
    interval: int
    num_samples: int

    # HISR params
    bucket_len_stage1: int
    bucket_len_stage2: int
    bucket_len_stage3: int
    topk_l1: int
    topk_l2: int

    # training
    device: str
    num_envs: int
    train_steps: int
    lr: float
    irm_lambda: float
    inv_lambda: float
    sparse_lambda: float


# -------------------------
# Loss functions
# -------------------------


def measurement_loss(graph: BucketGraph, x_hat: torch.Tensor) -> torch.Tensor:
    """Self-supervised measurement consistency loss: ||A x - y||^2."""
    y_hat = predict_counters_from_x(x_hat, graph)
    return torch.mean((y_hat - graph.y) ** 2)


def sparsity_loss(x_hat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Sparsity-promoting surrogate: sum log(1+|x|/eps)."""
    return torch.mean(torch.log1p(torch.abs(x_hat) / eps))


def irm_penalty(risks: Sequence[torch.Tensor], w: torch.nn.Parameter) -> torch.Tensor:
    """IRM penalty: sum_e ||∇_w R_e||^2.

    We implement the standard scalar classifier trick: predictions are scaled
    by a learnable scalar w. The penalty enforces the same optimal w across
    environments.
    """
    grads = []
    for r in risks:
        g = torch.autograd.grad(r, w, create_graph=True)[0]
        grads.append(g)
    gstack = torch.stack(grads)
    return torch.sum(gstack ** 2)


def invariant_alignment_loss(z_c_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Simple invariant alignment: variance across envs for same bucket should be small."""
    if len(z_c_list) <= 1:
        return torch.tensor(0.0, device=z_c_list[0].device)
    z = torch.stack(z_c_list, dim=0)
    return torch.mean(torch.var(z, dim=0))


# -------------------------
# Pipeline
# -------------------------


class HISRPipeline:
    def __init__(self, cfg: HISRConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.hierarchy = PrefixHierarchy(
            key_mode="u64",  # UCL datasets typically store opaque bytes; change if needed.
            levels_bits=(16, 24, 32),
            key_byteorder="little",
        )

    def load_traces(self) -> List[bytes]:
        size, traces = readTraces(self.cfg.data_path, self.cfg.data, self.cfg.key_size)
        if size > self.cfg.break_number:
            traces = traces[: self.cfg.break_number]
        return list(traces)

    def build_env_views(self, traces: Sequence[bytes]) -> Tuple[List[EnvView], Dict[bytes, int], List[bytes]]:
        """Build multi-environment sketch views and the shared CM candidate key set."""
        cfg = self.cfg
        ground_truth: Dict[bytes, int] = {}

        # We first build a reference view s=0 to get the key set (flowKeys+evictKeys)
        ref = UCLSketch(cfg.slot_num, cfg.width, cfg.depth, cfg.bf_width, cfg.bf_hash, cfg.key_size, decode_mode="CM")
        patch_cm_seeds(ref, seed_offset=0)

        # Sampling setup: follow UCL main.py
        size = len(traces)
        sample_initial = size - (cfg.interval + 12) * cfg.num_samples
        samples_ref = np.empty((0, ref.cm.depth, ref.cm.width), dtype=np.int64)

        for idx, trace in enumerate(tqdm(traces, desc="[HISR] Inserting traces (ref view)", leave=False)):
            ground_truth[trace] = ground_truth.get(trace, 0) + 1
            ref.insert(trace)
            if idx > sample_initial and idx % cfg.interval == 0:
                sample = ref.get_current_state(return_A=False)
                samples_ref = np.row_stack([samples_ref, sample])

        cm_keys = ref.flowKeys + ref.evictKeys

        env_views: List[EnvView] = []
        env_views.append(EnvView(s=0, sketch=ref, samples=samples_ref))

        # Other environments: different CM hash seeds, same heavy-filter behavior
        for s in range(1, cfg.num_envs):
            sk = UCLSketch(cfg.slot_num, cfg.width, cfg.depth, cfg.bf_width, cfg.bf_hash, cfg.key_size, decode_mode="CM")
            patch_cm_seeds(sk, seed_offset=s)
            samples = np.empty((0, sk.cm.depth, sk.cm.width), dtype=np.int64)

            # Insert same traces, sample snapshots
            for idx, trace in enumerate(tqdm(traces, desc=f"[HISR] Inserting traces (view s={s})", leave=False)):
                sk.insert(trace)
                if idx > sample_initial and idx % cfg.interval == 0:
                    sample = sk.get_current_state(return_A=False)
                    samples = np.row_stack([samples, sample])

            env_views.append(EnvView(s=s, sketch=sk, samples=samples))

        return env_views, ground_truth, cm_keys

    def build_stages(self, cm_keys: List[bytes], gt: Dict[bytes, int]) -> List[StageSpec]:
        """Build 3-stage key-space phases π based on prefix granularity.

        Stage 1 (L1): all CM keys (coarse scan).
        Stage 2 (L2): keys within top-K /16 prefixes (hot subnets).
        Stage 3 (L3): keys within top-K /24 prefixes *within hot /16*.

        This matches the coarse-to-fine intuition in the user's description.
        """
        cfg = self.cfg
        # Stage 1
        b1 = build_buckets(cm_keys, cfg.bucket_len_stage1, strategy="prefix", prefix_hierarchy=self.hierarchy, prefix_level=0)
        stage1 = StageSpec(name="L1_all", keys=list(cm_keys), bucket_index=b1)

        # Identify hot /16 by GT (for evaluation-stage oracle) or by Stage1 prediction (optional).
        # For the experimental plan doc, we usually report both: GT-hot oracle and predicted-hot.
        gt_l1 = aggregate_ground_truth_by_prefix({k: gt.get(k, 0) for k in cm_keys}, self.hierarchy, level=0)
        hot_l1 = select_topk_prefixes(gt_l1, cfg.topk_l1)
        keys_l2 = filter_keys_by_prefix(cm_keys, self.hierarchy, level=0, allowed_prefixes=hot_l1)
        b2 = build_buckets(keys_l2, cfg.bucket_len_stage2, strategy="prefix", prefix_hierarchy=self.hierarchy, prefix_level=1)
        stage2 = StageSpec(name=f"L2_hot16_top{cfg.topk_l1}", keys=keys_l2, bucket_index=b2)

        # Identify hot /24 within those keys
        gt_l2 = aggregate_ground_truth_by_prefix({k: gt.get(k, 0) for k in keys_l2}, self.hierarchy, level=1)
        hot_l2 = select_topk_prefixes(gt_l2, cfg.topk_l2)
        keys_l3 = filter_keys_by_prefix(keys_l2, self.hierarchy, level=1, allowed_prefixes=hot_l2)
        b3 = build_buckets(keys_l3, cfg.bucket_len_stage3, strategy="sorted")
        stage3 = StageSpec(name=f"L3_hot24_top{cfg.topk_l2}", keys=keys_l3, bucket_index=b3)

        return [stage1, stage2, stage3]

    def build_model(self) -> Tuple[BipartiteGNNEncoder, PrefixTreeDecoder, torch.nn.Parameter, torch.optim.Optimizer]:
        enc = BipartiteGNNEncoder(d_node=128, d_msg=128, d_z=128, num_layers=3, use_gru=True, dropout=0.1).to(self.device)
        dec = PrefixTreeDecoder(d_z=128, d_hidden=128, dropout=0.1).to(self.device)
        # IRM scalar
        w = torch.nn.Parameter(torch.tensor(1.0, device=self.device))

        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + [w], lr=self.cfg.lr)
        return enc, dec, w, opt

    def train(
        self,
        envs: List[EnvView],
        stage: StageSpec,
        enc: BipartiteGNNEncoder,
        dec: PrefixTreeDecoder,
        w: torch.nn.Parameter,
        opt: torch.optim.Optimizer,
    ) -> None:
        cfg = self.cfg
        enc.train()
        dec.train()

        # Use the *latest* snapshot index as default training horizon
        T = min(ev.samples.shape[0] for ev in envs)
        if T == 0:
            raise RuntimeError("No samples were collected; increase break_number or adjust interval/num_samples")

        for step in tqdm(range(cfg.train_steps), desc=f"[HISR] Training {stage.name}"):
            # Sample a time window τ and a bucket b
            tau = random.randrange(T)
            b = random.randrange(stage.bucket_index.num_buckets)
            keys_b = stage.bucket_index.bucket_keys(b)
            if len(keys_b) == 0:
                continue

            risks = []
            zc_list = []

            for ev in envs:
                # Load snapshot y_{τ} for this environment and build a temporary cm sketch matrix
                cm = ev.sketch.cm
                cm.matrix = ev.samples[tau].astype(np.int64)  # patch state

                graph = extract_bucket_bipartite(cm_sketch=cm, keys_in_bucket=keys_b, device=self.device)
                out = enc(graph)

                tree = build_prefix_tree(keys_b, self.hierarchy)
                x_hat, _ = dec(out.z_c, tree)
                x_hat = torch.relu(x_hat)  # ensure non-negativity (decoder already positive, but keep safe)

                # IRM scalar scaling
                x_hat_scaled = w * x_hat

                r = measurement_loss(graph, x_hat_scaled) + cfg.sparse_lambda * sparsity_loss(x_hat_scaled)
                risks.append(r)
                zc_list.append(out.z_c)

            risk_mean = torch.stack(risks).mean()
            penalty = irm_penalty(risks, w)
            inv_loss = invariant_alignment_loss(zc_list)

            loss = risk_mean + cfg.irm_lambda * penalty + cfg.inv_lambda * inv_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), max_norm=5.0)
            opt.step()

            if (step + 1) % 200 == 0:
                tqdm.write(
                    f"step={step+1} loss={loss.item():.4f} risk={risk_mean.item():.4f} irm={penalty.item():.4f} inv={inv_loss.item():.4f}"
                )

    @torch.no_grad()
    def eval_stage(
        self,
        env: EnvView,
        stage: StageSpec,
        enc: BipartiteGNNEncoder,
        dec: PrefixTreeDecoder,
        w: torch.nn.Parameter,
        gt: Dict[bytes, int],
        *,
        tau: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate reconstruction quality for a given stage on one environment."""
        enc.eval()
        dec.eval()

        if tau is None:
            tau = env.samples.shape[0] - 1

        # Patch cm state to snapshot τ
        cm = env.sketch.cm
        cm.matrix = env.samples[tau].astype(np.int64)

        # Predict x for all buckets and collect per-key predictions
        pred: Dict[bytes, float] = {}

        for b in tqdm(range(stage.bucket_index.num_buckets), desc=f"[HISR] Eval {stage.name} (s={env.s})", leave=False):
            keys_b = stage.bucket_index.bucket_keys(b)
            if len(keys_b) == 0:
                continue
            graph = extract_bucket_bipartite(cm_sketch=cm, keys_in_bucket=keys_b, device=self.device)
            out = enc(graph)
            tree = build_prefix_tree(keys_b, self.hierarchy)
            x_hat, _ = dec(out.z_c, tree)
            x_hat = torch.relu(w * x_hat)

            for k, v in zip(keys_b, x_hat.detach().cpu().numpy().tolist()):
                pred[k] = float(v)

        # Build vectors aligned to stage.keys for metrics
        GT = [gt.get(k, 0) for k in stage.keys]
        ET = [max(0.0, pred.get(k, 0.0)) for k in stage.keys]

        AAE = average_absolute_error(GT, ET)
        ARE = average_relative_error(GT, ET)
        WMRD = weighted_mean_relative_difference(GT, ET)
        EAE = entropy_absolute_error(GT, ET)

        return {"AAE": float(AAE), "ARE": float(ARE), "WMRD": float(WMRD), "EAE": float(EAE)}


def parse_args() -> HISRConfig:
    p = argparse.ArgumentParser(description="HISR training/evaluation on UCL dataset")
    p.add_argument("--data", type=str, default="network", choices=["retail", "kosarak", "network", "synthetic"])
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--key_size", type=int, default=8)
    p.add_argument("--break_number", type=int, default=1000000)

    # sketch params (match UCL config)
    p.add_argument("--slot_num", type=int, default=20000)
    p.add_argument("--width", type=int, default=20000)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--bf_width", type=int, default=50000)
    p.add_argument("--bf_hash", type=int, default=3)

    # sampling
    p.add_argument("--interval", type=int, default=1000)
    p.add_argument("--num_samples", type=int, default=128)

    # stages
    p.add_argument("--bucket_len_stage1", type=int, default=2048)
    p.add_argument("--bucket_len_stage2", type=int, default=1024)
    p.add_argument("--bucket_len_stage3", type=int, default=512)
    p.add_argument("--topk_l1", type=int, default=256)
    p.add_argument("--topk_l2", type=int, default=512)

    # training
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_envs", type=int, default=3)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--irm_lambda", type=float, default=1.0)
    p.add_argument("--inv_lambda", type=float, default=0.1)
    p.add_argument("--sparse_lambda", type=float, default=0.01)

    args = p.parse_args()
    return HISRConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    set_global_seed(12345)

    pipe = HISRPipeline(cfg)
    traces = pipe.load_traces()

    envs, gt, cm_keys = pipe.build_env_views(traces)
    stages = pipe.build_stages(cm_keys, gt)

    enc, dec, w, opt = pipe.build_model()

    # Train on Stage 1 by default; optionally continue on Stage 2/3.
    pipe.train(envs, stages[0], enc, dec, w, opt)

    # Evaluate
    for st in stages:
        metrics = pipe.eval_stage(envs[0], st, enc, dec, w, gt)
        print(f"[HISR] Stage={st.name} metrics={metrics}")


if __name__ == "__main__":
    main()
