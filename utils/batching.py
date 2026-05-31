from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from graph.temporal_store import TemporalGraphStore
from graph.temporal_features import TemporalFeatureExtractor
from graph.walk_sampler import CausalAnonymousWalkSampler, WalkStepTuple
from data.datasets import Event


def pad_walks(walks: List[List[WalkStepTuple]], role_feat_dim: int):
    bsz = len(walks)
    max_len = max((len(w) for w in walks), default=0)
    max_len = max(1, max_len)

    role_feat_np = np.zeros((bsz, max_len, role_feat_dim), dtype=np.float32)
    dt_np = np.zeros((bsz, max_len, 1), dtype=np.float32)
    direction_np = np.zeros((bsz, max_len), dtype=np.int64)
    amount_bucket_np = np.zeros((bsz, max_len), dtype=np.int64)
    anon_count_np = np.zeros((bsz, max_len), dtype=np.int64)
    valid_mask_np = np.zeros((bsz, max_len), dtype=bool)

    for i, walk in enumerate(walks):
        for j, step in enumerate(walk):
            role_feat, dt, direction, amount_bucket, anon_count = step
            role_feat_np[i, j] = role_feat
            dt_np[i, j, 0] = dt
            direction_np[i, j] = direction
            amount_bucket_np[i, j] = amount_bucket
            anon_count_np[i, j] = anon_count
            valid_mask_np[i, j] = True

    return {
        "role_feat": torch.from_numpy(role_feat_np),
        "dt": torch.from_numpy(dt_np),
        "direction": torch.from_numpy(direction_np),
        "amount_bucket": torch.from_numpy(amount_bucket_np),
        "anon_count": torch.from_numpy(anon_count_np),
        "valid_mask": torch.from_numpy(valid_mask_np),
    }


class EdgeBatchBuilder:
    def __init__(
        self,
        store: TemporalGraphStore,
        feat_extractor: TemporalFeatureExtractor,
        sampler: CausalAnonymousWalkSampler,
        role_feat_dim: int,
        edge_feat_dim: int,
    ) -> None:
        self.store = store
        self.feat_extractor = feat_extractor
        self.sampler = sampler
        self.role_feat_dim = role_feat_dim
        self.edge_feat_dim = edge_feat_dim

        # cache for source-side last outgoing gap
        self._gap_cache: Dict[Tuple[int, int], float] = {}

        # cache sampled walks by (node, int(ts))
        self._walk_cache: Dict[Tuple[int, int], List[List[WalkStepTuple]]] = {}

    def _get_cached_walks(self, node: int, ts: float) -> List[List[WalkStepTuple]]:
        key = (node, int(ts))
        cached = self._walk_cache.get(key)
        if cached is not None:
            return cached

        walks = self.sampler.sample_walks(node, ts)
        while len(walks) < self.sampler.num_walks:
            walks.append([])

        self._walk_cache[key] = walks
        return walks

    def _get_cached_last_gap(self, src: int, ts: float) -> float:
        gap_key = (src, int(ts))
        cached_gap = self._gap_cache.get(gap_key)
        if cached_gap is not None:
            return cached_gap

        past_out = self.store.get_past_out_edges(
            src, ts, self.feat_extractor.long_window
        )
        if past_out:
            last_gap = float(ts - past_out[-1][0])
        else:
            last_gap = float(self.feat_extractor.long_window)

        self._gap_cache[gap_key] = last_gap
        return last_gap

    def build_batch(self, events: List[Event]):
        bsz = len(events)

        src_walk_items: List[List[WalkStepTuple]] = []
        dst_walk_items: List[List[WalkStepTuple]] = []

        edge_attr_np = np.empty((bsz, self.edge_feat_dim), dtype=np.float32)
        edge_time_gap_np = np.empty((bsz, 1), dtype=np.float32)
        labels_np = np.empty((bsz,), dtype=np.float32)

        for i, e in enumerate(events):
            src_walks = self._get_cached_walks(e.src, e.ts)
            dst_walks = self._get_cached_walks(e.dst, e.ts)

            src_walk_items.extend(src_walks)
            dst_walk_items.extend(dst_walks)

            edge_attr_np[i] = e.edge_attr
            labels_np[i] = float(e.label)
            edge_time_gap_np[i, 0] = self._get_cached_last_gap(e.src, e.ts)

        src_batch = pad_walks(src_walk_items, self.role_feat_dim)
        dst_batch = pad_walks(dst_walk_items, self.role_feat_dim)

        edge_attr_t = torch.from_numpy(edge_attr_np)
        edge_time_gap_t = torch.from_numpy(edge_time_gap_np)
        labels_t = torch.from_numpy(labels_np)

        return src_batch, dst_batch, edge_attr_t, edge_time_gap_t, labels_t