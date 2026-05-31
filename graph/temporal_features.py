from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
from tqdm import tqdm

from graph.temporal_store import TemporalGraphStore
from data.datasets import Event


class TemporalFeatureExtractor:
    def __init__(
        self,
        store: TemporalGraphStore,
        short_window: float,
        long_window: float,
        eps: float = 1e-6,
    ) -> None:
        self.store = store
        self.short_window = short_window
        self.long_window = long_window
        self.eps = eps
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _mean_gap(self, ts_list: List[float]) -> float:
        if len(ts_list) < 2:
            return 0.0
        gaps = np.diff(np.asarray(ts_list, dtype=np.float32))
        return float(np.mean(gaps)) if len(gaps) > 0 else 0.0

    def _last_recency(self, ts_list: List[float], current_ts: float) -> float:
        if not ts_list:
            return self.long_window
        return float(current_ts - ts_list[-1])

    def get_features(self, node: int, ts: float) -> np.ndarray:
        key = (node, int(ts))
        if key in self._cache:
            return self._cache[key]

        in_s = self.store.get_past_in_edges(node, ts, self.short_window)
        out_s = self.store.get_past_out_edges(node, ts, self.short_window)
        in_l = self.store.get_past_in_edges(node, ts, self.long_window)
        out_l = self.store.get_past_out_edges(node, ts, self.long_window)

        in_deg_s = len(in_s)
        out_deg_s = len(out_s)
        in_deg_l = len(in_l)
        out_deg_l = len(out_l)

        in_vol_s = float(sum(x[2] for x in in_s))
        out_vol_s = float(sum(x[2] for x in out_s))
        in_vol_l = float(sum(x[2] for x in in_l))
        out_vol_l = float(sum(x[2] for x in out_l))

        fan_ratio_s = in_deg_s / (out_deg_s + self.eps)
        fan_ratio_l = in_deg_l / (out_deg_l + self.eps)
        pass_through = out_vol_s / (in_vol_s + self.eps)

        in_ts_s = [x[0] for x in in_s]
        out_ts_s = [x[0] for x in out_s]

        mean_in_gap_s = self._mean_gap(in_ts_s)
        mean_out_gap_s = self._mean_gap(out_ts_s)
        rec_last_in = self._last_recency(in_ts_s, ts)
        rec_last_out = self._last_recency(out_ts_s, ts)

        unique_in_neighbors_s = len(set(x[1] for x in in_s))
        unique_out_neighbors_s = len(set(x[1] for x in out_s))

        counterparties_s = [x[1] for x in in_s] + [x[1] for x in out_s]
        repeated_rate_s = 0.0
        if len(counterparties_s) > 0:
            repeated_rate_s = 1.0 - (len(set(counterparties_s)) / len(counterparties_s))

        feat = np.array(
            [
                in_deg_s,
                out_deg_s,
                in_deg_l,
                out_deg_l,
                in_vol_s,
                out_vol_s,
                in_vol_l,
                out_vol_l,
                fan_ratio_s,
                fan_ratio_l,
                pass_through,
                mean_in_gap_s,
                mean_out_gap_s,
                rec_last_in,
                rec_last_out,
                unique_in_neighbors_s,
                unique_out_neighbors_s,
                repeated_rate_s,
            ],
            dtype=np.float32,
        )

        feat[:17] = np.log1p(np.maximum(feat[:17], 0.0))
        self._cache[key] = feat
        return feat


def precompute_event_role_features(events: List[Event], feat_extractor: TemporalFeatureExtractor):
    n = len(events)
    role_dim = len(feat_extractor.get_features(events[0].src, events[0].ts))

    src_role_feat = np.zeros((n, role_dim), dtype=np.float32)
    dst_role_feat = np.zeros((n, role_dim), dtype=np.float32)

    for i, e in tqdm(enumerate(events), total=n, desc="Precompute role feats"):
        src_role_feat[i] = feat_extractor.get_features(e.src, e.ts)
        dst_role_feat[i] = feat_extractor.get_features(e.dst, e.ts)

    return src_role_feat, dst_role_feat