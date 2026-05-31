from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from graph.temporal_store import TemporalGraphStore


def amount_to_bucket(amount: float, num_buckets: int = 16) -> int:
    x = math.log1p(max(float(amount), 0.0))
    return int(min(num_buckets - 1, math.floor(x)))


# step tuple:
# (
#   role_feat: np.ndarray,
#   dt: float,
#   direction: int,
#   amount_bucket: int,
#   anon_count: int,
# )
WalkStepTuple = Tuple[np.ndarray, float, int, int, int]


class CausalAnonymousWalkSampler:
    def __init__(
        self,
        store: TemporalGraphStore,
        src_role_feat: np.ndarray,
        dst_role_feat: np.ndarray,
        walk_length: int = 6,
        num_walks: int = 4,
        history_window: float = 30 * 24 * 3600,
        neighbor_sample_size: int = 20,
    ) -> None:
        self.store = store
        self.src_role_feat = src_role_feat
        self.dst_role_feat = dst_role_feat
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.history_window = history_window
        self.neighbor_sample_size = neighbor_sample_size

    def _sample_next(self, node: int, ts: float):
        candidates = self.store.get_recent_neighbors(node, ts, self.history_window)
        if not candidates:
            return None

        candidates = candidates[: self.neighbor_sample_size]
        ages = np.array([max(ts - c[1], 0.0) for c in candidates], dtype=np.float32)
        probs = np.exp(-ages / max(self.history_window, 1e-6))
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        idx = np.random.choice(len(candidates), p=probs)
        return candidates[idx]

    def _lookup_role_feat(self, direction: int, event_idx: int) -> np.ndarray:
        # direction == 1 means current node -> neighbor, so neighbor is dst
        if direction == 1:
            return self.dst_role_feat[event_idx]
        return self.src_role_feat[event_idx]

    def sample_walks(self, start_node: int, start_ts: float) -> List[List[WalkStepTuple]]:
        walks: List[List[WalkStepTuple]] = []

        for _ in range(self.num_walks):
            current_node = start_node
            current_ts = start_ts
            anon_counter: Dict[int, int] = {}
            walk: List[WalkStepTuple] = []

            for _step in range(self.walk_length):
                nxt = self._sample_next(current_node, current_ts)
                if nxt is None:
                    break

                neighbor, edge_ts, amount, direction, event_idx = nxt
                anon_counter[neighbor] = anon_counter.get(neighbor, 0) + 1
                role_feat = self._lookup_role_feat(direction, event_idx)

                walk.append(
                    (
                        role_feat,
                        float(current_ts - edge_ts),
                        int(direction),
                        int(amount_to_bucket(amount)),
                        int(anon_counter[neighbor]),
                    )
                )

                current_node = neighbor
                current_ts = edge_ts

            walks.append(walk)

        return walks