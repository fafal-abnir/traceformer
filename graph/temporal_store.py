from __future__ import annotations

from bisect import bisect_left
from typing import Dict, List, Tuple, Optional

from data.datasets import Event


class TemporalGraphStore:
    def __init__(self, events: List[Event]) -> None:
        self.events = sorted(events, key=lambda e: e.ts)

        # each entry: (ts, neighbor, amount, event_idx)
        self.in_edges: Dict[int, List[Tuple[float, int, float, int]]] = {}
        self.out_edges: Dict[int, List[Tuple[float, int, float, int]]] = {}
        self.in_times: Dict[int, List[float]] = {}
        self.out_times: Dict[int, List[float]] = {}

        # cache for recent-neighbor queries
        self._recent_cache: Dict[Tuple[int, int, int], List[Tuple[int, float, float, int, int]]] = {}

        in_tmp: Dict[int, List[Tuple[float, int, float, int]]] = {}
        out_tmp: Dict[int, List[Tuple[float, int, float, int]]] = {}

        for idx, e in enumerate(self.events):
            out_tmp.setdefault(e.src, []).append((e.ts, e.dst, e.amount, idx))
            in_tmp.setdefault(e.dst, []).append((e.ts, e.src, e.amount, idx))

        self.in_edges = in_tmp
        self.out_edges = out_tmp
        self.in_times = {k: [x[0] for x in v] for k, v in self.in_edges.items()}
        self.out_times = {k: [x[0] for x in v] for k, v in self.out_edges.items()}

    def _slice_past(
        self,
        edge_list: List[Tuple[float, int, float, int]],
        time_list: List[float],
        ts: float,
        window: Optional[float],
    ) -> List[Tuple[float, int, float, int]]:
        hi = bisect_left(time_list, ts)
        if window is None:
            return edge_list[:hi]
        lo_ts = ts - window
        lo = bisect_left(time_list, lo_ts)
        return edge_list[lo:hi]

    def get_past_in_edges(
        self,
        node: int,
        ts: float,
        window: Optional[float] = None,
    ) -> List[Tuple[float, int, float, int]]:
        edge_list = self.in_edges.get(node, [])
        time_list = self.in_times.get(node, [])
        if not edge_list:
            return []
        return self._slice_past(edge_list, time_list, ts, window)

    def get_past_out_edges(
        self,
        node: int,
        ts: float,
        window: Optional[float] = None,
    ) -> List[Tuple[float, int, float, int]]:
        edge_list = self.out_edges.get(node, [])
        time_list = self.out_times.get(node, [])
        if not edge_list:
            return []
        return self._slice_past(edge_list, time_list, ts, window)

    def get_recent_neighbors(
        self,
        node: int,
        ts: float,
        window: float,
        tail_k: int = 64,
    ) -> List[Tuple[int, float, float, int, int]]:
        """
        Returns:
            (neighbor_node, edge_ts, amount, direction, event_idx)

        direction = 1 means node -> neighbor
        direction = 0 means neighbor -> node
        """
        key = (node, int(ts), int(window))
        cached = self._recent_cache.get(key)
        if cached is not None:
            return cached

        out_edges = self.get_past_out_edges(node, ts, window)
        in_edges = self.get_past_in_edges(node, ts, window)

        # only keep the recent tail from each already-time-sorted history
        out_tail = out_edges[-tail_k:] if len(out_edges) > tail_k else out_edges
        in_tail = in_edges[-tail_k:] if len(in_edges) > tail_k else in_edges

        results: List[Tuple[int, float, float, int, int]] = []

        # reverse tail -> recent first
        for edge_ts, dst, amount, event_idx in reversed(out_tail):
            results.append((dst, edge_ts, amount, 1, event_idx))
        for edge_ts, src, amount, event_idx in reversed(in_tail):
            results.append((src, edge_ts, amount, 0, event_idx))

        # small sort only on recent candidates
        results.sort(key=lambda x: x[1], reverse=True)

        self._recent_cache[key] = results
        return results