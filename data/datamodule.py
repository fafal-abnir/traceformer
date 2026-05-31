from __future__ import annotations

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lightning as L

from data.datasets import (
    build_events_from_dataset,
    temporal_split_indices,
    rebalance_train_events,
    compute_pos_weight,
)
from graph.temporal_store import TemporalGraphStore
from graph.temporal_features import TemporalFeatureExtractor, precompute_event_role_features
from graph.walk_sampler import CausalAnonymousWalkSampler
from utils.batching import EdgeBatchBuilder


class PrebuiltBatchDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class AMLDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.role_feat_dim = None
        self.edge_feat_dim = None
        self.pos_weight = None
        self.batch_builder = None
        self.num_nodes = None
        self._is_prepared = False

    def _chunk_events(self, events, batch_size):
        for i in range(0, len(events), batch_size):
            yield events[i:i + batch_size]

    def _prebuild_batches(self, events, batch_size, split_name="split"):
        batches = []
        total = (len(events) + batch_size - 1) // batch_size

        for chunk in tqdm(
                self._chunk_events(events, batch_size),
                total=total,
                desc=f"Prebuilding {split_name}",
        ):
            batches.append(self.batch_builder.build_batch(chunk))

        print(f"Prebuilt {len(batches)} {split_name} batches")
        return batches

    def setup(self, stage=None):
        if self._is_prepared:
            return
        events, num_nodes, edge_feat_dim = build_events_from_dataset(
            dataset=self.args.dataset,
            csv_path=self.args.csv,
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
            max_events=self.args.max_events,
        )

        events = sorted(events, key=lambda e: e.ts)
        n = len(events)
        train_end, val_end = temporal_split_indices(
            n,
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
        )

        train_events_raw = events[:train_end]
        val_events = events[train_end:val_end]
        test_events = events[val_end:]

        train_events = rebalance_train_events(
            train_events_raw,
            neg_to_pos_ratio=self.args.neg_to_pos_ratio,
            seed=self.args.seed,
        )
        pos_weight = compute_pos_weight(
            train_events,
            max_pos_weight=self.args.max_pos_weight,
        )

        store = TemporalGraphStore(events)
        feat_extractor = TemporalFeatureExtractor(
            store=store,
            short_window=self.args.short_window_days * 24 * 3600,
            long_window=self.args.long_window_days * 24 * 3600,
        )
        role_feat_dim = len(
            feat_extractor.get_features(events[0].src, events[min(5, len(events) - 1)].ts)
        )

        src_role_feat, dst_role_feat = precompute_event_role_features(events, feat_extractor)

        sampler = CausalAnonymousWalkSampler(
            store=store,
            src_role_feat=src_role_feat,
            dst_role_feat=dst_role_feat,
            walk_length=self.args.walk_length,
            num_walks=self.args.num_walks,
            history_window=self.args.history_window_days * 24 * 3600,
            neighbor_sample_size=self.args.neighbor_sample_size,
        )

        self.batch_builder = EdgeBatchBuilder(
            store=store,
            feat_extractor=feat_extractor,
            sampler=sampler,
            role_feat_dim=role_feat_dim,
            edge_feat_dim=edge_feat_dim,
        )

        self.role_feat_dim = role_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.pos_weight = pos_weight
        self.num_nodes = num_nodes

        # Precompute batches once
        print("Precomputing train batches...")
        train_batches = self._prebuild_batches(
            train_events,
            self.args.batch_size,
            split_name="train",
        )

        print("Precomputing val batches...")
        val_batches = self._prebuild_batches(
            val_events,
            self.args.eval_batch_size,
            split_name="val",
        )

        print("Precomputing test batches...")
        test_batches = self._prebuild_batches(
            test_events,
            self.args.eval_batch_size,
            split_name="test",
        )

        self.train_ds = PrebuiltBatchDataset(train_batches)
        self.val_ds = PrebuiltBatchDataset(val_batches)
        self.test_ds = PrebuiltBatchDataset(test_batches)
        self._is_prepared = True
    def _identity_collate(self, batch):
        # batch is a list with exactly one already-built batch
        return batch[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=self._identity_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self._identity_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self._identity_collate,
        )