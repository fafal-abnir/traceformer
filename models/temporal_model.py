from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders import RoleEncoder, StepEncoder, WalkTransformer, MultiWalkAggregator


class MeanWalkEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (x * mask).sum(dim=1) / denom
        return pooled


class MeanMultiWalkAggregator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, walk_repr: torch.Tensor) -> torch.Tensor:
        pooled = walk_repr.mean(dim=1)
        return self.proj(pooled)


class TemporalRIWalkModel(nn.Module):
    def __init__(
        self,
        role_feat_dim: int,
        edge_feat_dim: int,
        num_walks: int,
        role_dim: int = 32,
        step_dim: int = 128,
        walk_ctx_dim: int = 128,
        graph_ctx_dim: int = 128,
        ablate_edge_only: bool = False,
        ablate_edge_role_only: bool = False,
        ablate_no_role: bool = False,
        ablate_no_walks: bool = False,
        ablate_no_transformer: bool = False,
        ablate_no_walk_attn: bool = False,
        ablate_no_edge_gap: bool = False,
    ) -> None:
        super().__init__()
        self.num_walks = num_walks
        self.role_dim = role_dim
        self.graph_ctx_dim = graph_ctx_dim

        self.ablate_edge_only = ablate_edge_only
        self.ablate_edge_role_only = ablate_edge_role_only
        self.ablate_no_role = ablate_no_role
        self.ablate_no_walks = ablate_no_walks
        self.ablate_no_transformer = ablate_no_transformer
        self.ablate_no_walk_attn = ablate_no_walk_attn
        self.ablate_no_edge_gap = ablate_no_edge_gap

        self.role_encoder = RoleEncoder(role_feat_dim, role_dim)

        self.step_encoder = StepEncoder(
            role_dim=role_dim,
            dt_dim=16,
            dir_emb_dim=8,
            amt_emb_dim=16,
            anon_emb_dim=8,
            out_dim=step_dim,
        )

        if self.ablate_no_transformer:
            self.walk_encoder = MeanWalkEncoder()
        else:
            self.walk_encoder = WalkTransformer(
                model_dim=step_dim,
                nhead=4,
                num_layers=2,
                dropout=0.1,
            )

        if self.ablate_no_walk_attn:
            self.walk_agg = MeanMultiWalkAggregator(walk_ctx_dim, graph_ctx_dim)
        else:
            self.walk_agg = MultiWalkAggregator(walk_ctx_dim, graph_ctx_dim)

        edge_in_dim = edge_feat_dim if self.ablate_no_edge_gap else edge_feat_dim + 1

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.role_pair_proj = nn.Sequential(
            nn.Linear(role_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        if self.ablate_edge_only or self.ablate_no_walks:
            head_in_dim = 64
        elif self.ablate_edge_role_only:
            head_in_dim = 64 + 64
        else:
            head_in_dim = graph_ctx_dim * 2 + 64

        self.edge_head = nn.Sequential(
            nn.Linear(head_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def encode_roles_direct(self, src_walks, dst_walks, batch_size: int) -> torch.Tensor:
        # use the first walk's first valid step role feature as an event-time role proxy
        src_role_feat = src_walks["role_feat"].view(batch_size, self.num_walks, *src_walks["role_feat"].shape[1:])
        dst_role_feat = dst_walks["role_feat"].view(batch_size, self.num_walks, *dst_walks["role_feat"].shape[1:])
        src_valid = src_walks["valid_mask"].view(batch_size, self.num_walks, *src_walks["valid_mask"].shape[1:])
        dst_valid = dst_walks["valid_mask"].view(batch_size, self.num_walks, *dst_walks["valid_mask"].shape[1:])

        src_first = torch.zeros((batch_size, src_role_feat.size(-1)), device=src_role_feat.device, dtype=src_role_feat.dtype)
        dst_first = torch.zeros((batch_size, dst_role_feat.size(-1)), device=dst_role_feat.device, dtype=dst_role_feat.dtype)

        for b in range(batch_size):
            found = False
            for m in range(self.num_walks):
                valid_positions = torch.nonzero(src_valid[b, m], as_tuple=False)
                if len(valid_positions) > 0:
                    src_first[b] = src_role_feat[b, m, valid_positions[0, 0]]
                    found = True
                    break
            if not found:
                src_first[b].zero_()

            found = False
            for m in range(self.num_walks):
                valid_positions = torch.nonzero(dst_valid[b, m], as_tuple=False)
                if len(valid_positions) > 0:
                    dst_first[b] = dst_role_feat[b, m, valid_positions[0, 0]]
                    found = True
                    break
            if not found:
                dst_first[b].zero_()

        src_role = self.role_encoder(src_first)
        dst_role = self.role_encoder(dst_first)
        return self.role_pair_proj(torch.cat([src_role, dst_role], dim=-1))

    def encode_walk_batch(
        self,
        role_feat: torch.Tensor,
        dt: torch.Tensor,
        direction: torch.Tensor,
        amount_bucket: torch.Tensor,
        anon_count: torch.Tensor,
        valid_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        bwm, L, Fdim = role_feat.shape

        if self.ablate_no_role:
            role_emb = torch.zeros(
                (bwm, L, self.role_dim),
                device=role_feat.device,
                dtype=role_feat.dtype,
            )
        else:
            role_emb = self.role_encoder(role_feat.reshape(bwm * L, Fdim)).reshape(bwm, L, -1)

        step_emb = self.step_encoder(role_emb, dt, direction, amount_bucket, anon_count)

        has_any_valid = valid_mask.any(dim=1)
        walk_repr = torch.zeros(
            (bwm, step_emb.size(-1)),
            device=step_emb.device,
            dtype=step_emb.dtype,
        )

        if has_any_valid.any():
            walk_repr_valid = self.walk_encoder(step_emb[has_any_valid], valid_mask[has_any_valid])
            walk_repr[has_any_valid] = walk_repr_valid

        walk_repr = walk_repr.view(batch_size, self.num_walks, -1)
        return self.walk_agg(walk_repr)

    def forward(self, src_walks, dst_walks, edge_attr, edge_time_gap):
        batch_size = edge_attr.size(0)

        if self.ablate_no_edge_gap:
            edge_x = edge_attr
        else:
            edge_x = torch.cat([edge_attr, torch.log1p(edge_time_gap)], dim=-1)

        edge_x = self.edge_proj(edge_x)

        if self.ablate_edge_only or self.ablate_no_walks:
            logits = self.edge_head(edge_x).squeeze(-1)
            return logits

        if self.ablate_edge_role_only:
            role_x = self.encode_roles_direct(src_walks, dst_walks, batch_size)
            logits = self.edge_head(torch.cat([role_x, edge_x], dim=-1)).squeeze(-1)
            return logits

        src_ctx = self.encode_walk_batch(batch_size=batch_size, **src_walks)
        dst_ctx = self.encode_walk_batch(batch_size=batch_size, **dst_walks)

        logits = self.edge_head(torch.cat([src_ctx, dst_ctx, edge_x], dim=-1)).squeeze(-1)
        return logits