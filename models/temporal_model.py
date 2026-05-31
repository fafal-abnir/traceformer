from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders import RoleEncoder, StepEncoder, WalkTransformer, MultiWalkAggregator


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
    ) -> None:
        super().__init__()
        self.num_walks = num_walks

        self.role_encoder = RoleEncoder(role_feat_dim, role_dim)
        self.step_encoder = StepEncoder(
            role_dim=role_dim,
            dt_dim=16,
            dir_emb_dim=8,
            amt_emb_dim=16,
            anon_emb_dim=8,
            out_dim=step_dim,
        )
        self.walk_encoder = WalkTransformer(
            model_dim=step_dim,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        )
        self.walk_agg = MultiWalkAggregator(walk_ctx_dim, graph_ctx_dim)

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feat_dim + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.edge_head = nn.Sequential(
            nn.Linear(graph_ctx_dim * 2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

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

        src_ctx = self.encode_walk_batch(batch_size=batch_size, **src_walks)
        dst_ctx = self.encode_walk_batch(batch_size=batch_size, **dst_walks)

        edge_x = torch.cat([edge_attr, torch.log1p(edge_time_gap)], dim=-1)
        edge_x = self.edge_proj(edge_x)

        logits = self.edge_head(torch.cat([src_ctx, dst_ctx, edge_x], dim=-1)).squeeze(-1)
        return logits