from __future__ import annotations

import torch
import torch.nn as nn


class RoleEncoder(nn.Module):
    def __init__(self, in_dim: int, role_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, role_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StepEncoder(nn.Module):
    def __init__(
        self,
        role_dim: int,
        dt_dim: int,
        dir_emb_dim: int,
        amt_emb_dim: int,
        anon_emb_dim: int,
        num_amount_buckets: int = 16,
        max_anon_count: int = 16,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.dir_emb = nn.Embedding(2, dir_emb_dim)
        self.amt_emb = nn.Embedding(num_amount_buckets, amt_emb_dim)
        self.anon_emb = nn.Embedding(max_anon_count + 1, anon_emb_dim)

        self.dt_mlp = nn.Sequential(
            nn.Linear(1, dt_dim),
            nn.ReLU(),
            nn.Linear(dt_dim, dt_dim),
        )

        total_dim = role_dim + dt_dim + dir_emb_dim + amt_emb_dim + anon_emb_dim
        self.proj = nn.Linear(total_dim, out_dim)

    def forward(self, role_emb, dt, direction, amount_bucket, anon_count):
        dt_enc = self.dt_mlp(torch.log1p(dt))
        dir_enc = self.dir_emb(direction)
        amt_enc = self.amt_emb(amount_bucket)
        anon_enc = self.anon_emb(torch.clamp(anon_count, min=0, max=self.anon_emb.num_embeddings - 1))
        x = torch.cat([role_emb, dt_enc, dir_enc, amt_enc, anon_enc], dim=-1)
        return self.proj(x)


class WalkTransformer(nn.Module):
    def __init__(
        self,
        model_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, model_dim))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        cls = self.cls.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones((bsz, 1), dtype=valid_mask.dtype, device=valid_mask.device)
        valid_mask = torch.cat([cls_mask, valid_mask], dim=1)

        key_padding_mask = ~valid_mask.bool()
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return out[:, 0, :]


class MultiWalkAggregator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, walk_repr: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.score(walk_repr).squeeze(-1), dim=1)
        pooled = torch.sum(walk_repr * attn.unsqueeze(-1), dim=1)
        return self.proj(pooled)