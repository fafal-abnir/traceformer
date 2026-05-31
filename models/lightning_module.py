from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from models.temporal_model import TemporalRIWalkModel
from utils.metrics import classification_metrics


class LitTemporalRIWalk(L.LightningModule):
    def __init__(self, args, role_feat_dim: int, edge_feat_dim: int, pos_weight: torch.Tensor):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=["pos_weight"])

        self.model = TemporalRIWalkModel(
            role_feat_dim=role_feat_dim,
            edge_feat_dim=edge_feat_dim,
            num_walks=args.num_walks,
            role_dim=args.role_dim,
            step_dim=args.step_dim,
            walk_ctx_dim=args.step_dim,
            graph_ctx_dim=args.graph_ctx_dim,
        )

        self.register_buffer("pos_weight_tensor", pos_weight.clone().detach())

        self.val_probs = []
        self.val_labels = []
        self.test_probs = []
        self.test_labels = []

    def forward(self, src_batch, dst_batch, edge_attr, edge_time_gap):
        return self.model(src_batch, dst_batch, edge_attr, edge_time_gap)

    def training_step(self, batch, batch_idx):
        src_batch, dst_batch, edge_attr, edge_time_gap, labels = batch
        logits = self(src_batch, dst_batch, edge_attr, edge_time_gap)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight_tensor,
        )
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_batch, dst_batch, edge_attr, edge_time_gap, labels = batch
        logits = self(src_batch, dst_batch, edge_attr, edge_time_gap)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight_tensor,
        )
        probs = torch.sigmoid(logits).detach().cpu()
        self.val_probs.append(probs)
        self.val_labels.append(labels.detach().cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        probs = torch.cat(self.val_probs).numpy()
        labels = torch.cat(self.val_labels).numpy()
        self.val_probs.clear()
        self.val_labels.clear()

        metrics = classification_metrics(labels, probs, threshold=self.args.threshold)
        self.log_dict(
            {
                "val_roc_auc": metrics["roc_auc"],
                "val_pr_auc": metrics["pr_auc"],
                "val_f1": metrics["f1"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        src_batch, dst_batch, edge_attr, edge_time_gap, labels = batch
        logits = self(src_batch, dst_batch, edge_attr, edge_time_gap)
        probs = torch.sigmoid(logits).detach().cpu()
        self.test_probs.append(probs)
        self.test_labels.append(labels.detach().cpu())

    def on_test_epoch_end(self):
        probs = torch.cat(self.test_probs).numpy()
        labels = torch.cat(self.test_labels).numpy()
        self.test_probs.clear()
        self.test_labels.clear()

        metrics = classification_metrics(labels, probs, threshold=self.args.threshold)
        self.log_dict(
            {
                "test_roc_auc": metrics["roc_auc"],
                "test_pr_auc": metrics["pr_auc"],
                "test_f1": metrics["f1"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )