from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    pred = (probs >= threshold).astype(np.int32)

    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(labels, probs)
        pr_auc = average_precision_score(labels, probs)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1_score(labels, pred, zero_division=0),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "positive_rate": float(labels.mean()) if len(labels) > 0 else float("nan"),
    }