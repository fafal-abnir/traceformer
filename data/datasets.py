from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


@dataclass
class Event:
    src: int
    dst: int
    ts: float
    amount: float
    label: int
    edge_attr: np.ndarray


def temporal_split_indices(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return n_train, n_train + n_val


def compute_pos_weight(events, max_pos_weight: float = 100.0):
    y = np.array([e.label for e in events], dtype=np.float32)
    pos = y.sum()
    neg = len(y) - pos
    if pos <= 0:
        import torch
        return torch.tensor(1.0, dtype=torch.float32)
    w = neg / pos
    w = min(w, max_pos_weight)
    import torch
    return torch.tensor(w, dtype=torch.float32)


def rebalance_train_events(events, neg_to_pos_ratio: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos_events = [e for e in events if e.label == 1]
    neg_events = [e for e in events if e.label == 0]

    n_pos = len(pos_events)
    if n_pos == 0:
        return events

    n_neg_keep = min(len(neg_events), neg_to_pos_ratio * n_pos)
    keep_idx = rng.choice(len(neg_events), size=n_neg_keep, replace=False)
    neg_sample = [neg_events[i] for i in keep_idx]

    balanced = pos_events + neg_sample
    rng.shuffle(balanced)
    return balanced


def build_events_from_samld(
        csv_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_events: Optional[int] = None,
) -> Tuple[List[Event], int, int]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    df.rename(columns={
        "sender_account": "sender",
        "receiver_account": "receiver",
        "is_laundering": "label",
    }, inplace=True)

    required = ["sender", "receiver", "amount", "label", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SAML-D missing required columns: {missing}")

    if max_events is not None:
        df = df.iloc[:max_events].copy()

    df["date_time"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_time", "sender", "receiver", "amount", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df.sort_values("date_time").reset_index(drop=True)

    n = len(df)
    train_end, _ = temporal_split_indices(n, train_ratio=train_ratio, val_ratio=val_ratio)

    all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
    df["sender"] = df["sender"].map(id_mapping).astype(int)
    df["receiver"] = df["receiver"].map(id_mapping).astype(int)

    df["amount"] = np.log1p(df["amount"].clip(lower=0))
    scaler = StandardScaler()
    df.loc[: train_end - 1, ["amount"]] = scaler.fit_transform(df.loc[: train_end - 1, ["amount"]])
    if train_end < len(df):
        df.loc[train_end:, ["amount"]] = scaler.transform(df.loc[train_end:, ["amount"]])

    cat_cols = [
        "payment_currency",
        "received_currency",
        "sender_bank_location",
        "receiver_bank_location",
        "payment_type",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dtype=np.float32)
        dummy_prefixes = [f"{c}_" for c in cat_cols]
        dummy_cols = [c for c in df.columns if any(c.startswith(p) for p in dummy_prefixes)]
        edge_feat_cols = dummy_cols + ["amount"]
    else:
        edge_feat_cols = ["amount"]

    ts0 = df["date_time"].min()
    df["ts_float"] = (df["date_time"] - ts0).dt.total_seconds().astype(np.float64)

    edge_attr_matrix = df[edge_feat_cols].to_numpy(dtype=np.float32)
    senders = df["sender"].to_numpy(dtype=np.int64)
    receivers = df["receiver"].to_numpy(dtype=np.int64)
    timestamps = df["ts_float"].to_numpy(dtype=np.float64)
    amounts = df["amount"].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)

    events: List[Event] = []
    for i in range(len(df)):
        events.append(
            Event(
                src=int(senders[i]),
                dst=int(receivers[i]),
                ts=float(timestamps[i]),
                amount=float(amounts[i]),
                label=int(labels[i]),
                edge_attr=edge_attr_matrix[i],
            )
        )

    num_nodes = int(max(df["sender"].max(), df["receiver"].max()) + 1)
    edge_feat_dim = len(edge_feat_cols)
    return events, num_nodes, edge_feat_dim


def build_events_from_amlworld(
        csv_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_events: Optional[int] = None,
) -> Tuple[List[Event], int, int]:
    df = pd.read_csv(csv_path)

    df.rename(columns={
        "Account": "sender",
        "Account.1": "receiver",
        "Amount Received": "amount",
        "Is Laundering": "label",
    }, inplace=True)

    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

    required = ["sender", "receiver", "amount", "label", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"AMLWorld missing required columns: {missing}")

    if max_events is not None:
        df = df.iloc[:max_events].copy()

    df["date_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Found invalid timestamps after parsing.")

    df = df.dropna(subset=["date_time", "sender", "receiver", "amount", "label"]).copy()
    df["label"] = df["label"].astype(int)

    df["timestamp"] = (df["timestamp"].astype("int64") // 1_000_000_000).astype(np.int64)
    start_ts = df["timestamp"].min()
    df["ts_float"] = (df["timestamp"] - start_ts).astype(np.float64)
    df = df.sort_values("ts_float").reset_index(drop=True)

    n = len(df)
    train_end, _ = temporal_split_indices(n, train_ratio=train_ratio, val_ratio=val_ratio)

    all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
    df["sender"] = df["sender"].map(id_mapping).astype(int)
    df["receiver"] = df["receiver"].map(id_mapping).astype(int)

    df["amount"] = np.log1p(df["amount"].clip(lower=0))
    scaler = StandardScaler()
    df.loc[: train_end - 1, ["amount"]] = scaler.fit_transform(df.loc[: train_end - 1, ["amount"]])
    if train_end < len(df):
        df.loc[train_end:, ["amount"]] = scaler.transform(df.loc[train_end:, ["amount"]])

    cat_cols = ["from_bank", "to_bank", "receiving_currency", "payment_currency", "payment_format"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    edge_feat_cols = []
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_vals = df.loc[: train_end - 1, cat_cols].astype(str).fillna("__nan__")
        all_vals = df[cat_cols].astype(str).fillna("__nan__")
        enc.fit(train_vals)
        df[cat_cols] = enc.transform(all_vals).astype(np.float32)
        edge_feat_cols.extend(cat_cols)

    edge_feat_cols.append("amount")

    edge_attr_matrix = df[edge_feat_cols].to_numpy(dtype=np.float32)
    senders = df["sender"].to_numpy(dtype=np.int64)
    receivers = df["receiver"].to_numpy(dtype=np.int64)
    timestamps = df["ts_float"].to_numpy(dtype=np.float64)
    amounts = df["amount"].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)

    events: List[Event] = []
    for i in range(len(df)):
        events.append(
            Event(
                src=int(senders[i]),
                dst=int(receivers[i]),
                ts=float(timestamps[i]),
                amount=float(amounts[i]),
                label=int(labels[i]),
                edge_attr=edge_attr_matrix[i],
            )
        )

    num_nodes = int(max(df["sender"].max(), df["receiver"].max()) + 1)
    edge_feat_dim = len(edge_feat_cols)
    return events, num_nodes, edge_feat_dim


def build_events_from_amlsim(
        csv_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_events: Optional[int] = None,
) -> Tuple[List[Event], int, int]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    df.rename(columns={
        "sender_account_id": "sender",
        "receiver_account_id": "receiver",
        "tx_amount": "amount",
        "is_fraud": "label",
    }, inplace=True)

    required = ["sender", "receiver", "amount", "label", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"AMLSim missing required columns: {missing}")

    if max_events is not None:
        df = df.iloc[:max_events].copy()

    df = df.dropna(subset=["sender", "receiver", "amount", "label", "timestamp"]).copy()
    df["label"] = df["label"].astype(int)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Found invalid AMLSim timestamps after parsing.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end, _ = temporal_split_indices(n, train_ratio=train_ratio, val_ratio=val_ratio)

    all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
    df["sender"] = df["sender"].map(id_mapping).astype(int)
    df["receiver"] = df["receiver"].map(id_mapping).astype(int)

    min_time = df["timestamp"].min()
    df["ts_float"] = (df["timestamp"] - min_time).astype(np.float64)

    df["amount"] = np.log1p(df["amount"].clip(lower=0))
    scaler = StandardScaler()
    df.loc[: train_end - 1, ["amount"]] = scaler.fit_transform(df.loc[: train_end - 1, ["amount"]])
    if train_end < len(df):
        df.loc[train_end:, ["amount"]] = scaler.transform(df.loc[train_end:, ["amount"]])

    edge_feat_cols = []
    if "tx_type" in df.columns:
        df["tx_type"] = df["tx_type"].astype(str).fillna("__nan__")
        df = pd.get_dummies(df, columns=["tx_type"], dtype=np.float32)
        edge_feat_cols.extend([c for c in df.columns if c.startswith("tx_type_")])

    edge_feat_cols.append("amount")

    edge_attr_matrix = df[edge_feat_cols].to_numpy(dtype=np.float32)
    senders = df["sender"].to_numpy(dtype=np.int64)
    receivers = df["receiver"].to_numpy(dtype=np.int64)
    timestamps = df["ts_float"].to_numpy(dtype=np.float64)
    amounts = df["amount"].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)

    events: List[Event] = []
    for i in range(len(df)):
        events.append(
            Event(
                src=int(senders[i]),
                dst=int(receivers[i]),
                ts=float(timestamps[i]),
                amount=float(amounts[i]),
                label=int(labels[i]),
                edge_attr=edge_attr_matrix[i],
            )
        )

    num_nodes = int(max(df["sender"].max(), df["receiver"].max()) + 1)
    edge_feat_dim = len(edge_feat_cols)
    return events, num_nodes, edge_feat_dim


def build_events_from_bitcoin_alpha(
        csv_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_events: Optional[int] = None,
) -> Tuple[List[Event], int, int]:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["source", "target", "rating", "time"],
    )

    required = ["source", "target", "rating", "time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Bitcoin Alpha missing required columns: {missing}")

    if max_events is not None:
        df = df.iloc[:max_events].copy()

    df = df.dropna(subset=["source", "target", "rating", "time"]).copy()
    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["rating", "time"]).copy()

    # label only
    df["label"] = (df["rating"] < 0).astype(int)

    df = df.sort_values("time").reset_index(drop=True)

    all_ids = pd.unique(df[["source", "target"]].values.ravel())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
    df["sender"] = df["source"].map(id_mapping).astype(int)
    df["receiver"] = df["target"].map(id_mapping).astype(int)

    start_ts = df["time"].min()
    df["ts_float"] = (df["time"] - start_ts).astype(np.float64)

    # no leakage: do not use rating as feature
    # keep a tiny constant edge feature so the model still has edge_attr
    df["edge_const"] = 1.0

    edge_feat_cols = ["edge_const"]

    edge_attr_matrix = df[edge_feat_cols].to_numpy(dtype=np.float32)
    senders = df["sender"].to_numpy(dtype=np.int64)
    receivers = df["receiver"].to_numpy(dtype=np.int64)
    timestamps = df["ts_float"].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.int64)

    # amount is only used by walk tokenization; keep constant to avoid leakage
    amounts = np.ones(len(df), dtype=np.float32)

    events: List[Event] = []
    for i in range(len(df)):
        events.append(
            Event(
                src=int(senders[i]),
                dst=int(receivers[i]),
                ts=float(timestamps[i]),
                amount=float(amounts[i]),
                label=int(labels[i]),
                edge_attr=edge_attr_matrix[i],
            )
        )

    num_nodes = int(max(df["sender"].max(), df["receiver"].max()) + 1)
    edge_feat_dim = len(edge_feat_cols)
    return events, num_nodes, edge_feat_dim


def build_events_from_bitcoin_otc(
        csv_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_events: Optional[int] = None,
) -> Tuple[List[Event], int, int]:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["source", "target", "rating", "time"],
    )

    required = ["source", "target", "rating", "time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Bitcoin OTC missing required columns: {missing}")

    if max_events is not None:
        df = df.iloc[:max_events].copy()

    df = df.dropna(subset=["source", "target", "rating", "time"]).copy()
    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["rating", "time"]).copy()

    # label only
    df["label"] = (df["rating"] < 0).astype(int)

    df = df.sort_values("time").reset_index(drop=True)

    all_ids = pd.unique(df[["source", "target"]].values.ravel())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
    df["sender"] = df["source"].map(id_mapping).astype(int)
    df["receiver"] = df["target"].map(id_mapping).astype(int)

    start_ts = df["time"].min()
    df["ts_float"] = (df["time"] - start_ts).astype(np.float64)

    # no leakage: do not use rating as feature
    df["edge_const"] = 1.0
    edge_feat_cols = ["edge_const"]

    edge_attr_matrix = df[edge_feat_cols].to_numpy(dtype=np.float32)
    senders = df["sender"].to_numpy(dtype=np.int64)
    receivers = df["receiver"].to_numpy(dtype=np.int64)
    timestamps = df["ts_float"].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.int64)

    # constant amount for walk bucketing, no leakage
    amounts = np.ones(len(df), dtype=np.float32)

    events: List[Event] = []
    for i in range(len(df)):
        events.append(
            Event(
                src=int(senders[i]),
                dst=int(receivers[i]),
                ts=float(timestamps[i]),
                amount=float(amounts[i]),
                label=int(labels[i]),
                edge_attr=edge_attr_matrix[i],
            )
        )

    num_nodes = int(max(df["sender"].max(), df["receiver"].max()) + 1)
    edge_feat_dim = len(edge_feat_cols)
    return events, num_nodes, edge_feat_dim


def build_events_from_ascendexhacker(
        csv_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_events: Optional[int] = None,
) -> Tuple[List[Event], int, int]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = [
        "hash",
        "from",
        "to",
        "value",
        "timeStamp",
        "blockNumber",
        "tokenSymbol",
        "contractAddress",
        "isError",
        "gasPrice",
        "gasUsed",
        "label",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"AscendEXHacker missing required columns: {missing}")

    if max_events is not None:
        df = df.iloc[:max_events].copy()

    # rename to internal schema
    df = df.rename(
        columns={
            "from": "sender",
            "to": "receiver",
            "timeStamp": "timestamp",
            "blockNumber": "block_number",
            "tokenSymbol": "token_symbol",
            "contractAddress": "contract_address",
            "gasPrice": "gas_price",
            "gasUsed": "gas_used",
            "isError": "is_error",
        }
    )

    numeric_cols = ["value", "timestamp", "block_number", "gas_price", "gas_used", "label"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["is_error"] = pd.to_numeric(df["is_error"], errors="coerce").fillna(0)

    df = df.dropna(subset=["sender", "receiver", "value", "timestamp", "label"]).copy()

    df["label"] = (df["label"].astype(float) > 0).astype(int)

    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end, _ = temporal_split_indices(n, train_ratio=train_ratio, val_ratio=val_ratio)

    all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
    df["sender"] = df["sender"].map(id_mapping).astype(int)
    df["receiver"] = df["receiver"].map(id_mapping).astype(int)

    start_ts = df["timestamp"].min()
    df["ts_float"] = (df["timestamp"] - start_ts).astype(np.float64)

    df["amount_raw"] = df["value"].clip(lower=0).astype(np.float32)

    df["amount_feat"] = np.log1p(df["amount_raw"])
    df["gas_price_feat"] = np.log1p(df["gas_price"].clip(lower=0).fillna(0))
    df["gas_used_feat"] = np.log1p(df["gas_used"].clip(lower=0).fillna(0))
    df["block_number_feat"] = df["block_number"].fillna(df["block_number"].median())

    cont_cols = ["amount_feat", "gas_price_feat", "gas_used_feat", "block_number_feat"]
    scaler = StandardScaler()
    df.loc[: train_end - 1, cont_cols] = scaler.fit_transform(df.loc[: train_end - 1, cont_cols])
    if train_end < len(df):
        df.loc[train_end:, cont_cols] = scaler.transform(df.loc[train_end:, cont_cols])

    cat_cols = ["token_symbol", "contract_address"]
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("__nan__")

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_vals = df.loc[: train_end - 1, cat_cols]
    all_vals = df[cat_cols]
    df[cat_cols] = enc.fit(train_vals).transform(all_vals).astype(np.float32)

    df["is_error_feat"] = df["is_error"].astype(np.float32)

    edge_feat_cols = [
        "amount_feat",
        "gas_price_feat",
        "gas_used_feat",
        "block_number_feat",
        "token_symbol",
        "contract_address",
        "is_error_feat",
    ]

    edge_attr_matrix = df[edge_feat_cols].to_numpy(dtype=np.float32)
    senders = df["sender"].to_numpy(dtype=np.int64)
    receivers = df["receiver"].to_numpy(dtype=np.int64)
    timestamps = df["ts_float"].to_numpy(dtype=np.float64)
    amounts = df["amount_raw"].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)

    events: List[Event] = []
    for i in range(len(df)):
        events.append(
            Event(
                src=int(senders[i]),
                dst=int(receivers[i]),
                ts=float(timestamps[i]),
                amount=float(amounts[i]),
                label=int(labels[i]),
                edge_attr=edge_attr_matrix[i],
            )
        )

    num_nodes = int(max(df["sender"].max(), df["receiver"].max()) + 1)
    edge_feat_dim = len(edge_feat_cols)
    return events, num_nodes, edge_feat_dim


def build_events_from_upbithack(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_events: Optional[int] = None,
) -> Tuple[List[Event], int, int]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = [
        "hash",
        "from",
        "to",
        "value",
        "timeStamp",
        "blockNumber",
        "tokenSymbol",
        "contractAddress",
        "isError",
        "gasPrice",
        "gasUsed",
        "label",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"UpbitHack missing required columns: {missing}")

    if max_events is not None:
        df = df.iloc[:max_events].copy()

    df = df.rename(
        columns={
            "from": "sender",
            "to": "receiver",
            "timeStamp": "timestamp",
            "blockNumber": "block_number",
            "tokenSymbol": "token_symbol",
            "contractAddress": "contract_address",
            "gasPrice": "gas_price",
            "gasUsed": "gas_used",
            "isError": "is_error",
        }
    )

    # parse numeric columns
    numeric_cols = ["value", "timestamp", "block_number", "gas_price", "gas_used", "label", "is_error"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["sender", "receiver", "value", "timestamp", "label"]).copy()

    # binary label
    df["label"] = (df["label"].astype(float) > 0).astype(int)

    # sanitize numeric inputs
    df["value"] = df["value"].clip(lower=0)
    df["gas_price"] = df["gas_price"].fillna(0).clip(lower=0)
    df["gas_used"] = df["gas_used"].fillna(0).clip(lower=0)
    df["block_number"] = df["block_number"].fillna(df["block_number"].median())
    df["is_error"] = df["is_error"].fillna(0)

    # sort and reset index BEFORE split
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end, _ = temporal_split_indices(n, train_ratio=train_ratio, val_ratio=val_ratio)

    # reindex nodes
    all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
    df["sender"] = df["sender"].map(id_mapping).astype(int)
    df["receiver"] = df["receiver"].map(id_mapping).astype(int)

    # relative time
    start_ts = df["timestamp"].min()
    df["ts_float"] = (df["timestamp"] - start_ts).astype(np.float64)

    # IMPORTANT:
    # use float64 + log1p first, then cast later
    # use log-scaled amount as Event.amount to avoid overflow in temporal volume features
    value_float = df["value"].astype(np.float64)
    gas_price_float = df["gas_price"].astype(np.float64)
    gas_used_float = df["gas_used"].astype(np.float64)
    block_number_float = df["block_number"].astype(np.float64)

    df["amount_log"] = np.log1p(value_float)
    df["gas_price_feat"] = np.log1p(gas_price_float)
    df["gas_used_feat"] = np.log1p(gas_used_float)
    df["block_number_feat"] = block_number_float
    df["is_error_feat"] = df["is_error"].astype(np.float64)

    # categorical features
    df["token_symbol"] = df["token_symbol"].fillna("__nan__").astype(str)
    # drop contract_address for now; high-cardinality ordinal IDs are usually harmful here

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_vals = df.loc[: train_end - 1, ["token_symbol"]]
    all_vals = df[["token_symbol"]]
    df["token_symbol_enc"] = enc.fit(train_vals).transform(all_vals).astype(np.float64)

    # keep only finite rows for used features
    feature_cols = [
        "amount_log",
        "gas_price_feat",
        "gas_used_feat",
        "block_number_feat",
        "is_error_feat",
        "token_symbol_enc",
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ["sender", "receiver", "ts_float", "label"]).reset_index(drop=True)

    # recompute split after dropping non-finite rows
    n = len(df)
    train_end, _ = temporal_split_indices(n, train_ratio=train_ratio, val_ratio=val_ratio)

    # standardize only selected continuous features on train
    cont_cols = ["amount_log", "gas_price_feat", "gas_used_feat", "block_number_feat", "token_symbol_enc"]
    scaler = StandardScaler()
    df.loc[: train_end - 1, cont_cols] = scaler.fit_transform(df.loc[: train_end - 1, cont_cols])
    if train_end < len(df):
        df.loc[train_end:, cont_cols] = scaler.transform(df.loc[train_end:, cont_cols])

    edge_feat_cols = [
        "amount_log",
        "gas_price_feat",
        "gas_used_feat",
        "block_number_feat",
        "is_error_feat",
        "token_symbol_enc",
    ]

    edge_attr_matrix = df[edge_feat_cols].to_numpy(dtype=np.float32)
    senders = df["sender"].to_numpy(dtype=np.int64)
    receivers = df["receiver"].to_numpy(dtype=np.int64)
    timestamps = df["ts_float"].to_numpy(dtype=np.float64)

    # use log-scaled amount for temporal store / walk bucket / volume features
    amounts = np.log1p(df["value"].astype(np.float64)).to_numpy(dtype=np.float32)

    labels = df["label"].to_numpy(dtype=np.int64)

    events: List[Event] = []
    for i in range(len(df)):
        events.append(
            Event(
                src=int(senders[i]),
                dst=int(receivers[i]),
                ts=float(timestamps[i]),
                amount=float(amounts[i]),
                label=int(labels[i]),
                edge_attr=edge_attr_matrix[i],
            )
        )

    num_nodes = int(max(df["sender"].max(), df["receiver"].max()) + 1)
    edge_feat_dim = len(edge_feat_cols)
    return events, num_nodes, edge_feat_dim

def build_events_from_dataset(
        dataset: str,
        csv_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_events: Optional[int] = None,
):
    dataset = dataset.lower()
    if dataset == "samld":
        return build_events_from_samld(csv_path, train_ratio, val_ratio, max_events)
    if dataset == "amlworld":
        return build_events_from_amlworld(csv_path, train_ratio, val_ratio, max_events)
    if dataset == "amlsim":
        return build_events_from_amlsim(csv_path, train_ratio, val_ratio, max_events)
    if dataset == "bitcoin_alpha":
        return build_events_from_bitcoin_alpha(csv_path, train_ratio, val_ratio, max_events)
    if dataset == "bitcoin_otc":
        return build_events_from_bitcoin_otc(csv_path, train_ratio, val_ratio, max_events)
    if dataset == "ascendexhacker":
        return build_events_from_ascendexhacker(csv_path, train_ratio, val_ratio, max_events)
    if dataset == "upbithack":
        return build_events_from_upbithack(csv_path, train_ratio, val_ratio, max_events)
    raise ValueError(f"Unknown dataset: {dataset}")