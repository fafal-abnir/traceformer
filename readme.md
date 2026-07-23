# TraceFormer



Official implementation of:

> **TraceFormer: A Role-Aware Temporal Trace Transformer for Money Laundering Detection**


TraceFormer is a temporal graph model for suspicious-transaction classification in continuous-time financial networks. For each target transaction, it constructs multiple backward time-respecting traces from the sender and receiver, represents historical interactions using event-time behavioral roles and transaction attributes, and aggregates the resulting trace representations for prediction.


---

## Overview

Financial transactions naturally form directed temporal multigraphs in which multiple transfers may occur between the same accounts at different times. Suspicious activity is often difficult to identify from a transaction in isolation because relevant evidence may be distributed across earlier interactions.

TraceFormer models this historical context through:

- temporal behavioral features computed from strictly pre-event activity
- backward time-respecting walks sampled from both transaction endpoints
- role-conditioned tokens describing historical interactions
- Transformer-based trace encoding
- attention-based aggregation over multiple sampled traces
- current transaction attributes and sender-side recency information

All historical features and walks are constructed using only events that occurred before the target transaction timestamp.

---

## Supported Datasets

The current implementation supports seven datasets.

### AML-oriented transaction networks

- **SAML-D**
- **AMLWorld-HI-Small**
- **AMLSim**
- **AscendEXHacker**
- **UpbitHack**

### Auxiliary temporal interaction networks

- **Bitcoin-Alpha**
- **Bitcoin-OTC**

Dataset-specific preprocessing includes:

- chronological event ordering;
- sender and receiver reindexing;
- timestamp conversion;
- transaction-amount transformation;
- continuous-feature scaling using training data;
- categorical-feature encoding;
- construction of transaction-level labels and edge features.

Bitcoin-Alpha and Bitcoin-OTC are signed trust networks rather than financial transaction networks. Negative ratings are treated as positive-class edges, while the rating value itself is excluded from the input features.

---

---

## Repository Structure

```text
traceformer/
├── assets/traceformer_arch.png
├── data/
│   ├── datasets.py          # Dataset preprocessing and event construction
│   └── datamodule.py        # Splits, sampling, weighting, and batch preparation
├── graph/
│   ├── temporal_store.py    # Time-ordered historical edge lookup
│   ├── temporal_features.py # Temporal behavioral features
│   └── walk_sampler.py      # Backward time-respecting walks
├── models/
│   ├── encoders.py          # Role, step, Transformer, and attention encoders
│   ├── temporal_model.py    # TraceFormer and ablations
│   └── lightning_module.py  # Training, validation, and testing
├── utils/
│   ├── batching.py
│   ├── metrics.py
│   └── seed.py
├── train.py
└── requirements.txt
```



## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.4
- PyTorch Geometric
- PyTorch Lightning
- CUDA-capable GPU recommended for large datasets
- Poetry

---

## Installation

Clone the repository:

```bash
git clone https://github.com/fafal-abnir/traceformer
cd traceformer-aml
```

Create and activate the virtual environment and install dependencies:
```bash
poetry install
poetry shell
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch-geometric
pip3 install -r requirements.txt 
```

## Data Preparation

Pass the dataset file through `--csv`.

```text
raw_data/
├── SAML-D.csv
├── AMLWorld-HI-Small.csv
├── AMLSim.csv
├── soc-sign-bitcoinalpha.csv
├── soc-sign-bitcoinotc.csv
├── AscendEXHacker_transaction.csv
└── UpbitHack_transaction.csv
```

See `data/datasets.py` for expected schemas and preprocessing.

---

## Training

```bash
python train.py \
  --dataset <dataset-name> \
  --csv <path-to-csv>
```

Supported identifiers:

```text
samld
amlworld
amlsim
bitcoin_alpha
bitcoin_otc
ascendexhacker
upbithack
```

Example:

```bash
python train.py \
  --dataset amlsim \
  --csv raw_data/AMLSim.csv \
  --epochs 100 \
  --batch-size 2048 \
  --eval-batch-size 4096 \
  --short-window-days 7 \
  --long-window-days 30 \
  --history-window-days 90 \
  --walk-length 1 \
  --num-walks 16 \
  --neighbor-sample-size 10 \
  --lr 0.001 \
  --early-stop-patience 10 \
  --seed 42
```

CPU example:

```bash
python train.py \
  --dataset bitcoin_otc \
  --csv raw_data/soc-sign-bitcoinotc.csv \
  --cpu
```

---

## Main Arguments

| Group | Arguments and defaults                                                       |
|---|------------------------------------------------------------------------------|
| Data | `--dataset` required, `--csv` required                                       |
| Split | `--train-ratio 0.70`, `--val-ratio 0.15`                                     |
| Windows | `--short-window-days 7`, `--long-window-days 30`, `--history-window-days 90` |
| Walks | `--walk-length 4`, `--num-walks 4`, `--neighbor-sample-size 10`              |
| Dimensions | `--role-dim 32`, `--step-dim 128`, `--graph-ctx-dim 128`                     |
| Optimization | `--lr 1e-3`, `--weight-decay 1e-5`                                           |
| Imbalance | `--neg-to-pos-ratio 40`, `--max-pos-weight 50`                               |
| Runtime | `--seed 42`, `--cpu`                                                         |

Run `python train.py --help` for the complete interface.

---

---

## Temporal Leakage and Reproducibility

For every target event at  time  $`t_i`$:

- only interactions with $`t_j<t_i`$ are retrieved;
- the target event is excluded from its own history;
- equal-timestamp events are excluded from one another's history;
- preprocessing statistics are fitted on the training portion where applicable;
- validation and test events do not affect earlier representations.

Walks for the same node-time query are cached within a run. Different seeds may produce different sampled histories.

---

## Reference Results

The following results are computed over seeds **1–5** and reported as mean ± population standard deviation. The selected configurations are the runs that match the results reported in the paper.


| Dataset | Walk length L | Walks per endpoint M | Candidate size K | AUCPR | AUROC |
|:--|--:|--:|--:|:--|:--|
| Bitcoin-OTC | 1 | 16 | 10 | 0.394 ± 0.013 | 0.771 ± 0.006 |
| Bitcoin-Alpha | 1 | 16 | 10 | 0.306 ± 0.021 | 0.701 ± 0.033 |
| AMLSim | 1 | 16 | 10 | 0.670 ± 0.045 | 0.986 ± 0.002 |
| SAML-D | 1 | 16 | 10 | 0.951 ± 0.007 | 0.999 ± 0.000 |
| AMLWorld-HI-Small | 2 | 40 | 20 | 0.152 ± 0.028 | 0.951 ± 0.003 |
| AscendEXHacker | 1 | 16 | 10 | 0.528 ± 0.057 | 0.917 ± 0.015 |
| UpbitHack | 2 | 40 | 10 | 0.523 ± 0.009 | 0.776 ± 0.012 |