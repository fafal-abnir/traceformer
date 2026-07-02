from __future__ import annotations

import argparse
from datetime import datetime
from termcolor import colored
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from utils.seed import set_seed
from data.datamodule import AMLDataModule
from models.lightning_module import LitTemporalRIWalk


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["samld", "amlworld", "amlsim", "bitcoin_alpha", "bitcoin_otc"],
    )
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--max-events", type=int, default=None)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--short-window-days", type=float, default=7.0)
    parser.add_argument("--long-window-days", type=float, default=30.0)
    parser.add_argument("--history-window-days", type=float, default=90.0)

    parser.add_argument("--walk-length", type=int, default=4)
    parser.add_argument("--num-walks", type=int, default=4)
    parser.add_argument("--neighbor-sample-size", type=int, default=10)

    parser.add_argument("--role-dim", type=int, default=32)
    parser.add_argument("--step-dim", type=int, default=128)
    parser.add_argument("--graph-ctx-dim", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--neg-to-pos-ratio", type=int, default=40)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # ablations
    parser.add_argument("--ablate-edge-only", action="store_true")
    parser.add_argument("--ablate-edge-role-only", action="store_true")
    parser.add_argument("--ablate-no-role", action="store_true")
    parser.add_argument("--ablate-no-walks", action="store_true")
    parser.add_argument("--ablate-no-transformer", action="store_true")
    parser.add_argument("--ablate-no-walk-attn", action="store_true")
    parser.add_argument("--ablate-no-edge-gap", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    print(colored(vars(args), "red"))
    set_seed(args.seed)

    experiment_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    lightning_root_dir = "experiments"

    csv_logger = CSVLogger(
        save_dir=lightning_root_dir,
        name=f"{args.dataset}/{experiment_datetime}",
        version=None,
    )
    csv_logger.log_hyperparams(vars(args))

    dm = AMLDataModule(args)
    dm.setup()

    model = LitTemporalRIWalk(
        args=args,
        role_feat_dim=dm.role_feat_dim,
        edge_feat_dim=dm.edge_feat_dim,
        pos_weight=dm.pos_weight,
    )

    accelerator = "cpu" if args.cpu or not torch.cuda.is_available() else "gpu"

    callbacks = [
        EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=args.early_stop_patience,
        ),
        ModelCheckpoint(
            dirpath=f"{lightning_root_dir}/{args.dataset}/{experiment_datetime}/checkpoints",
            monitor="val_pr_auc",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_pr_auc:.4f}",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()