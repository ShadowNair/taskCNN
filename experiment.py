from __future__ import annotations

import argparse
import csv
from pathlib import Path

from plotting import plot_comparison
from train import train_model
from utils import ensure_dir


MODEL_DEFAULTS = {
    "lenet": {
        "epochs": 10,
        "batch_size": 128,
        "dropout": 0.0,
        "weight_decay": 1e-4,
        "lrs": {"sgd": 1e-2, "adadelta": 1.0, "nag": 1e-2, "adam": 1e-3},
        "regularizer": "weight_decay",
        "regularization_values": [0.0, 1e-4, 5e-4],
    },
    "vgg16": {
        "epochs": 20,
        "batch_size": 128,
        "dropout": 0.5,
        "weight_decay": 5e-4,
        "lrs": {"sgd": 5e-2, "adadelta": 1.0, "nag": 1e-2, "adam": 1e-3},
        "regularizer": "dropout",
        "regularization_values": [0.0, 0.3, 0.5],
    },
    "resnet34": {
        "epochs": 20,
        "batch_size": 128,
        "dropout": 0.3,
        "weight_decay": 5e-4,
        "lrs": {"sgd": 1e-1, "adadelta": 1.0, "nag": 5e-2, "adam": 1e-3},
        "regularizer": "dropout",
        "regularization_values": [0.0, 0.2, 0.4],
    },
}


def save_summary(summary_rows: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def run_optimizer_sweep(args: argparse.Namespace) -> None:
    defaults = MODEL_DEFAULTS[args.model]
    root_dir = ensure_dir(Path(args.output_dir) / args.model / "optimizers")
    histories = {}
    summary = []

    for optimizer_name in ["sgd", "adadelta", "nag", "adam"]:
        run_dir = root_dir / optimizer_name
        result = train_model(
            model_name=args.model,
            optimizer_name=optimizer_name,
            epochs=args.epochs or defaults["epochs"],
            batch_size=args.batch_size or defaults["batch_size"],
            output_dir=run_dir,
            data_root=args.data_root,
            lr=defaults["lrs"][optimizer_name] if args.lr is None else args.lr,
            dropout=args.dropout if args.dropout is not None else defaults["dropout"],
            weight_decay=args.weight_decay if args.weight_decay is not None else defaults["weight_decay"],
            seed=args.seed,
            num_workers=args.num_workers,
            subset_train=args.subset_train,
            subset_test=args.subset_test,
            force_cpu=args.force_cpu,
            label_smoothing=args.label_smoothing,
            disable_augmentation=args.disable_augmentation,
        )
        histories[optimizer_name.upper()] = result["history"]
        summary.append(result["metrics"])

    save_summary(summary, root_dir / "summary.csv")
    plot_comparison(histories, root_dir, "val_loss", f"{args.model}: сравнение loss для оптимизаторов")
    plot_comparison(histories, root_dir, "val_accuracy", f"{args.model}: сравнение accuracy для оптимизаторов")


def run_regularization_sweep(args: argparse.Namespace) -> None:
    defaults = MODEL_DEFAULTS[args.model]
    root_dir = ensure_dir(Path(args.output_dir) / args.model / "regularization")
    histories = {}
    summary = []

    regularizer = defaults["regularizer"] if args.regularizer is None else args.regularizer
    values = defaults["regularization_values"] if args.values is None else [float(value) for value in args.values.split(",")]

    for value in values:
        if regularizer == "dropout":
            dropout = value
            weight_decay = args.weight_decay if args.weight_decay is not None else defaults["weight_decay"]
            label = f"dropout={value}"
        elif regularizer == "weight_decay":
            dropout = args.dropout if args.dropout is not None else defaults["dropout"]
            weight_decay = value
            label = f"weight_decay={value}"
        else:
            raise ValueError("regularizer должен быть dropout или weight_decay")

        safe_label = label.replace("=", "_").replace(".", "_")
        run_dir = root_dir / safe_label
        result = train_model(
            model_name=args.model,
            optimizer_name=args.optimizer,
            epochs=args.epochs or defaults["epochs"],
            batch_size=args.batch_size or defaults["batch_size"],
            output_dir=run_dir,
            data_root=args.data_root,
            lr=defaults["lrs"][args.optimizer] if args.lr is None else args.lr,
            dropout=dropout,
            weight_decay=weight_decay,
            seed=args.seed,
            num_workers=args.num_workers,
            subset_train=args.subset_train,
            subset_test=args.subset_test,
            force_cpu=args.force_cpu,
            label_smoothing=args.label_smoothing,
            disable_augmentation=args.disable_augmentation,
        )
        histories[label] = result["history"]
        summary.append(result["metrics"])

    save_summary(summary, root_dir / "summary.csv")
    plot_comparison(histories, root_dir, "val_loss", f"{args.model}: loss для регуляризации")
    plot_comparison(histories, root_dir, "val_accuracy", f"{args.model}: accuracy для регуляризации")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Серия экспериментов для лабораторной работы по CNN")
    parser.add_argument("--model", choices=["lenet", "vgg16", "resnet34"], required=True)
    parser.add_argument("--mode", choices=["optimizers", "regularization"], required=True)
    parser.add_argument("--optimizer", choices=["sgd", "adadelta", "nag", "adam"], default="adam")
    parser.add_argument("--regularizer", choices=["dropout", "weight_decay"], default=None)
    parser.add_argument("--values", type=str, default=None, help="Список значений через запятую, например 0.0,0.3,0.5")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-train", type=int, default=None)
    parser.add_argument("--subset-test", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "optimizers":
        run_optimizer_sweep(args)
    else:
        run_regularization_sweep(args)
