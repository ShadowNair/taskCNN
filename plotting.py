from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def save_history_csv(history: list[dict], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_single_history(history: list[dict], destination_dir: str | Path, title_prefix: str) -> None:
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix}: Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination_dir / "loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix}: Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination_dir / "accuracy.png", dpi=160)
    plt.close()


def plot_comparison(series: dict[str, list[dict]], destination_dir: str | Path, metric: str, title: str) -> None:
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for label, history in series.items():
        epochs = [row["epoch"] for row in history]
        values = [row[metric] for row in history]
        plt.plot(epochs, values, label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination_dir / f"{metric}_comparison.png", dpi=160)
    plt.close()
