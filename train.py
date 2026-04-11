from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from datasets import build_datasets
from models import build_model
from plotting import plot_single_history, save_history_csv
from utils import Timer, accuracy_from_logits, count_parameters, ensure_dir, get_device, save_json, set_seed


MODEL_DATASETS = {
    "lenet": "mnist",
    "vgg16": "cifar10",
    "resnet34": "cifar100",
}

DEFAULT_LR = {
    "sgd": 1e-2,
    "adadelta": 1.0,
    "nag": 1e-2,
    "adam": 1e-3,
}


def build_optimizer(name: str, parameters, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adadelta":
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    if name == "nag":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Неизвестный оптимизатор: {name}")


def maybe_subset(dataset, subset_size: int | None):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    return Subset(dataset, list(range(subset_size)))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += batch_size

    avg_loss = total_loss / max(total_examples, 1)
    avg_accuracy = total_correct / max(total_examples, 1)
    return avg_loss, avg_accuracy


def train_model(
    model_name: str,
    optimizer_name: str,
    epochs: int,
    batch_size: int,
    output_dir: str | Path,
    data_root: str | Path,
    lr: float | None = None,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    seed: int = 42,
    num_workers: int = 0,
    subset_train: int | None = None,
    subset_test: int | None = None,
    force_cpu: bool = False,
    label_smoothing: float = 0.0,
    disable_augmentation: bool = False,
) -> dict:
    set_seed(seed)
    model_name = model_name.lower()
    optimizer_name = optimizer_name.lower()
    dataset_name = MODEL_DATASETS[model_name]
    lr = DEFAULT_LR[optimizer_name] if lr is None else lr

    output_dir = ensure_dir(output_dir)
    device = get_device(force_cpu=force_cpu)

    train_dataset, test_dataset = build_datasets(data_root, dataset_name, train_augment=not disable_augmentation)
    train_dataset = maybe_subset(train_dataset, subset_train)
    test_dataset = maybe_subset(test_dataset, subset_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(model_name, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    best_accuracy = 0.0
    checkpoint_path = output_dir / "best_model.pt"

    with Timer() as timer:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_epoch(model, test_loader, criterion, optimizer=None, device=device)

            epoch_row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
            }
            history.append(epoch_row)
            print(
                f"[{model_name}/{optimizer_name}] "
                f"epoch {epoch:03d}/{epochs:03d} | "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"test loss={val_loss:.4f} acc={val_acc:.4f}"
            )

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), checkpoint_path)

    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "optimizer": optimizer_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "best_accuracy": round(best_accuracy, 6),
        "last_accuracy": history[-1]["val_accuracy"],
        "train_seconds": round(timer.seconds, 3),
        "device": str(device),
        "parameter_count": count_parameters(model),
        "subset_train": subset_train,
        "subset_test": subset_test,
    }

    save_history_csv(history, output_dir / "history.csv")
    plot_single_history(history, output_dir, f"{model_name}/{optimizer_name}")
    save_json(output_dir / "metrics.json", metrics)

    return {
        "history": history,
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение CNN на PyTorch без torchvision")
    parser.add_argument("--model", choices=["lenet", "vgg16", "resnet34"], required=True)
    parser.add_argument("--optimizer", choices=["sgd", "adadelta", "nag", "adam"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs/single")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-train", type=int, default=None)
    parser.add_argument("--subset-test", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        model_name=args.model,
        optimizer_name=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        data_root=args.data_root,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
        force_cpu=args.force_cpu,
        label_smoothing=args.label_smoothing,
        disable_augmentation=args.disable_augmentation,
    )
