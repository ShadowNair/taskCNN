from __future__ import annotations

import gzip
import os
import pickle
import shutil
import struct
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

CIFAR_URLS = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
}


class Compose:
    def __init__(self, transforms: list[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.transforms = transforms

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image


class Normalize:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return torch.flip(image, dims=(2,))
        return image


class RandomCrop:
    def __init__(self, size: int, padding: int = 0, mode: str = "reflect") -> None:
        self.size = size
        self.padding = padding
        self.mode = mode

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            image = F.pad(image, (self.padding, self.padding, self.padding, self.padding), mode=self.mode)

        _, height, width = image.shape
        if height == self.size and width == self.size:
            return image

        top = torch.randint(0, height - self.size + 1, (1,)).item()
        left = torch.randint(0, width - self.size + 1, (1,)).item()
        return image[:, top : top + self.size, left : left + self.size]


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)


def _download_with_mirrors(file_name: str, mirrors: list[str], destination: Path) -> None:
    errors = []
    for mirror in mirrors:
        url = f"{mirror}{file_name}"
        try:
            _download(url, destination)
            return
        except Exception as exc:  # pragma: no cover - network-dependent path
            errors.append(f"{url}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Не удалось скачать файл {file_name}. Попытки:\n{joined}")


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Некорректный файл изображений MNIST: {path}")
        buffer = file.read()
    data = np.frombuffer(buffer, dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Некорректный файл меток MNIST: {path}")
        buffer = file.read()
    return np.frombuffer(buffer, dtype=np.uint8)


def ensure_mnist(root: str | Path) -> Path:
    root = Path(root)
    raw_dir = root / "mnist" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for file_name in MNIST_FILES.values():
        destination = raw_dir / file_name
        if not destination.exists():
            _download_with_mirrors(file_name, MNIST_MIRRORS, destination)

    return raw_dir


def ensure_cifar(root: str | Path, dataset_name: str) -> Path:
    root = Path(root)
    base_dir = root / dataset_name
    expected_dir = base_dir / ("cifar-10-batches-py" if dataset_name == "cifar10" else "cifar-100-python")
    if expected_dir.exists():
        return expected_dir

    archive_path = base_dir / Path(CIFAR_URLS[dataset_name]).name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        _download(CIFAR_URLS[dataset_name], archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(base_dir)
    return expected_dir


class MNISTDataset(Dataset):
    def __init__(self, root: str | Path, train: bool, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        raw_dir = ensure_mnist(root)
        if train:
            self.images = _read_idx_images(raw_dir / MNIST_FILES["train_images"])
            self.labels = _read_idx_labels(raw_dir / MNIST_FILES["train_labels"])
        else:
            self.images = _read_idx_images(raw_dir / MNIST_FILES["test_images"])
            self.labels = _read_idx_labels(raw_dir / MNIST_FILES["test_labels"])
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.tensor(self.images[index], dtype=torch.float32).unsqueeze(0) / 255.0
        image = F.pad(image, (2, 2, 2, 2))
        image = image.repeat(3, 1, 1)
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return image, target


class CIFARDataset(Dataset):
    def __init__(self, root: str | Path, dataset_name: str, train: bool, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        extracted_dir = ensure_cifar(root, dataset_name)
        self.dataset_name = dataset_name
        self.transform = transform

        if dataset_name == "cifar10":
            if train:
                batch_names = [f"data_batch_{idx}" for idx in range(1, 6)]
            else:
                batch_names = ["test_batch"]
            images = []
            labels = []
            for batch_name in batch_names:
                with (extracted_dir / batch_name).open("rb") as file:
                    batch = pickle.load(file, encoding="bytes")
                images.append(batch[b"data"])
                labels.extend(batch[b"labels"])
            self.images = np.concatenate(images, axis=0).reshape(-1, 3, 32, 32)
            self.labels = np.array(labels, dtype=np.int64)
        else:
            file_name = "train" if train else "test"
            with (extracted_dir / file_name).open("rb") as file:
                batch = pickle.load(file, encoding="bytes")
            self.images = batch[b"data"].reshape(-1, 3, 32, 32)
            self.labels = np.array(batch[b"fine_labels"], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.tensor(self.images[index], dtype=torch.float32) / 255.0
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return image, target


def build_datasets(
    data_root: str | Path,
    dataset_name: str,
    train_augment: bool = True,
) -> tuple[Dataset, Dataset]:
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        train_transform = Compose([
            Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]),
        ])
        test_transform = Compose([
            Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]),
        ])
        return MNISTDataset(data_root, train=True, transform=train_transform), MNISTDataset(
            data_root, train=False, transform=test_transform
        )

    if dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        train_transform = []
        if train_augment:
            train_transform.extend([RandomCrop(32, padding=4), RandomHorizontalFlip(0.5)])
        train_transform.append(Normalize(mean, std))
        test_transform = Compose([Normalize(mean, std)])
        return CIFARDataset(data_root, "cifar10", train=True, transform=Compose(train_transform)), CIFARDataset(
            data_root, "cifar10", train=False, transform=test_transform
        )

    if dataset_name == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        train_transform = []
        if train_augment:
            train_transform.extend([RandomCrop(32, padding=4), RandomHorizontalFlip(0.5)])
        train_transform.append(Normalize(mean, std))
        test_transform = Compose([Normalize(mean, std)])
        return CIFARDataset(data_root, "cifar100", train=True, transform=Compose(train_transform)), CIFARDataset(
            data_root, "cifar100", train=False, transform=test_transform
        )

    raise ValueError(f"Неизвестный датасет: {dataset_name}")
