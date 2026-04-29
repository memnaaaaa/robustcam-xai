# src/iq_othncc_dataset.py
# Lightweight dataset abstraction for IQ-OTH/NCCD lung cancer CT dataset.
# Returns (image_path, label, class_name) tuples for the evaluation pipeline.
# Separate from train.py's IQOTHNCCDTorchDataset (which returns tensors for training).

import os
import random


class IQOTHNCCDDataset:
    """
    Dataset loader for the IQ-OTH/NCCD lung cancer CT dataset.

    Directory structure expected:
        data/Normal cases/*.jpg      -> label 0
        data/Bengin cases/*.jpg      -> label 1   (note: "Bengin" is the actual folder name)
        data/Malignant cases/*.jpg   -> label 2

    Returns (image_path, label, class_name) 3-tuples. No tensors — pure Python.
    Splits are deterministic: seed=42 matches the split used in train.py.
    """

    CLASS_DIRS  = {0: "Normal cases", 1: "Bengin cases", 2: "Malignant cases"}
    CLASS_NAMES = {0: "Normal", 1: "Benign", 2: "Malignant"}
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        data_root: str,
        split: str = "all",
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        """
        Args:
            data_root:    Path to the dataset root (e.g. "data").
            split:        One of "train", "val", "test", "all".
            train_ratio:  Fraction of data for training (default 0.6).
            val_ratio:    Fraction of data for validation (default 0.2).
            seed:         Random seed for deterministic shuffling (must match train.py).
        """
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"split must be one of 'train','val','test','all'. Got: {split!r}")

        self._split = split
        self._data_root = data_root

        # 1. Discover all images, sorted by filename within each class.
        all_samples: list[tuple[str, int, str]] = []
        for label, class_dir in self.CLASS_DIRS.items():
            class_name = self.CLASS_NAMES[label]
            dir_path = os.path.join(data_root, class_dir)
            if not os.path.isdir(dir_path):
                print(f"[Warning] Class directory not found: {dir_path}")
                continue
            filenames = sorted(
                f for f in os.listdir(dir_path)
                if os.path.splitext(f)[1].lower() in self.VALID_EXTENSIONS
            )
            for fname in filenames:
                all_samples.append((os.path.join(dir_path, fname), label, class_name))

        # 2. Shuffle deterministically — use same RNG pattern as train.py.
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        # 3. Split into train / val / test.
        n_total = len(all_samples)
        n_train = int(n_total * train_ratio)
        n_val   = int(n_total * val_ratio)

        self._splits: dict[str, list[tuple[str, int, str]]] = {
            "train": all_samples[:n_train],
            "val":   all_samples[n_train : n_train + n_val],
            "test":  all_samples[n_train + n_val :],
            "all":   all_samples,
        }

        # 4. Print class distribution for the requested split.
        current = self._splits[split]
        counts = self._count_by_class(current)
        print(f"IQOTHNCCDDataset | split='{split}' | total={len(current)} images")
        for cn, cnt in counts.items():
            print(f"  {cn}: {cnt}")

    # ── public API ─────────────────────────────────────────────────────────────

    def get_all_samples(self) -> list[tuple[str, int, str]]:
        """
        Returns all samples for the current split as (image_path, label, class_name) tuples.
        """
        return list(self._splits[self._split])

    def class_counts(self) -> dict[str, int]:
        """Returns {class_name: count} for the current split."""
        return self._count_by_class(self._splits[self._split])

    def get_split_samples(self, split: str) -> list[tuple[str, int, str]]:
        """
        Returns samples for any named split regardless of self._split.
        Useful when you want train/val/test samples from a single instance.
        """
        if split not in self._splits:
            raise ValueError(f"split must be one of {list(self._splits.keys())}. Got: {split!r}")
        return list(self._splits[split])

    # ── helpers ────────────────────────────────────────────────────────────────

    def _count_by_class(self, samples: list[tuple[str, int, str]]) -> dict[str, int]:
        counts: dict[str, int] = {cn: 0 for cn in self.CLASS_NAMES.values()}
        for _, _, class_name in samples:
            counts[class_name] += 1
        return counts

    def __len__(self) -> int:
        return len(self._splits[self._split])

    def __repr__(self) -> str:
        return (
            f"IQOTHNCCDDataset(data_root={self._data_root!r}, "
            f"split={self._split!r}, n={len(self)})"
        )
