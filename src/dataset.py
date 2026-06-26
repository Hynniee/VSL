"""dataset.py - pipeline du lieu moi cho VSL.

- hands: feature engineering cho MLP/GRU/Transformer
- hands_raw: raw joints da canonicalize cho ST-GCN
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hand_utils import augment_hand_sequence, canonicalize_hands, preprocess_landmark_array
from pose_utils import (
    NUM_UNIFIED_JOINTS,
    augment_unified_sequence,
    canonicalize_unified,
    embed_hands_only,
    pack_unified,
)


class VSLDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        processed_dir,
        max_len=16,
        feature_mode="hands",
        train=False,
        augment=False,
        seed=42,
    ):
        self.df = pd.read_csv(metadata_csv)
        self.processed_dir = Path(processed_dir)
        self.max_len = max_len
        self.feature_mode = feature_mode
        self.train = train
        self.augment = augment and feature_mode in ("hands", "hands_raw", "unified")
        self.rng = np.random.RandomState(seed)
        self._feature_dim = None

    def __len__(self):
        return len(self.df)

    def _load_raw(self, filename):
        stem = Path(str(filename)).stem
        return np.load(self.processed_dir / f"{stem}.npy")

    def _prepare(self, arr):
        arr = np.asarray(arr)
        if self.feature_mode == "unified":
            return self._prepare_unified(arr)
        if arr.ndim == 3 and self.feature_mode in ("hands", "hands_raw"):
            if arr.shape[1] == 543:
                arr = arr[:, -42:, :]
            elif arr.shape[1] != 42:
                raise ValueError(f"Shape landmark la: {arr.shape}")
            arr = canonicalize_hands(arr)
            if self.train and self.augment:
                arr = augment_hand_sequence(arr, self.rng)
            return preprocess_landmark_array(arr, feature_mode=self.feature_mode)
        return preprocess_landmark_array(arr, feature_mode=self.feature_mode)

    def _prepare_unified(self, arr):
        if arr.ndim != 3:
            raise ValueError(f"Unified can mang 3 chieu, nhan {arr.shape}")
        if arr.shape[2] == 4:
            coords = arr[:, :, :3].astype(np.float32).copy()
            mask = arr[:, :, 3] > 0.5
        elif arr.shape[1] == 42 and arr.shape[2] == 3:
            coords, mask = embed_hands_only(arr)
        elif arr.shape[1] == NUM_UNIFIED_JOINTS and arr.shape[2] == 3:
            coords = arr.astype(np.float32).copy()
            mask = np.any(arr != 0, axis=2)
        elif arr.shape[1] == 543 and arr.shape[2] == 3:
            coords, mask = embed_hands_only(arr[:, -42:, :])
        else:
            raise ValueError(f"Layout unified khong ho tro: {arr.shape}")
        coords = canonicalize_unified(coords, mask)
        if self.train and self.augment:
            coords, mask = augment_unified_sequence(coords, mask, self.rng)
        return pack_unified(coords, mask)

    def _compute_feature_dim(self):
        row = self.df.iloc[0]
        arr = self._prepare(self._load_raw(row["filename"]))
        return int(arr.shape[1])

    @property
    def feature_dim(self):
        if self._feature_dim is None:
            self._feature_dim = self._compute_feature_dim()
        return self._feature_dim

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = self._prepare(self._load_raw(row["filename"]))
        if seq.shape[0] > self.max_len:
            ids = np.linspace(0, seq.shape[0] - 1, self.max_len).round().astype(int)
            seq = seq[ids]
        return torch.from_numpy(seq.astype(np.float32)), seq.shape[0], int(row["label_id"])


def collate_fn(batch):
    seqs, lengths, labels = zip(*batch)
    feat = int(seqs[0].shape[1])
    max_len = max(int(s.shape[0]) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, feat, dtype=torch.float32)
    for i, seq in enumerate(seqs):
        padded[i, :seq.shape[0]] = seq
    return padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
