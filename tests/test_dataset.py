"""
Tests for dataset.VSLDataset() and collate_fn().
Run: pytest tests/test_dataset.py

Luu y: can thu vien 'pytest' cai dat truoc: pip install pytest
"""
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import VSLDataset, collate_fn


@pytest.fixture()
def fake_dataset(tmp_path=None):
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    proc = Path(tmp_path) / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(5):
        label = "hello" if i < 3 else "bye"
        fname = f"{label}_{i:02d}.npy"
        arr = np.random.randn(10, 42).astype(np.float32)
        np.save(proc / fname, arr)
        rows.append(
            {
                "filename": fname,
                "label": label,
                "label_id": 0 if label == "hello" else 1,
                "signer": "ng1",
                "source": "test",
            }
        )
    meta = Path(tmp_path) / "metadata.csv"
    pd.DataFrame(rows).to_csv(meta, index=False)
    return VSLDataset(str(meta), str(proc), max_len=20, feature_mode="hands")


def test_dataset_length(fake_dataset):
    assert len(fake_dataset) == 5


def test_feature_dim_cached(fake_dataset):
    d1 = fake_dataset.feature_dim
    d2 = fake_dataset.feature_dim
    assert d1 == d2 == 42


def test_getitem_shape(fake_dataset):
    seq, length, label = fake_dataset[0]
    assert seq.shape == (10, 42)
    assert length == 10
    assert label in (0, 1)


def test_getitem_truncation(tmp_path):
    proc = Path(tmp_path) / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    fname = "long_00.npy"
    arr = np.random.randn(100, 42).astype(np.float32)
    np.save(proc / fname, arr)
    meta = Path(tmp_path) / "metadata.csv"
    pd.DataFrame(
        [
            {
                "filename": fname,
                "label": "long",
                "label_id": 0,
            }
        ]
    ).to_csv(meta, index=False)
    ds = VSLDataset(str(meta), str(proc), max_len=50, feature_mode="hands")
    seq, length, label = ds[0]
    assert seq.shape[0] == 50
    assert length == 50


def test_collate_fn_padding():
    batch = [
        (torch.zeros(4, 6), 4, 0),
        (torch.ones(7, 6), 7, 1),
        (torch.full((3, 6), 2.0), 3, 0),
    ]
    padded, lengths, labels = collate_fn(batch)
    assert padded.shape == (3, 7, 6)
    assert lengths.tolist() == [4, 7, 3]
    assert labels.tolist() == [0, 1, 0]
    assert padded[1, 4, 0].item() == 1.0
    assert padded[2, 3, 0].item() == 0.0
