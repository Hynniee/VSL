import numpy as np
import pandas as pd
import torch

from src.dataset import VSLDataset, collate_fn


def test_dataset_feature_dim(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    arr = np.random.randn(8, 42, 3).astype(np.float32)
    np.save(proc / "a.npy", arr)
    meta = tmp_path / "meta.csv"
    pd.DataFrame([{"filename": "a.npy", "label": "chu_a", "label_id": 0}]).to_csv(meta, index=False)
    ds = VSLDataset(str(meta), str(proc), max_len=8, feature_mode="hands", train=False)
    seq, length, label = ds[0]
    assert length == 8
    assert label == 0
    assert seq.shape[1] == ds.feature_dim


def test_collate_fn_shapes():
    batch = [
        (torch.zeros(4, 10), 4, 0),
        (torch.ones(6, 10), 6, 1),
    ]
    x, lengths, y = collate_fn(batch)
    assert x.shape == (2, 6, 10)
    assert lengths.tolist() == [4, 6]
    assert y.tolist() == [0, 1]
