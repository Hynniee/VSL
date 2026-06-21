"""
===============================================================================
  dataset.py - NAP DU LIEU KEYPOINT CHO MODEL (buoc 2)
===============================================================================

MUC DICH
--------
PyTorch can mot doi tuong "Dataset" de biet: co bao nhieu mau, va moi mau lay
ra nhu the nao. File nay doc metadata.csv, nap file .npy tuong ung, va tra ve
(chuoi keypoint, do dai, nhan) cho tung video.

KHAI NIEM:
    - Dataset   : "kho" du lieu, lay tung mau theo chi so.
    - DataLoader: (o train.py) gom nhieu mau thanh 1 "batch" de train nhanh.
    - collate_fn: cach gom 1 batch khi cac video DAI NGAN KHAC NHAU (phai padding).
===============================================================================
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VSLDataset(Dataset):
    """Dataset cho nhan dang TU VUNG ROI (moi video = 1 nhan).

    feature_mode quyet dinh lay bao nhieu diem lam dac trung:
        "hands" = CHI 2 ban tay (42 diem) -> tranh model "hoc khuon mat/danh tinh".
                  Day la mac dinh va NEN dung.
        "all"   = toan bo 543 diem (pose + face + 2 tay).
    """

    def __init__(self, metadata_csv, processed_dir, max_len=60, feature_mode="hands"):
        self.df = pd.read_csv(metadata_csv)       # bang liet ke cac mau
        self.processed_dir = Path(processed_dir)  # noi chua .npy
        self.max_len = max_len                    # cat video qua dai
        self.feature_mode = feature_mode
        self._feature_dim = None                  # cache sau lan dau tien

    def __len__(self):
        return len(self.df)

    def _load_npy(self, filename):
        """Nap 1 file .npy -> mang 2D (T, F) san sang cho model."""
        stem = Path(str(filename)).stem
        arr = np.load(self.processed_dir / f"{stem}.npy")
        if arr.ndim == 2:
            # Da o dang (T, F) -> dung luon
            return arr.astype(np.float32)
        # arr dang (T, 543, 3):
        if self.feature_mode == "hands":
            # Lay 42 DIEM CUOI = 2 ban tay (xem thu tu o extract_keypoints.py)
            arr = arr[:, -42:, :]
        # "Duoi phang" (T, diem, 3) -> (T, diem*3) de dua vao mang
        return arr.reshape(arr.shape[0], -1).astype(np.float32)

    def _compute_feature_dim(self):
        first = self._load_npy(self.df.iloc[0]["filename"])
        return first.shape[1]

    @property
    def feature_dim(self):
        """So chieu dac trung moi frame (model can de biet kich thuoc dau vao)."""
        if self._feature_dim is None:
            self._feature_dim = self._compute_feature_dim()
        return self._feature_dim

    def __getitem__(self, idx):
        """Tra 1 mau: (chuoi keypoint, do dai that, nhan dang so)."""
        row = self.df.iloc[idx]
        seq = self._load_npy(row["filename"])
        if seq.shape[0] > self.max_len:
            seq = seq[: self.max_len]  # cat neu qua dai
        return torch.from_numpy(seq), seq.shape[0], int(row["label_id"])


def collate_fn(batch):
    """Gom 1 batch + PADDING cac chuoi ve cung do dai.

    Vi moi video so frame khac nhau, khong the xep thang thanh 1 khoi.
    Cach lam: lay do dai lon nhat trong batch, cac chuoi ngan hon -> dem 0.
    'lengths' luu do dai THAT de model bo qua phan dem.
    """
    seqs, lengths, labels = zip(*batch)
    feat = seqs[0].shape[1]
    max_len = max(s.shape[0] for s in seqs)
    padded = torch.zeros(len(seqs), max_len, feat, dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0]] = s  # phan con lai giu nguyen 0 (dem)
    return padded, torch.tensor(lengths), torch.tensor(labels)
