"""predict.py - du doan nhanh tren file .npy va top-k."""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import faulthandler as _fh
    _fh.enable()
except Exception:
    pass

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from hand_utils import preprocess_landmark_array
from pose_utils import prepare_unified
from model import build_model


def load_sequence(npy_path, feature_mode, max_len):
    arr = np.load(npy_path)
    if feature_mode == "unified":
        seq = prepare_unified(arr)
    else:
        seq = preprocess_landmark_array(arr, feature_mode)
    if seq.shape[0] > max_len:
        ids = np.linspace(0, seq.shape[0] - 1, max_len).round().astype(int)
        seq = seq[ids]
    return torch.from_numpy(seq.astype(np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", nargs="+", required=True)
    p.add_argument("--ckpt", default="checkpoints/best.pth")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model = build_model(ckpt["model"], ckpt["feature_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    id2label = {v: k for k, v in ckpt["labels"].items()}
    max_len = int(ckpt.get("max_len", 16))
    feature_mode = ckpt.get("feature_mode", "hands")

    for path in args.npy:
        seq = load_sequence(path, feature_mode, max_len).unsqueeze(0).to(device)
        lengths = torch.tensor([seq.shape[1]], dtype=torch.long, device=device)
        with torch.no_grad():
            prob = F.softmax(model(seq, lengths), dim=1)[0]
        vals, idxs = torch.topk(prob, k=min(args.topk, prob.numel()))
        print(f"\n=== {path} ===")
        for rank, (v, i) in enumerate(zip(vals.tolist(), idxs.tolist()), 1):
            print(f"{rank:2d}. {id2label.get(i, str(i)):20s} {v*100:6.2f}%")


if __name__ == "__main__":
    main()
