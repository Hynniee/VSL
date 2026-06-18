"""
================================================================================
 predict.py - DU DOAN NHAN TU 1 FILE KEYPOINT (.npy)  [kiem tra nhanh]
================================================================================

Dung de thu model da train tren 1 file .npy bat ky, KHONG can webcam.

CACH CHAY:
    python src/predict.py --npy data/processed/xinchao_ng1_01.npy --ckpt checkpoints/best.pth
================================================================================
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model


def load_seq(npy_path, feature_mode, max_len):
    """Nap .npy va xu ly GIONG HET luc train (cung feature_mode, cung max_len)."""
    arr = np.load(npy_path)
    if arr.ndim == 3:  # (T, 543, 3)
        if feature_mode == "hands":
            arr = arr[:, -42:, :]  # chi 2 tay
        arr = arr.reshape(arr.shape[0], -1)
    return torch.from_numpy(arr.astype(np.float32)[:max_len])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True, help="file keypoint can doan")
    p.add_argument("--ckpt", default="checkpoints/best.pth", help="model da train")
    p.add_argument("--topk", type=int, default=3, help="hien may dap an kha di nhat")
    args = p.parse_args()

    # Nap checkpoint -> dung lai DUNG model luc train
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ckpt["model"], ckpt["feature_dim"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    id2label = {v: k for k, v in ckpt["labels"].items()}  # so -> ten nhan
    seq = load_seq(args.npy, ckpt["feature_mode"], ckpt["max_len"])
    x = seq.unsqueeze(0)                  # them chieu batch -> (1, T, F)
    lengths = torch.tensor([seq.shape[0]])
    with torch.no_grad():
        # softmax -> doi diem so thanh xac suat (%)
        prob = F.softmax(model(x, lengths), dim=1)[0]
    topk = torch.topk(prob, min(args.topk, prob.numel()))
    print("Ket qua du doan:")
    for score, idx in zip(topk.values, topk.indices):
        print(f"  {id2label.get(int(idx), int(idx)):20s}  {score.item()*100:.1f}%")


if __name__ == "__main__":
    main()
