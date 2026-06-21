"""
===============================================================================
  predict.py - DU DOAN NHAN TU MOT HOAC NHIEU FILE KEYPOINT (.npy)
===============================================================================

Dung de thu model da train tren file .npy, KHONG can webcam.

QUAN TRONG (da sua so voi ban cu):
  - Ban cu lay TOP-K cua 1 video roi nhoi vao tokens_to_text -> SAI, vi top-k la
    cac PHUONG AN khac nhau cho CUNG 1 ky hieu, khong phai 1 chuoi ky hieu.
  - Ban moi: nhan NHIEU file .npy theo dung THU TU ky hieu. Moi file -> lay
    du doan TOP-1, ghep cac top-1 lai thanh chuoi roi rap van (--alphabet).

CACH CHAY:
    # 1 ky hieu / 1 tu
    python src/predict.py --npy data/processed/xinchao_ng1_01.npy

    # nhieu ky hieu chu cai, rap thanh van ban tieng Viet (dung thu tu!)
    python src/predict.py --alphabet \
        --npy data/processed/chu_a_ng1_01.npy \
              data/processed/dau_mu_ng1_01.npy \
              data/processed/thanh_sac_ng1_01.npy
    # -> 'a^/' => 'a' + dau mu + dau sac = 'a'
===============================================================================
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model
from unicode_utils import tokens_to_text


def load_seq(npy_path, feature_mode, max_len):
    """Nap .npy va xu ly GIONG HET luc train (cung feature_mode, cung max_len)."""
    arr = np.load(npy_path)
    if arr.ndim == 3:
        if feature_mode == "hands":
            arr = arr[:, -42:, :]
        arr = arr.reshape(arr.shape[0], -1)
    return torch.from_numpy(arr.astype(np.float32)[:max_len])


def predict_one(model, ckpt, npy_path, topk):
    """Doan 1 file -> tra ve list [(label_id, score)] theo thu tu giam dan."""
    seq = load_seq(npy_path, ckpt["feature_mode"], ckpt["max_len"])
    x = seq.unsqueeze(0)
    lengths = torch.tensor([seq.shape[0]])
    with torch.no_grad():
        prob = F.softmax(model(x, lengths), dim=1)[0]
    k = min(topk, prob.numel())
    vals, idxs = torch.topk(prob, k)
    return [(int(i), float(v)) for v, i in zip(vals, idxs)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True, nargs="+",
                   help="mot hoac nhieu file .npy, theo dung THU TU ky hieu")
    p.add_argument("--ckpt", default="checkpoints/best.pth", help="model da train")
    p.add_argument("--topk", type=int, default=3, help="hien may dap an kha di nhat / file")
    p.add_argument("--alphabet", action="store_true",
                   help="ghep TOP-1 cua tung file thanh van ban tieng Viet (rap van)")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ckpt["model"], ckpt["feature_dim"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    id2label = {v: k for k, v in ckpt["labels"].items()}

    top1_tokens = []
    for path in args.npy:
        results = predict_one(model, ckpt, path, args.topk)
        print(f"\n=== {path} ===")
        for rank, (idx, score) in enumerate(results, 1):
            print(f"  {rank}. {id2label.get(idx, str(idx)):20s} {score*100:5.1f}%")
        top1_tokens.append(id2label.get(results[0][0], str(results[0][0])))

    print("\n--- TONG HOP (theo TOP-1 cua tung file) ---")
    if args.alphabet:
        text = tokens_to_text(top1_tokens)
        print(f"Van ban (rap van): '{text}'")
    else:
        print("Chuoi tu:", " ".join(top1_tokens))


if __name__ == "__main__":
    main()
