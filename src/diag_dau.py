"""diag_dau.py - Chan doan vi sao model KHONG nhan ky hieu DAU/THANH.

In ra cho tung lop:
  - so mau train (support)
  - recall (ty le nhan dung)
  - do tin cay TRUNG BINH model gan cho dung lop do (du co la top-1 hay khong)
  - lop bi NHAM thanh gi nhieu nhat

Tap trung lam noi bat cac lop dau_* / thanh_* (mau vang).

Chay:
  python src/diag_dau.py --metadata data/metadata_alphabet.csv \\
      --labels data/labels_alphabet.json --ckpt checkpoints/best.pth --device cuda
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import VSLDataset, collate_fn
from model import build_model


def is_dau(name):
    return str(name).startswith(("dau_", "thanh_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata_alphabet.csv")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--labels", default="data/labels_alphabet.json")
    ap.add_argument("--ckpt", default="checkpoints/best.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    if args.device == "cpu":
        device = "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    feature_mode = ckpt.get("feature_mode", "hands")
    max_len = int(ckpt.get("max_len", 16))
    name2id = ckpt["labels"]               # ten -> id
    id2name = {v: k for k, v in name2id.items()}
    num_classes = int(ckpt["num_classes"])

    model = build_model(ckpt["model"], ckpt["feature_dim"], num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = VSLDataset(args.metadata, args.processed, max_len=max_len,
                    feature_mode=feature_mode, train=False, augment=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    print("=" * 78)
    print(f"model={ckpt.get('model')}  device={device}  feature_mode={feature_mode}  "
          f"classes={num_classes}  max_len={max_len}")
    print("=" * 78)

    # Phan bo mau theo lop (tu metadata)
    import pandas as pd
    df = pd.read_csv(args.metadata)
    lab_col = "label" if "label" in df.columns else None
    support_meta = df[lab_col].value_counts().to_dict() if lab_col else {}

    ys, preds, confs = [], [], []
    conf_true = defaultdict(list)   # conf model gan cho DUNG lop
    confusion = defaultdict(Counter)
    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                prob = F.softmax(model(x, lengths), dim=1)
            prob = prob.float().cpu()
            p = prob.argmax(1)
            for i in range(len(y)):
                yi = int(y[i]); pi = int(p[i])
                ys.append(yi); preds.append(pi)
                confs.append(float(prob[i, pi]))
                conf_true[yi].append(float(prob[i, yi]))
                confusion[yi][pi] += 1

    ys = np.array(ys); preds = np.array(preds)
    overall = float((ys == preds).mean()) if len(ys) else 0.0
    print(f"Accuracy tong the: {overall*100:.1f}%   (tren {len(ys)} mau)\n")

    header = f"{'LOP':>14} | {'mau':>4} | {'recall':>6} | {'conf_dung':>9} | nham_nhieu_nhat"
    print(header)
    print("-" * len(header))

    rows = []
    for cid in range(num_classes):
        name = id2name.get(cid, str(cid))
        idx = (ys == cid)
        n = int(idx.sum())
        rec = float((preds[idx] == cid).mean()) if n else 0.0
        ct = float(np.mean(conf_true[cid])) if conf_true[cid] else 0.0
        # lop bi nham thanh gi nhieu nhat (khac chinh no)
        c = confusion[cid].copy()
        c.pop(cid, None)
        if c:
            wrong_id, wrong_n = c.most_common(1)[0]
            wrong = f"{id2name.get(wrong_id, wrong_id)} ({wrong_n})"
        else:
            wrong = "-"
        rows.append((name, n, rec, ct, wrong))

    # In: dau/thanh truoc cho de thay
    def fmt(r):
        name, n, rec, ct, wrong = r
        flag = "  <== DAU/THANH" if is_dau(name) else ""
        return (f"{name:>14} | {n:>4} | {rec*100:>5.0f}% | {ct*100:>8.1f}% | {wrong}{flag}")

    print("\n--- CAC LOP DAU / THANH ---")
    for r in sorted([r for r in rows if is_dau(r[0])], key=lambda x: x[2]):
        print(fmt(r))
    print("\n--- CAC LOP CHU CAI ---")
    for r in sorted([r for r in rows if not is_dau(r[0])], key=lambda x: x[2]):
        print(fmt(r))

    # Ket luan tu dong
    dau_rows = [r for r in rows if is_dau(r[0])]
    if dau_rows:
        avg_sup = np.mean([r[1] for r in dau_rows])
        avg_rec = np.mean([r[2] for r in dau_rows])
        avg_conf = np.mean([r[3] for r in dau_rows])
        chu_rows = [r for r in rows if not is_dau(r[0])]
        avg_sup_c = np.mean([r[1] for r in chu_rows]) if chu_rows else 0
        print("\n" + "=" * 78)
        print(f"DAU/THANH:  mau TB={avg_sup:.0f}  recall TB={avg_rec*100:.0f}%  conf_dung TB={avg_conf*100:.0f}%")
        print(f"CHU CAI  :  mau TB={avg_sup_c:.0f}")
        print("=" * 78)
        if avg_sup < 0.6 * avg_sup_c:
            print("[CHAN DOAN] Lop dau/thanh CO IT MAU hon han chu cai -> mat can bang du lieu.")
            print("  => Quay them mau cho dau/thanh, hoac tang trong so lop khi train.")
        if avg_conf < 0.4:
            print("[CHAN DOAN] conf_dung TB < 40% -> model gan ngo dau/thanh rat thap (kho chot).")
            print("  => Giam nguong cho dau (--dau-threshold trong run_realtime.py), va/ho train lai.")
        if avg_rec < 0.5:
            print("[CHAN DOAN] recall < 50% -> model hay nham dau sang lop khac (xem cot nham_nhieu_nhat).")


if __name__ == "__main__":
    main()
