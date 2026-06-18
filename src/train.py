"""
================================================================================
 train.py - HUAN LUYEN MODEL NHAN DANG TU VUNG (buoc 4)
================================================================================

Doc du lieu -> chia train/val -> lap nhieu EPOCH de model hoc -> luu ban tot nhat.

CACH CHAY:
    # chia ngau nhien theo mau (mac dinh)
    python src/train.py --model lstm --epochs 50 --batch 16

    # chia theo NGUOI: tu boc ngau nhien 1 nguoi lam test (can >=2 nguoi)
    python src/train.py --model lstm --val-signer-count 1

    # chi dinh cu the ai lam test
    python src/train.py --model lstm --val-signer ng3

KHAI NIEM NHANH:
    - epoch   : 1 lan model hoc het toan bo du lieu train.
    - loss    : "do sai" cua model, cang nho cang tot.
    - val_acc : do chinh xac tren tap VALIDATION (du lieu model khong train) -> dang tin.
================================================================================
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import VSLDataset, collate_fn
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="data/metadata.csv")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--labels", default="data/labels.json")
    p.add_argument("--model", default="lstm", choices=["lstm", "transformer"])
    p.add_argument("--epochs", type=int, default=50, help="so vong hoc")
    p.add_argument("--batch", type=int, default=16, help="so mau moi lan cap nhat")
    p.add_argument("--lr", type=float, default=1e-3, help="toc do hoc")
    p.add_argument("--max-len", type=int, default=60)
    p.add_argument("--feature-mode", default="hands", choices=["hands", "all"])
    p.add_argument("--val-size", type=float, default=0.2, help="ti le val khi chia ngau nhien")
    p.add_argument("--val-signer", default=None,
                   help="Chi dinh CU THE signer lam val (vd 'ng3'). Nhieu nguoi: 'ng2,ng3'")
    p.add_argument("--val-signer-count", type=int, default=0,
                   help="Neu >0: NGAU NHIEN boc bay nhieu nguoi lam val (vd 1 = giu lai 1 nguoi test)")
    p.add_argument("--out", default="checkpoints/best.pth", help="noi luu model tot nhat")
    p.add_argument("--seed", type=int, default=42, help="hat giong ngau nhien (de lap lai duoc)")
    return p.parse_args()


def evaluate(model, loader, device):
    """Tinh do chinh xac tren 1 tap (khong cap nhat trong so)."""
    model.eval()  # che do danh gia (tat dropout)
    total, correct = 0, 0
    with torch.no_grad():  # khong tinh gradient -> nhanh, do ton bo nho
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            correct += (model(x, lengths).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def main():
    args = parse_args()
    # Co dinh ngau nhien -> chay lai cho ket qua giong nhau (de so sanh)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Tu dung GPU (card RTX) neu co, khong thi CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Thiet bi: {device}")

    with open(args.labels, encoding="utf-8") as f:
        labels = json.load(f)
    num_classes = len(labels)

    ds = VSLDataset(args.metadata, args.processed, args.max_len, args.feature_mode)
    feature_dim = ds.feature_dim
    print(f"[INFO] So lop: {num_classes} | So mau: {len(ds)} | feature_dim: {feature_dim}")

    # ----------------------------------------------------------------------
    # CHIA TRAIN / VALIDATION - co 3 cach (uu tien tu tren xuong):
    #   1) --val-signer ng3      -> chi dinh CU THE ai lam test.
    #   2) --val-signer-count 1  -> NGAU NHIEN boc bay nhieu nguoi lam test,
    #                               phan con lai de train (do kha nang nhan NGUOI MOI).
    #   3) (mac dinh)            -> chia ngau nhien theo MAU, phan tang theo nhan (stratify).
    #
    # LUU Y: cach 1 & 2 phan anh thuc te hon (test nguoi chua tung thay),
    #        nhung can >= 2 nguoi quay. Chi 1 nguoi -> dung cach 3, va val_acc
    #        se "cao ao" vi train/test cung 1 nguoi.
    # ----------------------------------------------------------------------
    idx = np.arange(len(ds))
    signers = ds.df["signer"].astype(str).to_numpy()
    val_signers = None
    if args.val_signer:
        val_signers = [s.strip() for s in args.val_signer.split(",") if s.strip()]
    elif args.val_signer_count > 0:
        uniq = sorted(set(signers))
        if args.val_signer_count >= len(uniq):
            raise SystemExit(f"[LOI] Co {len(uniq)} signer ({uniq}), khong the giu "
                             f"{args.val_signer_count} lam val (phai con nguoi de train).")
        rng = np.random.RandomState(args.seed)
        val_signers = list(rng.choice(uniq, size=args.val_signer_count, replace=False))

    if val_signers is not None:
        mask_val = np.isin(signers, val_signers)
        va_idx, tr_idx = idx[mask_val], idx[~mask_val]
        if len(va_idx) == 0:
            raise SystemExit(f"[LOI] Khong co video nao cua signer {val_signers}")
        if len(tr_idx) == 0:
            raise SystemExit("[LOI] Khong con video nao de train. Can it nhat 2 signer.")
        print(f"[INFO] Chia theo NGUOI: val = {val_signers} ({len(va_idx)} mau) "
              f"| train = {len(tr_idx)} mau")
    else:
        label_ids = ds.df["label_id"].to_numpy()
        tr_idx, va_idx = train_test_split(
            idx, test_size=args.val_size, random_state=args.seed,
            stratify=label_ids if num_classes > 1 else None,
        )

    # DataLoader: tu chia batch + xao tron (train) + goi collate_fn de padding
    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=args.batch,
                           shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(Subset(ds, va_idx), batch_size=args.batch,
                           shuffle=False, collate_fn=collate_fn)

    model = build_model(args.model, feature_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)  # thuat toan cap nhat trong so
    crit = nn.CrossEntropyLoss()  # ham loss cho bai toan phan loai

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()  # che do hoc (bat dropout)
        total, correct, loss_sum = 0, 0, 0.0
        for x, lengths, y in tr_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            opt.zero_grad()           # xoa gradient cu
            logits = model(x, lengths)
            loss = crit(logits, y)    # tinh do sai
            loss.backward()           # lan truyen nguoc -> tinh gradient
            opt.step()                # cap nhat trong so
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        tr_acc = correct / max(total, 1)
        va_acc = evaluate(model, va_loader, device)
        print(f"Epoch {epoch:3d} | loss {loss_sum/max(total,1):.4f} "
              f"| train_acc {tr_acc:.3f} | val_acc {va_acc:.3f}")

        # Luu lai model co val_acc TOT NHAT (kem du thong tin de dung lai)
        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save({
                "model": args.model,
                "state_dict": model.state_dict(),
                "feature_dim": feature_dim,
                "num_classes": num_classes,
                "labels": labels,
                "max_len": args.max_len,
                "feature_mode": args.feature_mode,
            }, args.out)
    print(f"[DONE] Val accuracy tot nhat: {best_acc:.3f} | luu tai: {args.out}")


if __name__ == "__main__":
    main()
