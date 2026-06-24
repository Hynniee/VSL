"""
check_data.py - Kiem tra nhanh du lieu truoc khi train (chan doan loi "nap data crash").

Chay:
  python src/check_data.py --metadata data/metadata_alphabet.csv --labels data/labels_alphabet.json
  python src/check_data.py --metadata data/metadata_words.csv     --labels data/labels_words.json

No se cho biet:
  - metadata co bao nhieu dong (so mau)
  - labels.json co bao nhieu lop
  - thu muc data/processed co bao nhieu file .npy
  - co file .npy nao THIEU so voi metadata khong
  - shape cua mau dau tien (de chac chan doc duoc)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata_alphabet.csv")
    ap.add_argument("--labels", default="data/labels_alphabet.json")
    ap.add_argument("--processed", default="data/processed")
    args = ap.parse_args()

    print("=" * 64)
    meta = Path(args.metadata)
    if not meta.exists():
        print(f"[X] KHONG thay metadata: {meta.resolve()}")
        print("    -> Ban chua chay extract_keypoints.py, hoac sai duong dan.")
        return
    df = pd.read_csv(meta)
    print(f"[i] metadata    : {meta}  ->  {len(df)} dong (mau)")
    if len(df) == 0:
        print("[X] METADATA RONG! Day la nguyen nhan train crash o buoc nap data.")
        print("    -> Luc extract co the KHONG phat hien duoc tay (0 mau).")
        print("    -> Chay lai extract_keypoints.py, neu van 0 mau thi them --keep-empty")
        print("       hoac giam --min-detect-conf 0.3 va kiem tra video/anh co thay tay khong.")
        return
    if "filename" not in df.columns:
        print(f"[X] metadata thieu cot 'filename'. Cac cot dang co: {list(df.columns)}")
        return

    # phan bo theo lop
    if "label" in df.columns:
        print("[i] So mau theo lop:")
        for lab, n in df["label"].value_counts().sort_index().items():
            print(f"      {lab:>12} : {n}")

    labs = Path(args.labels)
    if labs.exists():
        d = json.loads(labs.read_text(encoding="utf-8"))
        print(f"[i] labels.json : {labs}  ->  {len(d)} lop")
    else:
        print(f"[!] Khong thay labels.json: {labs}")

    proc = Path(args.processed)
    if not proc.exists():
        print(f"[X] KHONG thay thu muc processed: {proc.resolve()}")
        print("    -> extract_keypoints.py chua ghi file .npy, hoac sai --processed.")
        return
    npy_files = list(proc.glob("*.npy"))
    print(f"[i] processed   : {proc}  ->  {len(npy_files)} file .npy")

    # doi chieu tung dong metadata voi file .npy
    missing = []
    for fn in df["filename"].astype(str):
        stem = Path(fn).stem
        if not (proc / f"{stem}.npy").exists():
            missing.append(stem)
    if missing:
        print(f"[X] THIEU {len(missing)}/{len(df)} file .npy trong {proc} (vd: {missing[:5]})")
        print("    -> Day la nguyen nhan crash. Chay lai extract_keypoints.py cho dong bo,")
        print("       hoac xoa metadata cu de tao lai tu dau.")
    else:
        print(f"[OK] Tat ca {len(df)} mau deu co file .npy tuong ung.")

    # thu doc mau dau tien
    first = Path(str(df.iloc[0]["filename"])).stem
    fpath = proc / f"{first}.npy"
    if fpath.exists():
        try:
            arr = np.load(fpath)
            print(f"[OK] Mau dau tien {fpath.name}: shape = {arr.shape}, dtype = {arr.dtype}")
            if arr.ndim != 3:
                print(f"[!] Canh bao: shape khong phai 3 chieu (T, joints, 3). Co the gay loi khi train.")
        except Exception as e:
            print(f"[X] Doc {fpath.name} loi: {e}")
    print("=" * 64)
    if not missing and len(df) > 0:
        print("[KET LUAN] Data OK. Neu train van crash, gui m/minh FULL traceback (dong mau do).")


if __name__ == "__main__":
    main()
