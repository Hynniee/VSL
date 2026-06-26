"""compare_models.py - so sanh nhanh cac model moi, co ca ST-GCN."""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import faulthandler as _fh
    _fh.enable()
except Exception:
    pass

import argparse
from pathlib import Path

import pandas as pd

from model import available_models
from train import parse_args as train_parse_args
from train import train_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(available_models()))
    p.add_argument("--metadata", default="data/metadata.csv")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--labels", default="data/labels.json")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=16)
    p.add_argument("--feature-mode", default="hands", choices=["hands", "hands_raw", "all", "unified"])
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--report", default="checkpoints/comparison")
    return p.parse_args()


def build_train_args(cargs, model_name):
    a = train_parse_args([])
    a.model = model_name
    a.metadata = cargs.metadata
    a.processed = cargs.processed
    a.labels = cargs.labels
    a.epochs = cargs.epochs
    a.batch = cargs.batch
    a.lr = cargs.lr
    a.weight_decay = cargs.weight_decay
    a.label_smoothing = cargs.label_smoothing
    a.max_len = cargs.max_len
    a.feature_mode = cargs.feature_mode
    a.val_size = cargs.val_size
    a.seed = cargs.seed
    a.workers = cargs.workers
    a.patience = cargs.patience
    a.device = cargs.device
    a.no_amp = cargs.no_amp
    a.out = str(Path(cargs.out_dir) / f"{model_name}.pth")
    a.metrics_out = str(Path(cargs.out_dir) / f"{model_name}_metrics.json")
    return a


def df_to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    cargs = parse_args()
    models = [m.strip() for m in cargs.models.split(",") if m.strip()]
    Path(cargs.out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for name in models:
        print("\n" + "=" * 70)
        print(f"TRAIN MODEL: {name}")
        print("=" * 70)
        m = train_model(build_train_args(cargs, name))
        rows.append({
            "Model": name,
            "Feature mode": m.get("feature_mode", cargs.feature_mode),
            "Val Acc (%)": round(m["val_accuracy"] * 100, 2),
            "Macro F1 (%)": round(m["macro_f1"] * 100, 2),
            "Precision (%)": round(m["macro_precision"] * 100, 2),
            "Recall (%)": round(m["macro_recall"] * 100, 2),
            "Infer (ms/mau)": round(m["inference_ms_per_sample"], 3),
            "Params": m["num_params"],
            "Train (s)": round(m["train_time_sec"], 1),
        })
    df = pd.DataFrame(rows).sort_values(["Macro F1 (%)", "Val Acc (%)"], ascending=False)
    csv_path = cargs.report + ".csv"
    md_path = cargs.report + ".md"
    df.to_csv(csv_path, index=False)
    Path(md_path).write_text("# Bang so sanh model\n\n" + df_to_markdown(df) + "\n", encoding="utf-8")
    print("\n" + "=" * 70)
    print("BANG SO SANH")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\n[DONE] Da luu: {csv_path} va {md_path}")


if __name__ == "__main__":
    main()
