"""train.py - RTX 4050 optimized training.

Thay đổi so với bản cũ:
  - build_optimizer() tách A_learn group (wd=0) -> graph học đúng
  - OneCycleLR thay CosineAnnealingLR -> warmup + annealing mượt hơn
  - --mixup flag: dùng collate_fn_mixup với alpha=0.2
  - torch.compile(model, mode="reduce-overhead") khi CUDA (RTX 4050)
  - Model warmup (1 dummy forward pass) trước khi train để BN ổn định
  - amp GradScaler cho FP16 trên 4050
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import faulthandler as _fh
    _fh.enable()
except Exception:
    pass

import argparse
import copy
import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import VSLDataset, collate_fn, collate_fn_mixup
from model import available_models, build_model, build_optimizer


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--metadata",      default="data/metadata.csv")
    p.add_argument("--processed",     default="data/processed")
    p.add_argument("--labels",        default="data/labels.json")
    p.add_argument("--model",         default="stgcn", choices=available_models())
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch",         type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--max-len",       type=int,   default=16)
    p.add_argument("--feature-mode",  default="hands_raw",
                   choices=["hands", "hands_raw", "all", "unified"])
    p.add_argument("--val-size",      type=float, default=0.2)
    p.add_argument("--out",           default="checkpoints/best.pth")
    p.add_argument("--metrics-out",   default=None)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--workers",       type=int,   default=0,
                   help="Windows nên để 0 cho ổn định")
    p.add_argument("--patience",      type=int,   default=12)
    p.add_argument("--device",        default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-amp",        action="store_true")
    p.add_argument("--no-compile",    action="store_true",
                   help="Tắt torch.compile (dùng nếu gặp lỗi compile)")
    p.add_argument("--mixup",         type=float, default=0.2,
                   help="Alpha cho Mixup augmentation (0=tắt)")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(ds, idxs, batch, workers, shuffle, device,
                mixup_alpha=0.0, num_classes=None):
    subset = Subset(ds, idxs)
    if shuffle and mixup_alpha > 0:
        cfn = partial(collate_fn_mixup, alpha=mixup_alpha, num_classes=num_classes)
    else:
        cfn = collate_fn
    return DataLoader(
        subset,
        batch_size=batch,
        shuffle=shuffle,
        collate_fn=cfn,
        num_workers=workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(workers > 0),
    )


def macro_metrics(ys, ps, label_names):
    acc = accuracy_score(ys, ps)
    prec, rec, f1, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    p_c, r_c, f_c, sup = precision_recall_fscore_support(
        ys, ps, labels=list(range(len(label_names))), zero_division=0
    )
    per_class = {
        label_names[i]: {
            "precision": float(p_c[i]),
            "recall":    float(r_c[i]),
            "f1":        float(f_c[i]),
            "support":   int(sup[i]),
        }
        for i in range(len(label_names))
    }
    cm = confusion_matrix(ys, ps, labels=list(range(len(label_names)))).tolist()
    return {
        "val_accuracy":      float(acc),
        "macro_precision":   float(prec),
        "macro_recall":      float(rec),
        "macro_f1":          float(f1),
        "per_class":         per_class,
        "confusion_matrix":  cm,
        "label_names":       label_names,
    }


def evaluate(model, loader, device, amp_enabled, label_names):
    model.eval()
    ys, ps = [], []
    n = 0
    t_total = 0.0
    with torch.no_grad():
        for batch in loader:
            x, lengths, y = batch
            # eval loader không dùng mixup nên y là tensor bình thường
            if isinstance(y, (tuple, list)):
                y = y[0]
            x       = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            y       = y.to(device, non_blocking=True)
            t0 = time.perf_counter()
            with torch.autocast(device_type=device, enabled=amp_enabled):
                logits = model(x, lengths)
            if device == "cuda":
                torch.cuda.synchronize()
            t_total += time.perf_counter() - t0
            pred = logits.argmax(1)
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())
            n += y.size(0)
    out = macro_metrics(ys, ps, label_names)
    out["inference_ms_per_sample"] = float(t_total / max(n, 1) * 1000)
    out["num_val_samples"] = int(n)
    return out


def mixup_loss(criterion, logits, y_tuple):
    """Tính CrossEntropyLoss cho Mixup: lam*CE(y_i) + (1-lam)*CE(y_j)."""
    y_i, y_j, lam = y_tuple
    return lam * criterion(logits, y_i) + (1 - lam) * criterion(logits, y_j)


def tune_args_for_model(args):
    if args.model == "unified":
        if args.feature_mode != "unified":
            args.feature_mode = "unified"
            print("[INFO] Model unified tự động chuyển feature_mode -> unified", flush=True)
        args.lr           = min(args.lr, 2e-4)
        args.weight_decay = max(args.weight_decay, 3e-4)
        args.label_smoothing = min(args.label_smoothing, 0.03)
        args.max_len      = max(args.max_len, 24)
        args.patience     = max(args.patience, 14)
        return args
    if args.model == "stgcn":
        if args.feature_mode == "hands":
            args.feature_mode = "hands_raw"
            print("[INFO] ST-GCN tự động chuyển feature_mode -> hands_raw", flush=True)
        args.lr           = min(args.lr, 2e-4)
        args.weight_decay = max(args.weight_decay, 3e-4)
        args.label_smoothing = min(args.label_smoothing, 0.03)
        args.max_len      = max(args.max_len, 12)
        args.patience     = max(args.patience, 12)
    return args


def warmup_model(model, device, feature_dim, max_len):
    """Chạy 1 dummy forward pass để warm up BN running stats và CUDA kernel.

    Quan trọng cho RTX 4050: tránh lag frame đầu tiên khi inference.
    """
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, max_len, feature_dim, device=device)
        dummy_len = torch.tensor([max_len, max_len // 2], dtype=torch.long, device=device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            model(dummy, dummy_len)
    if device == "cuda":
        torch.cuda.synchronize()
    model.train()
    print("[INFO] Model warmup xong", flush=True)


def train_model(args):
    args = tune_args_for_model(args)
    set_seed(args.seed)
    device      = resolve_device(args.device)
    amp_enabled = (device == "cuda" and not args.no_amp)
    if device == "cuda":
        # RTX 4050: benchmark=True + allow_tf32 để tận dụng tensor cores
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[INFO] Thiết bị: {device} | AMP: {amp_enabled} | Model: {args.model}", flush=True)

    labels      = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    num_classes = len(labels)
    id2label    = {v: k for k, v in labels.items()}
    label_names = [id2label[i] for i in range(num_classes)]

    print("[INFO] Đang nạp dataset...", flush=True)
    ds_train = VSLDataset(args.metadata, args.processed, max_len=args.max_len,
                          feature_mode=args.feature_mode, train=True, augment=True, seed=args.seed)
    ds_eval  = VSLDataset(args.metadata, args.processed, max_len=args.max_len,
                          feature_mode=args.feature_mode, train=False, augment=False, seed=args.seed)
    feature_dim = ds_train.feature_dim
    print(f"[INFO] Số lớp: {num_classes} | Số mẫu: {len(ds_train)} | "
          f"feature_dim: {feature_dim} | mode: {args.feature_mode}", flush=True)

    idx       = np.arange(len(ds_train))
    label_ids = ds_train.df["label_id"].to_numpy()
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=label_ids if num_classes > 1 else None,
    )
    print(f"[INFO] Train/Val = {len(tr_idx)}/{len(va_idx)}", flush=True)

    tr_loader = make_loader(ds_train, tr_idx, args.batch, args.workers, True,  device,
                            mixup_alpha=args.mixup, num_classes=num_classes)
    va_loader = make_loader(ds_eval,  va_idx, args.batch, args.workers, False, device)

    model = build_model(args.model, feature_dim, num_classes).to(device)

    # RTX 4050: torch.compile reduce-overhead (~30-50% faster inference per batch)
    # Triton chưa hỗ trợ Windows native -> tự động dùng backend eager trên win32
    use_compile = (device == "cuda" and not args.no_compile)
    compile_backend = "disabled"
    if use_compile:
        import sys, torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True  # fallback eager thay vì crash
        _has_triton = (sys.platform != "win32" and __import__("importlib").util.find_spec("triton") is not None)
        _backend = "inductor" if _has_triton else "eager"
        _mode    = "reduce-overhead" if _has_triton else "default"
        try:
            print(f"[INFO] torch.compile backend={_backend} mode={_mode} (lần đầu có thể chậm ~30s)", flush=True)
            model = torch.compile(model, backend=_backend, mode=_mode)
            compile_backend = _backend
            print(f"[INFO] torch.compile OK ({_backend})", flush=True)
        except Exception as e:
            print(f"[WARN] torch.compile thất bại ({e}), chạy eager", flush=True)
            use_compile = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Số tham số: {n_params:,}", flush=True)

    # Warmup model trước khi train
    warmup_model(model, device, feature_dim, args.max_len)

    class_counts  = np.bincount(label_ids[tr_idx], minlength=num_classes).astype(np.float32)
    class_weights = class_counts.sum() / np.clip(class_counts, 1.0, None)
    class_weights = torch.tensor(class_weights / class_weights.mean(),
                                  dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # FIX: dùng build_optimizer để A_learn có wd=0
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # OneCycleLR: warmup 10% epochs rồi annealing — hội tụ nhanh hơn CosineAnnealing
    steps_per_epoch = len(tr_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,           # 10% warmup
        anneal_strategy="cos",
        div_factor=10.0,         # lr bắt đầu = max_lr/10
        final_div_factor=100.0,  # lr kết thúc = max_lr/1000
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_score   = -1.0
    best_state   = None
    best_metrics = None
    bad_epochs   = 0
    t_train0     = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for batch in tr_loader:
            x, lengths, y_info = batch
            x       = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            use_mixup = isinstance(y_info, (tuple, list)) and len(y_info) == 3 and not torch.is_tensor(y_info)
            if use_mixup:
                y_i, y_j, lam = y_info
                y_i = y_i.to(device, non_blocking=True)
                y_j = y_j.to(device, non_blocking=True)
            else:
                y = y_info.to(device, non_blocking=True) if torch.is_tensor(y_info) else y_info[0].to(device, non_blocking=True)
                use_mixup = False

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=amp_enabled):
                logits = model(x, lengths)
                if use_mixup:
                    loss = mixup_loss(criterion, logits, (y_i, y_j, lam))
                    y_for_acc = y_i  # dùng label chính để tính accuracy xấp xỉ
                else:
                    loss = criterion(logits, y)
                    y_for_acc = y

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item() * y_for_acc.size(0)
            correct  += (logits.argmax(1) == y_for_acc).sum().item()
            total    += y_for_acc.size(0)

        train_acc   = correct / max(total, 1)
        val_metrics = evaluate(model, va_loader, device, amp_enabled, label_names)
        score       = val_metrics["macro_f1"]
        lr_now      = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | loss {loss_sum/max(total,1):.4f} | "
            f"train_acc {train_acc:.3f} | val_acc {val_metrics['val_accuracy']:.3f} | "
            f"val_f1 {val_metrics['macro_f1']:.3f} | lr {lr_now:.2e}",
            flush=True,
        )

        if score > best_score:
            best_score   = score
            # Lưu state_dict của model gốc (nếu đã compile thì lấy _orig_mod)
            raw_model = getattr(model, "_orig_mod", model)
            best_state   = copy.deepcopy(raw_model.state_dict())
            best_metrics = copy.deepcopy(val_metrics)
            bad_epochs   = 0
            torch.save({
                "model":       args.model,
                "state_dict":  best_state,
                "feature_dim": int(feature_dim),
                "num_classes": int(num_classes),
                "labels":      labels,
                "max_len":     int(args.max_len),
                "feature_mode": args.feature_mode,
                "input_representation": (
                    "unified_holistic_v1" if args.feature_mode == "unified"
                    else ("hands_raw_v1" if args.feature_mode == "hands_raw"
                          else "hand_features_v2")
                ),
                "infer_window": min(int(args.max_len), 8),
            }, args.out)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[INFO] Early stopping sau {bad_epochs} epoch không cải thiện", flush=True)
                break

    train_time = time.perf_counter() - t_train0

    # Load lại best state để eval cuối
    raw_model = getattr(model, "_orig_mod", model)
    if best_state is not None:
        raw_model.load_state_dict(best_state)
    metrics = best_metrics or evaluate(model, va_loader, device, amp_enabled, label_names)
    metrics.update({
        "model":             args.model,
        "epochs":            int(args.epochs),
        "num_params":        int(n_params),
        "train_time_sec":    float(train_time),
        "feature_dim":       int(feature_dim),
        "num_classes":       int(num_classes),
        "num_train_samples": int(len(tr_idx)),
        "checkpoint":        str(args.out),
        "device":            device,
        "amp":               bool(amp_enabled),
        "feature_mode":      args.feature_mode,
        "mixup_alpha":       float(args.mixup),
        "torch_compile":     bool(use_compile),
    })

    metrics_out = args.metrics_out or str(Path(args.out).with_suffix("").as_posix() + "_metrics.json")
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "-" * 64, flush=True)
    print(f"KẾT QUẢ [{args.model}]", flush=True)
    print("-" * 64, flush=True)
    print(f"Val accuracy    : {metrics['val_accuracy']*100:6.2f}%", flush=True)
    print(f"Macro F1        : {metrics['macro_f1']*100:6.2f}%", flush=True)
    print(f"Precision       : {metrics['macro_precision']*100:6.2f}%", flush=True)
    print(f"Recall          : {metrics['macro_recall']*100:6.2f}%", flush=True)
    print(f"Infer speed     : {metrics['inference_ms_per_sample']:.3f} ms/mẫu", flush=True)
    print(f"Train time      : {train_time:.1f}s", flush=True)
    print(f"Model lưu tại   : {args.out}", flush=True)
    print(f"Metrics lưu tại : {metrics_out}", flush=True)
    print("-" * 64, flush=True)
    return metrics


def main():
    train_model(parse_args())


if __name__ == "__main__":
    main()
