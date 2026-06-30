"""
extract_holistic.py - trich xuat keypoint THAN TREN + 2 TAY + MAT bang MediaPipe Holistic.

Dung cho BO DU LIEU MOI (co than tren va bieu cam guong mat). Ket qua luu dang
(T, 80, 4) = [x, y, z, presence] theo schema thong nhat trong pose_utils.

Du lieu CU (chi co tay, (T,42,3)) KHONG can chay lai: dataset tu nhung vao 80
khop khi train voi --model unified. Chi can de chung .npy + gop metadata.

Vi du:
  python src/extract_holistic.py \
      --raw data/raw_words --processed data/processed \
      --metadata data/metadata_words.csv --labels data/labels_words.json \
      --image-frames 8
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
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hand_utils import IMAGE_EXTS, VIDEO_EXTS
from pose_utils import mp_holistic, result_to_unified_array


def parse_signer(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    return parts[-2] if len(parts) >= 3 else "unknown"


def _frame_to_packed(res):
    """Holistic result 1 frame -> (80,4) [x,y,z,presence]."""
    coords, mask = result_to_unified_array(res)
    packed = np.concatenate([coords, mask[:, None].astype(np.float32)], axis=1)
    return packed.astype(np.float32), bool(mask.any())


def extract_video(path: Path, holistic, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    detected = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        packed, has = _frame_to_packed(res)
        if has:
            detected += 1
        frames.append(packed)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        return None, fps, 0
    return np.stack(frames).astype(np.float32), fps, detected


def extract_image(path: Path, holistic, n_frames: int):
    img = cv2.imread(str(path))
    if img is None:
        return None, 0
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)
    packed, has = _frame_to_packed(res)
    return np.repeat(packed[None, :, :], n_frames, axis=0).astype(np.float32), int(has)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_words")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--metadata", default="data/metadata_words.csv")
    ap.add_argument("--labels", default="data/labels_words.json")
    ap.add_argument("--label-map", default=None)
    ap.add_argument("--image-frames", type=int, default=8)
    ap.add_argument("--keep-empty", action="store_true")
    ap.add_argument("--min-detect-conf", type=float, default=0.5)
    ap.add_argument("--min-track-conf", type=float, default=0.5)
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--no-face", action="store_true",
                    help="Tat lay mat (refine_face_landmarks=False, nhanh hon)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = doc het video; >0 = cat den so frame nay")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Bo qua file da co .npy trong processed (van giu metadata cu)")
    args = ap.parse_args()

    if mp_holistic is None:
        raise SystemExit("[LOI] Chua cai mediapipe. Chay: pip install mediapipe")

    raw = Path(args.raw)
    proc = Path(args.processed)
    proc.mkdir(parents=True, exist_ok=True)

    label_map = {}
    if args.label_map and Path(args.label_map).exists():
        label_map = json.loads(Path(args.label_map).read_text(encoding="utf-8"))

    label_dirs = sorted([d for d in raw.iterdir() if d.is_dir()])
    if not label_dirs:
        raise SystemExit(f"[LOI] Khong thay folder nhan trong {raw}")

    displays = sorted({label_map.get(d.name, d.name) for d in label_dirs})
    labels = {name: i for i, name in enumerate(displays)}
    Path(args.labels).write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    skipped = 0
    kept_existing = 0

    # Khi --skip-existing: nap metadata cu de giu lai dong cua cac file da trich.
    old_meta = {}
    if args.skip_existing and Path(args.metadata).exists():
        try:
            prev = pd.read_csv(args.metadata, encoding="utf-8-sig")
            old_meta = {str(r["filename"]): r.to_dict() for _, r in prev.iterrows()}
        except Exception:
            old_meta = {}

    common = dict(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        refine_face_landmarks=(not args.no_face),
        min_detection_confidence=args.min_detect_conf,
        min_tracking_confidence=args.min_track_conf,
    )
    img_common = dict(common)
    img_common["static_image_mode"] = True
    with mp_holistic.Holistic(**common) as holistic_video, \
            mp_holistic.Holistic(**img_common) as holistic_image:
        for d in label_dirs:
            display = label_map.get(d.name, d.name)
            lid = labels[display]
            files = sorted(p for p in d.iterdir() if p.is_file())
            videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
            images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]

            for vid in videos:
                out_npy = proc / f"{vid.stem}.npy"
                if args.skip_existing and out_npy.exists():
                    kept_existing += 1
                    if vid.name in old_meta:
                        rows.append(old_meta[vid.name])
                    else:
                        try:
                            nf = int(np.load(out_npy).shape[0])
                        except Exception:
                            nf = 0
                        signer = parse_signer(vid.name)
                        rows.append(dict(
                            filename=vid.name, label=display, label_id=lid,
                            modality="video", source="internet" if signer == "net" else "self",
                            signer=signer, frames=nf, fps=0.0,
                            date=date.today().isoformat(),
                        ))
                    print(f"[GIU][video] {vid.name}: da co .npy -> bo qua")
                    continue
                arr, fps, detected = extract_video(
                    vid, holistic_video, None if args.max_frames <= 0 else args.max_frames)
                if arr is None:
                    print(f"[LOI] Video hong: {vid}")
                    continue
                if detected == 0 and not args.keep_empty:
                    skipped += 1
                    print(f"[BO QUA][video] {vid.name}: khong thay khop nao")
                    continue
                np.save(proc / f"{vid.stem}.npy", arr)
                signer = parse_signer(vid.name)
                rows.append(dict(
                    filename=vid.name, label=display, label_id=lid,
                    modality="video", source="internet" if signer == "net" else "self",
                    signer=signer, frames=int(arr.shape[0]), fps=round(float(fps), 1),
                    date=date.today().isoformat(),
                ))
                print(f"[OK][video] {vid.name}: {arr.shape[0]} frame | co khop {detected}/{arr.shape[0]}")

            for imgp in images:
                out_npy = proc / f"{imgp.stem}.npy"
                if args.skip_existing and out_npy.exists():
                    kept_existing += 1
                    if imgp.name in old_meta:
                        rows.append(old_meta[imgp.name])
                    else:
                        try:
                            nf = int(np.load(out_npy).shape[0])
                        except Exception:
                            nf = 0
                        signer = parse_signer(imgp.name)
                        rows.append(dict(
                            filename=imgp.name, label=display, label_id=lid,
                            modality="image", source="internet" if signer == "net" else "self",
                            signer=signer, frames=nf, fps=0.0,
                            date=date.today().isoformat(),
                        ))
                    print(f"[GIU][anh] {imgp.name}: da co .npy -> bo qua")
                    continue
                arr, has = extract_image(imgp, holistic_image, args.image_frames)
                if arr is None:
                    print(f"[LOI] Anh hong: {imgp}")
                    continue
                if not has and not args.keep_empty:
                    skipped += 1
                    print(f"[BO QUA][anh] {imgp.name}: khong thay khop nao")
                    continue
                np.save(proc / f"{imgp.stem}.npy", arr)
                signer = parse_signer(imgp.name)
                rows.append(dict(
                    filename=imgp.name, label=display, label_id=lid,
                    modality="image", source="internet" if signer == "net" else "self",
                    signer=signer, frames=int(arr.shape[0]), fps=0.0,
                    date=date.today().isoformat(),
                ))
                print(f"[OK][anh] {imgp.name}: {arr.shape[0]} frame lap | co khop={has}")

    pd.DataFrame(rows).to_csv(args.metadata, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] {len(rows)} mau (T,80,4) -> {args.processed} | {args.metadata} | {args.labels}")
    if skipped:
        print(f"[INFO] Da loai {skipped} mau khong thay khop")
    if kept_existing:
        print(f"[INFO] Da bo qua {kept_existing} mau da co .npy (--skip-existing)")


if __name__ == "__main__":
    main()
