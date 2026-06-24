"""
extract_keypoints.py - phien ban moi, uu tien MediaPipe Hands.
Nhanh hon Holistic, hop hon voi chu cai tinh va webcam can tay.
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

from hand_utils import IMAGE_EXTS, VIDEO_EXTS, mp_hands, result_to_two_hand_array


def parse_signer(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    return parts[-2] if len(parts) >= 3 else "unknown"


def extract_video(path: Path, hands, max_frames: int | None = None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    detected = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        arr = result_to_two_hand_array(res)
        if np.any(arr):
            detected += 1
        frames.append(arr)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        return None, fps, 0
    return np.stack(frames).astype(np.float32), fps, detected


def extract_image(path: Path, hands, n_frames: int):
    img = cv2.imread(str(path))
    if img is None:
        return None, 0
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    arr = result_to_two_hand_array(res)
    return np.repeat(arr[None, :, :], n_frames, axis=0).astype(np.float32), int(np.any(arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_alphabet")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--labels", default="data/labels.json")
    ap.add_argument("--label-map", default=None)
    ap.add_argument("--image-frames", type=int, default=8)
    ap.add_argument("--keep-empty", action="store_true")
    ap.add_argument("--min-detect-conf", type=float, default=0.35)
    ap.add_argument("--min-track-conf", type=float, default=0.35)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = doc het video; >0 = cat den so frame nay")
    args = ap.parse_args()

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
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detect_conf,
        min_tracking_confidence=args.min_track_conf,
    ) as hands_video, mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=args.min_detect_conf,
        min_tracking_confidence=args.min_track_conf,
    ) as hands_image:
        for d in label_dirs:
            display = label_map.get(d.name, d.name)
            lid = labels[display]
            files = sorted(p for p in d.iterdir() if p.is_file())
            videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
            images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]

            for vid in videos:
                arr, fps, detected = extract_video(vid, hands_video, None if args.max_frames <= 0 else args.max_frames)
                if arr is None:
                    print(f"[LOI] Video hong: {vid}")
                    continue
                if detected == 0 and not args.keep_empty:
                    skipped += 1
                    print(f"[BO QUA][video] {vid.name}: khong thay tay")
                    continue
                np.save(proc / f"{vid.stem}.npy", arr)
                signer = parse_signer(vid.name)
                rows.append(dict(
                    filename=vid.name,
                    label=display,
                    label_id=lid,
                    modality="video",
                    source="internet" if signer == "net" else "self",
                    signer=signer,
                    frames=int(arr.shape[0]),
                    fps=round(float(fps), 1),
                    date=date.today().isoformat(),
                ))
                print(f"[OK][video] {vid.name}: {arr.shape[0]} frame | thay tay {detected}/{arr.shape[0]}")

            for imgp in images:
                arr, has_hand = extract_image(imgp, hands_image, args.image_frames)
                if arr is None:
                    print(f"[LOI] Anh hong: {imgp}")
                    continue
                if not has_hand and not args.keep_empty:
                    skipped += 1
                    print(f"[BO QUA][anh] {imgp.name}: khong thay tay")
                    continue
                np.save(proc / f"{imgp.stem}.npy", arr)
                signer = parse_signer(imgp.name)
                rows.append(dict(
                    filename=imgp.name,
                    label=display,
                    label_id=lid,
                    modality="image",
                    source="internet" if signer == "net" else "self",
                    signer=signer,
                    frames=int(arr.shape[0]),
                    fps=0.0,
                    date=date.today().isoformat(),
                ))
                print(f"[OK][anh] {imgp.name}: {arr.shape[0]} frame lap | thay tay={has_hand}")

    pd.DataFrame(rows).to_csv(args.metadata, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] {len(rows)} mau -> {args.processed} | {args.metadata} | {args.labels}")
    if skipped:
        print(f"[INFO] Da loai {skipped} mau khong thay tay")


if __name__ == "__main__":
    main()
