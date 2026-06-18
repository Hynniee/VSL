"""
================================================================================
 extract_keypoints.py - TRICH KEYPOINT TU VIDEO (buoc 1 cua pipeline)
================================================================================

MUC DICH
--------
Doc cac video ky hieu (.mp4), dung MediaPipe Holistic de lay toa do cac diem
tren co the (tay/mat/than) o TUNG frame, roi luu thanh mang so (.npy) de model
hoc. Day la buoc bien "hinh anh" -> "so".

LUONG:
    data/raw_words/<nhan>/*.mp4  ->  data/processed/<ten>.npy
                                 ->  data/metadata.csv  (1 dong / video)
                                 ->  data/labels.json   (nhan -> so)

CACH CHAY:
    python src/extract_keypoints.py --raw data/raw_words
    python src/extract_keypoints.py --raw data/raw_alphabet --label-map data/label_map.json

CAI THU VIEN:
    pip install mediapipe opencv-python numpy pandas
================================================================================
"""
import argparse
import json
from datetime import date
from pathlib import Path

import cv2          # OpenCV: doc video, doi mau anh
import numpy as np  # tinh toan mang so
import pandas as pd # ghi file metadata.csv
import mediapipe as mp  # trich keypoint (pose + face + 2 tay)

# "Holistic" = mo hinh tong hop cua MediaPipe, cho ra CUNG LUC:
#   pose (than), face (mat), tay trai, tay phai.
mp_holistic = mp.solutions.holistic

# --- SO DIEM (landmark) MOI BO PHAN -----------------------------------------
# Thu tu nay CO DINH, cac file khac (dataset, app) deu dua vao no:
#   33 diem pose  +  468 diem mat  +  21 diem tay trai  +  21 diem tay phai
POSE_N, FACE_N, HAND_N = 33, 468, 21
TOTAL = POSE_N + FACE_N + HAND_N * 2  # = 543 diem/frame
# => 2 ban tay LUON la 42 DIEM CUOI CUNG
#    (rat quan trong: dataset lay arr[:, -42:, :] de chi giu 2 tay)


def landmarks_to_array(res):
    """Gom toan bo landmark cua 1 frame -> mang (543, 3).

    Moi diem co 3 toa do (x, y, z) da chuan hoa [0..1].
    Bo phan nao KHONG thay trong frame (vd tay ra khoi khung) -> dien 0.
    """
    frame = np.zeros((TOTAL, 3), dtype=np.float32)  # mac dinh tat ca = 0
    off = 0  # vi tri ghi hien tai trong mang

    # 1) POSE (than tren) - 33 diem
    if res.pose_landmarks:
        for i, l in enumerate(res.pose_landmarks.landmark):
            frame[off + i] = (l.x, l.y, l.z)
    off += POSE_N

    # 2) FACE (mat) - 468 diem
    if res.face_landmarks:
        for i, l in enumerate(res.face_landmarks.landmark):
            frame[off + i] = (l.x, l.y, l.z)
    off += FACE_N

    # 3) TAY TRAI - 21 diem
    if res.left_hand_landmarks:
        for i, l in enumerate(res.left_hand_landmarks.landmark):
            frame[off + i] = (l.x, l.y, l.z)
    off += HAND_N

    # 4) TAY PHAI - 21 diem
    if res.right_hand_landmarks:
        for i, l in enumerate(res.right_hand_landmarks.landmark):
            frame[off + i] = (l.x, l.y, l.z)

    return frame


def extract_video(path, holistic):
    """Doc 1 video -> mang (T, 543, 3) voi T = so frame.

    Tra them:
      - fps: khung hinh/giay (ghi vao metadata de biet toc do quay)
      - hand_detected: so frame CO thay tay (de canh bao neu quay loi)
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, hand_detected = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break  # het video
        # MediaPipe can anh RGB, con OpenCV doc ra BGR -> phai doi mau
        res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.left_hand_landmarks or res.right_hand_landmarks:
            hand_detected += 1
        frames.append(landmarks_to_array(res))
    cap.release()
    if not frames:
        return None, fps, 0
    return np.stack(frames), fps, hand_detected  # (T, 543, 3)


def parse_signer(filename):
    """Lay 'nguoi quay' tu ten file theo quy uoc <ma_nhan>_<signer>_<lan>.mp4.

    Vi du 'xinchao_ng1_01.mp4' -> 'ng1'.
    Dung de: (1) ghi vao metadata, (2) sau nay chia train/test theo nguoi.
    """
    parts = Path(filename).stem.split("_")
    return parts[-2] if len(parts) >= 3 else "unknown"


def main():
    # --- Khai bao tham so dong lenh ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_words", help="thu muc chua cac folder nhan")
    ap.add_argument("--processed", default="data/processed", help="noi luu .npy")
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--labels", default="data/labels.json")
    ap.add_argument("--label-map", default=None,
                    help='JSON anh xa {ten_folder: "ten hien thi"}, vd {"xinchao": "xin chao"}')
    args = ap.parse_args()

    raw, proc = Path(args.raw), Path(args.processed)
    proc.mkdir(parents=True, exist_ok=True)  # tao folder processed neu chua co

    # (Tuy chon) doc bang anh xa ten folder khong dau -> ten hien thi co dau
    label_map = {}
    if args.label_map and Path(args.label_map).exists():
        with open(args.label_map, encoding="utf-8") as f:
            label_map = json.load(f)

    # Moi FOLDER con trong --raw = 1 nhan (vd folder 'xinchao' -> nhan "xin chao")
    label_dirs = sorted([d for d in raw.iterdir() if d.is_dir()])
    if not label_dirs:
        print(f"[LOI] Khong thay folder nhan nao trong {raw}")
        return

    # Gan moi nhan 1 so (label_id) va luu labels.json
    displays = sorted({label_map.get(d.name, d.name) for d in label_dirs})
    labels = {name: i for i, name in enumerate(displays)}
    with open(args.labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # --- Duyet tung video va trich keypoint ---
    rows = []
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                              refine_face_landmarks=False) as holistic:
        for d in label_dirs:
            display = label_map.get(d.name, d.name)
            lid = labels[display]
            for vid in sorted(d.glob("*.mp4")):
                arr, fps, hand_detected = extract_video(vid, holistic)
                if arr is None:
                    print(f"[LOI] Khong doc duoc: {vid}")
                    continue
                # Luu mang keypoint -> file .npy (ten trung ten video)
                np.save(proc / f"{vid.stem}.npy", arr)
                signer = parse_signer(vid.name)
                ratio = hand_detected / max(arr.shape[0], 1)  # ti le frame thay tay
                rows.append(dict(
                    filename=vid.name, label=display, label_id=lid,
                    source="internet" if signer == "net" else "self",
                    signer=signer, frames=arr.shape[0], fps=round(fps, 1),
                    date=date.today().isoformat(),
                ))
                # Canh bao neu qua it frame thay tay (quay thieu sang / tay ra khoi khung)
                warn = "" if ratio > 0.5 else "  <-- CANH BAO: it thay tay, kiem tra lai!"
                print(f"[OK] {vid.name}: {arr.shape[0]} frame, thay tay {ratio*100:.0f}%{warn}")

    # Ghi metadata.csv (utf-8-sig de Excel mo khong loi font tieng Viet)
    pd.DataFrame(rows).to_csv(args.metadata, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] {len(rows)} video -> {args.processed} | {args.metadata} | {args.labels}")


if __name__ == "__main__":
    main()
