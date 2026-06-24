"""
===============================================================================
 record_data.py - QUAY DU LIEU TU WEBCAM (chon signer + tu xu ly dau tieng Viet)
===============================================================================

Mo cua so webcam roi dung phim:
  - L      : nhap / doi NHAN (go truc tiep tren cua so, vd: chu a | xin chao)
  - S      : doi SIGNER (cuon qua danh sach: Chau -> Huynh -> Nhi -> ...)
  - N      : them SIGNER moi (go ten, code tu sinh)
  - C      : chup 1 anh NGAY
  - SPACE  : quay 1 clip video (mac dinh ~5 giay)
  - Q      : thoat

DAU TIENG VIET:
  - Trong cua so OpenCV chi go duoc ASCII (khong bo dau). Cu go 'xin chao', 'chu a'.
  - Neu muon LUU ban co dau de hien thi (label_map / doc tieng Viet), truyen luc chay:
        python src/record_data.py --label "xin chao"      (folder: xinchao)
        python src/record_data.py --label "Xin chao"      (van ra folder: xinchao)
    Code TU BO DAU de dat ten thu muc/file, va GIU ban gan dau lam 'display'.
  - Vi the ban co the go tieng Viet co dau o terminal qua --label, code tu chinh lai.

TU DONG DAT TEN  ->  <prefix>_<signer>_<NN>
  - signer: chau / huynh / nhi (hoac code tu sinh tu ten khac)
  - NN    : dien tiep theo cac mau DA CO cua dung signer do trong thu muc
      vd dang co  chu_a_chau_07     -> luu  chu_a_chau_08
      vd thu muc rong               -> bat dau  chu_a_chau_01

CAI: pip install opencv-python mediapipe numpy
===============================================================================
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

try:
    from hand_utils import draw_guide_box, draw_hand_landmarks, mp_hands
    _HAS_MP = True
except Exception:  # mediapipe chua cai cung khong sao, chi mat overlay
    _HAS_MP = False
    mp_hands = None

    def draw_guide_box(img, text="Dat tay vao khung"):
        h, w = img.shape[:2]
        bw, bh = int(w * 0.44), int(h * 0.58)
        x1, y1 = (w - bw) // 2, (h - bh) // 2
        cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), (0, 220, 255), 2)

    def draw_hand_landmarks(img, lms, color=(255, 0, 255)):
        return None


ALPHABET_PREFIXES = ("chu", "dau", "thanh")
DEFAULT_SIGNERS = ["Chau", "Huynh", "Nhi"]


def to_ascii(text: str) -> str:
    """Bo dau tieng Viet, ve chu thuong. 'Xin Cha\u0300o' -> 'xin chao', '\u0110' -> 'd'."""
    text = text.strip().lower().replace("\u0111", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def slug(text: str) -> str:
    """Chi giu a-z0-9 (cho ten thu muc / signer code)."""
    return re.sub(r"[^a-z0-9]+", "", to_ascii(text))


def signer_code(name: str) -> str:
    code = slug(name)
    if code == "net":
        # 'net' danh rieng cho du lieu internet -> doi thanh khac
        code = "net_local"
    return code or "self"


def parse_label(label: str):
    """'chu a'  -> (raw_alphabet, 'chu_a', 'chu_a', 'chu_a')
       'Xin chao' -> (raw_words, 'xinchao', 'xinchao', 'xin chao')   # display giu dau
    Tra ve: (raw_subdir, folder, prefix, display)
    """
    raw = label.strip()
    disp_tokens = [t for t in re.split(r"\s+", raw) if t]
    tokens = [t for t in (slug(t) for t in disp_tokens) if t]
    if not tokens:
        raise ValueError("Nhan rong")
    if tokens[0] in ALPHABET_PREFIXES:
        folder = "_".join(tokens)            # chu_a / dau_mu / thanh_sac
        return "raw_alphabet", folder, folder, folder
    folder = "".join(tokens)                 # xinchao
    display = " ".join(disp_tokens).lower()  # giu dau neu nguoi dung go co dau
    return "raw_words", folder, folder, display


def next_stem(folder_dir: Path, prefix: str, signer: str) -> str:
    """Dien so thu tu con thieu THEO DUNG signer, bam kieu dat ten dang co.
      chu_a_chau_07 -> chu_a_chau_08 ; (rong) -> chu_a_chau_01
    """
    base = f"{prefix}_{signer}_" if signer else f"{prefix}_"
    best_num, best_width, found = 0, 2, False
    if folder_dir.exists():
        pat = re.compile(rf"^{re.escape(base)}(\d+)$")
        for p in folder_dir.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.stem)
            if m:
                found = True
                n = int(m.group(1))
                if n >= best_num:
                    best_num, best_width = n, len(m.group(1))
    if not found:
        return f"{base}01" if signer else f"{base}1"
    return f"{base}{best_num + 1:0{best_width}d}"


def update_label_map(label_map_path: Path, folder: str, display: str):
    data = {}
    if label_map_path.exists():
        try:
            data = json.loads(label_map_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if data.get(folder) != display:
        data[folder] = display
        label_map_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Cap nhat label_map: {folder} -> '{display}'")


def make_writer(path: Path, fps: float, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size)


def resolve_target(args, label):
    raw_subdir, folder, prefix, display = parse_label(label)
    root = Path(args.out_root)
    folder_dir = root / raw_subdir / folder
    folder_dir.mkdir(parents=True, exist_ok=True)
    if raw_subdir == "raw_words":
        update_label_map(root / args.label_map_name, folder, display)
    return folder_dir, prefix, display, raw_subdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="", help='vd: "chu a" hoac "xin chao" (co the nhap sau tren cua so)')
    ap.add_argument("--signer", default="", help="ten nguoi quay (vd Chau/Huynh/Nhi hoac ten khac). KHONG dung 'net'")
    ap.add_argument("--seconds", type=float, default=5.0, help="do dai moi clip video (mac dinh 5s)")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--label-map-name", default="label_map.json")
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    # Danh sach signer (display, code). Them --signer neu la nguoi moi.
    signers = [(n, signer_code(n)) for n in DEFAULT_SIGNERS]
    if args.signer.strip():
        code = signer_code(args.signer)
        if code not in [c for _, c in signers]:
            signers.insert(0, (args.signer.strip(), code))
        sidx = [c for _, c in signers].index(code)
    else:
        sidx = 0
    cur_signer = signers[sidx][1]

    # Trang thai nhan
    label = args.label.strip()
    folder_dir = prefix = display = raw_subdir = None
    stem = None
    saved = 0

    def refresh():
        nonlocal stem
        stem = next_stem(folder_dir, prefix, cur_signer) if folder_dir is not None else None

    def set_label(lbl):
        nonlocal label, folder_dir, prefix, display, raw_subdir, saved
        label = lbl
        folder_dir, prefix, display, raw_subdir = resolve_target(args, lbl)
        saved = 0
        refresh()
        print(f"[INFO] Nhan '{label}' (signer={cur_signer}) -> {folder_dir} | ke tiep: {stem}")

    if label:
        set_label(label)

    mode = "label" if not label else "normal"   # chua co nhan -> nhap luon
    buf = ""

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("[LOI] Khong mo duoc webcam")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    hands = None
    if _HAS_MP and mp_hands is not None:
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                               min_detection_confidence=0.4, min_tracking_confidence=0.4)

    print("=" * 66)
    print("  L=nhan | S=doi signer | N=signer moi | C=chup | SPACE=quay | Q=thoat")
    print(f"  Signer: {cur_signer}  | Clip ~{args.seconds:g}s")
    print("=" * 66)

    recording = False
    writer = None
    rec_end = 0.0

    def overlay(frame):
        vis = frame.copy()
        draw_guide_box(vis)
        if hands is not None:
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if getattr(res, "multi_hand_landmarks", None):
                for i, lm in enumerate(res.multi_hand_landmarks):
                    draw_hand_landmarks(vis, lm, (0, 255, 0) if i == 0 else (255, 0, 255))
        return vis

    def put(vis, text, y, color=(0, 255, 255), scale=0.6):
        cv2.putText(vis, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[LOI] Khong doc duoc frame")
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        vis = overlay(frame)

        if mode in ("label", "signer"):
            cv2.rectangle(vis, (0, 0), (width, 74), (40, 40, 40), -1)
            cur = "_" if int(time.time() * 2) % 2 == 0 else " "
            title = "Nhap NHAN" if mode == "label" else "Nhap SIGNER moi"
            put(vis, f"{title} (Enter=OK, Esc=huy):", 26, (0, 255, 255), 0.6)
            put(vis, buf + cur, 60, (255, 255, 255), 0.8)
        else:
            put(vis, f"Signer: {cur_signer}  [S doi | N moi]", 28, (0, 220, 120), 0.6)
            put(vis, f"Nhan: {label if label else '(chua co - nhan L)'}", 56, (0, 255, 255), 0.7)
            if stem:
                put(vis, f"Ke tiep: {stem}  (da luu {saved})", 84, (0, 255, 255), 0.6)
            if recording:
                remain = max(0.0, rec_end - time.time())
                put(vis, f"REC {remain:.1f}s", 112, (0, 0, 255), 0.8)
                cv2.circle(vis, (width - 30, 30), 10, (0, 0, 255), -1)

        cv2.imshow("VSL Recorder", vis)

        if recording and writer is not None:
            writer.write(frame)
            if time.time() >= rec_end:
                writer.release()
                writer = None
                recording = False
                saved += 1
                print(f"[OK] Da luu video #{saved}: {stem}.mp4")
                refresh()

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue

        # ----- Che do nhap (nhan hoac signer) -----
        if mode in ("label", "signer"):
            if key in (13, 10):              # Enter
                val = buf.strip()
                if mode == "label":
                    if val:
                        set_label(val)
                else:  # signer
                    if val:
                        code = signer_code(val)
                        if code not in [c for _, c in signers]:
                            signers.append((val, code))
                        cur_signer = code
                        refresh()
                        print(f"[INFO] Signer -> {cur_signer}")
                mode = "normal"
                buf = ""
            elif key == 27:                  # Esc
                mode = "normal"
                buf = ""
            elif key in (8, 127):            # Backspace
                buf = buf[:-1]
            elif key == ord(" "):
                buf += " "
            elif 65 <= key <= 90:            # A-Z -> thuong
                buf += chr(key + 32)
            elif 97 <= key <= 122:           # a-z
                buf += chr(key)
            elif 48 <= key <= 57:            # 0-9
                buf += chr(key)
            continue

        # ----- Che do binh thuong -----
        if key == ord("q"):
            break
        elif key == ord("l"):
            mode = "label"
            buf = label
        elif key == ord("n"):
            mode = "signer"
            buf = ""
        elif key == ord("s"):
            sidx = (sidx + 1) % len(signers)
            cur_signer = signers[sidx][1]
            refresh()
            print(f"[INFO] Signer -> {cur_signer}")
        elif key == ord("c") and not recording:
            if not label:
                print("[!] Chua co nhan. Nhan L de nhap truoc.")
                continue
            out_path = folder_dir / f"{stem}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            print(f"[OK] Da chup anh #{saved}: {out_path.name}")
            refresh()
        elif key == ord(" ") and not recording:
            if not label:
                print("[!] Chua co nhan. Nhan L de nhap truoc.")
                continue
            out_path = folder_dir / f"{stem}.mp4"
            writer = make_writer(out_path, args.fps, (width, height))
            rec_end = time.time() + args.seconds
            recording = True
            print(f"[REC] Bat dau quay: {out_path.name} (~{args.seconds:g}s)")

    if writer is not None:
        writer.release()
    cap.release()
    if hands is not None:
        hands.close()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Tong cong da luu {saved} mau cho phien nay.")
    if raw_subdir is not None:
        print("[GOI Y] Sau khi quay, chay lai trich keypoint, vi du:")
        if raw_subdir == "raw_alphabet":
            print("  python src/extract_keypoints.py --raw data/raw_alphabet --metadata data/metadata_alphabet.csv --labels data/labels_alphabet.json --image-frames 8")
        else:
            print("  python src/extract_keypoints.py --raw data/raw_words --metadata data/metadata_words.csv --labels data/labels_words.json --label-map data/label_map.json")


if __name__ == "__main__":
    main()
