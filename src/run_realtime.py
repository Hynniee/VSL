"""run_realtime.py - App desktop OpenCV, FPS cao (thay cho ban Streamlit lag).

Vi sao nhanh hon Streamlit:
  - Hien thi bang cv2.imshow (cua so native) -> KHONG day frame qua websocket.
  - 1 LUONG doc camera rieng (FreshestFrame) -> vong xu ly khong cho cap.read().
  - MediaPipe xu ly o anh thu nho (--proc-scale), du doan moi N khung (--infer-every).
  - Model warmup san -> khung dau khong giat.

Chay:
  python src/run_realtime.py
  python src/run_realtime.py --ckpt checkpoints/best.pth --device cuda
  python src/run_realtime.py --proc-scale 0.5 --infer-every 2 --width 640

Phim tat (bam khi cua so camera dang focus):
  q / ESC : thoat            c : xoa cau          x / Backspace : xoa 1 ky tu
  s       : phat am cau      a : bat/tat tu phat am
  p       : an/hien diem     f : an/hien mat       o : Holistic <-> chi 2 tay
  m       : lat guong        [ / ] : giam/tang nguong
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("GLOG_minloglevel", "2")
try:
    import faulthandler as _fh
    _fh.enable()
except Exception:
    pass

import argparse
import collections
import platform
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))
from hand_utils import (  # noqa: E402
    draw_guide_box,
    draw_hand_landmarks,
    mp_hands,
    preprocess_landmark_array,
    result_to_two_hand_array,
)
from pose_utils import (  # noqa: E402
    LHAND_OFF,
    NUM_HAND,
    RHAND_OFF,
    draw_holistic_landmarks,
    mp_holistic,
    prepare_unified,
    result_to_unified_array,
)
from model import build_model  # noqa: E402
from tts import synthesize  # noqa: E402
from unicode_utils import tokens_to_text  # noqa: E402

EMPTY_TXT = "(chua co)"

# ---- PIL de ve tieng Viet co dau len panel (fallback neu thieu) ----
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _find_font(size):
    cands = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in cands:
        if _os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


class Panel:
    """Ve panel thong tin ben phai (tieng Viet co dau bang PIL)."""

    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.ok = _HAS_PIL
        if self.ok:
            self.f_big = _find_font(34)
            self.f_mid = _find_font(22)
            self.f_small = _find_font(18)
            self.ok = self.f_big is not None

    def render(self, info):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = (24, 24, 28)
        if not self.ok:
            # Fallback: cv2 putText (khong dau)
            y = 40
            for line in info["plain"]:
                cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (235, 235, 235), 1, cv2.LINE_AA)
                y += 30
            return img
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        x = 18
        y = 16
        d.text((x, y), "DU DOAN", font=self.f_small, fill=(150, 150, 160))
        y += 26
        col = (90, 220, 120) if info["conf"] >= info["threshold"] else (245, 190, 70)
        d.text((x, y), info["label"], font=self.f_big, fill=col)
        y += 46
        # thanh tin cay
        bw = self.w - 2 * x
        d.rectangle([x, y, x + bw, y + 16], fill=(60, 62, 70))
        d.rectangle([x, y, x + int(bw * max(0.0, min(1.0, info["conf"]))), y + 16], fill=col)
        y += 22
        d.text((x, y), f"{info['conf'] * 100:.1f}%", font=self.f_small, fill=(200, 200, 205))
        y += 34
        d.text((x, y), "TOP-3", font=self.f_small, fill=(150, 150, 160))
        y += 26
        for line in info["top3"]:
            d.text((x, y), line, font=self.f_mid, fill=(220, 220, 225))
            y += 28
        y += 10
        d.text((x, y), "VAN BAN", font=self.f_small, fill=(150, 150, 160))
        y += 26
        for line in _wrap(info["sentence"] or EMPTY_TXT, self.f_mid, bw, d):
            d.text((x, y), line, font=self.f_mid, fill=(245, 245, 250))
            y += 28
        y += 8
        d.text((x, y), f"Da chot: {info['last_commit'] or EMPTY_TXT}",
               font=self.f_small, fill=(180, 200, 230))
        # footer
        fy = self.h - 96
        d.text((x, fy), f"Nguong: {info['threshold']:.2f}   Auto-doc: {'BAT' if info['auto_speak'] else 'TAT'}",
               font=self.f_small, fill=(160, 200, 160))
        fy += 24
        d.text((x, fy), "q thoat  c xoa cau  x xoa 1  s doc",
               font=self.f_small, fill=(140, 140, 150))
        fy += 22
        d.text((x, fy), "p diem  f mat  o holistic  m guong  [ ] nguong",
               font=self.f_small, fill=(140, 140, 150))
        return np.array(pil)


def _wrap(text, font, max_w, drawer):
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        try:
            tw = drawer.textlength(trial, font=font)
        except Exception:
            tw = len(trial) * 11
        if tw <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:4]


# ============================== Camera thread ==============================
class FreshestFrame(threading.Thread):
    """Doc camera lien tuc o luong rieng, luon giu khung MOI NHAT."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._running = True

    def run(self):
        while self._running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = f

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False


# ============================== Audio (nen) ==============================
_audio_lock = threading.Lock()


def _play_file(path):
    try:
        if platform.system() == "Windows":
            _os.startfile(path)  # mo bang trinh phat mac dinh
        elif platform.system() == "Darwin":
            subprocess.Popen(["afplay", path])
        else:
            subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])
    except Exception:
        pass


def speak_async(text):
    def _job():
        if not _audio_lock.acquire(blocking=False):
            return
        try:
            t = (text or "").strip()
            if not t:
                return
            path = f"tmp_audio_{uuid.uuid4().hex}.mp3"
            synthesize(t, out_path=path)
            _play_file(path)
        except Exception as e:
            print("[TTS]", e)
        finally:
            _audio_lock.release()
    threading.Thread(target=_job, daemon=True).start()


# ============================== Model ==============================
def load_model(ckpt_path, prefer_cuda=True):
    device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], ckpt["feature_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    max_len = int(ckpt.get("max_len", 16))
    fdim = int(ckpt["feature_dim"])
    # Warmup
    with torch.no_grad():
        dummy = torch.zeros(1, max_len, fdim, device=device)
        dlen = torch.tensor([max_len], dtype=torch.long, device=device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            model(dummy, dlen)
    if device == "cuda":
        torch.cuda.synchronize()
    id2label = {v: k for k, v in ckpt["labels"].items()}
    return model, ckpt, id2label, device


def is_alphabet_mode(id2label):
    return any(str(n).startswith(("chu_", "dau_", "thanh_")) for n in id2label.values())


def decode_tokens(tokens, alphabet_mode):
    return tokens_to_text(tokens) if alphabet_mode else " ".join(tokens)


def predict_prob(model, device, feature_mode, seq, max_len):
    if feature_mode == "unified":
        seq = prepare_unified(seq)
    else:
        seq = preprocess_landmark_array(seq, feature_mode)
    if seq.shape[0] > max_len:
        ids = np.linspace(0, seq.shape[0] - 1, max_len).round().astype(int)
        seq = seq[ids]
    x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
    lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            prob = F.softmax(model(x, lengths), dim=1)[0]
    return prob.detach().float().cpu()


# ============================== Main ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--proc-scale", type=float, default=0.5)
    ap.add_argument("--infer-every", type=int, default=2)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--vote-k", type=int, default=2)
    ap.add_argument("--cooldown", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--dau-threshold", type=float, default=0.40,
                    help="Nguong RIENG (thap hon) cho ky hieu dau/thanh de de chot hon")
    ap.add_argument("--dau-vote-k", type=int, default=1,
                    help="So frame can on dinh de chot dau/thanh (mac dinh 1 = chot nhanh)")
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--model-complexity", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--refine-face", action="store_true")
    ap.add_argument("--min-det", type=float, default=0.4)
    ap.add_argument("--min-track", type=float, default=0.4)
    ap.add_argument("--no-holistic", action="store_true",
                    help="Chi dung 2 tay (nhanh nhat, neu model khong phai unified)")
    ap.add_argument("--disp-width", type=int, default=720)
    args = ap.parse_args()

    prefer_cuda = args.device in ("auto", "cuda")
    print("[*] Dang nap model...")
    model, ckpt, id2label, device = load_model(args.ckpt, prefer_cuda=prefer_cuda)
    feature_mode = ckpt.get("feature_mode", "hands")
    max_len = int(ckpt.get("max_len", 16))
    alphabet_mode = is_alphabet_mode(id2label)
    is_unified = (feature_mode == "unified")
    print(f"[*] model={ckpt.get('model')} device={device} feature_mode={feature_mode} "
          f"classes={ckpt['num_classes']} max_len={max_len}")

    use_holistic = (is_unified or (not args.no_holistic)) and (mp_holistic is not None)
    if is_unified and mp_holistic is None:
        print("[!] Checkpoint unified can mediapipe Holistic: pip install mediapipe")
        return
    if not use_holistic and mp_hands is None:
        print("[!] Can mediapipe: pip install mediapipe")
        return

    # Mo camera
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[!] Khong mo duoc webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.width * 3 / 4))
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass
    grabber = FreshestFrame(cap)
    grabber.start()

    # MediaPipe
    if use_holistic:
        processor = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            refine_face_landmarks=args.refine_face,
            min_detection_confidence=args.min_det,
            min_tracking_confidence=args.min_track,
        )
    else:
        processor = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=args.min_det,
            min_tracking_confidence=args.min_track,
        )

    # Trang thai
    buffer = collections.deque(maxlen=min(args.window, max_len))
    recent = collections.deque(maxlen=args.vote_k)
    tokens = []
    last_commit = ""
    cooldown = 0
    ema_prob = None
    threshold = args.threshold
    mirror = True
    show_points = True
    show_face = True
    auto_speak = False
    label, conf = "...", 0.0
    top3 = ["- chua du du lieu -"]
    fps = 0.0
    t_prev = time.perf_counter()
    frame_count = 0

    win = "VSL - Nhan dang ngon ngu ky hieu (q=thoat)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    panel = None
    print("[*] San sang. Bam vao cua so camera roi dung phim tat.")

    try:
        while True:
            img = grabber.read()
            if img is None:
                time.sleep(0.005)
                continue
            # QUAN TRONG: KHONG lat guong truoc khi cho MediaPipe xu ly.
            # Du lieu train (record_data.py mac dinh KHONG --mirror) duoc trich
            # tu khung GOC, nen MediaPipe gan tay trai/phai theo khung goc.
            # Neu o day lat guong, tay trai<->phai bi DAO -> roi sai o (LHAND/RHAND)
            # -> model khong nhan dien duoc (nhat la chu cai 1 tay).
            # Vi vay: xu ly khung GOC, chi lat guong cho HIEN THI o cuoi.
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ps = args.proc_scale
            proc_in = cv2.resize(rgb, None, fx=ps, fy=ps, interpolation=cv2.INTER_AREA) if ps < 0.99 else rgb
            proc_in.flags.writeable = False
            res = processor.process(proc_in)

            vis = img
            draw_guide_box(vis)

            if use_holistic:
                coords, pmask = result_to_unified_array(res)
                if is_unified:
                    frame_arr = np.concatenate([coords, pmask[:, None].astype(np.float32)], axis=1)
                else:
                    frame_arr = np.zeros((42, 3), dtype=np.float32)
                    frame_arr[:21] = coords[LHAND_OFF:LHAND_OFF + NUM_HAND]
                    frame_arr[21:] = coords[RHAND_OFF:RHAND_OFF + NUM_HAND]
                hands_present = bool(pmask[LHAND_OFF:LHAND_OFF + NUM_HAND].any()
                                    or pmask[RHAND_OFF:RHAND_OFF + NUM_HAND].any())
                if show_points:
                    draw_holistic_landmarks(vis, res, draw_pose=True, draw_hands=True,
                                            draw_face=show_face, draw_face_oval=False, draw_iris=False)
            else:
                frame_arr = result_to_two_hand_array(res)
                hands_present = bool(np.any(frame_arr))
                if show_points and getattr(res, "multi_hand_landmarks", None):
                    for i, hand_lms in enumerate(res.multi_hand_landmarks):
                        color = (0, 255, 0) if i == 0 else (255, 0, 255)
                        draw_hand_landmarks(vis, hand_lms, color=color)

            if hands_present:
                buffer.append(frame_arr)
            else:
                buffer.clear()
                recent.clear()
                ema_prob = None

            if not buffer:
                label, conf = "...", 0.0
                top3 = ["- chua du du lieu -"]
            elif frame_count % args.infer_every == 0:
                prob = predict_prob(model, device, feature_mode, np.stack(buffer), max_len)
                if args.smooth > 0 and ema_prob is not None and ema_prob.shape == prob.shape:
                    ema_prob = args.smooth * ema_prob + (1.0 - args.smooth) * prob
                else:
                    ema_prob = prob
                pp = ema_prob
                k = min(3, pp.numel())
                vals, idxs = torch.topk(pp, k=k)
                conf = float(vals[0])
                label = id2label.get(int(idxs[0]), "?")
                top3 = [f"{id2label.get(int(i), '?')}  {float(v) * 100:.1f}%" for v, i in zip(vals, idxs)]
                if cooldown > 0:
                    cooldown -= 1
                else:
                    is_dau = str(label).startswith(("dau_", "thanh_"))
                    eff_thr = min(threshold, args.dau_threshold) if is_dau else threshold
                    eff_votek = max(1, args.dau_vote_k) if is_dau else args.vote_k
                    if conf >= eff_thr:
                        recent.append(label)
                    else:
                        recent.clear()
                    if len(recent) >= eff_votek and len(set(recent)) == 1:
                        tokens.append(label)
                        last_commit = label
                        recent.clear()
                        buffer.clear()
                        ema_prob = None
                        cooldown = args.cooldown
                        if auto_speak:
                            speak_async(decode_tokens(tokens, alphabet_mode))

            frame_count += 1
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0.0 else (0.9 * fps + 0.1 * inst)

            # Lat guong CHI cho hien thi (sau khi da ve landmark de van khop diem).
            if mirror:
                vis = cv2.flip(vis, 1)
            cv2.putText(vis, f"FPS {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
            if cooldown > 0:
                cv2.putText(vis, f"Reset {cooldown}", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255), 2, cv2.LINE_AA)

            # Resize hien thi
            dw = args.disp_width
            if dw and vis.shape[1] != dw:
                sc = dw / float(vis.shape[1])
                vis = cv2.resize(vis, (int(dw), int(vis.shape[0] * sc)), interpolation=cv2.INTER_LINEAR)

            if panel is None or panel.h != vis.shape[0]:
                panel = Panel(380, vis.shape[0])
            sentence = decode_tokens(tokens, alphabet_mode)
            info = {
                "label": label, "conf": conf, "top3": top3, "threshold": threshold,
                "sentence": sentence, "last_commit": last_commit, "auto_speak": auto_speak,
                "plain": [f"Pred: {label} {conf*100:.0f}%", f"Text: {' '.join(tokens)[:30]}",
                          f"Thr: {threshold:.2f}"],
            }
            panel_img = panel.render(info)
            disp = np.hstack([vis, panel_img])
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                tokens.clear(); last_commit = ""
            elif key in (ord("x"), 8):
                if tokens:
                    tokens.pop()
            elif key == ord("s"):
                speak_async(decode_tokens(tokens, alphabet_mode))
            elif key == ord("a"):
                auto_speak = not auto_speak
            elif key == ord("p"):
                show_points = not show_points
            elif key == ord("f"):
                show_face = not show_face
            elif key == ord("o") and not is_unified:
                # doi giua holistic va chi-2-tay (tao lai processor)
                try:
                    processor.close()
                except Exception:
                    pass
                use_holistic = not use_holistic and (mp_holistic is not None)
                if use_holistic:
                    processor = mp_holistic.Holistic(
                        static_image_mode=False, model_complexity=args.model_complexity,
                        refine_face_landmarks=args.refine_face,
                        min_detection_confidence=args.min_det, min_tracking_confidence=args.min_track)
                else:
                    processor = mp_hands.Hands(
                        static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=args.min_det, min_tracking_confidence=args.min_track)
                buffer.clear(); recent.clear(); ema_prob = None
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("["):
                threshold = max(0.0, round(threshold - 0.05, 2))
            elif key == ord("]"):
                threshold = min(1.0, round(threshold + 0.05, 2))
            # cua so bi dong bang nut X
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        grabber.stop()
        time.sleep(0.05)
        cap.release()
        try:
            processor.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[*] Da thoat.")


if __name__ == "__main__":
    main()
