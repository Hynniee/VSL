"""app.py - webcam demo moi: reset lien tuc, top-3, overlay points, uu tien toc do."""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import faulthandler as _fh
    _fh.enable()
except Exception:
    pass

import collections
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
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
from model import build_model  # noqa: E402
from tts import synthesize  # noqa: E402
from unicode_utils import tokens_to_text  # noqa: E402


st.set_page_config(page_title="VSL Webcam Demo", layout="wide")
st.title("Nhan dang ngon ngu ky hieu tieng Viet - Webcam")


@st.cache_resource
def load_model(ckpt_path: str, prefer_cuda: bool = True):
    device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["model"], ckpt["feature_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    id2label = {v: k for k, v in ckpt["labels"].items()}
    return model, ckpt, id2label, device


def decode_tokens(tokens, alphabet_mode):
    return tokens_to_text(tokens) if alphabet_mode else " ".join(tokens)


def is_alphabet_mode(id2label):
    names = list(id2label.values())
    return any(str(n).startswith(("chu_", "dau_", "thanh_")) for n in names)


def predict_prob(model, device, feature_mode, seq, max_len):
    seq = preprocess_landmark_array(seq, feature_mode)
    if seq.shape[0] > max_len:
        ids = np.linspace(0, seq.shape[0] - 1, max_len).round().astype(int)
        seq = seq[ids]
    x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
    lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            prob = F.softmax(model(x, lengths), dim=1)[0]
    return prob.detach().float().cpu()


if "running" not in st.session_state:
    st.session_state.running = False
if "tokens" not in st.session_state:
    st.session_state.tokens = []
if "last_commit" not in st.session_state:
    st.session_state.last_commit = ""

ckpt_path = st.sidebar.text_input("Checkpoint", "checkpoints/best.pth")
use_cuda = st.sidebar.checkbox("Dung GPU neu co", value=True)
threshold = st.sidebar.slider("Nguong tin cay", 0.0, 1.0, 0.75)
vote_k = st.sidebar.slider("So khung lap de chot", 1, 8, 3)
window = st.sidebar.slider("So khung nhin lai", 1, 16, 8)
cooldown_frames = st.sidebar.slider("So khung reset sau khi chot", 0, 30, 10)
show_points = st.sidebar.checkbox("Hien landmark points", value=True)
show_guide = st.sidebar.checkbox("Hien khung can tay", value=True)
mirror = st.sidebar.checkbox("Lat guong webcam", value=True)
min_det = st.sidebar.slider("Hands detect conf", 0.1, 0.9, 0.35)
min_track = st.sidebar.slider("Hands track conf", 0.1, 0.9, 0.35)

b1, b2, b3, b4 = st.sidebar.columns(4)
if b1.button("Bat dau"):
    st.session_state.running = True
if b2.button("Dung"):
    st.session_state.running = False
if b3.button("Xoa cau"):
    st.session_state.tokens = []
if b4.button("Xoa 1 ky tu") and st.session_state.tokens:
    st.session_state.tokens.pop()

col1, col2 = st.columns([3, 2])
frame_box = col1.empty()
info_box = col2.empty()
text_box = col2.empty()
help_box = st.empty()

try:
    model, ckpt, id2label, device = load_model(ckpt_path, prefer_cuda=use_cuda)
    feature_mode = ckpt.get("feature_mode", "hands")
    max_len = int(ckpt.get("max_len", 16))
    alphabet_mode = is_alphabet_mode(id2label)
    st.sidebar.success(f"Da nap {ckpt.get('model', '?')} tren {device}")
except FileNotFoundError:
    model = ckpt = id2label = None
    feature_mode = "hands"
    max_len = 16
    alphabet_mode = True
    st.sidebar.error("Khong thay checkpoint. Chon dung file .pth")

if st.session_state.running:
    if model is None:
        st.error("Chua nap duoc checkpoint.")
        st.stop()

    buffer = collections.deque(maxlen=min(window, max_len))
    recent = collections.deque(maxlen=vote_k)
    cooldown = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Khong mo duoc webcam")
        st.stop()

    t0 = time.perf_counter()
    frame_count = 0
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track,
    ) as hands:
        while st.session_state.running and cap.isOpened():
            ok, img = cap.read()
            if not ok:
                st.warning("Khong doc duoc frame webcam")
                break
            if mirror:
                img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            hand_arr = result_to_two_hand_array(res)
            if np.any(hand_arr):
                buffer.append(hand_arr)
            else:
                # reset mem khi mat tay de tranh keo theo ky tu cu
                buffer.clear()
                recent.clear()

            vis = img.copy()
            if show_guide:
                draw_guide_box(vis)
            if show_points and getattr(res, 'multi_hand_landmarks', None):
                for i, hand_lms in enumerate(res.multi_hand_landmarks):
                    color = (0, 255, 0) if i == 0 else (255, 0, 255)
                    draw_hand_landmarks(vis, hand_lms, color=color)

            label = "..."
            conf = 0.0
            top3_lines = ["- chua du du lieu -"]

            if buffer:
                seq = np.stack(buffer)
                prob = predict_prob(model, device, feature_mode, seq, max_len)
                topk = min(3, prob.numel())
                vals, idxs = torch.topk(prob, k=topk)
                conf = float(vals[0])
                label = id2label.get(int(idxs[0]), "?")
                top3_lines = [f"{rank+1}. {id2label.get(int(i), '?')} - {float(v)*100:.1f}%"
                              for rank, (v, i) in enumerate(zip(vals, idxs))]

                if cooldown > 0:
                    cooldown -= 1
                else:
                    if conf >= threshold:
                        recent.append(label)
                    else:
                        recent.clear()
                    if len(recent) == vote_k and len(set(recent)) == 1:
                        st.session_state.tokens.append(label)
                        st.session_state.last_commit = label
                        recent.clear()
                        buffer.clear()  # reset lien tuc sau moi ky tu
                        cooldown = cooldown_frames

            frame_count += 1
            elapsed = max(time.perf_counter() - t0, 1e-6)
            fps = frame_count / elapsed
            cv2.putText(vis, f"FPS {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            if cooldown > 0:
                cv2.putText(vis, f"Reset {cooldown}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255), 2, cv2.LINE_AA)

            frame_box.image(vis, channels="BGR")
            info_box.markdown(
                "\n".join([
                    f"### Du doan hien tai: **{label}**",
                    f"Do tin cay: **{conf*100:.1f}%**",
                    f"Top-3:<br>{'<br>'.join(top3_lines)}",
                    f"Da chot gan nhat: **{st.session_state.last_commit or '(chua co)'}**",
                ]),
                unsafe_allow_html=True,
            )
            text_box.markdown(f"## Van ban: {decode_tokens(st.session_state.tokens, alphabet_mode) or '(chua co)'}")
            help_box.info("Mo tay ro, dat trong khung, giu yen ~0.5-1s. Sau khi chot, he thong tu reset de nhan ky tu tiep theo.")
    cap.release()
else:
    st.info("Nhan Bat dau de mo webcam. App moi se tu reset sau moi ky tu de tranh dinh ky tu cu.")
    text = decode_tokens(st.session_state.tokens, alphabet_mode)
    st.markdown(f"## Van ban da nhan: {text or '(chua co)'}")
    if text.strip() and st.button("Phat am van ban"):
        out_path = f"tmp_audio_{uuid.uuid4().hex}.mp3"
        try:
            synthesize(text.strip(), out_path=out_path)
            st.audio(out_path)
        except Exception as e:
            st.error(f"Loi TTS: {e}")
