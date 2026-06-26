"""app.py - webcam demo: hien diem (pose + 2 tay + mat), top-3, reset lien tuc.

UI moi:
  - Bo cuc 2 cot: video ben trai, bang du doan ben phai.
  - Thanh do tin cay + chu giai mau diem.
  - Tu nhan dien checkpoint 'unified' (Holistic: than tren + mat + 2 tay) hay
    checkpoint cu (chi 2 tay).
"""
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


st.set_page_config(page_title="VSL Webcam Demo", page_icon="\U0001F91F", layout="wide")

# ---------------------------- CSS ----------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 1rem;}
      .vsl-title {font-size: 1.9rem; font-weight: 800; margin-bottom: .1rem;}
      .vsl-sub {color: #6b7280; margin-bottom: .6rem;}
      .vsl-card {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px;
                 padding: 16px 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04);}
      .vsl-pred {font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin: 2px 0 6px;}
      .vsl-bar-bg {background:#eef0f3; border-radius: 8px; height: 16px; width: 100%; overflow: hidden;}
      .vsl-bar-fg {height: 100%; border-radius: 8px; transition: width .15s ease;}
      .vsl-top {margin: 4px 0; font-size: 1rem;}
      .vsl-top b {display:inline-block; min-width: 18px;}
      .vsl-chip {display:inline-block; padding: 3px 10px; border-radius: 999px;
                 font-size: .8rem; font-weight: 600; margin-right: 6px; color:#fff;}
      .vsl-textbox {background:#0f172a; color:#f8fafc; border-radius: 12px; padding: 14px 16px;
                    font-size: 1.4rem; font-weight: 700; min-height: 56px; word-break: break-word;}
      .vsl-legend span {display:inline-flex; align-items:center; margin-right: 14px; font-size:.85rem; color:#374151;}
      .vsl-dot {width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="vsl-title">\U0001F91F Nh\u1eadn d\u1ea1ng ng\u00f4n ng\u1eef k\u00fd hi\u1ec7u ti\u1ebfng Vi\u1ec7t</div>', unsafe_allow_html=True)
st.markdown('<div class="vsl-sub">Webcam th\u1eddi gian th\u1ef1c \u00b7 hi\u1ec3n th\u1ecb \u0111i\u1ec3m khung x\u01b0\u01a1ng \u00b7 t\u1ef1 reset sau m\u1ed7i k\u00fd t\u1ef1</div>', unsafe_allow_html=True)


# ---------------------------- Helpers ----------------------------
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
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            prob = F.softmax(model(x, lengths), dim=1)[0]
    return prob.detach().float().cpu()


def render_info(label, conf, top3_lines, last_commit, threshold):
    bar_color = "#16a34a" if conf >= threshold else "#f59e0b"
    pct = max(0.0, min(1.0, conf)) * 100
    top_html = "".join(f'<div class="vsl-top"><b>{i+1}.</b> {line}</div>'
                       for i, line in enumerate(top3_lines))
    return f"""
    <div class="vsl-card">
      <div style="color:#6b7280;font-size:.85rem;">D\u1ef1 \u0111o\u00e1n hi\u1ec7n t\u1ea1i</div>
      <div class="vsl-pred">{label}</div>
      <div class="vsl-bar-bg"><div class="vsl-bar-fg" style="width:{pct:.0f}%;background:{bar_color};"></div></div>
      <div style="text-align:right;font-size:.85rem;color:#374151;margin-top:2px;">{conf*100:.1f}%</div>
      <hr style="margin:10px 0;border:none;border-top:1px solid #eef0f3;">
      <div style="color:#6b7280;font-size:.85rem;margin-bottom:4px;">Top-3</div>
      {top_html}
      <hr style="margin:10px 0;border:none;border-top:1px solid #eef0f3;">
      <div style="color:#6b7280;font-size:.85rem;">\u0110\u00e3 ch\u1ed1t g\u1ea7n nh\u1ea5t</div>
      <div style="font-weight:700;font-size:1.1rem;">{last_commit or '(chưa có)'}</div>
    </div>
    """


LEGEND_HTML = (
    '<div class="vsl-legend">'
    '<span><span class="vsl-dot" style="background:#ffd700;"></span>Th\u00e2n tr\u00ean</span>'
    '<span><span class="vsl-dot" style="background:#00ff00;"></span>Tay tr\u00e1i</span>'
    '<span><span class="vsl-dot" style="background:#ff00ff;"></span>Tay ph\u1ea3i</span>'
    '<span><span class="vsl-dot" style="background:#00c8ff;"></span>G\u01b0\u01a1ng m\u1eb7t</span>'
    '</div>'
)


# ---------------------------- Session state ----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "tokens" not in st.session_state:
    st.session_state.tokens = []
if "last_commit" not in st.session_state:
    st.session_state.last_commit = ""

# ---------------------------- Sidebar ----------------------------
st.sidebar.header("\u2699\ufe0f C\u00e0i \u0111\u1eb7t")
ckpt_path = st.sidebar.text_input("Checkpoint", "checkpoints/best.pth")
use_cuda = st.sidebar.checkbox("D\u00f9ng GPU n\u1ebfu c\u00f3", value=True)

with st.sidebar.expander("\U0001F3AF Ng\u01b0\u1ee1ng & ch\u1ed1t k\u00fd t\u1ef1", expanded=True):
    threshold = st.slider("Ng\u01b0\u1ee1ng tin c\u1eady", 0.0, 1.0, 0.75)
    vote_k = st.slider("S\u1ed1 khung l\u1eb7p \u0111\u1ec3 ch\u1ed1t", 1, 8, 3)
    window = st.slider("S\u1ed1 khung nh\u00ecn l\u1ea1i", 1, 32, 12)
    cooldown_frames = st.slider("S\u1ed1 khung reset sau khi ch\u1ed1t", 0, 30, 10)

with st.sidebar.expander("\U0001F441\ufe0f Hi\u1ec3n th\u1ecb", expanded=True):
    show_points = st.checkbox("Hi\u1ec7n \u0111i\u1ec3m khung x\u01b0\u01a1ng", value=True)
    show_pose = st.checkbox("  \u00b7 Th\u00e2n tr\u00ean (pose)", value=True)
    show_face = st.checkbox("  \u00b7 G\u01b0\u01a1ng m\u1eb7t (m\u00f4i/m\u1eaft/m\u00e0y)", value=True)
    show_face_oval = st.checkbox("  \u00b7 Vi\u1ec1n m\u1eb7t", value=False)
    show_iris = st.checkbox("  \u00b7 Tr\u00f2ng m\u1eaft (c\u1ea7n b\u1eadt tinh ch\u1ec9nh m\u1eb7t)", value=False)
    full_skeleton = st.checkbox(
        "Lu\u00f4n d\u00f9ng Holistic (xem th\u00e2n + m\u1eb7t k\u1ec3 c\u1ea3 model ch\u1ec9 d\u00f9ng tay)",
        value=True)
    show_guide = st.checkbox("Hi\u1ec7n khung c\u0103n tay", value=True)
    mirror = st.checkbox("L\u1eadt g\u01b0\u01a1ng webcam", value=True)

with st.sidebar.expander("\U0001F39B\ufe0f MediaPipe", expanded=False):
    min_det = st.slider("Detect conf", 0.1, 0.9, 0.4)
    min_track = st.slider("Track conf", 0.1, 0.9, 0.4)

with st.sidebar.expander("⚡ Hiệu năng (FPS)", expanded=True):
    model_complexity = st.select_slider("Độ phức tạp model", options=[0, 1, 2], value=0)
    refine_face = st.checkbox("Tinh chỉnh điểm mặt (nặng)", value=False)
    proc_scale = st.slider("Thu nhỏ ảnh khi xử lý", 0.4, 1.0, 0.6, 0.1)
    cap_width = st.select_slider("Độ phân giải webcam", options=[480, 640, 800, 1280], value=640)
    infer_every = st.slider("Dự đoán mỗi N khung", 1, 5, 2)

b1, b2 = st.sidebar.columns(2)
if b1.button("\u25B6 B\u1eaft \u0111\u1ea7u", use_container_width=True):
    st.session_state.running = True
if b2.button("\u23F9 D\u1eebng", use_container_width=True):
    st.session_state.running = False
b3, b4 = st.sidebar.columns(2)
if b3.button("\U0001F5D1 X\u00f3a c\u00e2u", use_container_width=True):
    st.session_state.tokens = []
if b4.button("\u232B X\u00f3a 1 k\u00fd t\u1ef1", use_container_width=True) and st.session_state.tokens:
    st.session_state.tokens.pop()

# ---------------------------- Load model ----------------------------
try:
    model, ckpt, id2label, device = load_model(ckpt_path, prefer_cuda=use_cuda)
    feature_mode = ckpt.get("feature_mode", "hands")
    max_len = int(ckpt.get("max_len", 16))
    alphabet_mode = is_alphabet_mode(id2label)
    is_unified = (feature_mode == "unified")
    badge = "Holistic (th\u00e2n + m\u1eb7t + 2 tay)" if is_unified else "Hands (2 tay)"
    st.sidebar.success(f"\u0110\u00e3 n\u1ea1p {ckpt.get('model','?')} \u00b7 {device} \u00b7 {badge}")
except FileNotFoundError:
    model = ckpt = id2label = None
    feature_mode = "hands"
    max_len = 16
    alphabet_mode = True
    is_unified = False
    st.sidebar.error("Kh\u00f4ng th\u1ea5y checkpoint. Ch\u1ecdn \u0111\u00fang file .pth")

if is_unified and mp_holistic is None:
    st.sidebar.warning("Checkpoint unified c\u1ea7n mediapipe Holistic. H\u00e3y: pip install mediapipe")

# Co dung Holistic de VE khung xuong (than + mat) hay khong, du model chi dung tay
use_holistic = (is_unified or full_skeleton) and (mp_holistic is not None)
if full_skeleton and not is_unified and mp_holistic is None:
    st.sidebar.info("Mu\u1ed1n xem th\u00e2n + m\u1eb7t c\u1ea7n mediapipe Holistic: pip install mediapipe")
if use_holistic and not is_unified:
    st.sidebar.caption("\U0001F50E \u0110ang xem khung x\u01b0\u01a1ng \u0111\u1ea7y \u0111\u1ee7 b\u1eb1ng Holistic. Model hi\u1ec7n t\u1ea1i v\u1eabn ch\u1ec9 d\u1ef1 \u0111o\u00e1n b\u1eb1ng 2 tay.")

# ---------------------------- Layout ----------------------------
col1, col2 = st.columns([3, 2])
frame_box = col1.empty()
legend_box = col1.empty()
info_box = col2.empty()
text_label = col2.empty()
text_box = col2.empty()
help_box = st.empty()

if show_points:
    legend_box.markdown(LEGEND_HTML, unsafe_allow_html=True)


def make_processor():
    if use_holistic:
        return mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            refine_face_landmarks=refine_face,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track,
    )


if st.session_state.running:
    if model is None:
        st.error("Ch\u01b0a n\u1ea1p \u0111\u01b0\u1ee3c checkpoint.")
        st.stop()
    if is_unified and mp_holistic is None:
        st.error("Mode unified c\u1ea7n mediapipe Holistic. H\u00e3y: pip install mediapipe")
        st.stop()

    buffer = collections.deque(maxlen=min(window, max_len))
    recent = collections.deque(maxlen=vote_k)
    cooldown = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Kh\u00f4ng m\u1edf \u0111\u01b0\u1ee3c webcam")
        st.stop()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cap_width * 3 / 4))
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    last_label, last_conf = "...", 0.0
    last_top3 = ["- ch\u01b0a \u0111\u1ee7 d\u1eef li\u1ec7u -"]
    t0 = time.perf_counter()
    frame_count = 0
    with make_processor() as processor:
        while st.session_state.running and cap.isOpened():
            ok, img = cap.read()
            if not ok:
                st.warning("Kh\u00f4ng \u0111\u1ecdc \u0111\u01b0\u1ee3c frame webcam")
                break
            if mirror:
                img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if proc_scale < 0.99:
                proc_in = cv2.resize(rgb, None, fx=proc_scale, fy=proc_scale,
                                     interpolation=cv2.INTER_AREA)
            else:
                proc_in = rgb
            proc_in.flags.writeable = False
            res = processor.process(proc_in)

            vis = img.copy()
            if show_guide:
                draw_guide_box(vis)

            if use_holistic:
                coords, pmask = result_to_unified_array(res)
                if is_unified:
                    frame_arr = np.concatenate(
                        [coords, pmask[:, None].astype(np.float32)], axis=1)  # (80,4)
                else:
                    # Model chi dung tay: rut (42,3) [tay trai | tay phai] tu Holistic
                    frame_arr = np.zeros((42, 3), dtype=np.float32)
                    frame_arr[:21] = coords[LHAND_OFF:LHAND_OFF + NUM_HAND]
                    frame_arr[21:] = coords[RHAND_OFF:RHAND_OFF + NUM_HAND]
                hands_present = bool(
                    pmask[LHAND_OFF:LHAND_OFF + NUM_HAND].any()
                    or pmask[RHAND_OFF:RHAND_OFF + NUM_HAND].any()
                )
                if show_points:
                    draw_holistic_landmarks(
                        vis, res, draw_pose=show_pose, draw_hands=True, draw_face=show_face,
                        draw_face_oval=show_face_oval, draw_iris=show_iris)
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

            if not buffer:
                last_label, last_conf = "...", 0.0
                last_top3 = ["- ch\u01b0a \u0111\u1ee7 d\u1eef li\u1ec7u -"]
            elif frame_count % infer_every == 0:
                seq = np.stack(buffer)
                prob = predict_prob(model, device, feature_mode, seq, max_len)
                topk = min(3, prob.numel())
                vals, idxs = torch.topk(prob, k=topk)
                last_conf = float(vals[0])
                last_label = id2label.get(int(idxs[0]), "?")
                last_top3 = [f"{id2label.get(int(i), '?')} \u2014 {float(v)*100:.1f}%"
                             for v, i in zip(vals, idxs)]

                if cooldown > 0:
                    cooldown -= 1
                else:
                    if last_conf >= threshold:
                        recent.append(last_label)
                    else:
                        recent.clear()
                    if len(recent) == vote_k and len(set(recent)) == 1:
                        st.session_state.tokens.append(last_label)
                        st.session_state.last_commit = last_label
                        recent.clear()
                        buffer.clear()
                        cooldown = cooldown_frames

            label, conf, top3_lines = last_label, last_conf, last_top3

            frame_count += 1
            elapsed = max(time.perf_counter() - t0, 1e-6)
            fps = frame_count / elapsed
            cv2.putText(vis, f"FPS {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            if cooldown > 0:
                cv2.putText(vis, f"Reset {cooldown}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255), 2, cv2.LINE_AA)

            frame_box.image(vis, channels="BGR", use_container_width=True)
            if show_points:
                legend_box.markdown(LEGEND_HTML, unsafe_allow_html=True)
            else:
                legend_box.empty()
            info_box.markdown(render_info(label, conf, top3_lines,
                                          st.session_state.last_commit, threshold),
                              unsafe_allow_html=True)
            text_label.markdown("**V\u0103n b\u1ea3n nh\u1eadn \u0111\u01b0\u1ee3c**")
            text_box.markdown(
                f'<div class="vsl-textbox">{decode_tokens(st.session_state.tokens, alphabet_mode) or "(chưa có)"}</div>',
                unsafe_allow_html=True)
            help_box.info("M\u1edf tay r\u00f5, \u0111\u1eb7t trong khung, gi\u1eef y\u00ean ~0.5-1s. Sau khi ch\u1ed1t, h\u1ec7 th\u1ed1ng t\u1ef1 reset.")
    cap.release()
else:
    text = decode_tokens(st.session_state.tokens, alphabet_mode)
    info_box.markdown(render_info("\u2014", 0.0, ["- ch\u01b0a ch\u1ea1y -"],
                                  st.session_state.last_commit, threshold),
                      unsafe_allow_html=True)
    text_label.markdown("**V\u0103n b\u1ea3n nh\u1eadn \u0111\u01b0\u1ee3c**")
    text_box.markdown(
        f'<div class="vsl-textbox">{text or "(chưa có)"}</div>',
        unsafe_allow_html=True)
    frame_box.info("Nh\u1ea5n **\u25B6 B\u1eaft \u0111\u1ea7u** \u0111\u1ec3 m\u1edf webcam. App t\u1ef1 reset sau m\u1ed7i k\u00fd t\u1ef1.")
    if text.strip() and st.button("\U0001F50A Ph\u00e1t \u00e2m v\u0103n b\u1ea3n"):
        out_path = f"tmp_audio_{uuid.uuid4().hex}.mp3"
        try:
            synthesize(text.strip(), out_path=out_path)
            st.audio(out_path)
        except Exception as e:
            st.error(f"L\u1ed7i TTS: {e}")
