"""app.py - VSL Webcam Demo (RTX 4050 optimized).

Cải tiến so với bản gốc:
  - Model warmup (dummy forward pass) sau khi load -> không lag frame đầu
  - torch.compile safe: warmup TRƯỚC compile, detect Windows/Triton, fallback eager
  - Fix: torch.autocast dùng device_type=device (không hard-code "cuda")
  - EMA smoothing xác suất (smooth_alpha) để dự đoán ổn định
  - JPEG encode + thu nhỏ ảnh hiển thị (disp_width / jpeg_quality) -> FPS cao hơn
  - Gián nhịp cập nhật panel (ui_every) -> giảm Streamlit re-render
  - Adaptive FPS EMA thay vì tính từ đầu
  - Phát âm tự động hoặc thủ công qua TTS
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
from tts import synthesize     # noqa: E402
from unicode_utils import tokens_to_text  # noqa: E402

EMPTY_TXT = "(chưa có)"

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
      .vsl-textbox {background:#0f172a; color:#f8fafc; border-radius: 12px; padding: 14px 16px;
                    font-size: 1.4rem; font-weight: 700; min-height: 56px; word-break: break-word;}
      .vsl-legend span {display:inline-flex; align-items:center; margin-right: 14px; font-size:.85rem; color:#374151;}
      .vsl-dot {width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="vsl-title">\U0001F91F Nhận dạng ngôn ngữ ký hiệu tiếng Việt</div>',
            unsafe_allow_html=True)
st.markdown('<div class="vsl-sub">Webcam thời gian thực · RTX 4050 · tự reset sau mỗi ký tự</div>',
            unsafe_allow_html=True)

# ---------------------------- Helpers ----------------------------
def _triton_available() -> bool:
    """Triton chưa hỗ trợ Windows native -> trả False trên win32."""
    import sys
    if sys.platform == "win32":
        return False
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def _try_compile(model):
    """torch.compile an toàn: suppress_errors + chọn backend phù hợp OS.
    
    - Windows (không có Triton): backend='eager' (graph capture, không JIT kernel)
    - Linux/Mac có Triton: backend='inductor', mode='reduce-overhead'
    Returns (compiled_model, backend_str)
    """
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True   # fallback sang eager thay vì crash
    if _triton_available():
        backend, mode = "inductor", "reduce-overhead"
    else:
        backend, mode = "eager", "default"
    try:
        return torch.compile(model, backend=backend, mode=mode), backend
    except Exception as e:
        return model, f"disabled({e})"


@st.cache_resource
def load_model(ckpt_path: str, prefer_cuda: bool = True, compile_model: bool = True):
    device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], ckpt["feature_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    max_len = int(ckpt.get("max_len", 16))
    fdim    = int(ckpt["feature_dim"])

    # Bước 1: Warmup TRƯỚC compile — tránh crash Triton khi warmup dùng compiled graph
    with torch.no_grad():
        dummy = torch.zeros(1, max_len, fdim, device=device)
        dlen  = torch.tensor([max_len], dtype=torch.long, device=device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            model(dummy, dlen)
    if device == "cuda":
        torch.cuda.synchronize()

    # Bước 2: Compile SAU warmup (an toàn hơn; Windows dùng eager backend)
    compile_backend = "disabled"
    if compile_model and device == "cuda":
        model, compile_backend = _try_compile(model)

    id2label = {v: k for k, v in ckpt["labels"].items()}
    return model, ckpt, id2label, device, compile_backend


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
        # FIX: device_type=device (không hard-code "cuda")
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            prob = F.softmax(model(x, lengths), dim=1)[0]
    return prob.detach().float().cpu()


def play_audio(box, path):
    try:
        box.audio(path, format="audio/mp3", autoplay=True)
    except TypeError:
        box.audio(path, format="audio/mp3")


def speak_to_file(text):
    if not text or not text.strip():
        return None
    path = f"tmp_audio_{uuid.uuid4().hex}.mp3"
    synthesize(text.strip(), out_path=path)
    return path


def render_info(label, conf, top3_lines, last_commit, threshold):
    bar_color = "#16a34a" if conf >= threshold else "#f59e0b"
    pct = max(0.0, min(1.0, conf)) * 100
    top_html = "".join(f'<div class="vsl-top"><b>{i+1}.</b> {line}</div>'
                       for i, line in enumerate(top3_lines))
    return f"""
    <div class="vsl-card">
      <div style="color:#6b7280;font-size:.85rem;">Dự đoán hiện tại</div>
      <div class="vsl-pred">{label}</div>
      <div class="vsl-bar-bg"><div class="vsl-bar-fg" style="width:{pct:.0f}%;background:{bar_color};"></div></div>
      <div style="text-align:right;font-size:.85rem;color:#374151;margin-top:2px;">{conf*100:.1f}%</div>
      <hr style="margin:10px 0;border:none;border-top:1px solid #eef0f3;">
      <div style="color:#6b7280;font-size:.85rem;margin-bottom:4px;">Top-3</div>
      {top_html}
      <hr style="margin:10px 0;border:none;border-top:1px solid #eef0f3;">
      <div style="color:#6b7280;font-size:.85rem;">Đã chốt gần nhất</div>
      <div style="font-weight:700;font-size:1.1rem;">{last_commit or EMPTY_TXT}</div>
    </div>
    """


LEGEND_HTML = (
    '<div class="vsl-legend" style="margin-top:6px;">'
    '<span><span class="vsl-dot" style="background:#ffd700;"></span>Thân trên</span>'
    '<span><span class="vsl-dot" style="background:#00ff00;"></span>Tay trái</span>'
    '<span><span class="vsl-dot" style="background:#ff00ff;"></span>Tay phải</span>'
    '<span><span class="vsl-dot" style="background:#00c8ff;"></span>Gương mặt</span>'
    '</div>'
)

# ---------------------------- Session state ----------------------------
for k, v in [("running", False), ("tokens", []), ("last_commit", ""), ("audio_path", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------- Sidebar ----------------------------
st.sidebar.header("⚙️ Cài đặt")
ckpt_path    = st.sidebar.text_input("Checkpoint", "checkpoints/best.pth")
use_cuda     = st.sidebar.checkbox("Dùng GPU nếu có (RTX 4050)", value=True)
compile_flag = st.sidebar.checkbox("torch.compile (nhanh hơn ~30%, cần PyTorch 2.x)", value=True)

with st.sidebar.expander("🎯 Ngưỡng & chốt ký tự", expanded=True):
    threshold      = st.slider("Ngưỡng tin cậy",             0.0, 1.0, 0.70)
    vote_k         = st.slider("Số khung lặp để chốt",       1,   8,   2)
    window         = st.slider("Số khung nhìn lại",           1,  32,  12)
    cooldown_frames = st.slider("Số khung reset sau khi chốt", 0, 30,   6)
    smooth_alpha   = st.slider("Làm mượt xác suất (EMA)",    0.0, 0.9, 0.5, 0.1)

with st.sidebar.expander("👁️ Hiển thị", expanded=False):
    show_points    = st.checkbox("Hiện điểm khung xương", value=True)
    show_pose      = st.checkbox("  · Thân trên (pose)",  value=True)
    show_face      = st.checkbox("  · Gương mặt (môi/mắt/mày)", value=True)
    show_face_oval = st.checkbox("  · Viền mặt",          value=False)
    show_iris      = st.checkbox("  · Tròng mắt",         value=False)
    full_skeleton  = st.checkbox(
        "Luôn dùng Holistic (xem thân + mặt kể cả model chỉ dùng tay)",
        value=True)
    show_guide     = st.checkbox("Hiện khung căn tay",    value=True)
    mirror         = st.checkbox("Lật gương webcam",      value=True)

with st.sidebar.expander("🎛️ MediaPipe", expanded=False):
    min_det   = st.slider("Detect conf",  0.1, 0.9, 0.4)
    min_track = st.slider("Track conf",   0.1, 0.9, 0.4)

with st.sidebar.expander("⚡ Hiệu năng (FPS)", expanded=True):
    model_complexity = st.select_slider("Độ phức tạp model", options=[0, 1, 2], value=0)
    refine_face      = st.checkbox("Tinh chỉnh điểm mặt (nặng)", value=False)
    proc_scale       = st.slider("Thu nhỏ ảnh khi xử lý", 0.3, 1.0, 0.5, 0.1)
    cap_width        = st.select_slider("Độ phân giải webcam",
                                        options=[480, 640, 800, 1280], value=640)
    infer_every      = st.slider("Dự đoán mỗi N khung", 1, 5, 2)
    ui_every         = st.slider("Cập nhật panel mỗi N khung", 1, 10, 4)
    disp_width       = st.select_slider("Bề rộng hiển thị",
                                        options=[420, 480, 540, 640, 800], value=540)
    jpeg_quality     = st.slider("Chất lượng JPEG hiển thị", 40, 95, 70, 5)

with st.sidebar.expander("🔊 Âm thanh", expanded=True):
    auto_speak = st.checkbox("Tự phát âm khi chốt ký tự", value=False)

b1, b2 = st.sidebar.columns(2)
if b1.button("▶ Bắt đầu", use_container_width=True):
    st.session_state.running = True
if b2.button("⏹ Dừng", use_container_width=True):
    st.session_state.running = False
b3, b4 = st.sidebar.columns(2)
if b3.button("🗑 Xóa câu", use_container_width=True):
    st.session_state.tokens      = []
    st.session_state.last_commit = ""
if b4.button("⌫ Xóa 1 ký tự", use_container_width=True) and st.session_state.tokens:
    st.session_state.tokens.pop()

# ---------------------------- Load model ----------------------------
try:
    model, ckpt, id2label, device, compile_backend = load_model(
        ckpt_path, prefer_cuda=use_cuda, compile_model=compile_flag)
    feature_mode = ckpt.get("feature_mode", "hands")
    max_len      = int(ckpt.get("max_len", 16))
    alphabet_mode = is_alphabet_mode(id2label)
    is_unified   = (feature_mode == "unified")
    badge = "Holistic (thân + mặt + 2 tay)" if is_unified else "Hands (2 tay)"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    compile_info = f" · compile={compile_backend}" if device == "cuda" else ""
    st.sidebar.success(f"✅ {ckpt.get('model','?')} · {gpu_name} · {badge}{compile_info}")
except (FileNotFoundError, ValueError, RuntimeError) as _load_err:
    model = ckpt = id2label = None
    feature_mode = "hands"
    max_len      = 16
    alphabet_mode = True
    is_unified   = False
    device       = "cpu"
    st.sidebar.error(f"Lỗi nạp checkpoint: {_load_err}")

# Nút phát âm thủ công (hoạt động khi DỪNG)
st.sidebar.markdown("---")
_cur_text = decode_tokens(st.session_state.tokens, alphabet_mode) if id2label else ""
if st.sidebar.button("🔊 Phát âm văn bản", use_container_width=True,
                     disabled=not _cur_text.strip()):
    try:
        p = speak_to_file(_cur_text)
        if p:
            st.session_state.audio_path = p
    except Exception as e:
        st.sidebar.error(f"Lỗi TTS: {e}")
if st.session_state.running:
    st.sidebar.caption("ⓘ Nút phát âm hoạt động sau khi bấm ⏹ Dừng. "
                       "Khi đang chạy, bật 'Tự phát âm'.")

if is_unified and mp_holistic is None:
    st.sidebar.warning("Checkpoint unified cần mediapipe Holistic: pip install mediapipe")

use_holistic = (is_unified or full_skeleton) and (mp_holistic is not None)
if use_holistic and not is_unified:
    st.sidebar.caption("🔎 Đang xem khung xương đầy đủ bằng Holistic. "
                       "Model hiện tại vẫn chỉ dự đoán bằng 2 tay.")

# ---------------------------- Layout ----------------------------
col1, col2 = st.columns([3, 2])
frame_box  = col1.empty()
legend_box = col1.empty()
info_box   = col2.empty()
text_label = col2.empty()
text_box   = col2.empty()
audio_box  = col2.empty()
help_box   = st.empty()

# Phần TĨNH: chỉ vẽ 1 lần (không lặp mỗi khung)
if show_points:
    legend_box.markdown(LEGEND_HTML, unsafe_allow_html=True)
else:
    legend_box.empty()
text_label.markdown("**Văn bản nhận được**")
if st.session_state.audio_path:
    play_audio(audio_box, st.session_state.audio_path)


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


def show_frame(vis, fps, cooldown):
    """Encode JPEG + thu nhỏ ảnh -> đẩy lên Streamlit nhanh hơn."""
    cv2.putText(vis, f"FPS {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    if cooldown > 0:
        cv2.putText(vis, f"Reset {cooldown}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 255), 2, cv2.LINE_AA)
    disp = vis
    if disp_width and vis.shape[1] > disp_width:
        scale = disp_width / float(vis.shape[1])
        disp  = cv2.resize(vis, (int(disp_width), int(vis.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    ok_enc, buf = cv2.imencode(".jpg", disp,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if ok_enc:
        frame_box.image(buf.tobytes(), use_container_width=True)
    else:
        frame_box.image(disp, channels="BGR", use_container_width=True)


# ============================== MAIN LOOP ==============================
if st.session_state.running:
    if model is None:
        st.error("Chưa nạp được checkpoint.")
        st.stop()
    if is_unified and mp_holistic is None:
        st.error("Mode unified cần mediapipe Holistic: pip install mediapipe")
        st.stop()

    help_box.info("Mở tay rõ, đặt trong khung, giữ yên ~0.5-1s. Sau khi chốt, hệ thống tự reset.")

    buffer   = collections.deque(maxlen=min(window, max_len))
    recent   = collections.deque(maxlen=vote_k)
    cooldown = 0
    ema_prob = None

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Không mở được webcam")
        st.stop()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cap_width * 3 / 4))
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass

    last_label, last_conf = "...", 0.0
    last_top3    = ["- chưa đủ dữ liệu -"]
    prev_tok_len = -1
    fps          = 0.0
    t_prev       = time.perf_counter()
    frame_count  = 0

    with make_processor() as processor:
        while st.session_state.running and cap.isOpened():
            ok, img = cap.read()
            if not ok:
                st.warning("Không đọc được frame webcam")
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
                frame_arr     = result_to_two_hand_array(res)
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

            committed_now = None
            if not buffer:
                last_label, last_conf = "...", 0.0
                last_top3 = ["- chưa đủ dữ liệu -"]
            elif frame_count % infer_every == 0:
                seq  = np.stack(buffer)
                prob = predict_prob(model, device, feature_mode, seq, max_len)

                # EMA smoothing xác suất -> ổn định hơn
                if smooth_alpha > 0 and ema_prob is not None and ema_prob.shape == prob.shape:
                    ema_prob = smooth_alpha * ema_prob + (1.0 - smooth_alpha) * prob
                else:
                    ema_prob = prob
                pp = ema_prob

                topk = min(3, pp.numel())
                vals, idxs = torch.topk(pp, k=topk)
                last_conf  = float(vals[0])
                last_label = id2label.get(int(idxs[0]), "?")
                last_top3  = [f"{id2label.get(int(i), '?')} — {float(v)*100:.1f}%"
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
                        committed_now = last_label
                        recent.clear()
                        buffer.clear()
                        ema_prob  = None
                        cooldown  = cooldown_frames

            # Tự phát âm khi vừa chốt
            if committed_now is not None and auto_speak:
                try:
                    full_text = decode_tokens(st.session_state.tokens, alphabet_mode)
                    p = speak_to_file(full_text)
                    if p:
                        play_audio(audio_box, p)
                except Exception:
                    pass

            frame_count += 1
            # FPS EMA tức thời (mượt hơn đếm từ đầu)
            t_now = time.perf_counter()
            dt    = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                inst = 1.0 / dt
                fps  = inst if fps == 0.0 else (0.9 * fps + 0.1 * inst)

            show_frame(vis, fps, cooldown)

            # Cập nhật panel theo gián nhịp hoặc khi vừa chốt
            if (frame_count % ui_every == 0) or (committed_now is not None):
                info_box.markdown(render_info(last_label, last_conf, last_top3,
                                              st.session_state.last_commit, threshold),
                                  unsafe_allow_html=True)
                cur_len = len(st.session_state.tokens)
                if cur_len != prev_tok_len:
                    prev_tok_len = cur_len
                    txt = decode_tokens(st.session_state.tokens, alphabet_mode)
                    text_box.markdown(
                        f'<div class="vsl-textbox">{txt or EMPTY_TXT}</div>',
                        unsafe_allow_html=True)

    cap.release()

else:
    text = decode_tokens(st.session_state.tokens, alphabet_mode) if id2label else ""
    info_box.markdown(render_info("—", 0.0, ["- chưa chạy -"],
                                  st.session_state.last_commit, threshold),
                      unsafe_allow_html=True)
    text_box.markdown(
        f'<div class="vsl-textbox">{text or EMPTY_TXT}</div>',
        unsafe_allow_html=True)
    frame_box.info("Nhấn **▶ Bắt đầu** để mở webcam. App tự reset sau mỗi ký tự.")
