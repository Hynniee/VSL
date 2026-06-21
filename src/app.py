"""
===============================================================================
 app.py - DEMO REAL-TIME (Webcam -> Model -> van ban)  [Streamlit]
===============================================================================

Webcam -> MediaPipe (keypoint) -> Model -> chu/tu -> hien len man hinh -> TTS.

CHAY:  streamlit run src/app.py
CAI :  pip install streamlit mediapipe opencv-python torch gTTS requests aiohttp

DA SUA so voi ban cu:
  1) decode_buffer cu lam `int(idx)` tren CHUOI ten nhan -> crash. Bo han, dung
     truc tiep danh sach ten nhan da chot.
  2) `idx` co the chua duoc gan o nhung frame dau -> NameError. Da chan.
  3) Nut "Phat am" cu nam SAU vong while nen gan nhu khong bao gio bam toi.
     Nay luu cau vao st.session_state -> hien & phat am o nhanh "da dung".
  4) Them DEBOUNCE: chi chot 1 ky hieu khi no lap lai on dinh (stable_k lan),
     vuot nguong, va khac ky hieu vua chot -> bot nhieu/giat khi real-time.
  5) Chi doan moi N frame (predict_every) cho do nang may.
===============================================================================
"""
import collections
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))
from extract_keypoints import landmarks_to_array, mp_holistic
from model import build_model
from unicode_utils import tokens_to_text
from tts import synthesize


st.set_page_config(page_title="VSL - Nhan dang ngon ngu ky hieu", layout="wide")
st.title("Nhan dang ngon ngu ky hieu tieng Viet")


@st.cache_resource
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ckpt["model"], ckpt["feature_dim"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    id2label = {v: k for k, v in ckpt["labels"].items()}
    return model, ckpt, id2label


def is_alphabet_mode(id2label):
    """True neu nhan la token chu cai/dau (can rap van), False neu la tu hoan chinh."""
    names = list(id2label.values())
    return any(str(n).startswith(("chu_", "dau_", "thanh_")) for n in names)


def decode_tokens(tokens, alphabet_mode):
    """tokens: danh sach TEN NHAN da chot -> van ban hien thi."""
    if alphabet_mode:
        return tokens_to_text(tokens)
    return " ".join(tokens)


# --- Trang thai giu qua cac lan rerun cua Streamlit ---
if "running" not in st.session_state:
    st.session_state.running = False
if "tokens" not in st.session_state:
    st.session_state.tokens = []  # danh sach ten nhan da "chot"

ckpt_path = st.sidebar.text_input("Checkpoint", "checkpoints/best.pth")
window = st.sidebar.slider("So frame moi lan doan", 16, 80, 40)
predict_every = st.sidebar.slider("Doan moi N frame", 1, 15, 5)
threshold = st.sidebar.slider("Nguong tin cay", 0.0, 1.0, 0.6)
stable_k = st.sidebar.slider("So lan lap de chot 1 ky hieu", 1, 10, 3)

b1, b2, b3 = st.sidebar.columns(3)
if b1.button("Bat dau"):
    st.session_state.running = True
if b2.button("Dung"):
    st.session_state.running = False
if b3.button("Xoa cau"):
    st.session_state.tokens = []

col1, col2 = st.columns([3, 2])
frame_box = col1.empty()
result_box = col2.empty()
sentence_box = col2.empty()


if st.session_state.running:
    try:
        model, ckpt, id2label = load_model(ckpt_path)
    except FileNotFoundError:
        st.error("Chua tim thay checkpoint. Hay train truoc: python src/train.py")
        st.stop()

    feature_mode = ckpt["feature_mode"]
    alphabet_mode = is_alphabet_mode(id2label)

    buffer = collections.deque(maxlen=window)     # keypoint cac frame gan nhat
    recent = collections.deque(maxlen=stable_k)   # cac du doan gan nhat (debounce)
    frame_i = 0
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(model_complexity=1, refine_face_landmarks=False) as holistic:
        while st.session_state.running and cap.isOpened():
            ok, img = cap.read()
            if not ok:
                st.warning("Khong doc duoc frame tu webcam.")
                break
            res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            buffer.append(landmarks_to_array(res))
            frame_i += 1

            label, conf = "...", 0.0
            if len(buffer) == window and frame_i % predict_every == 0:
                arr_batch = np.stack(buffer)
                if feature_mode == "hands":
                    arr_batch = arr_batch[:, -42:, :]
                x = torch.from_numpy(
                    arr_batch.reshape(arr_batch.shape[0], -1).astype("float32")
                ).unsqueeze(0)
                lengths = torch.tensor([x.shape[1]])
                with torch.no_grad():
                    prob = F.softmax(model(x, lengths), dim=1)[0]
                conf, idx = float(prob.max()), int(prob.argmax())
                label = id2label.get(idx, "?")

                # DEBOUNCE: chi chot khi cung 1 nhan lap lai du stable_k lan,
                # vuot nguong, va khac ky hieu vua chot truoc do.
                if conf >= threshold:
                    recent.append(label)
                    last = st.session_state.tokens[-1] if st.session_state.tokens else None
                    if (len(recent) == stable_k and len(set(recent)) == 1
                            and label != last):
                        st.session_state.tokens.append(label)
                        recent.clear()
                else:
                    recent.clear()

            frame_box.image(img, channels="BGR")
            result_box.markdown(f"### Du doan: {label}\n\nDo tin cay: {conf*100:.0f}%")
            sentence_box.markdown(
                f"## Cau: {decode_tokens(st.session_state.tokens, alphabet_mode)}"
            )

    cap.release()
else:
    st.info("Nhan **Bat dau** o thanh ben de mo webcam.")
    # Hien cau da nhan + phat am. Chay duoc vi cau duoc luu trong session_state.
    alphabet_mode = False
    try:
        _, _, id2label = load_model(ckpt_path)
        alphabet_mode = is_alphabet_mode(id2label)
    except FileNotFoundError:
        pass
    text = decode_tokens(st.session_state.tokens, alphabet_mode)
    st.markdown(f"## Cau da nhan: {text or '(chua co)'}")
    if text.strip() and st.button("Phat am cau"):
        out_path = f"tmp_audio_{uuid.uuid4().hex}.mp3"
        try:
            synthesize(text.strip(), out_path=out_path)
            st.audio(out_path)
        except Exception as e:
            st.error(f"Loi phat am: {e}")
