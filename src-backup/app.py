"""
================================================================================
 app.py - DEMO REAL-TIME (Webcam -> Model -> van ban)  [Streamlit]
================================================================================

Noi tat ca lai thanh 1 ung dung web chay tren trinh duyet:
  Webcam -> MediaPipe (keypoint) -> Model -> chu/tu -> hien len man hinh.

CHAY:  streamlit run src/app.py
CAI :  pip install streamlit mediapipe opencv-python torch gTTS

LUU Y: phai co checkpoint da train (checkpoints/best.pth) truoc khi chay.
       (Phan phat loa bang tts.py va ghep van bang unicode_utils.py co the
        noi them vao day - hien app moi hien thi CHU.)
================================================================================
"""
import collections
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

# Cho phep import cac file cung thu muc src/
sys.path.append(str(Path(__file__).parent))
from extract_keypoints import landmarks_to_array, mp_holistic  # noqa: E402
from model import build_model  # noqa: E402

st.set_page_config(page_title="VSL - Nhan dang ngon ngu ky hieu", layout="wide")
st.title("🤟 Nhận dạng ngôn ngữ ký hiệu tiếng Việt")


@st.cache_resource  # nap model 1 lan roi giu lai (khong nap lai moi frame)
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ckpt["model"], ckpt["feature_dim"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    id2label = {v: k for k, v in ckpt["labels"].items()}
    return model, ckpt, id2label


# --- Thanh dieu khien ben trai ---
ckpt_path = st.sidebar.text_input("Checkpoint", "checkpoints/best.pth")
window = st.sidebar.slider("Số frame mỗi lần đoán", 16, 80, 40)
threshold = st.sidebar.slider("Ngưỡng tin cậy", 0.0, 1.0, 0.6)

if "running" not in st.session_state:
    st.session_state.running = False
if st.sidebar.button("▶️ Bắt đầu"):
    st.session_state.running = True
if st.sidebar.button("⏹️ Dừng"):
    st.session_state.running = False

# Bo cuc: cot trai rong (webcam), cot phai hep (ket qua)
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
    buffer = collections.deque(maxlen=window)  # cua so truot N frame gan nhat
    sentence = []
    cap = cv2.VideoCapture(0)  # 0 = webcam mac dinh
    with mp_holistic.Holistic(model_complexity=1, refine_face_landmarks=False) as holistic:
        while st.session_state.running and cap.isOpened():
            ok, img = cap.read()
            if not ok:
                break
            res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            buffer.append(landmarks_to_array(res))

            label, conf = "...", 0.0
            if len(buffer) == window:  # du N frame moi doan
                arr = np.stack(buffer)
                if feature_mode == "hands":
                    arr = arr[:, -42:, :]
                x = torch.from_numpy(arr.reshape(arr.shape[0], -1).astype("float32")).unsqueeze(0)
                lengths = torch.tensor([x.shape[1]])
                with torch.no_grad():
                    prob = F.softmax(model(x, lengths), dim=1)[0]
                conf, idx = float(prob.max()), int(prob.argmax())
                label = id2label.get(idx, "?")
                # Chi them vao cau khi du tin cay + khac tu vua them (chong lap)
                if conf >= threshold and (not sentence or sentence[-1] != label):
                    sentence.append(label)

            frame_box.image(img, channels="BGR")
            result_box.markdown(f"### Dự đoán: **{label}**\n\nĐộ tin cậy: {conf*100:.0f}%")
            sentence_box.markdown("## Câu: " + " ".join(sentence))
    cap.release()
else:
    st.info("Nhấn **Bắt đầu** ở thanh bên để mở webcam.")

st.sidebar.caption(
    "Với model đánh vần (chữ cái + dấu), dùng unicode_utils.tokens_to_text() "
    "để ghép ký hiệu thành chữ tiếng Việt."
)
