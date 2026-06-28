"""
hand_utils.py - utility dùng chung cho extract/train/predict/app.
Fixes vs bản cũ:
  - preprocess_landmark_array KHÔNG gọi canonicalize_hands nữa (tránh double call)
  - augment_hand_sequence thêm temporal_cutout (robust với frame drop từ webcam)
  - _prepare trong dataset chịu trách nhiệm gọi canonicalize_hands 1 lần duy nhất
"""
from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except Exception:
    mp = None
    mp_hands = None

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
# liên kết giữa 2 tay để ST-GCN học tư thế đối xứng / khoảng cách 2 tay
CROSS_HAND_EDGES = [
    (0, 21), (5, 26), (9, 30), (13, 34), (17, 38),
    (4, 25), (8, 29), (12, 33), (16, 37), (20, 41),
]
FINGERTIPS = [4, 8, 12, 16, 20]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def result_to_two_hand_array(res) -> np.ndarray:
    """MediaPipe Hands result -> (42, 3) = [left 21 | right 21]."""
    arr = np.zeros((42, 3), dtype=np.float32)
    if not getattr(res, "multi_hand_landmarks", None):
        return arr
    handedness = getattr(res, "multi_handedness", None) or []
    for i, hand_lms in enumerate(res.multi_hand_landmarks):
        label = "Right"
        if i < len(handedness) and handedness[i].classification:
            label = handedness[i].classification[0].label
        base = 0 if label == "Left" else 21
        for j, lm in enumerate(hand_lms.landmark):
            arr[base + j] = (lm.x, lm.y, lm.z)
    return arr


def _canonicalize_one_hand(hand: np.ndarray) -> np.ndarray:
    hand = hand.astype(np.float32).copy()
    if not np.any(hand):
        return hand
    wrist = hand[0:1].copy()
    hand -= wrist
    scale = float(np.sqrt((hand ** 2).sum(axis=1)).max())
    if scale > 1e-6:
        hand /= scale
    return hand


def canonicalize_hands(arr: np.ndarray) -> np.ndarray:
    """Chuẩn hóa 2 tay và đưa sign 1 tay về cùng một slot.

    - Nếu chỉ có 1 tay: đưa về slot đầu tiên để giảm phụ thuộc trái/phải.
    - Mỗi tay: trừ cổ tay + chuẩn hóa theo kích thước tay.
    """
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    else:
        squeeze = False
    out = arr.astype(np.float32).copy()
    for t in range(out.shape[0]):
        left = out[t, :21].copy()
        right = out[t, 21:].copy()
        has_left = np.any(left)
        has_right = np.any(right)
        if (not has_left) and has_right:
            left, right = right, np.zeros_like(right)
        left = _canonicalize_one_hand(left)
        right = _canonicalize_one_hand(right)
        out[t, :21] = left
        out[t, 21:] = right
    return out[0] if squeeze else out


def _bone_vectors(one_hand: np.ndarray) -> np.ndarray:
    vecs = []
    for a, b in HAND_EDGES:
        vecs.append(one_hand[b] - one_hand[a])
    return np.stack(vecs, axis=0) if vecs else np.zeros((0, 3), dtype=np.float32)


def extract_hand_features(seq42: np.ndarray, include_velocity: bool = True) -> np.ndarray:
    """(T, 42, 3) -> (T, F) feature engineering nhẹ nhưng hiệu quả.

    NOTE: Hàm này KHÔNG gọi canonicalize_hands nữa.
    Caller (dataset._prepare hoặc preprocess_landmark_array) phải gọi trước.
    """
    seq = seq42  # đã được canonicalize bởi caller
    coords = seq.reshape(seq.shape[0], -1)

    bones = []
    dists = []
    for frame in seq:
        per_frame_bones = []
        per_frame_dists = []
        for start in (0, 21):
            hand = frame[start:start + 21]
            bone = _bone_vectors(hand).reshape(-1)
            tip_dist = np.array([
                float(np.linalg.norm(hand[idx] - hand[0])) for idx in FINGERTIPS
            ], dtype=np.float32)
            per_frame_bones.append(bone)
            per_frame_dists.append(tip_dist)
        bones.append(np.concatenate(per_frame_bones, axis=0))
        dists.append(np.concatenate(per_frame_dists, axis=0))
    bones = np.stack(bones).astype(np.float32)
    dists = np.stack(dists).astype(np.float32)

    feats = [coords, bones, dists]
    if include_velocity:
        vel = np.diff(coords, axis=0, prepend=coords[:1])
        feats.append(vel.astype(np.float32))
    return np.concatenate(feats, axis=1).astype(np.float32)


def preprocess_landmark_array(arr: np.ndarray, feature_mode: str = "hands") -> np.ndarray:
    """Hỗ trợ cả định dạng cũ (T,543,3) lẫn mới (T,42,3).

    FIX: Không gọi canonicalize_hands nữa — caller phải gọi trước.
    Điều này tránh double canonicalize khi gọi qua dataset._prepare.

    feature_mode:
      - hands     : feature engineering phẳng, hợp MLP/GRU/Transformer
      - hands_raw : raw joints 42x3 đã canonicalize -> flatten 126, hợp ST-GCN
      - all       : fallback reshape toàn bộ
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Mảng landmark không hợp lệ: shape={arr.shape}")
    if arr.shape[1] == 543:
        arr = arr[:, -42:, :]
    elif arr.shape[1] != 42 and feature_mode in ("hands", "hands_raw"):
        raise ValueError(f"Không nhận dạng được số điểm tay: {arr.shape}")

    if feature_mode == "hands":
        # KHÔNG gọi canonicalize nữa — caller đã gọi rồi
        return extract_hand_features(arr)
    if feature_mode == "hands_raw":
        # KHÔNG gọi canonicalize nữa — caller đã gọi rồi
        return arr.reshape(arr.shape[0], -1).astype(np.float32)
    return arr.reshape(arr.shape[0], -1).astype(np.float32)


def rotate_xy(seq42: np.ndarray, deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    out = seq42.copy()
    xy = out[..., :2] @ rot.T
    out[..., :2] = xy
    return out


def _temporal_cutout(seq: np.ndarray, rng: np.random.RandomState,
                     max_cut_ratio: float = 0.25) -> np.ndarray:
    """Xóa ngẫu nhiên 1 đoạn liên tiếp tối đa max_cut_ratio*T frames.

    Mô phỏng mất frame từ webcam / chuyển động nhanh.
    Đoạn bị xóa được thay bằng frame lân cận (forward-fill).
    """
    T = seq.shape[0]
    if T < 4:
        return seq
    cut_len = max(1, int(rng.uniform(0.1, max_cut_ratio) * T))
    start = rng.randint(0, T - cut_len + 1)
    out = seq.copy()
    # forward fill từ frame trước đoạn cắt (hoặc frame sau nếu start=0)
    fill = out[start - 1] if start > 0 else out[start + cut_len]
    out[start:start + cut_len] = fill
    return out


def augment_hand_sequence(seq42: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Augmentation hình học + temporal cutout cho chuỗi 42 hand joints.

    Mới so với bản cũ:
    - Thêm temporal cutout (prob=0.25) để model chịu được frame drop
    """
    out = seq42.astype(np.float32).copy()
    if rng.rand() < 0.85:
        out = rotate_xy(out, rng.uniform(-12, 12))
    if rng.rand() < 0.85:
        out[..., :2] *= rng.uniform(0.92, 1.08)
    if rng.rand() < 0.85:
        out += rng.normal(0, 0.01, size=out.shape).astype(np.float32)
    if rng.rand() < 0.20:
        # đảo 2 slot tay để giảm nhạy trái/phải trên webcam mirror
        out = np.concatenate([out[:, 21:], out[:, :21]], axis=1)
    # NEW: temporal cutout — mô phỏng mất frame
    if rng.rand() < 0.25:
        out = _temporal_cutout(out, rng)
    return out


def draw_hand_landmarks(img, hand_landmarks, color=(255, 0, 255)) -> None:
    if not hand_landmarks:
        return
    h, w = img.shape[:2]
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    for a, b in HAND_EDGES:
        cv2.line(img, pts[a], pts[b], color, 2)
    for i, pt in enumerate(pts):
        cv2.circle(img, pt, 5 if i == 0 else 3, color, -1)


def draw_guide_box(img, text: str = "Đặt tay vào khung") -> None:
    h, w = img.shape[:2]
    bw, bh = int(w * 0.44), int(h * 0.58)
    x1, y1 = (w - bw) // 2, (h - bh) // 2
    x2, y2 = x1 + bw, y1 + bh
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(img, text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 220, 255), 2, cv2.LINE_AA)
