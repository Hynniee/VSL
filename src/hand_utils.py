"""
hand_utils.py - utility dung chung cho extract/train/predict/app.
Toan bo pipeline moi xoay quanh 2 ban tay (42 landmark), uu tien toc do va
su on dinh cho bai toan CHU CAI TINH tu webcam.
"""
from __future__ import annotations

import math

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
# lien ket giua 2 tay de ST-GCN hoc tu the doi xung / khoang cach 2 tay
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
    """Chuan hoa 2 tay va dua sign 1 tay ve cung mot slot.

    - Neu chi co 1 tay: dua ve slot dau tien de giam phu thuoc trai/phai.
    - Moi tay: tru co tay + chuan hoa theo kich thuoc tay.
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
    """(T, 42, 3) -> (T, F) feature engineering nhe nhung hieu qua."""
    seq = canonicalize_hands(seq42)
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
    """Ho tro ca dinh dang cu (T,543,3) lan moi (T,42,3).

    feature_mode:
      - hands     : feature engineering phang, hop MLP/GRU/Transformer
      - hands_raw : raw joints 42x3 da canonicalize -> flatten 126, hop ST-GCN
      - all       : fallback reshape toan bo
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Mang landmark khong hop le: shape={arr.shape}")
    if arr.shape[1] == 543:
        arr = arr[:, -42:, :]
    elif arr.shape[1] != 42 and feature_mode in ("hands", "hands_raw"):
        raise ValueError(f"Khong nhan dang duoc so diem tay: {arr.shape}")

    if feature_mode == "hands":
        return extract_hand_features(arr)
    if feature_mode == "hands_raw":
        arr = canonicalize_hands(arr)
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


def augment_hand_sequence(seq42: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = seq42.astype(np.float32).copy()
    if rng.rand() < 0.85:
        out = rotate_xy(out, rng.uniform(-12, 12))
    if rng.rand() < 0.85:
        out[..., :2] *= rng.uniform(0.92, 1.08)
    if rng.rand() < 0.85:
        out += rng.normal(0, 0.01, size=out.shape).astype(np.float32)
    if rng.rand() < 0.20:
        # dao 2 slot tay de giam nhay trai/phai tren webcam mirror
        out = np.concatenate([out[:, 21:], out[:, :21]], axis=1)
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


def draw_guide_box(img, text: str = "Dat tay vao khung") -> None:
    h, w = img.shape[:2]
    bw, bh = int(w * 0.44), int(h * 0.58)
    x1, y1 = (w - bw) // 2, (h - bh) // 2
    x2, y2 = x1 + bw, y1 + bh
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(img, text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 220, 255), 2, cv2.LINE_AA)
