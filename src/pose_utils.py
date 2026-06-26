"""
pose_utils.py - schema keypoint THONG NHAT cho mo hinh GOP (Cach 2).

Gop 4 bo phan thanh 1 do thi khop duy nhat:
  - POSE  : 9 diem than tren  (mui, 2 vai, 2 khuyu, 2 co tay, 2 hong)
  - LHAND : 21 diem tay trai
  - RHAND : 21 diem tay phai
  - FACE  : 29 diem mat (mat, long may, mieng, mui)

Moi khop co them 1 "presence flag" (1 = co, 0 = vang). Nho vay:
  - Du lieu CU (chi co tay, dang (T,42,3)) -> nhung vao slot tay, pose/mat = vang.
  - Du lieu MOI (Holistic, dang (T,80,4)) -> day du.
  => Hai loai dung CHUNG 1 model. Part-dropout khi train giup model khong
     hoc meo "vang mat = chu cai".

Chuan hoa THEO TUNG BO PHAN:
  - Tay : tru co tay + chia kich thuoc tay  => HINH DANG tay giong het du lieu cu
          (alphabet khong bi anh huong khi gop).
  - Pose: tru trung diem 2 vai + chia be rong vai.
  - Mat : tru trong tam mat + chia kich thuoc mat (bat bien voi kich thuoc dau).
Vi tri tay so voi than duoc giu qua chinh cac diem co tay trong POSE.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2 as _cv2
except Exception:  # cv2 co the vang trong vai moi truong
    _cv2 = None

from hand_utils import HAND_EDGES, CROSS_HAND_EDGES

try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    _HAS_MP = True
except Exception:  # sandbox / chua cai mediapipe van import duoc module
    mp = None
    mp_holistic = None
    _HAS_MP = False

# Cac nhom duong net FaceMesh - CHI de VE cho day (khong lien quan schema model)
try:
    from mediapipe.python.solutions import face_mesh_connections as _fmc
    _FACE_CONN = {
        "lips": _fmc.FACEMESH_LIPS,
        "left_eye": _fmc.FACEMESH_LEFT_EYE,
        "right_eye": _fmc.FACEMESH_RIGHT_EYE,
        "left_brow": _fmc.FACEMESH_LEFT_EYEBROW,
        "right_brow": _fmc.FACEMESH_RIGHT_EYEBROW,
        "left_iris": getattr(_fmc, "FACEMESH_LEFT_IRIS", frozenset()),
        "right_iris": getattr(_fmc, "FACEMESH_RIGHT_IRIS", frozenset()),
        "nose": getattr(_fmc, "FACEMESH_NOSE", frozenset()),
        "oval": _fmc.FACEMESH_FACE_OVAL,
    }
except Exception:
    _fmc = None
    _FACE_CONN = None

# --------------------------------------------------------------------------
# Chi so landmark goc cua MediaPipe Holistic
# --------------------------------------------------------------------------
# Pose (33 diem) -> lay 9 diem than tren
POSE_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
# local: 0 mui | 1 vaiT 2 vaiP | 3 khuyuT 4 khuyuP | 5 cotayT 6 cotayP | 7 hongT 8 hongP
POSE_LSHOULDER, POSE_RSHOULDER = 1, 2
POSE_LWRIST, POSE_RWRIST = 5, 6

# Face mesh (468 diem) -> 29 diem bieu cam
FACE_IDS = [
    1,                                  # 0  mui
    33, 160, 158, 133, 153, 144,        # 1..6   mat trai (vong)
    362, 385, 387, 263, 373, 380,       # 7..12  mat phai (vong)
    70, 63, 105, 66, 107,               # 13..17 long may trai
    336, 296, 334, 293, 300,            # 18..22 long may phai
    61, 0, 291, 17,                     # 23..26 mieng: khoe trai, tren, khoe phai, duoi
    13, 14,                             # 27,28  moi trong (tren/duoi)
]

NUM_POSE = len(POSE_IDS)    # 9
NUM_HAND = 21
NUM_FACE = len(FACE_IDS)    # 29

POSE_OFF = 0
LHAND_OFF = POSE_OFF + NUM_POSE      # 9
RHAND_OFF = LHAND_OFF + NUM_HAND     # 30
FACE_OFF = RHAND_OFF + NUM_HAND      # 51
NUM_UNIFIED_JOINTS = FACE_OFF + NUM_FACE   # 80

PART_SLICES = {
    "pose": slice(POSE_OFF, POSE_OFF + NUM_POSE),
    "lhand": slice(LHAND_OFF, LHAND_OFF + NUM_HAND),
    "rhand": slice(RHAND_OFF, RHAND_OFF + NUM_HAND),
    "face": slice(FACE_OFF, FACE_OFF + NUM_FACE),
}

# --------------------------------------------------------------------------
# Canh do thi (local trong tung bo phan) -> doi sang global
# --------------------------------------------------------------------------
_POSE_EDGES_LOCAL = [
    (0, 1), (0, 2), (1, 2),          # mui - 2 vai - noi vai
    (1, 3), (3, 5),                  # tay trai: vai-khuyu-cotay
    (2, 4), (4, 6),                  # tay phai
    (1, 7), (2, 8), (7, 8),          # than: vai-hong, noi hong
]
_FACE_EDGES_LOCAL = [
    # mat trai (vong 1..6)
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
    # mat phai (vong 7..12)
    (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 7),
    # long may trai 13..17 / phai 18..22
    (13, 14), (14, 15), (15, 16), (16, 17),
    (18, 19), (19, 20), (20, 21), (21, 22),
    # mieng 23..26 (vong) + moi trong 27,28
    (23, 24), (24, 25), (25, 26), (26, 23), (27, 28),
    # noi mui voi mat/mieng + may voi mat
    (0, 1), (0, 7), (0, 23), (13, 1), (18, 7),
]


def _build_unified_edges():
    edges = []
    # pose
    for a, b in _POSE_EDGES_LOCAL:
        edges.append((POSE_OFF + a, POSE_OFF + b))
    # 2 ban tay
    for off in (LHAND_OFF, RHAND_OFF):
        for a, b in HAND_EDGES:
            edges.append((off + a, off + b))
    # noi 2 tay (CROSS_HAND_EDGES dung layout 42: 0..20 trai, 21..41 phai)
    for a, b in CROSS_HAND_EDGES:
        ga = LHAND_OFF + a if a < 21 else RHAND_OFF + (a - 21)
        gb = LHAND_OFF + b if b < 21 else RHAND_OFF + (b - 21)
        edges.append((ga, gb))
    # face
    for a, b in _FACE_EDGES_LOCAL:
        edges.append((FACE_OFF + a, FACE_OFF + b))
    # cross-part: co tay pose <-> co tay ban tay ; mui pose <-> mui mat
    edges.append((POSE_OFF + POSE_LWRIST, LHAND_OFF + 0))
    edges.append((POSE_OFF + POSE_RWRIST, RHAND_OFF + 0))
    edges.append((POSE_OFF + 0, FACE_OFF + 0))
    return edges


UNIFIED_EDGES = _build_unified_edges()


# --------------------------------------------------------------------------
# Tu ket qua MediaPipe Holistic -> 1 frame (80,3) + mask (80,)
# --------------------------------------------------------------------------
def _landmarks_xyz(landmark_list):
    return [(lm.x, lm.y, lm.z) for lm in landmark_list.landmark]


def result_to_unified_array(res, vis_thresh: float = 0.3):
    """Holistic result -> (coords (80,3), mask (80,) bool) cho 1 frame."""
    coords = np.zeros((NUM_UNIFIED_JOINTS, 3), dtype=np.float32)
    mask = np.zeros((NUM_UNIFIED_JOINTS,), dtype=bool)

    pose = getattr(res, "pose_landmarks", None)
    if pose is not None:
        lms = pose.landmark
        for li, gi in enumerate(POSE_IDS):
            lm = lms[gi]
            vis = getattr(lm, "visibility", 1.0)
            if vis is None or vis >= vis_thresh:
                coords[POSE_OFF + li] = (lm.x, lm.y, lm.z)
                mask[POSE_OFF + li] = True

    lh = getattr(res, "left_hand_landmarks", None)
    if lh is not None:
        for j, lm in enumerate(lh.landmark):
            coords[LHAND_OFF + j] = (lm.x, lm.y, lm.z)
            mask[LHAND_OFF + j] = True

    rh = getattr(res, "right_hand_landmarks", None)
    if rh is not None:
        for j, lm in enumerate(rh.landmark):
            coords[RHAND_OFF + j] = (lm.x, lm.y, lm.z)
            mask[RHAND_OFF + j] = True

    face = getattr(res, "face_landmarks", None)
    if face is not None:
        lms = face.landmark
        for li, gi in enumerate(FACE_IDS):
            lm = lms[gi]
            coords[FACE_OFF + li] = (lm.x, lm.y, lm.z)
            mask[FACE_OFF + li] = True
    return coords, mask


# --------------------------------------------------------------------------
# Nhung du lieu CU (T,42,3) [left21|right21] vao schema 80 khop
# --------------------------------------------------------------------------
def embed_hands_only(seq42: np.ndarray):
    """(T,42,3) -> coords (T,80,3), mask (T,80). Pose/Face = vang."""
    seq42 = np.asarray(seq42, dtype=np.float32)
    if seq42.ndim == 2:
        seq42 = seq42[None, ...]
    T = seq42.shape[0]
    coords = np.zeros((T, NUM_UNIFIED_JOINTS, 3), dtype=np.float32)
    mask = np.zeros((T, NUM_UNIFIED_JOINTS), dtype=bool)
    left = seq42[:, :21, :]
    right = seq42[:, 21:42, :]
    coords[:, LHAND_OFF:LHAND_OFF + 21, :] = left
    coords[:, RHAND_OFF:RHAND_OFF + 21, :] = right
    mask[:, LHAND_OFF:LHAND_OFF + 21] = np.any(left != 0, axis=2)
    mask[:, RHAND_OFF:RHAND_OFF + 21] = np.any(right != 0, axis=2)
    return coords, mask


# --------------------------------------------------------------------------
# Chuan hoa theo tung bo phan
# --------------------------------------------------------------------------
def _canon_part(frame_coords, frame_mask, sl, origin, scale):
    idxs = np.arange(sl.start, sl.stop)
    present = idxs[frame_mask[idxs]]
    if present.size == 0:
        return
    frame_coords[present] = (frame_coords[present] - origin) / scale


def canonicalize_unified(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Chuan hoa (T,80,3) theo tung bo phan. Khop vang giu nguyen 0."""
    coords = np.asarray(coords, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    T = coords.shape[0]
    for t in range(T):
        fc = coords[t]
        fm = mask[t]
        # --- 2 ban tay: tru co tay (local 0) + chia kich thuoc tay ---
        for off in (LHAND_OFF, RHAND_OFF):
            sl = slice(off, off + NUM_HAND)
            if not np.any(fm[sl]):
                continue
            wrist = fc[off].copy()
            sub = fc[off:off + NUM_HAND][fm[sl]] - wrist
            scale = float(np.sqrt((sub ** 2).sum(axis=1)).max())
            scale = scale if scale > 1e-6 else 1.0
            _canon_part(fc, fm, sl, wrist, scale)
        # --- pose: trung diem 2 vai + be rong vai ---
        sl = PART_SLICES["pose"]
        if np.any(fm[sl]):
            ls, rs = POSE_OFF + POSE_LSHOULDER, POSE_OFF + POSE_RSHOULDER
            if fm[ls] and fm[rs]:
                origin = (fc[ls] + fc[rs]) / 2.0
                scale = float(np.linalg.norm(fc[ls] - fc[rs]))
            else:
                pres = fc[sl][fm[sl]]
                origin = pres.mean(axis=0)
                scale = float(np.sqrt(((pres - origin) ** 2).sum(axis=1)).max())
            scale = scale if scale > 1e-6 else 1.0
            _canon_part(fc, fm, sl, origin, scale)
        # --- mat: trong tam + kich thuoc mat ---
        sl = PART_SLICES["face"]
        if np.any(fm[sl]):
            pres = fc[sl][fm[sl]]
            origin = pres.mean(axis=0)
            scale = float(np.sqrt(((pres - origin) ** 2).sum(axis=1)).max())
            scale = scale if scale > 1e-6 else 1.0
            _canon_part(fc, fm, sl, origin, scale)
    return coords


# --------------------------------------------------------------------------
# Augmentation: hinh hoc + PART-DROPOUT (mau chot cua Cach 2)
# --------------------------------------------------------------------------
def _rotate_xy(coords, mask, deg):
    import math
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    out = coords.copy()
    xy = out[..., :2] @ rot.T
    out[..., :2] = xy
    out[~mask] = 0.0
    return out


def augment_unified_sequence(coords, mask, rng):
    """coords (T,80,3), mask (T,80). Tra ve coords, mask da augment."""
    coords = coords.astype(np.float32).copy()
    mask = mask.copy()
    # hinh hoc nhe
    if rng.rand() < 0.8:
        coords = _rotate_xy(coords, mask, rng.uniform(-12, 12))
    if rng.rand() < 0.8:
        coords[..., :2] *= rng.uniform(0.92, 1.08)
        coords[~mask] = 0.0
    if rng.rand() < 0.8:
        noise = rng.normal(0, 0.01, size=coords.shape).astype(np.float32)
        coords = coords + noise
        coords[~mask] = 0.0
    # --- PART-DROPOUT: ngau nhien bo mat / pose de model khong phu thuoc,
    #     va de mau du lieu day du "gia lap" thanh mau chi co tay (giong data cu) ---
    if rng.rand() < 0.30:
        sl = PART_SLICES["face"]
        coords[:, sl, :] = 0.0
        mask[:, sl] = False
    if rng.rand() < 0.20:
        sl = PART_SLICES["pose"]
        coords[:, sl, :] = 0.0
        mask[:, sl] = False
    return coords, mask


# --------------------------------------------------------------------------
# Dong goi -> (T, 80*4): [x,y,z,presence] cho tung khop
# --------------------------------------------------------------------------
def pack_unified(coords, mask):
    coords = np.asarray(coords, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    coords = coords * mask[..., None]  # khop vang -> 0
    packed = np.concatenate([coords, mask[..., None]], axis=2)  # (T,80,4)
    return packed.reshape(packed.shape[0], -1).astype(np.float32)


def prepare_unified(arr, mask=None):
    """Mot ham tien loi: tu mang raw bat ky -> (T, 80*4) da chuan hoa (KHONG augment).

    Ho tro:
      - (T,80,4): da co mask san
      - (T,42,3): du lieu cu chi co tay
      - (T,80,3): unified khong mask -> suy mask tu gia tri != 0
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Mang khong hop le cho unified: shape={arr.shape}")
    if arr.shape[2] == 4:
        coords = arr[:, :, :3].copy()
        mask = arr[:, :, 3] > 0.5
    elif arr.shape[1] == 42 and arr.shape[2] == 3:
        coords, mask = embed_hands_only(arr)
    elif arr.shape[1] == NUM_UNIFIED_JOINTS and arr.shape[2] == 3:
        coords = arr.copy()
        mask = np.any(arr != 0, axis=2)
    elif arr.shape[1] == 543 and arr.shape[2] == 3:
        # legacy holistic full -> lay 42 diem tay cuoi
        coords, mask = embed_hands_only(arr[:, -42:, :])
    else:
        raise ValueError(f"Khong nhan dang duoc layout: {arr.shape}")
    coords = canonicalize_unified(coords, mask)
    return pack_unified(coords, mask)


# --------------------------------------------------------------------------
# Ve landmark (pose + 2 tay + mat) len anh xem truoc - dung chung app & recorder
# --------------------------------------------------------------------------
# Mau ve theo he BGR (OpenCV)
COLOR_POSE = (0, 215, 255)    # vang cam  - than tren
COLOR_LHAND = (0, 255, 0)     # xanh la   - tay trai
COLOR_RHAND = (255, 0, 255)   # tim       - tay phai
COLOR_FACE = (255, 200, 0)    # xanh duong- guong mat
COLOR_LIPS = (0, 0, 255)      # do      - moi
COLOR_EYE = (255, 255, 0)     # cyan    - mat
COLOR_BROW = (0, 255, 255)    # vang    - long may
COLOR_IRIS = (210, 210, 210)  # xam     - trong mat
COLOR_NOSE = (180, 120, 255)  # hong    - song mui


def _px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))


def draw_holistic_landmarks(img, res, draw_pose=True, draw_hands=True, draw_face=True,
                            draw_face_oval=False, draw_iris=False):
    """Ve diem + canh cho pose / 2 tay / mat truc tiep tu ket qua MediaPipe Holistic.

    Chi ve len anh xem truoc (vis). KHONG goi ham nay tren frame dang ghi ra file.
    Tra ve chinh img (da ve tai cho).
    """
    if _cv2 is None or res is None or img is None:
        return img
    h, w = img.shape[:2]

    # --- POSE (than tren) ---
    pose = getattr(res, "pose_landmarks", None)
    if draw_pose and pose is not None:
        lms = pose.landmark
        pts = {}
        for local, gi in enumerate(POSE_IDS):
            if gi < len(lms):
                pts[local] = _px(lms[gi], w, h)
        for a, b in _POSE_EDGES_LOCAL:
            if a in pts and b in pts:
                _cv2.line(img, pts[a], pts[b], COLOR_POSE, 2)
        for pt in pts.values():
            _cv2.circle(img, pt, 4, COLOR_POSE, -1)

    # --- 2 BAN TAY ---
    if draw_hands:
        for attr, color in (("left_hand_landmarks", COLOR_LHAND),
                            ("right_hand_landmarks", COLOR_RHAND)):
            hand = getattr(res, attr, None)
            if hand is None:
                continue
            pts = [_px(lm, w, h) for lm in hand.landmark]
            for a, b in HAND_EDGES:
                if a < len(pts) and b < len(pts):
                    _cv2.line(img, pts[a], pts[b], color, 2)
            for i, pt in enumerate(pts):
                _cv2.circle(img, pt, 5 if i == 0 else 3, color, -1)

    # --- MAT: moi / mat / long may (ve day theo FaceMesh contours) ---
    face = getattr(res, "face_landmarks", None)
    if draw_face and face is not None:
        lms = face.landmark
        if _FACE_CONN is not None:
            groups = [
                ("left_brow", COLOR_BROW),
                ("right_brow", COLOR_BROW),
                ("left_eye", COLOR_EYE),
                ("right_eye", COLOR_EYE),
                ("lips", COLOR_LIPS),
                ("nose", COLOR_NOSE),
            ]
            if draw_iris:
                groups += [("left_iris", COLOR_IRIS), ("right_iris", COLOR_IRIS)]
            if draw_face_oval:
                groups.append(("oval", COLOR_FACE))
            for key, color in groups:
                conn = _FACE_CONN.get(key)
                if not conn:
                    continue
                idxs = set()
                for a, b in conn:
                    if a < len(lms) and b < len(lms):
                        _cv2.line(img, _px(lms[a], w, h), _px(lms[b], w, h), color, 1)
                        idxs.add(a)
                        idxs.add(b)
                for gi in idxs:
                    _cv2.circle(img, _px(lms[gi], w, h), 2, color, -1)
        else:
            # Fallback: 29 diem theo FACE_IDS neu khong co bang contour
            pts = {}
            for local, gi in enumerate(FACE_IDS):
                if gi < len(lms):
                    pts[local] = _px(lms[gi], w, h)
            for a, b in _FACE_EDGES_LOCAL:
                if a in pts and b in pts:
                    _cv2.line(img, pts[a], pts[b], COLOR_FACE, 1)
            for pt in pts.values():
                _cv2.circle(img, pt, 2, COLOR_FACE, -1)
    return img


if __name__ == "__main__":
    # Self-test chi dung numpy (khong can torch/mediapipe)
    print(f"NUM_UNIFIED_JOINTS = {NUM_UNIFIED_JOINTS}")
    print(f"offsets: pose={POSE_OFF} lhand={LHAND_OFF} rhand={RHAND_OFF} face={FACE_OFF}")
    print(f"so canh = {len(UNIFIED_EDGES)}")
    mx = max(max(a, b) for a, b in UNIFIED_EDGES)
    assert mx < NUM_UNIFIED_JOINTS, f"canh vuot so khop: {mx}"
    # gia lap du lieu cu (T=10, 42, 3)
    rng = np.random.RandomState(0)
    old = rng.rand(10, 42, 3).astype(np.float32)
    packed = prepare_unified(old)
    print(f"old (10,42,3) -> packed {packed.shape} (mong doi (10,{NUM_UNIFIED_JOINTS*4}))")
    assert packed.shape == (10, NUM_UNIFIED_JOINTS * 4)
    # gia lap du lieu moi day du (T=12, 80, 4)
    coords = rng.rand(12, NUM_UNIFIED_JOINTS, 3).astype(np.float32)
    m = (rng.rand(12, NUM_UNIFIED_JOINTS) > 0.1)
    full = np.concatenate([coords, m[..., None].astype(np.float32)], axis=2)
    packed2 = prepare_unified(full)
    print(f"new (12,80,4) -> packed {packed2.shape}")
    assert packed2.shape == (12, NUM_UNIFIED_JOINTS * 4)
    # augment + canonicalize
    c2, m2 = embed_hands_only(old)
    c2 = canonicalize_unified(c2, m2)
    c3, m3 = augment_unified_sequence(c2, m2, rng)
    print(f"augment ok: coords {c3.shape}, mask {m3.shape}")
    print("SELF-TEST OK")
