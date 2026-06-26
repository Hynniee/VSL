"""Test numpy-only cho schema gop (khong can torch/mediapipe).
Chay: python -m pytest tests/test_unified.py  (hoac) python tests/test_unified.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pose_utils as pu


def test_offsets():
    assert pu.NUM_UNIFIED_JOINTS == 80
    assert (pu.POSE_OFF, pu.LHAND_OFF, pu.RHAND_OFF, pu.FACE_OFF) == (0, 9, 30, 51)


def test_edges_valid():
    edges = pu.UNIFIED_EDGES
    assert max(max(a, b) for a, b in edges) < pu.NUM_UNIFIED_JOINTS
    assert all(a != b for a, b in edges)


def test_old_hands_embed():
    old = np.random.RandomState(0).rand(12, 42, 3).astype(np.float32)
    pk = pu.prepare_unified(old)
    assert pk.shape == (12, 320)
    pk3 = pk.reshape(12, 80, 4)
    assert pk3[:, pu.PART_SLICES["pose"], 3].sum() == 0
    assert pk3[:, pu.PART_SLICES["face"], 3].sum() == 0
    assert pk3[:, pu.LHAND_OFF:pu.RHAND_OFF + 21, 3].mean() == 1.0


def test_new_full_pack():
    rng = np.random.RandomState(2)
    coords = rng.rand(18, 80, 3).astype(np.float32)
    m = (rng.rand(18, 80) > 0.1).astype(np.float32)
    full = np.concatenate([coords, m[..., None]], axis=2).astype(np.float32)
    pk = pu.prepare_unified(full)
    assert pk.shape == (18, 320)


def test_part_dropout():
    rng = np.random.RandomState(3)
    old = rng.rand(10, 42, 3).astype(np.float32)
    c, m = pu.embed_hands_only(old)
    c = pu.canonicalize_unified(c, m)
    c2, m2 = pu.augment_unified_sequence(c, m, rng)
    assert c2.shape == (10, 80, 3) and m2.shape == (10, 80)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("ALL TESTS PASS")
