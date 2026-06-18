"""
================================================================================
 unicode_utils.py - GHEP VAN TIENG VIET + CHUAN HOA UNICODE (NFC)
================================================================================

Model chi nhan KY HIEU ROI: chu cai goc (chu_a, chu_o...), dau bien doi
(dau_mu ^, dau_trang a-breve, dau_rau o/u-horn) va dau thanh (sac, huyen,
hoi, nga, nang). File nay rap cac ky hieu roi do lai thanh CHU TIENG VIET
HOAN CHINH.

VI DU:
    chu_a + dau_mu  + thanh_sac    -> ấ
    chu_a + dau_trang              -> ă
    chu_o + dau_rau + thanh_huyen  -> ờ

GHI CHU KY THUAT:
    Dau duoc bieu dien bang "ky tu to hop" (combining marks) cua Unicode.
    Sau khi ghep, ta goi normalize("NFC") de bien 'a' + dau-trang thanh 1
    ky tu 'ă' chuan (go duoc, hien thi dung, copy ra Word/TTS khong loi).
================================================================================
"""
import unicodedata

# Dau thanh -> ky tu to hop (dat SAU nguyen am)
TONE = {
    "thanh_sac":   "\u0301",  # dau sac
    "thanh_huyen": "\u0300",  # dau huyen
    "thanh_hoi":   "\u0309",  # dau hoi
    "thanh_nga":   "\u0303",  # dau nga
    "thanh_nang":  "\u0323",  # dau nang
}

# Dau bien doi chu -> ky tu to hop
MODIFIER = {
    "dau_mu":    "\u0302",  # ^      (a o e -> â ô ê)
    "dau_trang": "\u0306",  # breve  (a -> ă)
    "dau_rau":   "\u031b",  # horn   (o u -> ơ ư)
}


def compose_char(base, modifier=None, tone=None):
    """Ghep 1 chu cai goc + (dau bien doi) + (dau thanh) -> ky tu NFC."""
    # Xu ly 'd-dai' (đ) rieng (khong phai ghep tu d + dau)
    if base in ("chu_dd", "đ", "dd"):
        s = "đ"
    elif base.startswith("chu_"):
        s = base[len("chu_"):]  # 'chu_a' -> 'a'
    else:
        s = base
    if modifier in MODIFIER:
        s = s + MODIFIER[modifier]  # gan dau bien doi
    if tone in TONE:
        s = s + TONE[tone]          # gan dau thanh
    return unicodedata.normalize("NFC", s)  # 'a' + dau-trang -> 'ă'


def tokens_to_text(tokens):
    """Chuyen 1 CHUOI token (chu_*/dau_*/thanh_*/space) -> van ban tieng Viet.

    Cach hoat dong (may trang thai):
      - Gap chu cai goc moi  -> "chot" chu dang dung truoc do, bat dau chu moi.
      - Gap dau bien doi/dau thanh -> gan vao chu DANG dung.
      - Gap 'space' -> chot chu + them khoang trang.
    """
    out, state = [], {"base": None, "modifier": None, "tone": None}

    def flush():
        # Hoan tat chu dang dung (neu co) roi reset trang thai
        if state["base"] is not None:
            out.append(compose_char(state["base"], state["modifier"], state["tone"]))
        state["base"] = state["modifier"] = state["tone"] = None

    for tok in tokens:
        if tok in ("space", "khoang_trang", " "):
            flush()
            out.append(" ")
        elif tok in MODIFIER:
            state["modifier"] = tok
        elif tok in TONE:
            state["tone"] = tok
        else:  # chu cai goc
            flush()
            state["base"] = tok
    flush()  # chot chu cuoi cung
    return "".join(out).strip()


# Chay thu nhanh: python src/unicode_utils.py
if __name__ == "__main__":
    tests = [
        (["chu_a", "dau_mu", "thanh_sac"], "ấ"),
        (["chu_a", "dau_trang"], "ă"),
        (["chu_o", "dau_rau", "thanh_huyen"], "ờ"),
        (["chu_x", "chu_i", "space", "chu_n"], "xi n"),
    ]
    for toks, expect in tests:
        got = tokens_to_text(toks)
        print(f"{toks} -> '{got}'  (mong doi '{expect}')  {'OK' if got==expect else 'SAI'}")
