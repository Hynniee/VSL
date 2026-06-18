"""
================================================================================
 tts.py - CHUYEN VAN BAN TIENG VIET -> GIONG NOI (Text-To-Speech)
================================================================================

Mac dinh dung gTTS (Google, mien phi, can internet).
Co san 'stub' FPT.AI de nang cap chat luong giong doc sau nay.

CAI: pip install gTTS
================================================================================
"""


def synthesize_gtts(text, out_path="output.mp3", lang="vi"):
    """Tao file mp3 tu van ban bang Google TTS."""
    from gtts import gTTS
    gTTS(text=text, lang=lang).save(out_path)
    return out_path


def synthesize_fptai(text, api_key, voice="banmai", out_path="output.mp3"):
    """(Tuy chon) Goi FPT.AI TTS - giong Viet tu nhien hon. Can API key cua FPT.AI."""
    import time
    import requests

    res = requests.post(
        "https://api.fpt.ai/hmi/tts/v5",
        headers={"api-key": api_key, "voice": voice, "speed": ""},
        data=text.encode("utf-8"),
    )
    audio_url = res.json().get("async")
    if not audio_url:
        raise RuntimeError(f"FPT.AI loi: {res.text}")
    time.sleep(3)  # FPT tra link bat dong bo -> doi vai giay roi tai
    audio = requests.get(audio_url)
    with open(out_path, "wb") as f:
        f.write(audio.content)
    return out_path


def synthesize(text, out_path="output.mp3", engine="gtts", **kwargs):
    """Cong chung: chon engine 'gtts' (mac dinh) hoac 'fptai'."""
    if engine == "gtts":
        return synthesize_gtts(text, out_path, **kwargs)
    if engine == "fptai":
        return synthesize_fptai(text, out_path=out_path, **kwargs)
    raise ValueError(f"TTS engine chua ho tro: {engine}")


if __name__ == "__main__":
    path = synthesize("xin chào, đây là hệ thống nhận dạng ngôn ngữ ký hiệu")
    print("Da luu giong noi tai:", path)
