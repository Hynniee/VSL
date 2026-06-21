"""
===============================================================================
  tts.py - CHUYEN VAN BAN TIENG VIET -> GIONG NOI (Text-To-Speech)
===============================================================================

Mac dinh dung gTTS (Google, mien phi, can internet).
Co san FPT.AI de nang cap chat luong giong doc sau nay (async, khong blocking).

CAI: pip install gTTS requests aiohttp
===============================================================================
"""
import asyncio

from aiohttp import ClientSession


def synthesize_gtts(text, out_path="output.mp3", lang="vi"):
    """Tao file mp3 tu van ban bang Google TTS."""
    from gtts import gTTS
    gTTS(text=text, lang=lang).save(out_path)
    return out_path


async def synthesize_fptai(text, api_key, voice="banmai", out_path="output.mp3",
                           poll_interval=1.0, timeout=30.0):
    """Goi FPT.AI TTS - giong Viet tu nhien hon. Khong blocking."""
    async with ClientSession() as session:
        async with session.post(
            "https://api.fpt.ai/hmi/tts/v5",
            headers={"api-key": api_key, "voice": voice, "speed": ""},
            data=text.encode("utf-8"),
        ) as res:
            data = await res.json()
        audio_url = data.get("async")
        if not audio_url:
            raise RuntimeError(f"FPT.AI loi: {data}")
        waited = 0.0
        while waited < timeout:
            await asyncio.sleep(poll_interval)
            waited += poll_interval
            async with session.get(audio_url) as audio_res:
                if audio_res.status == 200:
                    content = await audio_res.read()
                    with open(out_path, "wb") as f:
                        f.write(content)
                    return out_path
    raise TimeoutError("FPT.AI: qua thoi gian cho audio.")


def synthesize(text, out_path="output.mp3", engine="gtts", **kwargs):
    """Cong chung: chon engine 'gtts' (mac dinh) hoac 'fptai'."""
    if engine == "gtts":
        return synthesize_gtts(text, out_path, **kwargs)
    if engine == "fptai":
        if "api_key" not in kwargs:
            raise ValueError(
                "FPT.AI can 'api_key'. Vi du: "
                "synthesize(text, engine='fptai', api_key='YOUR_KEY')"
            )
        return asyncio.run(synthesize_fptai(text, out_path=out_path, **kwargs))
    raise ValueError(f"TTS engine chua ho tro: {engine}")


if __name__ == "__main__":
    path = synthesize("xin chào, đây là hệ thống nhận dạng ngôn ngữ ký hiệu")
    print("Da luu giong noi tai:", path)
