# src/ — Mã nguồn dự án VSL (bản chuẩn, có chú thích chi tiết)

Thứ tự pipeline:
1. extract_keypoints.py  — video .mp4 -> keypoint .npy + metadata.csv + labels.json
2. dataset.py            — nạp dữ liệu cho model (Dataset + collate_fn padding)
3. model.py              — LSTM (làm trước) / Transformer (nâng cấp)
4. train.py              — huấn luyện, chia train/val (3 cách), lưu best.pth
5. predict.py            — thử model trên 1 file .npy
6. unicode_utils.py      — ghép vần tiếng Việt + chuẩn hoá NFC (chữ cái + dấu)
7. tts.py                — văn bản -> giọng nói (gTTS / FPT.AI)
8. app.py                — demo webcam real-time (Streamlit)

Chạy nhanh:
  python src/extract_keypoints.py --raw data/raw_words
  python src/train.py --model lstm --epochs 50
  streamlit run src/app.py
