# VSL Project - Nhận dạng ngôn ngữ ký hiệu tiếng Việt

Repo này chứa mã nguồn cho hệ thống nhận dạng ngôn ngữ ký hiệu tiếng Việt, bao gồm:
- trích keypoint từ video bằng MediaPipe
- huấn luyện và so sánh các model `mlp`, `gru`, `transformer`, `stgcn`
- demo webcam bằng Streamlit
- tạo dữ liệu từ webcam

## Yêu cầu

- Python 3.8 / 3.9 / 3.10
- Windows hoặc Linux
- GPU nếu muốn train nhanh hơn với `--device cuda`

## Cài đặt

1. Tạo virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

2. Cài phụ thuộc:
```powershell
pip install -r requirements.txt
```

## Cấu trúc chính

- `src/`: mã nguồn chính
- `data/`: dữ liệu đầu vào và thư mục `processed/`
- `checkpoints/`: lưu model đã huấn luyện
- `tests/`: test đơn vị

## Chạy nhanh

### Chạy demo webcam

```powershell
streamlit run src/app.py
```

Từ giao diện Streamlit, chọn checkpoint `checkpoints/best.pth` hoặc checkpoint phù hợp của bạn.

### Huấn luyện model

```powershell
python src/train.py --metadata data/metadata_alphabet.csv --labels data/labels_alphabet.json --model stgcn --epochs 60 --batch 48 --device cuda
```

Nếu không có GPU, dùng `--device cpu`.

### So sánh nhiều model

```powershell
python src/compare_models.py --metadata data/metadata_alphabet.csv --labels data/labels_alphabet.json --models mlp,gru,stgcn --epochs 40 --batch 48 --device cuda
```

## Tạo dữ liệu từ webcam

```powershell
python src/record_data.py
```

Script này cho phép quay ảnh/video cho nhãn mới và tự động đặt tên file theo quy ước.

## Trích keypoint từ video

Dùng `extract_keypoints.py` để chuyển video trong `data/raw_alphabet` hoặc `data/raw_words` thành file `.npy` và tạo metadata/labels.

Ví dụ với dataset chữ cái:

```powershell
python src/extract_keypoints.py --raw data/raw_alphabet --metadata data/metadata_alphabet.csv --labels data/labels_alphabet.json --image-frames 8
```

## Lưu ý quan trọng

- `data/processed/` chứa file `.npy` keypoint sinh tự động.
- `data/raw_alphabet/` và `data/raw_words/` chứa video đầu vào.
- `data/labels.json` và `data/metadata.csv` được sinh tự động từ `extract_keypoints.py`.
- `checkpoints/` chứa file mô hình `.pth` để dùng cho demo và dự đoán.

## Ghi chú

- Model `stgcn` ưu tiên `feature_mode=hands_raw` và thường cho độ chính xác tốt nhất trên dữ liệu chữ cái và từ vựng ngắn.
- Trên Windows, `src/train.py` đã set `--workers 0` làm mặc định để tránh lỗi multiprocessing.

## Tham khảo dữ liệu

Xem thêm `data/README_DATA.md` để biết cấu trúc dataset, cách đặt tên video và pipeline dữ liệu.
