# VSL Sign Language Recognition — Source Code

## Cấu trúc

```
src/
├── model.py          # ST-GCN + MLP/GRU/Transformer classifiers
├── train.py          # Training pipeline (RTX 4050 optimized)
├── dataset.py        # VSLDataset + collate_fn + Mixup
├── hand_utils.py     # MediaPipe hand processing + augmentation
├── pose_utils.py     # Holistic schema (pose+hand+face)
├── app.py            # Streamlit webcam demo (RTX 4050 optimized)
├── predict.py        # CLI batch inference
├── record_data.py    # Thu thập dữ liệu
├── extract_keypoints.py   # Trích xuất keypoints từ video/ảnh
├── extract_holistic.py    # Trích xuất Holistic keypoints
├── check_data.py     # Kiểm tra dataset
├── compare_models.py # So sánh các model
├── tts.py            # Text-to-Speech (gTTS / FPT.AI)
└── unicode_utils.py  # Ghép chữ tiếng Việt từ tokens
```

## Changelog (version này so với bản gốc)

### model.py
- **`build_optimizer()`**: A_learn có `weight_decay=0` riêng → graph học đúng, không bị AdamW shrink
- **`STGCNClassifier.forward`**: std joint tính weighted (nhất quán với HolisticSTGCN)
- **`BatchNorm` momentum=0.05**: ổn định hơn với batch nhỏ/lệch phân phối
- Tất cả BN trong block đều dùng momentum=0.05

### hand_utils.py
- **`extract_hand_features`**: KHÔNG gọi `canonicalize_hands` nội bộ nữa → loại bỏ double canonicalize
- **`preprocess_landmark_array`**: KHÔNG gọi `canonicalize_hands` nữa → caller chịu trách nhiệm
- **`augment_hand_sequence`**: thêm `temporal_cutout` (prob=25%) → robust với frame drop từ webcam
- `draw_guide_box`: text tiếng Việt đúng encoding

### dataset.py
- **`_prepare`**: `canonicalize_hands` chỉ gọi 1 lần duy nhất ở đây (loại bỏ ~15% CPU overhead)
- **`collate_fn_mixup`**: Mixup augmentation trên chuỗi (alpha=0.2) → giảm overfit khi ít data

### train.py
- **`build_optimizer()`**: dùng hàm từ model.py (A_learn wd=0)
- **`OneCycleLR`**: thay CosineAnnealingLR → warmup 10% + annealing cos
- **`torch.compile(mode="reduce-overhead")`**: ~30-50% nhanh hơn per batch trên RTX 4050
- **`warmup_model()`**: dummy forward pass trước train → BN running stats ổn định ngay
- **`allow_tf32=True`**: tận dụng Tensor Cores của RTX 4050
- **`--mixup`**: flag để bật Mixup (default alpha=0.2)
- **`--no-compile`**: flag tắt torch.compile nếu gặp lỗi
- AMP GradScaler FP16 cho RTX 4050

### app.py
- **Model warmup**: dummy forward pass sau `load_model` → không lag frame đầu
- **`torch.compile(mode="reduce-overhead")`**: inference nhanh hơn ~30%
- **Fix `torch.autocast`**: `device_type=device` thay vì hard-code `"cuda"`
- **JPEG encode + resize**: `cv2.imencode(".jpg")` với quality tunable → FPS tăng đáng kể
- **`ui_every`**: gián nhịp cập nhật panel → giảm Streamlit re-render overhead
- **`smooth_alpha` EMA**: làm mượt xác suất theo thời gian → ít flip kết quả hơn
- **`torch.load(weights_only=False)`**: tránh warning PyTorch 2.4+
- Adaptive FPS EMA (0.9*prev + 0.1*inst) thay vì tính từ đầu phiên

## Training (RTX 4050)

```bash
# ST-GCN cơ bản (khuyến nghị)
python train.py --model stgcn --epochs 50 --batch 64

# Với Mixup (ít data / dễ overfit)
python train.py --model stgcn --epochs 50 --batch 64 --mixup 0.2

# Tắt torch.compile nếu gặp lỗi
python train.py --model stgcn --no-compile

# Holistic (pose + tay + mặt)
python train.py --model unified --feature-mode unified --epochs 60 --batch 48
```

## Demo

```bash
streamlit run app.py
```

Cài đặt FPS tối ưu cho RTX 4050:
- Độ phức tạp model: **0**
- Thu nhỏ ảnh xử lý: **0.5**
- Dự đoán mỗi N khung: **2**
- Cập nhật panel: **4**
- Bề rộng hiển thị: **540**
- JPEG quality: **70**

## Inference CLI

```bash
python predict.py --npy path/to/sample.npy --ckpt checkpoints/best.pth --topk 5
```

## Yêu cầu

```
torch>=2.0          # để dùng torch.compile
mediapipe
opencv-python
streamlit>=1.28
scikit-learn
numpy
pandas
gtts
```
