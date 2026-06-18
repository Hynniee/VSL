# 📂 data/ — Bộ dữ liệu dự án VSL (bản chuẩn, đầy đủ)

Folder này chứa **toàn bộ dữ liệu** cho hệ thống nhận dạng ngôn ngữ ký hiệu tiếng Việt.
Có **2 nhánh model riêng** (từ vựng & chữ cái), dùng chung pipeline trích keypoint.

---

## 🗂️ Cấu trúc thư mục

```
data/
├── raw_words/          # (BẠN TỰ ĐỔ VÀO) Video TỪ VỰNG — mỗi từ 1 folder
│   ├── xinchao/        #   vd: xinchao_ng1_01.mp4, xinchao_ng2_01.mp4 ...
│   ├── camon/
│   └── ... (15 từ)
│
├── raw_alphabet/       # (BẠN TỰ ĐỔ VÀO) Video CHỮ CÁI + DẤU
│   ├── chu_a/ ... chu_y, chu_dd   # 23 chữ cái
│   ├── dau_mu/ dau_trang/ dau_rau/ # 3 dấu biến đổi (^  ˘  ơ/ư)
│   └── thanh_sac/ thanh_huyen/ thanh_hoi/ thanh_nga/ thanh_nang/  # 5 dấu thanh
│
├── raw_sentences/      # (GĐ3 — để sau) Video CÂU LIÊN TỤC
│   ├── videos/         #   vd: cau001_ng1_01.mp4 ...
│   └── sentences.csv   #   nhãn câu (gloss + text) — xem ví dụ sẵn
│
├── processed/          # (TỰ SINH) file .npy keypoint, do extract_keypoints.py tạo
│
├── label_map.json      # (INPUT) ánh xạ tên folder → tên hiển thị (15 từ)
├── labels.example.json            # VÍ DỤ output (tự sinh khi chạy extract)
├── labels_alphabet.example.json   # VÍ DỤ output cho model chữ cái
└── metadata.example.csv           # VÍ DỤ output (tự sinh)
```

> Các folder rỗng có file `.gitkeep` để git giữ lại cấu trúc. Khi bạn bỏ video vào thì cứ để hoặc xoá `.gitkeep` đều được.

---

## 📛 Quy ước đặt tên file video

```
<mã_nhãn>_<người_quay>_<lần>.mp4
```

| Thành phần | Ý nghĩa | Ví dụ |
|---|---|---|
| `mã_nhãn` | trùng TÊN FOLDER | `xinchao`, `chu_a`, `dau_mu` |
| `người_quay` | ai quay | `ng1` `ng2` `ng3`, hoặc `net` (lấy từ internet) |
| `lần` | số thứ tự clip | `01` `02` `03` ... |

Ví dụ: `xinchao_ng1_03.mp4` = từ "xin chào", người 1 quay, clip thứ 3.

> ⚠️ Tên file quyết định `signer` (người quay) → dùng để chia train/test theo người. Đặt sai → chia sai.

---

## ⚙️ 2 pipeline — chạy thế nào

### 1️⃣ Model TỪ VỰNG (raw_words)
```bash
# Trích keypoint (DÙNG label_map để ra tên có dấu)
python src/extract_keypoints.py \
    --raw data/raw_words \
    --label-map data/label_map.json \
    --metadata data/metadata.csv \
    --labels data/labels.json

# Train
python src/train.py --model lstm --metadata data/metadata.csv --labels data/labels.json --out checkpoints/best.pth
```

### 2️⃣ Model CHỮ CÁI + DẤU (raw_alphabet)
```bash
# KHÔNG dùng label-map → nhãn = TÊN FOLDER (chu_a, dau_mu, thanh_sac...)
# Vì unicode_utils.py cần đúng các mã này để ghép vần.
python src/extract_keypoints.py \
    --raw data/raw_alphabet \
    --metadata data/metadata_alphabet.csv \
    --labels data/labels_alphabet.json

# Train model chữ cái RIÊNG
python src/train.py --model lstm \
    --metadata data/metadata_alphabet.csv \
    --labels data/labels_alphabet.json \
    --out checkpoints/alphabet.pth
```

> 🔑 **Khác biệt mấu chốt:** từ vựng DÙNG `label_map` (ra "xin chào"); chữ cái KHÔNG dùng (giữ mã `chu_a`/`dau_mu`/`thanh_sac` để ghép vần bằng `unicode_utils.py`).

---

## 🔤 Bảng nhãn chữ cái (31 ký hiệu)

- **23 chữ cái:** chu_a, chu_b, chu_c, chu_d, **chu_dd** (đ), chu_e, chu_g, chu_h, chu_i, chu_k, chu_l, chu_m, chu_n, chu_o, chu_p, chu_q, chu_r, chu_s, chu_t, chu_u, chu_v, chu_x, chu_y
- **3 dấu biến đổi:** dau_mu (â ô ê), dau_trang (ă), dau_rau (ơ ư)
- **5 dấu thanh:** thanh_sac, thanh_huyen, thanh_hoi, thanh_nga, thanh_nang

→ `unicode_utils.tokens_to_text(["chu_a","dau_mu","thanh_sac"])` = **ấ**

---

## 🚫 Lưu ý Git (quan trọng)

Không commit dữ liệu nặng. Trong `.gitignore` của repo nên có:
```
data/raw_words/**/*.mp4
data/raw_alphabet/**/*.mp4
data/raw_sentences/videos/*.mp4
data/processed/
data/metadata*.csv
data/labels*.json
```
Chỉ commit: cấu trúc folder (.gitkeep), `label_map.json`, `raw_sentences/sentences.csv`.
Video chia sẻ qua Google Drive / OneDrive (hoặc DVC / Git LFS).

---

## ✅ Checklist khi thêm dữ liệu mới
1. Quay video → đặt tên đúng `<mã>_<signer>_<lần>.mp4`.
2. Bỏ vào đúng folder nhãn (`raw_words/<từ>/` hoặc `raw_alphabet/<mã>/`).
3. Chạy lại `extract_keypoints.py` → cập nhật `processed/` + `metadata.csv` + `labels.json`.
4. Train lại. Xong!
