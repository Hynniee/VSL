# src/ - Ban toi uu co them ST-GCN

## Dung truong hop nao?
- **mlp**: nhanh, demo webcam on dinh
- **gru**: manh va de train
- **transformer**: thu nghiem them
- **stgcn**: uu tien **do chinh xac cao** cho **chu cai** va **tu vung ngan**, hoc truc tiep tren **raw hand joints**

## Diem toi uu ST-GCN ban nay
- graph 42 joints cho 2 ban tay
- co **cross-hand edges**
- tu tinh them **velocity stream**
- temporal conv **da ti le** (3/5/7)
- squeeze-excitation de nhan manh kenh quan trong
- train tu dong doi sang `feature_mode=hands_raw`

## Lenh nen dung cho ST-GCN
### 1) Trich lai keypoint
```bash
python src/extract_keypoints.py --raw data/raw_alphabet --metadata data/metadata_alphabet.csv --labels data/labels_alphabet.json --image-frames 8
```

### 2) Train ST-GCN (uu tien accuracy)
```bash
python -u src/train.py --metadata data/metadata_alphabet.csv --labels data/labels_alphabet.json --model stgcn --epochs 60 --batch 48 --device cuda
```

### 3) So sanh voi model khac
```bash
python -u src/compare_models.py --metadata data/metadata_alphabet.csv --labels data/labels_alphabet.json --models mlp,gru,stgcn --epochs 40 --batch 48 --device cuda
```

## Goi y
- Neu muon **demo nhanh**: dung `mlp`
- Neu muon **ket qua tot nhat cho bao cao**: test ky `stgcn`
- ST-GCN hop nhat khi ban bo sung them **video tu quay** cho tu vung ngan / ky hieu co chuyen dong
