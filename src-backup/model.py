"""
================================================================================
 model.py - DINH NGHIA "BO NAO" NHAN DANG KY HIEU (buoc 3)
================================================================================

Co 2 kien truc, cung nhan 1 chuoi keypoint -> doan ra 1 nhan:

  1) LSTMClassifier        : don gian, hoc nhanh. LAM TRUOC de kiem tra pipeline.
  2) TransformerClassifier : manh hon, can nhieu du lieu hon. Dung khi nang cap.

Ca hai deu dung 'lengths' de BO QUA phan padding (dem 0) khi tinh toan,
nho vay phan dem khong lam sai ket qua.
================================================================================
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """BiLSTM doc chuoi theo thoi gian roi phan loai."""

    def __init__(self, input_size, num_classes, hidden=128, num_layers=2,
                 bidirectional=True, dropout=0.3):
        super().__init__()
        # LSTM: mang chuyen xu ly CHUOI (nho thong tin theo thoi gian).
        # bidirectional=True: doc chuoi theo CA 2 chieu (xuoi + nguoc) -> hieu ngu canh tot hon.
        self.lstm = nn.LSTM(
            input_size, hidden, num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        # "Dau" phan loai: chuan hoa -> dropout (chong hoc vet) -> Linear ra so lop
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x, lengths):
        # 'pack' bao cho LSTM biet do dai that cua tung chuoi -> bo qua phan dem
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # MEAN POOLING: lay trung binh theo thoi gian, CHI tren frame hop le.
        # mask = True o cac frame that, False o phan dem.
        mask = torch.arange(out.size(1), device=x.device)[None, :] < lengths[:, None].to(x.device)
        mask = mask.unsqueeze(-1).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(mean)  # -> diem so cho tung lop (logits)


class TransformerClassifier(nn.Module):
    """Transformer Encoder: xu ly ca chuoi cung luc nho co che attention."""

    def __init__(self, input_size, num_classes, d_model=128, nhead=4,
                 num_layers=3, dropout=0.3, max_len=200):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)  # dua keypoint len khong gian d_model
        # "positional encoding": Transformer khong co khai niem thu tu, nen phai
        # cong them thong tin VI TRI cua frame.
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x, lengths):
        t = x.size(1)
        h = self.proj(x) + self.pos[:, :t]  # dac trung + vi tri
        # pad_mask = True o cac frame DEM -> attention bo qua chung
        pad_mask = torch.arange(t, device=x.device)[None, :] >= lengths[:, None].to(x.device)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()
        mean = (h * valid).sum(1) / valid.sum(1).clamp(min=1)  # pooling bo dem
        return self.head(mean)


def build_model(name, input_size, num_classes):
    """Chon model theo ten: 'lstm' hoac 'transformer'."""
    name = name.lower()
    if name == "lstm":
        return LSTMClassifier(input_size, num_classes)
    if name == "transformer":
        return TransformerClassifier(input_size, num_classes)
    raise ValueError(f"Model khong ho tro: {name}")
