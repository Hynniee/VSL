"""model.py - model moi uu tien nhanh, on dinh; co them ST-GCN toi uu cho hand joints."""
from __future__ import annotations

import torch
import torch.nn as nn

from hand_utils import CROSS_HAND_EDGES, HAND_EDGES


class MaskedPooling(nn.Module):
    def forward(self, x, lengths):
        t = x.size(1)
        mask = (torch.arange(t, device=x.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        mask_f = mask.float()
        mean = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        x_max = x.masked_fill(~mask, -1e4).amax(1)
        var = ((x - mean[:, None, :]) ** 2 * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        std = torch.sqrt(var.clamp(min=1e-6))
        return mean, std, x_max


class TemporalMLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden=256, dropout=0.25):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.pool = MaskedPooling()
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, x, lengths):
        h = self.frame_encoder(x)
        mean, std, hmax = self.pool(h, lengths)
        t = h.size(1)
        mask = torch.arange(t, device=h.device)[None, :] < lengths[:, None]
        attn = self.attn(h).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(attn, dim=1).unsqueeze(-1)
        pooled = (h * weight).sum(1)
        return self.head(torch.cat([pooled, mean, std, hmax], dim=1))


class GRUAttentionClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden=192, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden * 2
        self.pool = MaskedPooling()
        self.attn = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Tanh(), nn.Linear(out_dim, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mean, std, hmax = self.pool(out, lengths)
        t = out.size(1)
        mask = torch.arange(t, device=out.device)[None, :] < lengths[:, None]
        attn = self.attn(out).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(attn, dim=1).unsqueeze(-1)
        pooled = (out * w).sum(1)
        return self.head(torch.cat([pooled, mean, std, hmax], dim=1))


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, d_model=192, nhead=4, num_layers=3, dropout=0.2, max_len=64):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pool = MaskedPooling()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )
        self.max_len = max_len

    def forward(self, x, lengths):
        t = min(x.size(1), self.max_len)
        x = x[:, :t]
        lengths = lengths.clamp(max=t)
        h = self.proj(x) + self.pos[:, :t]
        pad = torch.arange(t, device=x.device)[None, :] >= lengths[:, None]
        h = self.encoder(h, src_key_padding_mask=pad)
        mean, std, hmax = self.pool(h, lengths)
        return self.head(torch.cat([mean, std, hmax], dim=1))


def build_adjacency(num_joints: int) -> torch.Tensor:
    A = torch.eye(num_joints, dtype=torch.float32)
    if num_joints == 42:
        for off in (0, 21):
            for a, b in HAND_EDGES:
                A[off + a, off + b] = 1.0
                A[off + b, off + a] = 1.0
        for a, b in CROSS_HAND_EDGES:
            A[a, b] = 1.0
            A[b, a] = 1.0
    deg = A.sum(1)
    d_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    A = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return A


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.register_buffer("A_base", A)
        self.A_learn = nn.Parameter(torch.zeros_like(A))
        self.theta = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        A = self.A_base + self.A_learn
        y = self.theta(x)
        y = torch.einsum("bctv,vw->bctw", y, A)
        return self.bn(y)


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, channels, stride=1, dropout=0.2):
        super().__init__()
        branch = channels // 4
        self.b1 = nn.Sequential(
            nn.Conv2d(channels, branch, 1),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch, branch, (3, 1), (stride, 1), (1, 0)),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(channels, branch, 1),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch, branch, (5, 1), (stride, 1), (2, 0)),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(channels, branch, 1),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch, branch, (7, 1), (stride, 1), (3, 0)),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.Conv2d(channels, branch, 1),
        )
        self.bn = nn.BatchNorm2d(branch * 4)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        y = self.bn(y)
        return self.drop(y)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, dropout=0.2, residual=True):
        super().__init__()
        self.gcn = SpatialGraphConv(in_ch, out_ch, A)
        self.tcn = MultiScaleTemporalConv(out_ch, stride=stride, dropout=dropout)
        self.se = SEBlock(out_ch)
        if not residual:
            self.residual = None
        elif in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, (stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        res = 0 if self.residual is None else self.residual(x)
        x = self.gcn(x)
        x = self.tcn(torch.relu(x))
        x = self.se(x)
        return self.relu(x + res)


class STGCNClassifier(nn.Module):
    """ST-GCN tối ưu cho 42 joints bàn tay, hợp chữ cái và từ vựng ngắn.

    Điểm tối ưu:
    - input raw joints chuẩn hóa (hands_raw)
    - motion stream nội bộ bằng velocity theo thời gian
    - graph có cả intra-hand + cross-hand edges
    - temporal conv đa tỉ lệ để bắt dáng tĩnh lẫn chuyển động ngắn
    """
    def __init__(self, input_size, num_classes, hidden=64, dropout=0.2):
        super().__init__()
        if input_size % 3 != 0:
            raise ValueError(f"ST-GCN can input la raw joints 3D, nhan duoc input_size={input_size}")
        self.num_joints = input_size // 3
        self.num_coords = 3
        A = build_adjacency(self.num_joints)
        in_ch = self.num_coords * 2  # coords + velocity
        self.data_bn = nn.BatchNorm1d(self.num_joints * in_ch)
        self.blocks = nn.ModuleList([
            STGCNBlock(in_ch, hidden, A, stride=1, dropout=dropout, residual=False),
            STGCNBlock(hidden, hidden, A, stride=1, dropout=dropout),
            STGCNBlock(hidden, hidden * 2, A, stride=2, dropout=dropout),
            STGCNBlock(hidden * 2, hidden * 2, A, stride=1, dropout=dropout),
            STGCNBlock(hidden * 2, hidden * 4, A, stride=2, dropout=dropout),
        ])
        out_ch = hidden * 4
        self.head = nn.Sequential(
            nn.LayerNorm(out_ch * 3),
            nn.Linear(out_ch * 3, out_ch * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_ch * 2, num_classes),
        )

    def forward(self, x, lengths):
        b, t, f = x.shape
        v = self.num_joints
        coords = x.view(b, t, v, 3).permute(0, 3, 1, 2).contiguous()  # (B,3,T,V)
        vel = torch.diff(coords, dim=2, prepend=coords[:, :, :1])
        h = torch.cat([coords, vel], dim=1)  # (B,6,T,V)
        h = h.permute(0, 2, 3, 1).contiguous().view(b, t, v * 6)
        h = self.data_bn(h.transpose(1, 2)).transpose(1, 2)
        h = h.view(b, t, v, 6).permute(0, 3, 1, 2).contiguous()

        cur_len = lengths.clone()
        for blk in self.blocks:
            h = blk(h)
            if blk.stride > 1:
                cur_len = (cur_len + blk.stride - 1) // blk.stride

        # masked pooling tren time sau khi da qua cac stride
        t2 = h.size(2)
        mask = (torch.arange(t2, device=h.device)[None, :] < cur_len[:, None]).float()
        mask4 = mask[:, None, :, None]
        h_mean_joint = (h * mask4).sum(dim=2) / mask4.sum(dim=2).clamp(min=1.0)  # (B,C,V)
        h_max_joint = h.masked_fill(mask4 == 0, -1e4).amax(dim=2)                 # (B,C,V)
        mean = h_mean_joint.mean(dim=2)
        hmax = h_max_joint.max(dim=2).values
        std = h_mean_joint.std(dim=2)
        return self.head(torch.cat([mean, std, hmax], dim=1))


_MODELS = {
    "mlp": TemporalMLPClassifier,
    "gru": GRUAttentionClassifier,
    "transformer": TransformerClassifier,
    "stgcn": STGCNClassifier,
}


def available_models():
    return list(_MODELS.keys())


def build_model(name, input_size, num_classes):
    key = name.lower().replace("-", "").replace("_", "")
    alias = {"temporalmlp": "mlp", "fast": "mlp", "bigru": "gru", "stcgn": "stgcn", "stgcn": "stgcn"}
    key = alias.get(key, key)
    if key not in _MODELS:
        raise ValueError(f"Model chua ho tro: {name}. Chon 1 trong {available_models()}")
    return _MODELS[key](input_size, num_classes)
