# v4_MNIST_10_20 — CELL 1: Xây dựng Multi-Crop/Patch Views (109 Writers) + Huấn luyện TỪ ĐẦU Hierarchical Multi-MLP (5 Folds) + Kiểm định Raw & Capped_Style.
# Colab T4 GPU: Dán TOÀN BỘ vào Cell 1 của notebook mới v4_MNIST_10_20 và chạy (~7-8 phút).
# Tự động lưu checkpoint sau mỗi fold, tái sử dụng cache strokes/recomb đã sinh; không sửa bất kỳ file v5 gốc nào.
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
from functools import lru_cache
import csv, hashlib, json, math, random, sys, time, traceback, unicodedata, uuid, zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

V4_ROOT = '/content/drive/MyDrive/MNIST_data'
V4_BASE_ID = 'e55217c88b0c'
V4_FT_ID = '12861935de9d'
V4_SEED29_ID = 'dd71a4292e16'
V4_CLEAN_ID = 'c813298ffe24'
V4_FINAL_V5_ID = '4b1e77287434'
V4_V5_EXCLUSIONS = [458, 1767, 2213, 4355]
V4_V31_EXCLUSIONS = [9, 15, 16, 25, 29, 78, 95, 222, 224, 231, 233, 234, 235, 238, 244, 251, 253, 256]


def v4_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def v4_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def v4_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def v4_save_npz(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.pending.npz')
    np.savez_compressed(tmp, **obj)
    os.replace(tmp, path)


def v4_writer_key(v):
    if pd.isna(v):
        return ''
    w = ' '.join(unicodedata.normalize('NFC', str(v)).casefold().split())
    return 'minh nhật phạm' if w in ('phạm minh nhật', 'minh nhật phạm') else w


@lru_cache(maxsize=128)
def v4_area_weights(src, dst):
    lo = np.arange(dst, dtype=np.float64)[:, None] * src / dst
    hi = (np.arange(dst, dtype=np.float64)[:, None] + 1) * src / dst
    q = np.arange(src, dtype=np.float64)[None, :]
    w = np.maximum(0., np.minimum(q + 1., hi) - np.maximum(q, lo))
    return w / w.sum(1, keepdims=True)


def v4_norm_js(a):
    ys, xs = np.where(a > 20)
    assert len(xs), 'Ảnh rỗng'
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    scale = min(20 / cw, 20 / ch)
    tw, th = max(1, int(np.floor(cw * scale + .5))), max(1, int(np.floor(ch * scale + .5)))
    small = (v4_area_weights(ch, th) @ crop.astype(np.float64) @ v4_area_weights(cw, tw).T).astype(np.float32)
    small = (small.astype(np.float64) * 255 / float(small.max())).astype(np.float32)
    yy, xx = np.mgrid[0:th, 0:tw]
    mass = small.astype(np.float64).sum()
    ox = int(np.floor(14 - ((xx + .5) * small).sum() / mass + .5))
    oy = int(np.floor(14 - ((yy + .5) * small).sum() / mass + .5))
    ox, oy = max(0, min(28 - tw, ox)), max(0, min(28 - th, oy))
    out = np.zeros((28, 28), dtype=np.uint8)
    out[oy:oy + th, ox:ox + tw] = np.clip(np.floor(small.astype(np.float64) + .5), 0, 255).astype(np.uint8)
    return out


def v4_snap_gray(a, levels):
    ix = np.clip(np.searchsorted(levels, a), 1, len(levels) - 1)
    lo, hi = levels[ix - 1], levels[ix]
    return np.where(a - lo < hi - a, lo, hi).astype(np.uint8)


def v4_render_unclipped_raw(strokes_json):
    d = json.loads(strokes_json)
    source = [(np.asarray(st['p'], dtype=np.float64).reshape(-1, 2), float(st['w'])) for st in d['s']]
    low = np.min([p.min(0) - w / 2 for p, w in source], axis=0)
    high = np.max([p.max(0) + w / 2 for p, w in source], axis=0)
    if low[0] >= 0 and low[1] >= 0 and high[0] <= 600 and high[1] <= 400:
        return None
    origin = np.floor(low).astype(int) - 3
    size = np.ceil(high).astype(int) - origin + 4
    width, height = map(int, size)
    im = Image.new('L', (width * 2, height * 2), 0)
    draw = ImageDraw.Draw(im)
    for p, w in source:
        p = (p - origin) * 2
        w *= 2
        if len(p) == 1:
            p = np.vstack([p, p])
        draw.line([tuple(q) for q in p], fill=255, width=max(1, int(w)), joint='curve')
        for x, y in p[::max(1, len(p) // 40)].tolist() + [p[0].tolist(), p[-1].tolist()]:
            draw.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=255)
    a = np.asarray(im.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    return v4_norm_js(a)


def v4_render_stroke_cap(strokes_json, seed, settings, levels, cap=1.25):
    d = json.loads(strokes_json)
    source = [(np.asarray(st['p'], dtype=np.float64).reshape(-1, 2), float(st['w'])) for st in d['s']]
    mean_w = np.mean([w for _, w in source])
    r = np.random.default_rng(seed)
    shear = r.uniform(*settings['shear'])
    sx = r.uniform(*settings['squeeze_x'])
    pen_scale = min(r.uniform(*settings['pen']) / mean_w, cap)
    points, widths = [], []
    for p0, w0 in source:
        p = p0.copy()
        p[:, 0] = (p[:, 0] + shear * (p[:, 1] - 200)) * sx
        points.append(p); widths.append(w0 * pen_scale)
    low = np.min([p.min(0) - w / 2 for p, w in zip(points, widths)], axis=0)
    high = np.max([p.max(0) + w / 2 for p, w in zip(points, widths)], axis=0)
    origin = np.floor(low).astype(int) - 3
    size = np.ceil(high).astype(int) - origin + 4
    width, height = map(int, size)
    im = Image.new('L', (width * 2, height * 2), 0)
    draw = ImageDraw.Draw(im)
    for p, w in zip(points, widths):
        p = (p - origin) * 2
        w *= 2
        if len(p) == 1:
            p = np.vstack([p, p])
        draw.line([tuple(q) for q in p], fill=255, width=max(1, int(w)), joint='curve')
        for x, y in p[::max(1, len(p) // 40)].tolist() + [p[0].tolist(), p[-1].tolist()]:
            draw.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=255)
    a = np.asarray(im.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    return v4_snap_gray(v4_norm_js(a), levels)


@torch.inference_mode()
def v4_precompute_multi_mlp_views_gpu(images_u8, levels_np, dev, chunk=4096):
    """
    Trích xuất 4 đầu vào chuyên biệt cho Hệ thống Đa-MLP Phân cấp (HierMultiMLP):
    1. out_xd (N, 784): Ảnh 28x28 sau khi khử nghiêng nội chữ bằng Bicubic + Khôi phục Bão hòa Lõi + V3_LEVELS.
    2. out_xl (N, 324): Khung cắt 18x18 (không kéo giãn, giữ nguyên tỷ lệ 1:1) căn chính giữa Tâm khối Chữ số Hàng chục (Trái).
    3. out_xr (N, 324): Khung cắt 18x18 (không kéo giãn, giữ nguyên tỷ lệ 1:1) căn chính giữa Tâm khối Chữ số Hàng đơn vị (Phải).
    4. out_g6 (N, 6):   6 chỉ số hình học tương đối [(cyl-cyr)/10, (cxr-cxl-9)/6, true_skew, valley_ratio, mass_ratio, width_ratio].
    """
    n_total = len(images_u8)
    out_xd = np.empty((n_total, 784), dtype=np.uint8)
    out_xl = np.empty((n_total, 324), dtype=np.uint8)
    out_xr = np.empty((n_total, 324), dtype=np.uint8)
    out_g6 = np.empty((n_total, 6), dtype=np.float32)

    yy, xx = torch.meshgrid(
        torch.arange(28, dtype=torch.float32, device=dev),
        torch.arange(28, dtype=torch.float32, device=dev),
        indexing='ij'
    )
    cols = torch.arange(9, 18, dtype=torch.float32, device=dev).unsqueeze(0)
    c_grid = xx.unsqueeze(0)
    grid_18_y, grid_18_x = torch.meshgrid(
        torch.arange(-8.5, 9.5, 1.0, device=dev),
        torch.arange(-8.5, 9.5, 1.0, device=dev),
        indexing='ij'
    )

    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        x = torch.from_numpy(np.ascontiguousarray(images_u8[start:end].reshape(-1, 28, 28))).to(dev).float() / 255.0
        b = len(x)

        col_sum = x.sum(dim=1)
        col_max = col_sum.amax(dim=1, keepdim=True).clamp(min=1e-5)
        penalty = 0.08 * col_max * ((cols - 12.5) / 4.5) ** 2
        best_rel = torch.argmin(col_sum[:, 9:18] + penalty, dim=1)
        cut_col = (9 + best_rel).unsqueeze(1).unsqueeze(2).float()

        w_left = torch.sigmoid((cut_col + 0.5 - c_grid) * 2.0)
        w_right = 1.0 - w_left
        xl_p = x * w_left
        xr_p = x * w_right

        ml = xl_p.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        mr = xr_p.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        cxl = (xl_p * xx).sum(dim=(1, 2), keepdim=True) / ml
        cyl = (xl_p * yy).sum(dim=(1, 2), keepdim=True) / ml
        cxr = (xr_p * xx).sum(dim=(1, 2), keepdim=True) / mr
        cyr = (xr_p * yy).sum(dim=(1, 2), keepdim=True) / mr

        mu11_w = (xl_p * (xx - cxl) * (yy - cyl) + xr_p * (xx - cxr) * (yy - cyr)).sum(dim=(1, 2))
        mu02_w = (xl_p * (yy - cyl) ** 2 + xr_p * (yy - cyr) ** 2).sum(dim=(1, 2)).clamp(min=1.0)
        true_skew = (mu11_w / mu02_w).clamp(-0.30, 0.30)

        theta_d = torch.zeros((b, 2, 3), dtype=torch.float32, device=dev)
        theta_d[:, 0, 0] = 1.0
        theta_d[:, 0, 1] = 0.85 * true_skew
        theta_d[:, 1, 1] = 1.0
        grid_d = F.affine_grid(theta_d, (b, 1, 28, 28), align_corners=False)
        xd = F.grid_sample(x.unsqueeze(1), grid_d, mode='bicubic', padding_mode='zeros', align_corners=False).squeeze(1)
        xd = (xd.clamp(0.0, 1.0) / 0.91).clamp(0.0, 1.0)

        col_sum_d = xd.sum(dim=1)
        col_max_d = col_sum_d.amax(dim=1, keepdim=True).clamp(min=1e-5)
        best_rel_d = torch.argmin(col_sum_d[:, 9:18] + 0.08 * col_max_d * ((cols - 12.5) / 4.5) ** 2, dim=1)
        cut_d = (9 + best_rel_d).unsqueeze(1).unsqueeze(2).float()
        valley_d = (col_sum_d.gather(1, (9 + best_rel_d).unsqueeze(1)) / col_max_d).squeeze(1)

        soft_l = torch.sigmoid((cut_d + 1.5 - c_grid) * 1.5)
        soft_r = torch.sigmoid((c_grid - (cut_d - 1.5)) * 1.5)
        xd_l = xd * soft_l
        xd_r = xd * soft_r

        ml_d = xd_l.sum(dim=(1, 2)).clamp(min=1e-5)
        mr_d = xd_r.sum(dim=(1, 2)).clamp(min=1e-5)
        cxl_d = (xd_l * xx).sum(dim=(1, 2)) / ml_d
        cyl_d = (xd_l * yy).sum(dim=(1, 2)) / ml_d
        cxr_d = (xd_r * xx).sum(dim=(1, 2)) / mr_d
        cyr_d = (xd_r * yy).sum(dim=(1, 2)) / mr_d

        def sample_18(img28, cx_v, cy_v):
            gx = (cx_v.view(b, 1, 1) + grid_18_x.unsqueeze(0) - 13.5) / 13.5
            gy = (cy_v.view(b, 1, 1) + grid_18_y.unsqueeze(0) - 13.5) / 13.5
            g = torch.stack([gx, gy], dim=-1)
            cr = F.grid_sample(img28.unsqueeze(1), g, mode='bicubic', padding_mode='zeros', align_corners=False).squeeze(1)
            cr = (cr.clamp(0.0, 1.0) / 0.92).clamp(0.0, 1.0)
            return cr.reshape(b, 324)

        cl_flat = sample_18(xd_l, cxl_d, cyl_d)
        cr_flat = sample_18(xd_r, cxr_d, cyr_d)

        out_xd[start:end] = v4_snap_gray(np.round(xd.reshape(b, 784).cpu().numpy() * 255.0), levels_np)
        out_xl[start:end] = v4_snap_gray(np.round(cl_flat.cpu().numpy() * 255.0), levels_np)
        out_xr[start:end] = v4_snap_gray(np.round(cr_flat.cpu().numpy() * 255.0), levels_np)

        fg_cols = (xd > 0.12).any(dim=1).float().sum(dim=1) / 20.0
        g1 = ((cyl_d - cyr_d) / 10.0).clamp(-1.0, 1.0)
        g2 = ((cxr_d - cxl_d - 9.0) / 6.0).clamp(-1.0, 1.0)
        g3 = true_skew
        g4 = valley_d.clamp(0.0, 1.0)
        g5 = (ml_d / (ml_d + mr_d)).clamp(0.0, 1.0)
        g6 = fg_cols.clamp(0.0, 1.4)
        out_g6[start:end] = torch.stack([g1, g2, g3, g4, g5, g6], dim=1).cpu().numpy()

    return out_xd, out_xl, out_xr, out_g6


class V4HierMultiMLP(nn.Module):
    """
    Hệ thống 6 MLP Phân cấp (100% nn.Linear, LayerNorm, BatchNorm1d, GELU, Dropout — KHÔNG dùng Conv2d):
    1. patch_embed:  MLP cục bộ (100 -> 192 -> 192) dùng chung trên 16 ô cửa sổ 10x10 (stride 6).
    2. mixer_blocks: 3 khối MLP-Mixer trộn không gian (16 -> 32 -> 16) và trộn kênh (192 -> 384 -> 192).
    3. left_mlp:     MLP chuyên gia Chữ số Hàng chục trên khung cắt 18x18 (324 -> 256 -> 128 -> 2: '1' vs '2').
    4. right_mlp:    MLP chuyên gia Chữ số Hàng đơn vị trên khung cắt 18x18 (324 -> 384 -> 256 -> 10: '0'..'9').
    5. global_mlp:   MLP toàn cảnh trên ảnh 28x28 đã khử nghiêng (784 -> 512 -> 256 -> 11).
    6. coordinator:  MLP điều phối hợp nhất (384 + 128 + 256 + 256 + 6 = 1030 -> 512 -> 11).
    """
    def __init__(self, dropout=0.32):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Linear(100, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
        )
        self.mixer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(192),
                'token_mix': nn.Sequential(
                    nn.Linear(16, 32),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(32, 16),
                    nn.Dropout(dropout * 0.5),
                ),
                'norm2': nn.LayerNorm(192),
                'channel_mix': nn.Sequential(
                    nn.Linear(192, 384),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(384, 192),
                    nn.Dropout(dropout * 0.5),
                ),
            })
            for _ in range(3)
        ])
        self.mixer_proj = nn.Sequential(
            nn.Linear(192 * 3, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_mixer = nn.Linear(384, 11)

        self.left_mlp = nn.Sequential(
            nn.Linear(324, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_tens = nn.Linear(128, 2)

        self.right_mlp = nn.Sequential(
            nn.Linear(324, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_units = nn.Linear(256, 10)

        self.global_mlp = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_global = nn.Linear(256, 11)

        self.coordinator = nn.Sequential(
            nn.Linear(384 + 128 + 256 + 256 + 6, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 11),
        )

    def forward(self, x_full_784, x_left_324, x_right_324, geo6):
        B = x_full_784.shape[0]
        x2d = x_full_784.reshape(B, 1, 28, 28)

        patches = F.unfold(x2d, kernel_size=10, stride=6).transpose(1, 2)
        tok = self.patch_embed(patches)

        for blk in self.mixer_blocks:
            y = blk['norm1'](tok).transpose(1, 2)
            tok = tok + blk['token_mix'](y).transpose(1, 2)
            tok = tok + blk['channel_mix'](blk['norm2'](tok))

        tok_grid = tok.reshape(B, 4, 4, 192)
        f_mix_glob = tok.mean(dim=1)
        f_mix_left = tok_grid[:, :, :2, :].mean(dim=(1, 2))
        f_mix_right = tok_grid[:, :, 2:, :].mean(dim=(1, 2))
        f_mixer = self.mixer_proj(torch.cat([f_mix_glob, f_mix_left, f_mix_right], dim=1))

        f_left = self.left_mlp(x_left_324)
        f_right = self.right_mlp(x_right_324)
        f_glob = self.global_mlp(x_full_784)

        l_mix = self.head_mixer(f_mixer)
        l_tens = self.head_tens(f_left)
        l_units = self.head_units(f_right)
        l_glob = self.head_global(f_glob)

        f_all = torch.cat([f_mixer, f_left, f_right, f_glob, geo6], dim=1)
        l_coord = self.coordinator(f_all)
        return l_coord, l_mix, l_glob, l_tens, l_units

    def predict_proba(self, x_full_784, x_left_324, x_right_324, geo6):
        l_coord, l_mix, l_glob, l_tens, l_units = self.forward(x_full_784, x_left_324, x_right_324, geo6)
        p_coord = l_coord.softmax(dim=1)
        p_mix = l_mix.softmax(dim=1)
        p_glob = l_glob.softmax(dim=1)

        pt = l_tens.softmax(dim=1)
        pu = lu_soft = l_units.softmax(dim=1)
        p_fact = torch.cat([pt[:, 0:1] * lu_soft[:, 0:10], pt[:, 1:2] * lu_soft[:, 0:1]], dim=1)
        p_fact = p_fact / p_fact.sum(dim=1, keepdim=True).clamp(min=1e-7)

        return 0.40 * p_coord + 0.25 * p_mix + 0.15 * p_glob + 0.20 * p_fact


def v4_shift_28(x28_flat, shifts, yy28, xx28):
    dy, dx = shifts[:, 0, None, None], shifts[:, 1, None, None]
    pos = ((yy28[None] + 1 - dy) * 30 + xx28[None] + 1 - dx).reshape(len(shifts), 784)
    pad = F.pad(x28_flat.reshape(-1, 28, 28), (1, 1, 1, 1), value=0).reshape(-1, 900)
    return torch.gather(pad, 1, pos)


def v4_shift_18(x18_flat, shifts, yy18, xx18):
    dy, dx = shifts[:, 0, None, None], shifts[:, 1, None, None]
    pos = ((yy18[None] + 1 - dy) * 20 + xx18[None] + 1 - dx).reshape(len(shifts), 324)
    pad = F.pad(x18_flat.reshape(-1, 18, 18), (1, 1, 1, 1), value=0).reshape(-1, 400)
    return torch.gather(pad, 1, pos)


@torch.inference_mode()
def v4_predict_hier_shift9(model, xd_u8, xl_u8, xr_u8, g6_f32, dev, batch_size=512):
    model.eval()
    n = len(xd_u8)
    out = []
    for j in range(0, n, batch_size):
        xd = torch.from_numpy(np.ascontiguousarray(xd_u8[j:j + batch_size].reshape(-1, 28, 28))).to(dev).float() / 255.0
        xl = torch.from_numpy(np.ascontiguousarray(xl_u8[j:j + batch_size].reshape(-1, 18, 18))).to(dev).float() / 255.0
        xr = torch.from_numpy(np.ascontiguousarray(xr_u8[j:j + batch_size].reshape(-1, 18, 18))).to(dev).float() / 255.0
        g6 = torch.from_numpy(np.ascontiguousarray(g6_f32[j:j + batch_size])).to(dev).float()

        pad_d = F.pad(xd, (1, 1, 1, 1), value=0)
        pad_l = F.pad(xl, (1, 1, 1, 1), value=0)
        pad_r = F.pad(xr, (1, 1, 1, 1), value=0)

        prob = torch.zeros((len(xd), 11), dtype=torch.float32, device=dev)
        w_sum = 0.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = 1.5 if (dy == 0 and dx == 0) else (1.0 if (dy == 0 or dx == 0) else 0.75)
                vd = pad_d[:, 1 - dy:29 - dy, 1 - dx:29 - dx].reshape(-1, 784)
                vl = pad_l[:, 1 - dy:19 - dy, 1 - dx:19 - dx].reshape(-1, 324)
                vr = pad_r[:, 1 - dy:19 - dy, 1 - dx:19 - dx].reshape(-1, 324)
                prob += w * model.predict_proba(vd, vl, vr, g6)
                w_sum += w
        out.append((prob / w_sum).cpu().numpy())
    res = np.concatenate(out, axis=0)
    assert res.shape == (n, 11) and np.isfinite(res).all()
    return res


def v4_metrics(prob, y, writer):
    ok = (prob.argmax(1) + 10) == y
    return dict(
        n=int(len(y)),
        writers=int(len(np.unique(writer))),
        correct=int(ok.sum()),
        errors=int((~ok).sum()),
        micro_acc=round(float(ok.mean()), 6),
        macro_class=round(float(np.mean([ok[y == c].mean() for c in np.unique(y)])), 6),
        macro_writer=round(float(np.mean([ok[writer == w].mean() for w in np.unique(writer)])), 6),
    )


def v4_paired_bootstrap(base_prob, cand_prob, y, writer, seed=440007, draws=4000):
    a = (base_prob.argmax(1) + 10) == y
    b = (cand_prob.argmax(1) + 10) == y
    names, inv = np.unique(writer, return_inverse=True)
    n_w = np.bincount(inv, minlength=len(names))
    diff = np.bincount(inv, weights=b.astype(float) - a.astype(float), minlength=len(names))
    counts = np.random.default_rng(seed).multinomial(len(names), np.full(len(names), 1.0 / len(names)), size=draws)
    micro_d = (counts @ diff) / (counts @ n_w) * 100.0
    macro_d = (counts @ (diff / n_w)) / len(names) * 100.0
    lo, hi = np.percentile(micro_d, [2.5, 97.5])
    mlo, mhi = np.percentile(macro_d, [2.5, 97.5])
    return dict(
        fixed=int((~a & b).sum()),
        broken=int((a & ~b).sum()),
        net_gain=int(b.sum() - a.sum()),
        delta_pp=round(float((b.mean() - a.mean()) * 100.0), 4),
        ci95_low_pp=round(float(lo), 4),
        ci95_high_pp=round(float(hi), 4),
        macro_writer_delta_pp=round(float((diff / n_w).mean() * 100.0), 4),
        macro_ci95_low_pp=round(float(mlo), 4),
        macro_ci95_high_pp=round(float(mhi), 4),
    )


class V4Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, s):
        self.stream.write(s); self.file.write(s); self.file.flush()
    def flush(self):
        self.stream.flush(); self.file.flush()
    def isatty(self):
        return False


def run_v4_cell1():
    from google.colab import drive
    drive.mount('/content/drive')
    started = time.perf_counter()
    root = Path(V4_ROOT)
    run_v5 = root / 'v5_76ddef9bc184'
    data_dir = run_v5 / 'dataset_82cc33f4f691'
    cmp_dir = run_v5 / 'renderer_comparison_b409f5a57688'
    assert run_v5.is_dir() and cmp_dir.is_dir(), 'Không tìm thấy thư mục dữ liệu gốc trên Drive.'

    v4_home = root / 'v4_hier_multi_mlp'
    v4_home.mkdir(parents=True, exist_ok=True)
    session = v4_home / 'cell1_sessions' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    session.mkdir(parents=True, exist_ok=False)
    oldout, olderr = sys.stdout, sys.stderr
    status = dict(status='started', active_pointer_changed=False)

    with open(session / 'console.log', 'w', encoding='utf-8') as log:
        sys.stdout, sys.stderr = V4Tee(oldout, log), V4Tee(olderr, log)
        try:
            assert torch.cuda.is_available(), 'Hãy chọn Runtime > Change runtime type > T4 GPU rồi chạy lại Cell 1!'
            dev = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            torch.set_num_threads(2)
            torch.set_default_dtype(torch.float32)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False

            # Kiểm tra kiến trúc V4HierMultiMLP tuyệt đối không chứa bất kỳ lớp Conv nào
            chk_m = V4HierMultiMLP()
            assert not any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in chk_m.modules())
            n_params = sum(p.numel() for p in chk_m.parameters())
            del chk_m
            print(f'v4_MNIST_10_20 — CELL 1: Huấn luyện TỪ ĐẦU HierMultiMLP ({n_params:,} params, 100% nn.Linear) | GPU: {gpu_name}')

            # 1. Nạp dữ liệu v5 (cả Raw và Capped_Style validation) + 451 mẫu mới sạch (cả Raw và Capped_Style)
            tr = v4_npz(data_dir / 'train.npz')
            va = v4_npz(data_dir / 'validation.npz')
            st_cap = v4_npz(cmp_dir / 'strokes_capped.npz')
            vc_cap = v4_npz(cmp_dir / 'validation_capped.npz')
            cmp_cfg = v4_read(cmp_dir / 'config.json')
            settings = cmp_cfg['renderer']
            levels = np.asarray(settings['gray_levels'], dtype=np.float64)

            x_tr_orig = tr['X'].copy().reshape(-1, 784)
            x_tr_orig[np.flatnonzero(tr['source'] == 1)] = st_cap['X'].reshape(-1, 784)
            y_tr = tr['y'].astype(np.int64)
            fa_tr, fb_tr, src_tr = tr['fold_a'].copy(), tr['fold_b'].copy(), tr['source'].copy()
            ra_tr, rb_tr = tr['row_a'].copy(), tr['row_b'].copy()
            del tr, st_cap

            excl_v5 = np.array(V4_V5_EXCLUSIONS, dtype=np.int64)
            clean_tr_mask = ~(np.isin(ra_tr, excl_v5) | np.isin(rb_tr, excl_v5))
            valid_oof_mask = ~np.isin(va['rows'], excl_v5)
            assert int(valid_oof_mask.sum()) == 4932

            old_writer_fold = {str(w): int(f) for w, f in zip(va['writer'], va['fold'])}

            df1 = pd.read_csv(root / 've_tay_10_20 - data.csv')
            df2 = pd.read_csv(root / 've_tay_10_20 - data_v31.csv')
            pc = [f'pixel{i}' for i in range(784)]
            new178 = df1.iloc[5452:].copy()
            v31 = df2.copy()
            v31_pix = v31[pc].to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
            for idx_r in range(len(v31)):
                fixed_im = v4_render_unclipped_raw(v31['strokes'].iloc[idx_r])
                if fixed_im is not None:
                    v31_pix[idx_r] = fixed_im
            keep_v31 = ~v31.index.isin(V4_V31_EXCLUSIONS)
            X_hold_raw = np.concatenate([
                new178[pc].to_numpy(dtype=np.uint8).reshape(-1, 784),
                v31_pix[keep_v31].reshape(-1, 784)
            ], axis=0)
            y_hold = np.concatenate([
                new178['label'].to_numpy(dtype=np.int64),
                v31.loc[keep_v31, 'label'].to_numpy(dtype=np.int64)
            ], axis=0)
            w_hold = np.concatenate([
                new178['writer'].map(v4_writer_key).to_numpy(dtype=str),
                v31.loc[keep_v31, 'writer_name'].map(v4_writer_key).to_numpy(dtype=str)
            ], axis=0)
            strokes_hold = np.concatenate([
                new178['strokes'].to_numpy(dtype=str),
                v31.loc[keep_v31, 'strokes'].to_numpy(dtype=str)
            ], axis=0)

            new_writer_fold = {}
            strict_new_writers = sorted(
                [w for w in np.unique(w_hold) if w not in old_writer_fold],
                key=lambda w: (-int((w_hold == w).sum()), w)
            )
            fold_loads = np.zeros(5, dtype=np.int64)
            for w in strict_new_writers:
                k_best = int(np.argmin(fold_loads))
                new_writer_fold[w] = k_best
                fold_loads[k_best] += int((w_hold == w).sum())
            for w in np.unique(w_hold):
                if w in old_writer_fold:
                    new_writer_fold[w] = old_writer_fold[w]
            fold_hold = np.array([new_writer_fold[w] for w in w_hold], dtype=np.int64)

            # Sinh hoặc nạp Capped_Style validation cho 451 mẫu mới + tái sử dụng 1.804 strokes & 2.200 recomb từ Cell 42
            v12_aug_cache = cmp_dir / 'standalone_fdeskew_109w_v12' / 'new_451_strokes_and_recomb.npz'
            assert v12_aug_cache.exists(), 'Không thấy cache new_451_strokes_and_recomb.npz từ Cell 42.'
            z_aug = v4_npz(v12_aug_cache)
            X_new_st, y_new_st, f_new_st = z_aug['X_st'], z_aug['y_st'], z_aug['f_st']
            X_new_rc, y_new_rc, fa_new_rc, fb_new_rc = z_aug['X_rc'], z_aug['y_rc'], z_aug['fa_rc'], z_aug['fb_rc']

            ho_style_cache = v4_home / 'holdout_451_val_capped_style.npz'
            if ho_style_cache.exists():
                X_hold_style = v4_npz(ho_style_cache)['X_style']
            else:
                imgs_hs = [
                    v4_render_stroke_cap(strokes_hold[i], 45_000_000 + i, settings, levels, cap=1.25)
                    for i in range(len(strokes_hold))
                ]
                X_hold_style = np.stack(imgs_hs, axis=0).reshape(-1, 784)
                v4_save_npz(ho_style_cache, dict(X_style=X_hold_style))

            test_df = pd.read_csv(root / 'test.csv', dtype={'id': str}, keep_default_na=False)
            test_ids = test_df['id'].to_numpy(dtype=str)
            X_test_orig = test_df[pc].to_numpy(dtype=np.uint8).reshape(-1, 784)

            # 2. Ghép tập dữ liệu huấn luyện 109 Người viết và tiền xử lý Multi-MLP Views (28x28 + 18x18 Left + 18x18 Right + Geo6)
            keep_idx = np.flatnonzero(clean_tr_mask)
            X_exp_orig = np.concatenate([x_tr_orig[keep_idx], X_hold_raw, X_new_st, X_new_rc], axis=0)
            y_exp = np.concatenate([y_tr[keep_idx], y_hold, y_new_st, y_new_rc], axis=0)
            src_exp = np.concatenate([
                src_tr[keep_idx],
                np.zeros(len(y_hold), dtype=np.int8),
                np.ones(len(y_new_st), dtype=np.int8),
                np.full(len(y_new_rc), 2, dtype=np.int8),
            ], axis=0)
            fa_exp = np.concatenate([fa_tr[keep_idx], fold_hold, f_new_st, fa_new_rc], axis=0)
            fb_exp = np.concatenate([fb_tr[keep_idx], fold_hold, f_new_st, fb_new_rc], axis=0)
            del x_tr_orig

            t_pre = time.perf_counter()
            print(f'Đang trích xuất 4 góc nhìn (28x28 Deskew + 18x18 Left-CoM + 18x18 Right-CoM + Geo6) cho {len(y_exp):,} ảnh...')
            xd_exp, xl_exp, xr_exp, g6_exp = v4_precompute_multi_mlp_views_gpu(X_exp_orig, levels, dev)
            xd_va_raw, xl_va_raw, xr_va_raw, g6_va_raw = v4_precompute_multi_mlp_views_gpu(va['X_raw'].reshape(-1, 784), levels, dev)
            xd_va_cap, xl_va_cap, xr_va_cap, g6_va_cap = v4_precompute_multi_mlp_views_gpu(vc_cap['X'].reshape(-1, 784), levels, dev)
            xd_ho_raw, xl_ho_raw, xr_ho_raw, g6_ho_raw = v4_precompute_multi_mlp_views_gpu(X_hold_raw, levels, dev)
            xd_ho_cap, xl_ho_cap, xr_ho_cap, g6_ho_cap = v4_precompute_multi_mlp_views_gpu(X_hold_style, levels, dev)
            xd_te, xl_te, xr_te, g6_te = v4_precompute_multi_mlp_views_gpu(X_test_orig, levels, dev)
            del X_exp_orig
            print(f'Hoàn tất trích xuất Multi-MLP Views trong {time.perf_counter() - t_pre:.2f}s.')

            # 3. Huấn luyện TỪ ĐẦU (From Scratch) V4HierMultiMLP trên cả 5 Fold (Curriculum Sampling + Dual-Crop Shift)
            xd_gpu = torch.from_numpy(xd_exp).to(dev)
            xl_gpu = torch.from_numpy(xl_exp).to(dev)
            xr_gpu = torch.from_numpy(xr_exp).to(dev)
            g6_gpu = torch.from_numpy(g6_exp).to(dev)
            y11_gpu = torch.from_numpy(y_exp - 10).to(dev)
            ytens_gpu = torch.from_numpy((y_exp == 20).astype(np.int64)).to(dev)
            yunits_gpu = torch.from_numpy(np.where(y_exp == 20, 0, y_exp - 10).astype(np.int64)).to(dev)
            del xd_exp, xl_exp, xr_exp, g6_exp

            yy28, xx28 = torch.meshgrid(torch.arange(28, device=dev), torch.arange(28, device=dev), indexing='ij')
            yy18, xx18 = torch.meshgrid(torch.arange(18, device=dev), torch.arange(18, device=dev), indexing='ij')

            oof_hier_raw = np.full((len(va['y']), 11), np.nan, dtype=np.float32)
            oof_hier_cap = np.full((len(va['y']), 11), np.nan, dtype=np.float32)
            ho_hier_raw = np.full((len(y_hold), 11), np.nan, dtype=np.float32)
            ho_hier_cap = np.full((len(y_hold), 11), np.nan, dtype=np.float32)
            te_hier_folds = []

            for k in range(5):
                f_dir = v4_home / f'fold_{k}'
                f_dir.mkdir(parents=True, exist_ok=True)
                pt_path = f_dir / 'hier_mlp_from_scratch_s7.pt'
                pred_path = f_dir / 'hier_preds_s7.npz'
                vi_va = np.flatnonzero(va['fold'] == k)
                vi_ho = np.flatnonzero(fold_hold == k)

                if pt_path.exists() and pred_path.exists():
                    zf = v4_npz(pred_path)
                    oof_hier_raw[vi_va] = zf['oof_raw']
                    oof_hier_cap[vi_va] = zf['oof_cap']
                    ho_hier_raw[vi_ho] = zf['ho_raw']
                    ho_hier_cap[vi_ho] = zf['ho_cap']
                    te_hier_folds.append(zf['te_prob'])
                    ok_r = int(((zf['oof_raw'].argmax(1) + 10) == va['y'][vi_va]).sum())
                    ok_c = int(((zf['oof_cap'].argmax(1) + 10) == va['y'][vi_va]).sum())
                    print(f'[HierMultiMLP Fold {k}] Nạp cache hợp lệ | OOF Raw: {ok_r}/{len(vi_va)} | OOF Capped_Style: {ok_c}/{len(vi_va)}')
                    continue

                tr_ix = np.flatnonzero((fa_exp != k) & (fb_exp != k))
                # Stage 1 Curriculum Pools (BS=256: 48 raw + 96 strokes + 64 recomb + 48 composite)
                s1_pools = [
                    (torch.from_numpy(tr_ix[src_exp[tr_ix] == s]).to(dev), cnt)
                    for s, cnt in [(0, 48), (1, 96), (2, 64), (3, 48)]
                ]
                # Stage 2 Precision Pools (BS=128: 36 raw + 60 strokes + 32 recomb, ZERO composite)
                s2_pools = [
                    (torch.from_numpy(tr_ix[src_exp[tr_ix] == s]).to(dev), cnt)
                    for s, cnt in [(0, 36), (1, 60), (2, 32)]
                ]

                seed_k = 44000 + 100 * k + 7
                random.seed(seed_k); np.random.seed(seed_k); torch.manual_seed(seed_k); torch.cuda.manual_seed_all(seed_k)
                aug_rng = torch.Generator(device=dev).manual_seed(54000 + 100 * k + 7)

                model = V4HierMultiMLP(dropout=0.30).to(dev)
                t_fk = time.perf_counter()

                # Stage 1: Huấn luyện TỪ ĐẦU (12 epochs × 800 steps = 9.600 steps, OneCycleLR max_lr=1.8e-3)
                epochs_s1, steps_s1 = 12, 800
                opt1 = torch.optim.AdamW(model.parameters(), lr=1.8e-3 / 25.0, weight_decay=2.5e-5)
                sched1 = torch.optim.lr_scheduler.OneCycleLR(
                    opt1, max_lr=1.8e-3, total_steps=epochs_s1 * steps_s1,
                    pct_start=0.12, div_factor=25.0, final_div_factor=1e3
                )
                for ep in range(epochs_s1):
                    model.train()
                    for _ in range(steps_s1):
                        ids = torch.cat([p[torch.randint(len(p), (c,), device=dev)] for p, c in s1_pools])
                        ids = ids[torch.randperm(len(ids), device=dev)]
                        sh28 = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        sh18_l = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        sh18_r = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)

                        vd = v4_shift_28(xd_gpu[ids].float() / 255.0, sh28, yy28, xx28)
                        vl = v4_shift_18(xl_gpu[ids].float() / 255.0, sh18_l, yy18, xx18)
                        vr = v4_shift_18(xr_gpu[ids].float() / 255.0, sh18_r, yy18, xx18)
                        g6 = g6_gpu[ids]

                        l_coord, l_mix, l_glob, l_tens, l_units = model(vd, vl, vr, g6)
                        y_b = y11_gpu[ids]
                        loss = (
                            0.40 * F.cross_entropy(l_coord, y_b)
                            + 0.25 * F.cross_entropy(l_mix, y_b)
                            + 0.20 * F.cross_entropy(l_glob, y_b)
                            + 0.20 * F.cross_entropy(l_tens, ytens_gpu[ids])
                            + 0.30 * F.cross_entropy(l_units, yunits_gpu[ids])
                        )
                        opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step(); sched1.step()

                # Stage 2: Tinh chỉnh miền thật (4 epochs × 450 steps = 1.800 steps, CosineAnnealingLR 2.5e-4 -> 1.5e-5)
                epochs_s2, steps_s2 = 4, 450
                opt2 = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=2.5e-5)
                sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_s2 * steps_s2, eta_min=1.5e-5)
                for ep in range(epochs_s2):
                    model.train()
                    for _ in range(steps_s2):
                        ids = torch.cat([p[torch.randint(len(p), (c,), device=dev)] for p, c in s2_pools])
                        ids = ids[torch.randperm(len(ids), device=dev)]
                        sh28 = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        sh18_l = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        sh18_r = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)

                        vd = v4_shift_28(xd_gpu[ids].float() / 255.0, sh28, yy28, xx28)
                        vl = v4_shift_18(xl_gpu[ids].float() / 255.0, sh18_l, yy18, xx18)
                        vr = v4_shift_18(xr_gpu[ids].float() / 255.0, sh18_r, yy18, xx18)
                        g6 = g6_gpu[ids]

                        l_coord, l_mix, l_glob, l_tens, l_units = model(vd, vl, vr, g6)
                        y_b = y11_gpu[ids]
                        loss = (
                            0.45 * F.cross_entropy(l_coord, y_b)
                            + 0.25 * F.cross_entropy(l_mix, y_b)
                            + 0.15 * F.cross_entropy(l_glob, y_b)
                            + 0.20 * F.cross_entropy(l_tens, ytens_gpu[ids])
                            + 0.30 * F.cross_entropy(l_units, yunits_gpu[ids])
                        )
                        opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step(); sched2.step()

                p_va_raw = v4_predict_hier_shift9(model, xd_va_raw[vi_va], xl_va_raw[vi_va], xr_va_raw[vi_va], g6_va_raw[vi_va], dev)
                p_va_cap = v4_predict_hier_shift9(model, xd_va_cap[vi_va], xl_va_cap[vi_va], xr_va_cap[vi_va], g6_va_cap[vi_va], dev)
                p_ho_raw = v4_predict_hier_shift9(model, xd_ho_raw[vi_ho], xl_ho_raw[vi_ho], xr_ho_raw[vi_ho], g6_ho_raw[vi_ho], dev)
                p_ho_cap = v4_predict_hier_shift9(model, xd_ho_cap[vi_ho], xl_ho_cap[vi_ho], xr_ho_cap[vi_ho], g6_ho_cap[vi_ho], dev)
                p_te = v4_predict_hier_shift9(model, xd_te, xl_te, xr_te, g6_te, dev)

                oof_hier_raw[vi_va] = p_va_raw
                oof_hier_cap[vi_va] = p_va_cap
                ho_hier_raw[vi_ho] = p_ho_raw
                ho_hier_cap[vi_ho] = p_ho_cap
                te_hier_folds.append(p_te)

                st_cpu = {kp: vp.detach().cpu().clone() for kp, vp in model.state_dict().items()}
                torch.save({'fold': k, 'seed': seed_k, 'model': st_cpu}, pt_path)
                v4_save_npz(pred_path, dict(
                    oof_raw=p_va_raw, oof_cap=p_va_cap,
                    ho_raw=p_ho_raw, ho_cap=p_ho_cap, te_prob=p_te,
                ))
                ok_r = int(((p_va_raw.argmax(1) + 10) == va['y'][vi_va]).sum())
                ok_c = int(((p_va_cap.argmax(1) + 10) == va['y'][vi_va]).sum())
                ok_hc = int(((p_ho_cap.argmax(1) + 10) == y_hold[vi_ho]).sum())
                print(f'HOÀN TẤT [HierMultiMLP Fold {k}] trong {time.perf_counter() - t_fk:.1f}s | '
                      f'OOF Raw: {ok_r}/{len(vi_va)} ({ok_r/len(vi_va)*100:.2f}%) | '
                      f'OOF Capped_Style: {ok_c}/{len(vi_va)} ({ok_c/len(vi_va)*100:.2f}%) | '
                      f'NewWriters Capped_Style: {ok_hc}/{len(vi_ho)} ({ok_hc/len(vi_ho)*100:.2f}%)')
                del model, opt1, sched1, opt2, sched2, s1_pools, s2_pools

            del xd_gpu, xl_gpu, xr_gpu, g6_gpu, y11_gpu, ytens_gpu, yunits_gpu
            torch.cuda.empty_cache()

            # 4. Nạp dự đoán Raw & Capped_Style của Incumbent v5 và các mô hình v12 (Cell 42) để đối chứng
            z_inc_oof = v4_npz(cmp_dir / ('ft_cv_' + V4_FT_ID) / 'capped' / 'oof_all_folds.npz')
            oof_inc_raw = z_inc_oof['shift9_raw_prob']
            oof_inc_cap = z_inc_oof['shift9_capped_style_prob']

            z_s29_oof = v4_npz(cmp_dir / ('seed29_ensemble_cv_' + V4_SEED29_ID) / 'oof_all_folds.npz')
            oof_s29_raw = z_s29_oof['seed29_shift9_raw_prob']
            oof_s29_cap = z_s29_oof['seed29_shift9_capped_style_prob']

            # Chấm Incumbent v5 trên 451 mẫu mới (cả Raw và Capped_Style) theo đúng Fold k held-out
            base_cfg = v4_read(cmp_dir / ('baseline_cv_' + V4_BASE_ID) / 'config.json')
            width_std, dropout_std = base_cfg['width'], base_cfg['dropout']

            def make_std_mlp():
                layers, d = [], 784
                for h in width_std:
                    layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout_std)]
                    d = h
                return nn.Sequential(*layers, nn.Linear(d, 11)).to(dev)

            @torch.inference_mode()
            def pred_std_shift9(m, im_u8):
                m.eval()
                x = torch.from_numpy(np.ascontiguousarray(im_u8.reshape(-1, 28, 28))).to(dev).float() / 255.0
                pad = F.pad(x, (1, 1, 1, 1), value=0)
                p = torch.zeros((len(x), 11), dtype=torch.float32, device=dev)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        p += m(pad[:, 1 - dy:29 - dy, 1 - dx:29 - dx].reshape(-1, 784)).softmax(1)
                return (p / 9.0).cpu().numpy()

            ho_inc_raw = np.full((len(y_hold), 11), np.nan, dtype=np.float32)
            ho_inc_cap = np.full((len(y_hold), 11), np.nan, dtype=np.float32)
            te_inc_folds = []
            for k in range(5):
                vi_ho = np.flatnonzero(fold_hold == k)
                cp = torch.load(cmp_dir / ('ft_cv_' + V4_FT_ID) / 'capped' / f'fold_{k}' / 'model_final.pt', map_location='cpu', weights_only=False)
                m = make_std_mlp()
                m.load_state_dict(cp['model']); del cp
                ho_inc_raw[vi_ho] = pred_std_shift9(m, X_hold_raw[vi_ho])
                ho_inc_cap[vi_ho] = pred_std_shift9(m, X_hold_style[vi_ho])
                te_inc_folds.append(pred_std_shift9(m, X_test_orig))
                del m

            cp_full = torch.load(cmp_dir / ('final_full_capped_' + V4_FINAL_V5_ID) / 'model_final.pt', map_location='cpu', weights_only=False)
            m_full = make_std_mlp()
            m_full.load_state_dict(cp_full['model']); del cp_full
            te_v5_full_s9 = pred_std_shift9(m_full, X_test_orig)
            del m_full

            # Nạp thêm dự đoán OOF Raw & Test của v12 (Cell 42)
            v12_dir = cmp_dir / 'standalone_fdeskew_109w_v12'
            oof_v12_fd = np.full((len(va['y']), 11), np.nan, dtype=np.float32)
            ho_v12_fd = np.full((len(y_hold), 11), np.nan, dtype=np.float32)
            te_v12_l = []
            for k in range(5):
                zf12 = v4_npz(v12_dir / f'fold_{k}' / 'fold_preds.npz')
                oof_v12_fd[va['fold'] == k] = 0.50 * zf12['oof_fd'] + 0.50 * (0.55 * zf12['oof_o_s7'] + 0.45 * zf12['oof_o_s29'])
                ho_v12_fd[fold_hold == k] = 0.50 * zf12['ho_fd'] + 0.50 * (0.55 * zf12['ho_o_s7'] + 0.45 * zf12['ho_o_s29'])
                te_v12_l.append(0.50 * zf12['te_fd'] + 0.50 * (0.55 * zf12['te_o_s7'] + 0.45 * zf12['te_o_s29']))
            te_v12_ens = np.mean(te_v12_l, axis=0)
            te_hier_ens = np.mean(te_hier_folds, axis=0)

            # 5. ĐÁNH GIÁ TRÊN 4 CỔNG KIỂM ĐỊNH (ĐẶC BIỆT LÀ CAPPED_STYLE — ĐẠI DIỆN TRỰC TIẾP CỦA TEST.CSV)
            ym_oof = va['y'][valid_oof_mask]
            wm_oof = va['writer'][valid_oof_mask]

            oof_blend_cap = 0.45 * oof_hier_cap + 0.35 * oof_inc_cap + 0.20 * oof_s29_cap
            oof_blend_raw = 0.40 * oof_hier_raw + 0.35 * oof_v12_fd + 0.25 * oof_inc_raw
            ho_blend_cap = 0.50 * ho_hier_cap + 0.50 * ho_inc_cap
            ho_blend_raw = 0.40 * ho_hier_raw + 0.35 * ho_v12_fd + 0.25 * ho_inc_raw

            print('\n================ CỔNG 1A (QUAN TRỌNG NHẤT CHO TEST): OOF 4.932 CAPPED_STYLE ================')
            t_1a = [
                dict(candidate='1_incumbent_v5_seed7',    **v4_metrics(oof_inc_cap[valid_oof_mask], ym_oof, wm_oof)),
                dict(candidate='2_seed29_v5_ft',          **v4_metrics(oof_s29_cap[valid_oof_mask], ym_oof, wm_oof)),
                dict(candidate='3_hier_multi_mlp_scratch',**v4_metrics(oof_hier_cap[valid_oof_mask], ym_oof, wm_oof)),
                dict(candidate='4_blend_hier_plus_v5',    **v4_metrics(oof_blend_cap[valid_oof_mask], ym_oof, wm_oof)),
            ]
            print(pd.DataFrame(t_1a).to_string(index=False))
            p_1a = [
                dict(comparison='hier_scratch vs incumbent (capped_style)', **v4_paired_bootstrap(oof_inc_cap[valid_oof_mask], oof_hier_cap[valid_oof_mask], ym_oof, wm_oof)),
                dict(comparison='blend_hier_v5 vs incumbent (capped_style)',**v4_paired_bootstrap(oof_inc_cap[valid_oof_mask], oof_blend_cap[valid_oof_mask], ym_oof, wm_oof)),
            ]
            print(pd.DataFrame(p_1a).to_string(index=False))

            print('\n================ CỔNG 1B: OOF 4.932 RAW ================')
            t_1b = [
                dict(candidate='1_incumbent_v5_seed7',    **v4_metrics(oof_inc_raw[valid_oof_mask], ym_oof, wm_oof)),
                dict(candidate='2_hier_multi_mlp_scratch',**v4_metrics(oof_hier_raw[valid_oof_mask], ym_oof, wm_oof)),
                dict(candidate='3_blend_hier_v12_v5',     **v4_metrics(oof_blend_raw[valid_oof_mask], ym_oof, wm_oof)),
            ]
            print(pd.DataFrame(t_1b).to_string(index=False))

            print('\n================ CỔNG 2A: HELD-OUT 451 NEW WRITERS — CAPPED_STYLE ================')
            t_2a = [
                dict(candidate='1_incumbent_v5_heldout',  **v4_metrics(ho_inc_cap, y_hold, w_hold)),
                dict(candidate='2_hier_multi_mlp_scratch',**v4_metrics(ho_hier_cap, y_hold, w_hold)),
                dict(candidate='3_blend_hier_plus_v5',    **v4_metrics(ho_blend_cap, y_hold, w_hold)),
            ]
            print(pd.DataFrame(t_2a).to_string(index=False))

            print('\n================ CỔNG 2B: HELD-OUT 451 NEW WRITERS — RAW ================')
            t_2b = [
                dict(candidate='1_incumbent_v5_heldout',  **v4_metrics(ho_inc_raw, y_hold, w_hold)),
                dict(candidate='2_hier_multi_mlp_scratch',**v4_metrics(ho_hier_raw, y_hold, w_hold)),
                dict(candidate='3_blend_hier_v12_v5',     **v4_metrics(ho_blend_raw, y_hold, w_hold)),
            ]
            print(pd.DataFrame(t_2b).to_string(index=False))

            # 6. Kiểm tra độ độc lập của HierMultiMLP (From Scratch) trên 550 mẫu Test so với v5
            pred_v5 = te_v5_full_s9.argmax(axis=1) + 10
            conf_v5 = te_v5_full_s9.max(axis=1)
            pred_hier = te_hier_ens.argmax(axis=1) + 10
            conf_hier = te_hier_ens.max(axis=1)

            te_grand_v4 = 0.40 * te_hier_ens + 0.35 * te_v12_ens + 0.25 * (0.5 * te_v5_full_s9 + 0.5 * np.mean(te_inc_folds, axis=0))
            pred_grand_arg = te_grand_v4.argmax(axis=1) + 10

            # Xem các dòng mà HierMultiMLP (From Scratch) dự đoán khác với v5 (đặc biệt là các dòng v5 tự tin >= 0.60!)
            diff_hier_v5 = np.flatnonzero(pred_hier != pred_v5)
            high_conf_flips = [
                dict(
                    id=str(test_ids[i]),
                    v5_pred=int(pred_v5[i]), v5_conf=round(float(conf_v5[i]), 4),
                    hier_pred=int(pred_hier[i]), hier_conf=round(float(conf_hier[i]), 4),
                    grand_pred=int(pred_grand_arg[i]), grand_conf=round(float(te_grand_v4[i].max()), 4),
                )
                for i in diff_hier_v5
            ]

            print(f'\n================ PHÂN TÍCH ĐỘ ĐỘC LẬP TRÊN 550 MẪU TEST ================')
            print(f'Số dòng trên Test mà HierMultiMLP (From Scratch) dự đoán khác v5: {len(diff_hier_v5)}/550 dòng.')
            print('Chi tiết các dòng khác biệt giữa HierMultiMLP và v5:', high_conf_flips)

            # Lưu các dự đoán Test để phân tích trước khi quyết định bước tiếp theo
            pd.DataFrame({'id': test_ids, 'label': pred_hier}).to_csv(session / 'test_hier_scratch_argmax.csv', index=False)
            pd.DataFrame({'id': test_ids, 'label': pred_grand_arg}).to_csv(session / 'test_v4_grand_argmax.csv', index=False)

            report = dict(
                session_id=session.name,
                gate_1a_old4932_capped_style=t_1a,
                gate_1a_paired=p_1a,
                gate_1b_old4932_raw=t_1b,
                gate_2a_new451_capped_style=t_2a,
                gate_2b_new451_raw=t_2b,
                test_hier_vs_v5_diff_count=len(diff_hier_v5),
                test_hier_vs_v5_details=high_conf_flips,
            )
            v4_json(session / 'report.json', report)
            status.update(
                status='complete',
                old4932_capped_style_incumbent=int(t_1a[0]['correct']),
                old4932_capped_style_hier=int(t_1a[2]['correct']),
                old4932_capped_style_blend=int(t_1a[3]['correct']),
                new451_capped_style_incumbent=int(t_2a[0]['correct']),
                new451_capped_style_blend=int(t_2a[2]['correct']),
                elapsed_minutes=round((time.perf_counter() - started) / 60.0, 2),
            )
        except Exception as e:
            status.update(status='stopped_error', error_type=type(e).__name__, error=str(e))
            traceback.print_exc()
        finally:
            v4_json(session / 'status.json', status)
            print('\n=== GỬI LẠI LOG NÀY VÀ FILE ZIP BÁO CÁO ===')
            print(json.dumps(status, ensure_ascii=False, indent=2))
            sys.stdout, sys.stderr = oldout, olderr

    zip_path = session.parent / (session.name + '_v4_cell1_report.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(session.rglob('*')):
            if path.is_file():
                zf.write(path, path.relative_to(session))
    print('REPORT_ZIP:', zip_path)
    try:
        from google.colab import files
        files.download(str(zip_path))
    except Exception:
        print('Tải thủ công file ZIP tại:', zip_path)
    return str(zip_path)


V4_CELL1_ZIP = run_v4_cell1()