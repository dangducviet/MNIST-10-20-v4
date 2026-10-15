# v4_MNIST_10_20 — CELL 5: Hướng 1 (Giải phẫu Ma trận Xác suất & Cân bằng Phẫu thuật Dư->Thiếu)
# + Hướng 2 (Dual-Peak Cumulative-Mass Cut + Valley-Adaptive Gating + Hard-Example Focal Fine-Tuning trên 5 Folds & Full-Data).
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
from functools import lru_cache
import hashlib, json, math, random, sys, time, traceback, unicodedata, uuid, zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

V4_ROOT = '/content/drive/MyDrive/MNIST_data'
V4_BASE_ID = 'e55217c88b0c'
V4_FT_ID = '12861935de9d'
V4_SEED29_ID = 'dd71a4292e16'
V4_FINAL_V5_ID = '4b1e77287434'
V4_V5_EXCLUSIONS = [458, 1767, 2213, 4355]
V4_V31_EXCLUSIONS = [9, 15, 16, 25, 29, 78, 95, 222, 224, 231, 233, 234, 235, 238, 244, 251, 253, 256]


def c5_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def c5_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def c5_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def c5_save_npz(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.pending.npz')
    np.savez_compressed(tmp, **obj)
    os.replace(tmp, path)


def c5_writer_key(v):
    if pd.isna(v):
        return ''
    w = ' '.join(unicodedata.normalize('NFC', str(v)).casefold().split())
    return 'minh nhật phạm' if w in ('phạm minh nhật', 'minh nhật phạm') else w


@lru_cache(maxsize=128)
def c5_area_weights(src, dst):
    lo = np.arange(dst, dtype=np.float64)[:, None] * src / dst
    hi = (np.arange(dst, dtype=np.float64)[:, None] + 1) * src / dst
    q = np.arange(src, dtype=np.float64)[None, :]
    w = np.maximum(0., np.minimum(q + 1., hi) - np.maximum(q, lo))
    return w / w.sum(1, keepdims=True)


def c5_norm_js(a):
    ys, xs = np.where(a > 20)
    assert len(xs), 'Ảnh rỗng'
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    scale = min(20 / cw, 20 / ch)
    tw, th = max(1, int(np.floor(cw * scale + .5))), max(1, int(np.floor(ch * scale + .5)))
    small = (c5_area_weights(ch, th) @ crop.astype(np.float64) @ c5_area_weights(cw, tw).T).astype(np.float32)
    small = (small.astype(np.float64) * 255 / float(small.max())).astype(np.float32)
    yy, xx = np.mgrid[0:th, 0:tw]
    mass = small.astype(np.float64).sum()
    ox = int(np.floor(14 - ((xx + .5) * small).sum() / mass + .5))
    oy = int(np.floor(14 - ((yy + .5) * small).sum() / mass + .5))
    ox, oy = max(0, min(28 - tw, ox)), max(0, min(28 - th, oy))
    out = np.zeros((28, 28), dtype=np.uint8)
    out[oy:oy + th, ox:ox + tw] = np.clip(np.floor(small.astype(np.float64) + .5), 0, 255).astype(np.uint8)
    return out


def c5_snap_gray(a, levels):
    ix = np.clip(np.searchsorted(levels, a), 1, len(levels) - 1)
    lo, hi = levels[ix - 1], levels[ix]
    return np.where(a - lo < hi - a, lo, hi).astype(np.uint8)


def c5_render_unclipped_raw(strokes_json):
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
    return c5_norm_js(a)


@torch.inference_mode()
def c5_extract_dualpeak_voronoi_20x20_gpu(images_u8, levels_np, dev, delta_skew=0.0, delta_cut=0.0, chunk=4096):
    """
    Nâng cấp Hướng 2:
    - Định vị khe cắt bằng Ràng buộc Khối lượng Tích lũy & Đỉnh Kép (Dual-Peak & Cumulative-Mass Guard:
      chặn tuyệt đối lỗi nhảy vào cắt giữa bụng chữ số '0, 4, 6, 8, 9' ở cột 15..17 khi 2 chữ số viết sát nhau).
    - Trích xuất khung 20x20 Voronoi-Cleaned + biến độ sâu thung lũng valley_d để điều tiết trọng số nhánh cắt.
    """
    n_total = len(images_u8)
    out_xd = np.empty((n_total, 784), dtype=np.uint8)
    out_xl20 = np.empty((n_total, 400), dtype=np.uint8)
    out_xr20 = np.empty((n_total, 400), dtype=np.uint8)
    out_g6 = np.empty((n_total, 6), dtype=np.float32)

    yy, xx = torch.meshgrid(
        torch.arange(28, dtype=torch.float32, device=dev),
        torch.arange(28, dtype=torch.float32, device=dev),
        indexing='ij'
    )
    cols8_15 = torch.arange(8, 16, dtype=torch.float32, device=dev).unsqueeze(0)
    c_grid = xx.unsqueeze(0)
    grid_20_y, grid_20_x = torch.meshgrid(
        torch.arange(-9.5, 10.5, 1.0, device=dev),
        torch.arange(-9.5, 10.5, 1.0, device=dev),
        indexing='ij'
    )

    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        x = torch.from_numpy(np.ascontiguousarray(images_u8[start:end].reshape(-1, 28, 28))).to(dev).float() / 255.0
        b = len(x)

        col_sum = x.sum(dim=1)
        col_max = col_sum.amax(dim=1, keepdim=True).clamp(min=1e-5)
        tot_mass = col_sum.sum(dim=1, keepdim=True).clamp(min=1e-5)
        cum_ratio = torch.cumsum(col_sum, dim=1)[:, 8:16] / tot_mass
        mass_pen = (torch.relu(0.16 - cum_ratio) * 4.0 + torch.relu(cum_ratio - 0.56) * 4.5) * col_max
        pos_pen = 0.07 * col_max * ((cols8_15 - 11.5) / 3.5) ** 2
        best_rel = torch.argmin(col_sum[:, 8:16] + pos_pen + mass_pen, dim=1)
        cut_col = (8 + best_rel).unsqueeze(1).unsqueeze(2).float()

        w_left = torch.sigmoid((cut_col + 0.5 - c_grid) * 2.0)
        w_right = 1.0 - w_left
        xl_p, xr_p = x * w_left, x * w_right
        ml = xl_p.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        mr = xr_p.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        cxl = (xl_p * xx).sum(dim=(1, 2), keepdim=True) / ml
        cyl = (xl_p * yy).sum(dim=(1, 2), keepdim=True) / ml
        cxr = (xr_p * xx).sum(dim=(1, 2), keepdim=True) / mr
        cyr = (xr_p * yy).sum(dim=(1, 2), keepdim=True) / mr

        mu11_w = (xl_p * (xx - cxl) * (yy - cyl) + xr_p * (xx - cxr) * (yy - cyr)).sum(dim=(1, 2))
        mu02_w = (xl_p * (yy - cyl) ** 2 + xr_p * (yy - cyr) ** 2).sum(dim=(1, 2)).clamp(min=1.0)
        true_skew = (mu11_w / mu02_w + delta_skew).clamp(-0.32, 0.32)

        theta_d = torch.zeros((b, 2, 3), dtype=torch.float32, device=dev)
        theta_d[:, 0, 0] = 1.0
        theta_d[:, 0, 1] = 0.85 * true_skew
        theta_d[:, 1, 1] = 1.0
        grid_d = F.affine_grid(theta_d, (b, 1, 28, 28), align_corners=False)
        xd = F.grid_sample(x.unsqueeze(1), grid_d, mode='bicubic', padding_mode='zeros', align_corners=False).squeeze(1)
        xd = (xd.clamp(0.0, 1.0) / 0.91).clamp(0.0, 1.0)

        col_sum_d = xd.sum(dim=1)
        col_max_d = col_sum_d.amax(dim=1, keepdim=True).clamp(min=1e-5)
        tot_mass_d = col_sum_d.sum(dim=1, keepdim=True).clamp(min=1e-5)
        cum_d = torch.cumsum(col_sum_d, dim=1)[:, 8:16] / tot_mass_d
        mass_pen_d = (torch.relu(0.16 - cum_d) * 4.0 + torch.relu(cum_d - 0.56) * 4.5) * col_max_d
        pos_pen_d = 0.07 * col_max_d * ((cols8_15 - 11.5) / 3.5) ** 2
        best_rel_d = torch.argmin(col_sum_d[:, 8:16] + pos_pen_d + mass_pen_d, dim=1)
        cut_d = ((8 + best_rel_d).float() + delta_cut).clamp(8.0, 16.0).unsqueeze(1).unsqueeze(2)
        valley_d = (col_sum_d.gather(1, (8 + best_rel_d).unsqueeze(1)) / col_max_d).squeeze(1)

        init_l = torch.sigmoid((cut_d + 1.0 - c_grid) * 1.8)
        init_r = torch.sigmoid((c_grid - (cut_d - 1.0)) * 1.8)
        xd_l0, xd_r0 = xd * init_l, xd * init_r
        ml_0 = xd_l0.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        mr_0 = xd_r0.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        cxl_0 = (xd_l0 * xx).sum(dim=(1, 2), keepdim=True) / ml_0
        cyl_0 = (xd_l0 * yy).sum(dim=(1, 2), keepdim=True) / ml_0
        cxr_0 = (xd_r0 * xx).sum(dim=(1, 2), keepdim=True) / mr_0
        cyr_0 = (xd_r0 * yy).sum(dim=(1, 2), keepdim=True) / mr_0

        dist_l = (xx.unsqueeze(0) - cxl_0) ** 2 + 0.35 * (yy.unsqueeze(0) - cyl_0) ** 2
        dist_r = (xx.unsqueeze(0) - cxr_0) ** 2 + 0.35 * (yy.unsqueeze(0) - cyr_0) ** 2
        vor_l = torch.sigmoid((dist_r - dist_l) * 0.15) * torch.sigmoid((cut_d + 2.0 - c_grid) * 1.5)
        vor_r = torch.sigmoid((dist_l - dist_r) * 0.15) * torch.sigmoid((c_grid - (cut_d - 2.0)) * 1.5)

        xd_l = xd * vor_l
        xd_r = xd * vor_r
        ml_d = xd_l.sum(dim=(1, 2)).clamp(min=1e-5)
        mr_d = xd_r.sum(dim=(1, 2)).clamp(min=1e-5)
        cxl_d = (xd_l * xx).sum(dim=(1, 2)) / ml_d
        cyl_d = (xd_l * yy).sum(dim=(1, 2)) / ml_d
        cxr_d = (xd_r * xx).sum(dim=(1, 2)) / mr_d
        cyr_d = (xd_r * yy).sum(dim=(1, 2)) / mr_d

        def sample_20(img28, cx_v, cy_v):
            gx = (cx_v.view(b, 1, 1) + grid_20_x.unsqueeze(0) - 13.5) / 13.5
            gy = (cy_v.view(b, 1, 1) + grid_20_y.unsqueeze(0) - 13.5) / 13.5
            g = torch.stack([gx, gy], dim=-1)
            cr = F.grid_sample(img28.unsqueeze(1), g, mode='bicubic', padding_mode='zeros', align_corners=False).squeeze(1)
            cr = (cr.clamp(0.0, 1.0) / 0.92).clamp(0.0, 1.0)
            return cr.reshape(b, 400)

        cl_flat = sample_20(xd_l, cxl_d, cyl_d)
        cr_flat = sample_20(xd_r, cxr_d, cyr_d)

        out_xd[start:end] = c5_snap_gray(np.round(xd.reshape(b, 784).cpu().numpy() * 255.0), levels_np)
        out_xl20[start:end] = c5_snap_gray(np.round(cl_flat.cpu().numpy() * 255.0), levels_np)
        out_xr20[start:end] = c5_snap_gray(np.round(cr_flat.cpu().numpy() * 255.0), levels_np)

        fg_cols = (xd > 0.12).any(dim=1).float().sum(dim=1) / 20.0
        g1 = ((cyl_d - cyr_d) / 10.0).clamp(-1.0, 1.0)
        g2 = ((cxr_d - cxl_d - 9.0) / 6.0).clamp(-1.0, 1.0)
        g3 = true_skew
        g4 = valley_d.clamp(0.0, 1.0)
        g5 = (ml_d / (ml_d + mr_d)).clamp(0.0, 1.0)
        g6 = fg_cols.clamp(0.0, 1.4)
        out_g6[start:end] = torch.stack([g1, g2, g3, g4, g5, g6], dim=1).cpu().numpy()

    return out_xd, out_xl20, out_xr20, out_g6


class V4Ultimate20x20HierMLP(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()
        self.coarse_embed = nn.Sequential(
            nn.Linear(100, 192), nn.LayerNorm(192), nn.GELU(),
            nn.Linear(192, 192), nn.LayerNorm(192),
        )
        self.coarse_blocks = nn.ModuleList([
            nn.ModuleDict({
                'n1': nn.LayerNorm(192),
                'tm': nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(32, 16), nn.Dropout(dropout * 0.5)),
                'n2': nn.LayerNorm(192),
                'cm': nn.Sequential(nn.Linear(192, 384), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(384, 192), nn.Dropout(dropout * 0.5)),
            }) for _ in range(2)
        ])
        self.coarse_proj = nn.Sequential(nn.Linear(192 * 3, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(dropout))

        self.fine_embed = nn.Sequential(
            nn.Linear(36, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 128), nn.LayerNorm(128),
        )
        self.fine_blocks = nn.ModuleList([
            nn.ModuleDict({
                'n1': nn.LayerNorm(128),
                'tm': nn.Sequential(nn.Linear(36, 64), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(64, 36), nn.Dropout(dropout * 0.5)),
                'n2': nn.LayerNorm(128),
                'cm': nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(256, 128), nn.Dropout(dropout * 0.5)),
            }) for _ in range(2)
        ])
        self.fine_proj = nn.Sequential(nn.Linear(128 * 3, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout))

        self.left_mlp = nn.Sequential(
            nn.Linear(400 + 36, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_tens = nn.Linear(128, 2)

        self.right_mlp = nn.Sequential(
            nn.Linear(400 + 36, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(384, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_units = nn.Linear(256, 10)

        self.holistic_mlp = nn.Sequential(
            nn.Linear(784 * 2, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
        )

        self.head_coarse = nn.Linear(384, 11)
        self.head_fine = nn.Linear(256, 11)
        self.head_holistic = nn.Linear(256, 11)

        self.coordinator = nn.Sequential(
            nn.Linear(384 + 256 + 128 + 256 + 256 + 6, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 11),
        )
        self.fft_rows = [0, 1, 2, 3, 18, 19]

    def _fft20(self, x400):
        B = x400.shape[0]
        f = torch.fft.rfft2(x400.reshape(B, 20, 20), norm='ortho').abs()
        return f[:, self.fft_rows, :6].reshape(B, 36)

    def forward(self, xo_784, xd_784, xl_400, xr_400, geo6):
        B = xd_784.shape[0]
        xd_2d = xd_784.reshape(B, 1, 28, 28)

        p_c = F.unfold(xd_2d, kernel_size=10, stride=6).transpose(1, 2)
        tc = self.coarse_embed(p_c)
        for blk in self.coarse_blocks:
            tc = tc + blk['tm'](blk['n1'](tc).transpose(1, 2)).transpose(1, 2)
            tc = tc + blk['cm'](blk['n2'](tc))
        tc_grid = tc.reshape(B, 4, 4, 192)
        fc = self.coarse_proj(torch.cat([
            tc.mean(dim=1),
            tc_grid[:, :, :2, :].mean(dim=(1, 2)),
            tc_grid[:, :, 2:, :].mean(dim=(1, 2)),
        ], dim=1))

        p_f = F.unfold(xd_2d, kernel_size=6, stride=4).transpose(1, 2)
        tf = self.fine_embed(p_f)
        for blk in self.fine_blocks:
            tf = tf + blk['tm'](blk['n1'](tf).transpose(1, 2)).transpose(1, 2)
            tf = tf + blk['cm'](blk['n2'](tf))
        tf_grid = tf.reshape(B, 6, 6, 128)
        ff = self.fine_proj(torch.cat([
            tf.mean(dim=1),
            tf_grid[:, :, :3, :].mean(dim=(1, 2)),
            tf_grid[:, :, 3:, :].mean(dim=(1, 2)),
        ], dim=1))

        fl = self.left_mlp(torch.cat([xl_400, self._fft20(xl_400)], dim=1))
        fr = self.right_mlp(torch.cat([xr_400, self._fft20(xr_400)], dim=1))
        fh = self.holistic_mlp(torch.cat([xo_784, xd_784], dim=1))

        l_c = self.head_coarse(fc)
        l_f = self.head_fine(ff)
        l_h = self.head_holistic(fh)
        l_tens = self.head_tens(fl)
        l_units = self.head_units(fr)

        l_coord = self.coordinator(torch.cat([fc, ff, fl, fr, fh, geo6], dim=1))
        return l_coord, l_c, l_f, l_h, l_tens, l_units

    def predict_proba(self, xo_784, xd_784, xl_400, xr_400, geo6):
        l_coord, l_c, l_f, l_h, l_tens, l_units = self.forward(xo_784, xd_784, xl_400, xr_400, geo6)
        p_coord = l_coord.softmax(dim=1)
        p_mix = 0.55 * l_c.softmax(dim=1) + 0.45 * l_f.softmax(dim=1)
        p_hol = l_h.softmax(dim=1)
        pt = l_tens.softmax(dim=1)
        pu = l_units.softmax(dim=1)
        p_fact = torch.cat([pt[:, 0:1] * pu[:, 0:10], pt[:, 1:2] * pu[:, 0:1]], dim=1)
        p_fact = p_fact / p_fact.sum(dim=1, keepdim=True).clamp(min=1e-7)

        # Valley-Reliability Adaptive Branch Weighting:
        # Khi khe cắt rõ (valley = geo6[:, 3] gần 0), p_fact nhận trọng số ~0.23;
        # khi 2 chữ số dính liền (valley cao), tự động giảm p_fact và tăng trọng số cho Patch-Mixer & Holistic (không cắt).
        valley = geo6[:, 3:4].clamp(0.0, 1.0)
        w_fact = 0.23 * torch.exp(-2.2 * valley)
        w_rem = 1.0 - w_fact
        return w_rem * (0.48 * p_coord + 0.32 * p_mix + 0.20 * p_hol) + w_fact * p_fact


def c5_focal_ce(logits, targets, gamma=1.75, label_smoothing=0.01):
    ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.gather(logits.softmax(dim=1), 1, targets.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
    return (((1.0 - pt) ** gamma) * ce).mean()


def c5_shift_28(x28_flat, shifts, yy28, xx28):
    dy, dx = shifts[:, 0, None, None], shifts[:, 1, None, None]
    pos = ((yy28[None] + 1 - dy) * 30 + xx28[None] + 1 - dx).reshape(len(shifts), 784)
    pad = F.pad(x28_flat.reshape(-1, 28, 28), (1, 1, 1, 1), value=0).reshape(-1, 900)
    return torch.gather(pad, 1, pos)


def c5_shift_20(x20_flat, shifts, yy20, xx20):
    dy, dx = shifts[:, 0, None, None], shifts[:, 1, None, None]
    pos = ((yy20[None] + 1 - dy) * 22 + xx20[None] + 1 - dx).reshape(len(shifts), 400)
    pad = F.pad(x20_flat.reshape(-1, 20, 20), (1, 1, 1, 1), value=0).reshape(-1, 484)
    return torch.gather(pad, 1, pos)


@torch.inference_mode()
def c5_predict_ult20_shift9(model, xo_u8, xd_u8, xl20_u8, xr20_u8, g6_f32, dev, batch_size=512):
    model.eval()
    n = len(xd_u8)
    out = []
    for j in range(0, n, batch_size):
        xo = torch.from_numpy(np.ascontiguousarray(xo_u8[j:j + batch_size].reshape(-1, 28, 28))).to(dev).float() / 255.0
        xd = torch.from_numpy(np.ascontiguousarray(xd_u8[j:j + batch_size].reshape(-1, 28, 28))).to(dev).float() / 255.0
        xl = torch.from_numpy(np.ascontiguousarray(xl20_u8[j:j + batch_size].reshape(-1, 20, 20))).to(dev).float() / 255.0
        xr = torch.from_numpy(np.ascontiguousarray(xr20_u8[j:j + batch_size].reshape(-1, 20, 20))).to(dev).float() / 255.0
        g6 = torch.from_numpy(np.ascontiguousarray(g6_f32[j:j + batch_size])).to(dev).float()

        pad_o = F.pad(xo, (1, 1, 1, 1), value=0)
        pad_d = F.pad(xd, (1, 1, 1, 1), value=0)
        pad_l = F.pad(xl, (1, 1, 1, 1), value=0)
        pad_r = F.pad(xr, (1, 1, 1, 1), value=0)

        prob = torch.zeros((len(xd), 11), dtype=torch.float32, device=dev)
        w_sum = 0.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = 1.5 if (dy == 0 and dx == 0) else (1.0 if (dy == 0 or dx == 0) else 0.75)
                vo = pad_o[:, 1 - dy:29 - dy, 1 - dx:29 - dx].reshape(-1, 784)
                vd = pad_d[:, 1 - dy:29 - dy, 1 - dx:29 - dx].reshape(-1, 784)
                vl = pad_l[:, 1 - dy:21 - dy, 1 - dx:21 - dx].reshape(-1, 400)
                vr = pad_r[:, 1 - dy:21 - dy, 1 - dx:21 - dx].reshape(-1, 400)
                prob += w * model.predict_proba(vo, vd, vl, vr, g6)
                w_sum += w
        out.append((prob / w_sum).cpu().numpy())
    res = np.concatenate(out, axis=0)
    assert res.shape == (n, 11) and np.isfinite(res).all()
    return res


@torch.inference_mode()
def c5_predict_ult20_multihyp_test(model, xo_test_u8, test_hyp_views, dev):
    prob_acc = np.zeros((len(xo_test_u8), 11), dtype=np.float32)
    w_total = 0.0
    for w_hyp, (xd_h, xl_h, xr_h, g6_h) in test_hyp_views:
        prob_acc += w_hyp * c5_predict_ult20_shift9(model, xo_test_u8, xd_h, xl_h, xr_h, g6_h, dev)
        w_total += w_hyp
    return prob_acc / w_total


def c5_metrics(prob, y, writer):
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


def c5_surgical_surplus_deficit_resolver(base_labels, p_grand22, p_hier_all, p_anchor, test_ids):
    """
    Hướng 1: Giải phẫu Ma trận Xác suất trên 550 mẫu Test:
    - Giữ nguyên 100% nền tảng 0.97272 (214/220) từ base_labels.
    - Quét toàn bộ các mẫu thuộc các lớp đang DƯ (>50 mẫu: 17, 19, 12, 15) có xác suất lớp ứng viên
      thuộc các lớp đang THIẾU (<50 mẫu: 11, 18, 13, 14).
    - Tạo:
      (A) surgical_labels: Chỉ chuyển đổi các ứng viên có bằng chứng xác suất mạnh (p_cand >= 0.30 & margin <= 0.16)
          theo đúng chiều Dư -> Thiếu.
      (B) balanced50_labels: Cân bằng có bảo vệ (Margin-Guarded OT) đưa các lớp thiếu/dư về sát 50 mẫu/lớp.
    """
    preds_surg = base_labels.copy()
    counts = {c: int((preds_surg == c).sum()) for c in range(10, 21)}
    surplus_classes = [c for c, cnt in counts.items() if cnt > 50]
    deficit_classes = [c for c, cnt in counts.items() if cnt < 50]

    # Bảng giải phẫu tất cả ứng viên có khả năng chuyển từ lớp Dư sang lớp Thiếu
    audit_candidates = []
    for i in range(len(preds_surg)):
        c_cur = int(preds_surg[i])
        if c_cur not in surplus_classes:
            continue
        p_cur = float(p_grand22[i, c_cur - 10])
        for c_def in deficit_classes:
            p_def = float(p_grand22[i, c_def - 10])
            p_h_def = float(p_hier_all[i, c_def - 10])
            p_a_def = float(p_anchor[i, c_def - 10])
            if max(p_def, p_h_def, p_a_def) >= 0.15:
                audit_candidates.append(dict(
                    row_idx=int(i),
                    id=str(test_ids[i]),
                    cur_class=c_cur,
                    target_deficit_class=c_def,
                    p_cur=round(p_cur, 4),
                    p_target_grand22=round(p_def, 4),
                    p_target_hier=round(p_h_def, 4),
                    p_target_anchor=round(p_a_def, 4),
                    margin=round(p_cur - max(p_def, p_h_def), 4),
                ))
    audit_candidates.sort(key=lambda d: d['margin'])

    # Bước phẫu thuật có kiểm soát (chỉ nhận swap Dư -> Thiếu khi margin rất hẹp <= 0.14 và xác suất lớp thiếu >= 0.28)
    surgical_swaps = []
    for cand in audit_candidates:
        i = cand['row_idx']
        c_cur = int(preds_surg[i])
        c_def = cand['target_deficit_class']
        if counts[c_cur] > 50 and counts[c_def] < 50:
            if cand['margin'] <= 0.14 and max(cand['p_target_grand22'], cand['p_target_hier'], cand['p_target_anchor']) >= 0.28:
                preds_surg[i] = c_def
                counts[c_cur] -= 1
                counts[c_def] += 1
                surgical_swaps.append(cand)

    # Tạo thêm bản Margin-Guarded Optimal Transport (cho phép lấp nốt các ca biên có xác suất >= 0.24 từ Dư sang Thiếu)
    preds_bal = preds_surg.copy()
    counts_b = {c: int((preds_bal == c).sum()) for c in range(10, 21)}
    bal_swaps = list(surgical_swaps)
    for cand in audit_candidates:
        i = cand['row_idx']
        c_cur = int(preds_bal[i])
        c_def = cand['target_deficit_class']
        if c_cur == cand['cur_class'] and counts_b[c_cur] > 50 and counts_b[c_def] < 50:
            if cand['margin'] <= 0.26 and max(cand['p_target_grand22'], cand['p_target_hier'], cand['p_target_anchor']) >= 0.22:
                preds_bal[i] = c_def
                counts_b[c_cur] -= 1
                counts_b[c_def] += 1
                bal_swaps.append(cand)

    return preds_surg, preds_bal, audit_candidates, surgical_swaps, bal_swaps


class V4C5Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, s):
        self.stream.write(s); self.file.write(s); self.file.flush()
    def flush(self):
        self.stream.flush(); self.file.flush()
    def isatty(self):
        return False


def run_v4_cell5():
    from google.colab import drive
    drive.mount('/content/drive')
    started = time.perf_counter()
    root = Path(V4_ROOT)
    run_v5 = root / 'v5_76ddef9bc184'
    data_dir = run_v5 / 'dataset_82cc33f4f691'
    cmp_dir = run_v5 / 'renderer_comparison_b409f5a57688'
    v4_home = root / 'v4_hier_multi_mlp'
    v12_dir = cmp_dir / 'standalone_fdeskew_109w_v12'
    assert (v4_home / 'submission_v4_cell4_ultimate_master.csv').exists(), 'Hãy đảm bảo Cell 4 đã chạy xong!'

    session = v4_home / 'cell5_sessions' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    session.mkdir(parents=True, exist_ok=False)
    oldout, olderr = sys.stdout, sys.stderr
    status = dict(status='started', active_pointer_changed=False)

    with open(session / 'console.log', 'w', encoding='utf-8') as log:
        sys.stdout, sys.stderr = V4C5Tee(oldout, log), V4C5Tee(olderr, log)
        try:
            assert torch.cuda.is_available(), 'Hãy chọn Runtime > Change runtime type > T4 GPU rồi chạy lại Cell 5!'
            dev = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            torch.set_num_threads(2)
            torch.set_default_dtype(torch.float32)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            print(f'v4_MNIST_10_20 — CELL 5: Hướng 1 (Giải phẫu Ma trận Xác suất) + Hướng 2 (Dual-Peak Cut + Hard-Example Focal FT) | GPU: {gpu_name}')

            # 1. Nạp dữ liệu và trích xuất Dual-Peak Cumulative-Mass Voronoi 20x20 Views
            tr = c5_npz(data_dir / 'train.npz')
            va = c5_npz(data_dir / 'validation.npz')
            st_cap = c5_npz(cmp_dir / 'strokes_capped.npz')
            vc_cap = c5_npz(cmp_dir / 'validation_capped.npz')
            cmp_cfg = c5_read(cmp_dir / 'config.json')
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
            old_writer_fold = {str(w): int(f) for w, f in zip(va['writer'], va['fold'])}

            df1 = pd.read_csv(root / 've_tay_10_20 - data.csv')
            df2 = pd.read_csv(root / 've_tay_10_20 - data_v31.csv')
            pc = [f'pixel{i}' for i in range(784)]
            new178 = df1.iloc[5452:].copy()
            v31 = df2.copy()
            v31_pix = v31[pc].to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
            for idx_r in range(len(v31)):
                fixed_im = c5_render_unclipped_raw(v31['strokes'].iloc[idx_r])
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
                new178['writer'].map(c5_writer_key).to_numpy(dtype=str),
                v31.loc[keep_v31, 'writer_name'].map(c5_writer_key).to_numpy(dtype=str)
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

            z_aug = c5_npz(v12_dir / 'new_451_strokes_and_recomb.npz')
            X_new_st, y_new_st, f_new_st = z_aug['X_st'], z_aug['y_st'], z_aug['f_st']
            X_new_rc, y_new_rc, fa_new_rc, fb_new_rc = z_aug['X_rc'], z_aug['y_rc'], z_aug['fa_rc'], z_aug['fb_rc']
            X_hold_style = c5_npz(v4_home / 'holdout_451_val_capped_style.npz')['X_style']
            z_an = c5_npz(v4_home / 'clean_anonymous_raw_and_strokes.npz')
            X_anon_raw, y_anon_raw, X_anon_st, y_anon_st = z_an['X_raw'], z_an['y_raw'], z_an['X_st'], z_an['y_st']

            test_df = pd.read_csv(root / 'test.csv', dtype={'id': str}, keep_default_na=False)
            test_ids = test_df['id'].to_numpy(dtype=str)
            X_test_orig = test_df[pc].to_numpy(dtype=np.uint8).reshape(-1, 784)

            # Chỉ giữ các nguồn dữ liệu chất lượng cao (raw + capped_strokes + recomb, loại bỏ composite ở bước Hard-FT)
            keep_idx = np.flatnonzero(clean_tr_mask & (src_tr != 3))
            X_ft_orig = np.concatenate([x_tr_orig[keep_idx], X_hold_raw, X_new_st, X_new_rc, X_anon_raw, X_anon_st], axis=0)
            y_ft = np.concatenate([y_tr[keep_idx], y_hold, y_new_st, y_new_rc, y_anon_raw, y_anon_st], axis=0)
            src_ft = np.concatenate([
                src_tr[keep_idx],
                np.zeros(len(y_hold), dtype=np.int8),
                np.ones(len(y_new_st), dtype=np.int8),
                np.full(len(y_new_rc), 2, dtype=np.int8),
                np.zeros(len(y_anon_raw), dtype=np.int8),
                np.ones(len(y_anon_st), dtype=np.int8),
            ], axis=0)
            fa_ft = np.concatenate([fa_tr[keep_idx], fold_hold, f_new_st, fa_new_rc, np.full(len(y_anon_raw) + len(y_anon_st), -1, dtype=np.int64)], axis=0)
            fb_ft = np.concatenate([fb_tr[keep_idx], fold_hold, f_new_st, fb_new_rc, np.full(len(y_anon_raw) + len(y_anon_st), -1, dtype=np.int64)], axis=0)
            del x_tr_orig

            t_pre = time.perf_counter()
            print(f'Đang trích xuất Dual-Peak Cumulative-Mass Voronoi 20x20 Views cho {len(y_ft):,} ảnh & 5 Giả thuyết TTA Test...')
            xd_ft, xl20_ft, xr20_ft, g6_ft = c5_extract_dualpeak_voronoi_20x20_gpu(X_ft_orig, levels, dev)
            xd_va_raw, xl20_va_raw, xr20_va_raw, g6_va_raw = c5_extract_dualpeak_voronoi_20x20_gpu(va['X_raw'].reshape(-1, 784), levels, dev)
            xd_va_cap, xl20_va_cap, xr20_va_cap, g6_va_cap = c5_extract_dualpeak_voronoi_20x20_gpu(vc_cap['X'].reshape(-1, 784), levels, dev)
            xd_ho_raw, xl20_ho_raw, xr20_ho_raw, g6_ho_raw = c5_extract_dualpeak_voronoi_20x20_gpu(X_hold_raw, levels, dev)
            xd_ho_cap, xl20_ho_cap, xr20_ho_cap, g6_ho_cap = c5_extract_dualpeak_voronoi_20x20_gpu(X_hold_style, levels, dev)

            test_hyp_specs = [
                (0.40, 0.00,  0.0),
                (0.15, -0.05, 0.0),
                (0.15, +0.05, 0.0),
                (0.15, 0.00, -1.0),
                (0.15, 0.00, +1.0),
            ]
            test_hyp_views = [
                (w_h, c5_extract_dualpeak_voronoi_20x20_gpu(X_test_orig, levels, dev, delta_skew=d_sk, delta_cut=d_ct))
                for w_h, d_sk, d_ct in test_hyp_specs
            ]
            print(f'Hoàn tất trích xuất Dual-Peak Voronoi 20x20 trong {time.perf_counter() - t_pre:.2f}s.')

            # 2. Nạp kết quả cache của 16 mô hình HierMultiMLP từ Cell 1, Cell 2, Cell 4 và Anchor v12/v5
            oof_s7_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_s7_cap = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_s29_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_s29_cap = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_u20_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_u20_cap = np.zeros((len(va['y']), 11), dtype=np.float32)
            ho_s7_raw = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_s7_cap = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_s29_raw = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_s29_cap = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_u20_raw = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_u20_cap = np.zeros((len(y_hold), 11), dtype=np.float32)

            te_s7_l, te_s29_l, te_u20_l, te_v12_l = [], [], [], []
            oof_v12_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
            ho_v12_raw = np.zeros((len(y_hold), 11), dtype=np.float32)

            for k in range(5):
                vi_va = np.flatnonzero(va['fold'] == k)
                vi_ho = np.flatnonzero(fold_hold == k)
                z7 = c5_npz(v4_home / f'fold_{k}' / 'hier_preds_s7.npz')
                z29 = c5_npz(v4_home / f'fold_{k}' / 'mscale_preds_s29.npz')
                z4 = c5_npz(v4_home / f'fold_{k}' / 'ult20_preds_s2026.npz')
                z12 = c5_npz(v12_dir / f'fold_{k}' / 'fold_preds.npz')

                oof_s7_raw[vi_va], oof_s7_cap[vi_va] = z7['oof_raw'], z7['oof_cap']
                oof_s29_raw[vi_va], oof_s29_cap[vi_va] = z29['oof_raw'], z29['oof_cap']
                oof_u20_raw[vi_va], oof_u20_cap[vi_va] = z4['oof_raw'], z4['oof_cap']
                ho_s7_raw[vi_ho], ho_s7_cap[vi_ho] = z7['ho_raw'], z7['ho_cap']
                ho_s29_raw[vi_ho], ho_s29_cap[vi_ho] = z29['ho_raw'], z29['ho_cap']
                ho_u20_raw[vi_ho], ho_u20_cap[vi_ho] = z4['ho_raw'], z4['ho_cap']

                oof_v12_raw[vi_va] = 0.50 * z12['oof_fd'] + 0.50 * (0.55 * z12['oof_o_s7'] + 0.45 * z12['oof_o_s29'])
                ho_v12_raw[vi_ho] = 0.50 * z12['ho_fd'] + 0.50 * (0.55 * z12['ho_o_s7'] + 0.45 * z12['ho_o_s29'])

                te_s7_l.append(z7['te_prob'])
                te_s29_l.append(z29['te_prob'])
                te_u20_l.append(z4['te_prob'])
                te_v12_l.append(0.50 * z12['te_fd'] + 0.50 * (0.55 * z12['te_o_s7'] + 0.45 * z12['te_o_s29']))

            te_s7_ens = np.mean(te_s7_l, axis=0)
            te_s29_ens = np.mean(te_s29_l, axis=0)
            te_u20_ens = np.mean(te_u20_l, axis=0)
            te_u20_full_c4 = c5_npz(v4_home / 'ult20_full_data_100pct_preds.npz')['te_prob']
            te_v12_ens = np.mean(te_v12_l, axis=0)

            # 3. Hướng 2: Tinh chỉnh Tập trung Mẫu Khó (Hard-Example Focal Fine-Tuning Stage 3) trên góc nhìn Dual-Peak 20x20
            xo_gpu = torch.from_numpy(X_ft_orig).to(dev)
            xd_gpu = torch.from_numpy(xd_ft).to(dev)
            xl_gpu = torch.from_numpy(xl20_ft).to(dev)
            xr_gpu = torch.from_numpy(xr20_ft).to(dev)
            g6_gpu = torch.from_numpy(g6_ft).to(dev)
            y11_gpu = torch.from_numpy(y_ft - 10).to(dev)
            ytens_gpu = torch.from_numpy((y_ft == 20).astype(np.int64)).to(dev)
            yunits_gpu = torch.from_numpy(np.where(y_ft == 20, 0, y_ft - 10).astype(np.int64)).to(dev)
            del X_ft_orig, xd_ft, xl20_ft, xr20_ft, g6_ft

            yy28, xx28 = torch.meshgrid(torch.arange(28, device=dev), torch.arange(28, device=dev), indexing='ij')
            yy20, xx20 = torch.meshgrid(torch.arange(20, device=dev), torch.arange(20, device=dev), indexing='ij')

            # Tập trung ưu tiên lấy mẫu thêm cho các lớp dễ nhầm lẫn (11, 13, 14, 15, 16, 17, 18, 19)
            hard_classes_mask = np.isin(y_ft, [11, 13, 14, 15, 16, 17, 18, 19])

            def run_hard_focal_ft(init_pt_path, tr_ix, seed_val, epochs=3, steps=380):
                pools = [
                    (torch.from_numpy(tr_ix[src_ft[tr_ix] == 0]).to(dev), 36),
                    (torch.from_numpy(tr_ix[src_ft[tr_ix] == 1]).to(dev), 48),
                    (torch.from_numpy(tr_ix[src_ft[tr_ix] == 2]).to(dev), 24),
                    (torch.from_numpy(tr_ix[(src_ft[tr_ix] <= 1) & hard_classes_mask[tr_ix]]).to(dev), 20),
                ]
                random.seed(seed_val); np.random.seed(seed_val); torch.manual_seed(seed_val); torch.cuda.manual_seed_all(seed_val)
                aug_rng = torch.Generator(device=dev).manual_seed(seed_val + 20000)

                model = V4Ultimate20x20HierMLP(dropout=0.28).to(dev)
                cp = torch.load(init_pt_path, map_location='cpu', weights_only=False)
                model.load_state_dict(cp['model']); del cp

                opt = torch.optim.AdamW(model.parameters(), lr=1.2e-4, weight_decay=2.5e-5)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps, eta_min=1.0e-5)
                for _ in range(epochs):
                    model.train()
                    for _ in range(steps):
                        ids = torch.cat([p[torch.randint(len(p), (c,), device=dev)] for p, c in pools])
                        ids = ids[torch.randperm(len(ids), device=dev)]
                        sh28 = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        sh20_l = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        sh20_r = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        vo = c5_shift_28(xo_gpu[ids].float() / 255.0, sh28, yy28, xx28)
                        vd = c5_shift_28(xd_gpu[ids].float() / 255.0, sh28, yy28, xx28)
                        vl = c5_shift_20(xl_gpu[ids].float() / 255.0, sh20_l, yy20, xx20)
                        vr = c5_shift_20(xr_gpu[ids].float() / 255.0, sh20_r, yy20, xx20)
                        g6 = g6_gpu[ids]
                        l_coord, l_c, l_f, l_h, l_tens, l_units = model(vo, vd, vl, vr, g6)
                        y_b = y11_gpu[ids]
                        loss = (
                            0.42 * c5_focal_ce(l_coord, y_b, gamma=1.75, label_smoothing=0.01)
                            + 0.16 * c5_focal_ce(l_c, y_b, gamma=1.50, label_smoothing=0.01)
                            + 0.16 * c5_focal_ce(l_f, y_b, gamma=1.50, label_smoothing=0.01)
                            + 0.14 * c5_focal_ce(l_h, y_b, gamma=1.50, label_smoothing=0.01)
                            + 0.20 * F.cross_entropy(l_tens, ytens_gpu[ids])
                            + 0.32 * c5_focal_ce(l_units, yunits_gpu[ids], gamma=1.50, label_smoothing=0.005)
                        )
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                return model

            oof_dp5_raw = np.full((len(va['y']), 11), np.nan, dtype=np.float32)
            oof_dp5_cap = np.full((len(va['y']), 11), np.nan, dtype=np.float32)
            ho_dp5_raw = np.full((len(y_hold), 11), np.nan, dtype=np.float32)
            ho_dp5_cap = np.full((len(y_hold), 11), np.nan, dtype=np.float32)
            te_dp5_folds = []

            for k in range(5):
                f_dir = v4_home / f'fold_{k}'
                pt_path = f_dir / 'dp20_hardft_s2027.pt'
                pred_path = f_dir / 'dp20_hardft_preds_s2027.npz'
                vi_va = np.flatnonzero(va['fold'] == k)
                vi_ho = np.flatnonzero(fold_hold == k)

                if pt_path.exists() and pred_path.exists():
                    zf = c5_npz(pred_path)
                    oof_dp5_raw[vi_va], oof_dp5_cap[vi_va] = zf['oof_raw'], zf['oof_cap']
                    ho_dp5_raw[vi_ho], ho_dp5_cap[vi_ho] = zf['ho_raw'], zf['ho_cap']
                    te_dp5_folds.append(zf['te_prob'])
                    ok_r = int(((zf['oof_raw'].argmax(1) + 10) == va['y'][vi_va]).sum())
                    ok_c = int(((zf['oof_cap'].argmax(1) + 10) == va['y'][vi_va]).sum())
                    print(f'[DualPeak-HardFT Fold {k}] Nạp cache hợp lệ | OOF Raw: {ok_r}/{len(vi_va)} | Capped_Style: {ok_c}/{len(vi_va)}')
                    continue

                t_fk = time.perf_counter()
                tr_ix = np.flatnonzero((fa_ft != k) & (fb_ft != k))
                init_pt = f_dir / 'ult20_mlp_s2026.pt'
                model = run_hard_focal_ft(init_pt, tr_ix, 47000 + 100 * k + 27, epochs=3, steps=380)

                p_va_raw = c5_predict_ult20_shift9(model, va['X_raw'][vi_va].reshape(-1, 784), xd_va_raw[vi_va], xl20_va_raw[vi_va], xr20_va_raw[vi_va], g6_va_raw[vi_va], dev)
                p_va_cap = c5_predict_ult20_shift9(model, vc_cap['X'][vi_va].reshape(-1, 784), xd_va_cap[vi_va], xl20_va_cap[vi_va], xr20_va_cap[vi_va], g6_va_cap[vi_va], dev)
                p_ho_raw = c5_predict_ult20_shift9(model, X_hold_raw[vi_ho], xd_ho_raw[vi_ho], xl20_ho_raw[vi_ho], xr20_ho_raw[vi_ho], g6_ho_raw[vi_ho], dev)
                p_ho_cap = c5_predict_ult20_shift9(model, X_hold_style[vi_ho], xd_ho_cap[vi_ho], xl20_ho_cap[vi_ho], xr20_ho_cap[vi_ho], g6_ho_cap[vi_ho], dev)
                p_te = c5_predict_ult20_multihyp_test(model, X_test_orig, test_hyp_views, dev)

                oof_dp5_raw[vi_va], oof_dp5_cap[vi_va] = p_va_raw, p_va_cap
                ho_dp5_raw[vi_ho], ho_dp5_cap[vi_ho] = p_ho_raw, p_ho_cap
                te_dp5_folds.append(p_te)

                st_cpu = {kp: vp.detach().cpu().clone() for kp, vp in model.state_dict().items()}
                torch.save({'fold': k, 'model': st_cpu}, pt_path)
                c5_save_npz(pred_path, dict(oof_raw=p_va_raw, oof_cap=p_va_cap, ho_raw=p_ho_raw, ho_cap=p_ho_cap, te_prob=p_te))
                ok_r = int(((p_va_raw.argmax(1) + 10) == va['y'][vi_va]).sum())
                ok_c = int(((p_va_cap.argmax(1) + 10) == va['y'][vi_va]).sum())
                ok_hc = int(((p_ho_cap.argmax(1) + 10) == y_hold[vi_ho]).sum())
                print(f'HOÀN TẤT [DualPeak-HardFT Fold {k}] trong {time.perf_counter() - t_fk:.1f}s | '
                      f'OOF Raw: {ok_r}/{len(vi_va)} ({ok_r/len(vi_va)*100:.2f}%) | '
                      f'OOF Capped_Style: {ok_c}/{len(vi_va)} ({ok_c/len(vi_va)*100:.2f}%) | '
                      f'NewWriters Capped: {ok_hc}/{len(vi_ho)} ({ok_hc/len(vi_ho)*100:.2f}%)')
                del model

            full_dp_pt = v4_home / 'dp20_hardft_full_100pct.pt'
            full_dp_pred = v4_home / 'dp20_hardft_full_100pct_preds.npz'
            if full_dp_pt.exists() and full_dp_pred.exists():
                te_dp5_full = c5_npz(full_dp_pred)['te_prob']
                print('[DualPeak-HardFT Full-Data 100%] Đã nạp cache hợp lệ.')
            else:
                t_fd = time.perf_counter()
                all_ix = np.arange(len(y_ft), dtype=np.int64)
                m_full = run_hard_focal_ft(v4_home / 'ult20_full_data_100pct.pt', all_ix, 47999, epochs=3, steps=420)
                te_dp5_full = c5_predict_ult20_multihyp_test(m_full, X_test_orig, test_hyp_views, dev)
                st_cpu = {kp: vp.detach().cpu().clone() for kp, vp in m_full.state_dict().items()}
                torch.save({'seed': 47999, 'model': st_cpu}, full_dp_pt)
                c5_save_npz(full_dp_pred, dict(te_prob=te_dp5_full))
                print(f'HOÀN TẤT [DualPeak-HardFT Full-Data 100%] trong {time.perf_counter() - t_fd:.1f}s.')
                del m_full

            del xo_gpu, xd_gpu, xl_gpu, xr_gpu, g6_gpu, y11_gpu, ytens_gpu, yunits_gpu
            torch.cuda.empty_cache()

            te_dp5_ens = np.mean(te_dp5_folds, axis=0)

            # 4. Tổng hợp Đại Hội Đồng 22 Mô hình Đa-MLP Phân cấp (4 Trụ cột CV + 2 Full-Data 100%)
            oof_hier20_raw = 0.25 * oof_s7_raw + 0.25 * oof_s29_raw + 0.25 * oof_u20_raw + 0.25 * oof_dp5_raw
            oof_hier20_cap = 0.25 * oof_s7_cap + 0.25 * oof_s29_cap + 0.25 * oof_u20_cap + 0.25 * oof_dp5_cap
            ho_hier20_raw = 0.25 * ho_s7_raw + 0.25 * ho_s29_raw + 0.25 * ho_u20_raw + 0.25 * ho_dp5_raw
            ho_hier20_cap = 0.25 * ho_s7_cap + 0.25 * ho_s29_cap + 0.25 * ho_u20_cap + 0.25 * ho_dp5_cap

            te_hier22 = (
                0.21 * te_s7_ens
                + 0.21 * te_s29_ens
                + 0.21 * te_u20_ens
                + 0.21 * te_dp5_ens
                + 0.08 * te_u20_full_c4
                + 0.08 * te_dp5_full
            )

            # Pha nhẹ với Neo toàn ảnh v12 qua cổng Sigmoid mượt để tạo ma trận xác suất tổng hợp p_grand22
            conf_h = te_hier22.max(axis=1, keepdims=True)
            conf_a = te_v12_ens.max(axis=1, keepdims=True)
            sig = 1.0 / (1.0 + np.exp(-22.0 * ((conf_h - conf_a) + 0.06)))
            w_h = (0.30 + 0.58 * sig).astype(np.float32)
            p_grand22 = w_h * te_hier22 + (1.0 - w_h) * te_v12_ens
            p_grand22 /= p_grand22.sum(axis=1, keepdims=True)

            # 5. Đánh giá OOF trên toàn bộ 4.932 mẫu cũ, 451 mẫu mới và 5.383 mẫu tổng hợp
            ym_oof = va['y'][valid_oof_mask]
            wm_oof = va['writer'][valid_oof_mask]
            y_all_5383 = np.concatenate([ym_oof, y_hold], axis=0)
            w_all_5383 = np.concatenate([wm_oof, w_hold], axis=0)
            p_all_cap = np.concatenate([oof_hier20_cap[valid_oof_mask], ho_hier20_cap], axis=0)
            p_all_raw = np.concatenate([oof_hier20_raw[valid_oof_mask], ho_hier20_raw], axis=0)

            summary_oof = [
                dict(split='old4932_capped_style', candidate='dp20_hardft_s2027 (Cell 5)', **c5_metrics(oof_dp5_cap[valid_oof_mask], ym_oof, wm_oof)),
                dict(split='old4932_capped_style', candidate='hier20_4pillars_ensemble',  **c5_metrics(oof_hier20_cap[valid_oof_mask], ym_oof, wm_oof)),
                dict(split='old4932_raw',          candidate='dp20_hardft_s2027 (Cell 5)', **c5_metrics(oof_dp5_raw[valid_oof_mask], ym_oof, wm_oof)),
                dict(split='old4932_raw',          candidate='hier20_4pillars_ensemble',  **c5_metrics(oof_hier20_raw[valid_oof_mask], ym_oof, wm_oof)),
                dict(split='new451_capped_style',  candidate='dp20_hardft_s2027 (Cell 5)', **c5_metrics(ho_dp5_cap, y_hold, w_hold)),
                dict(split='new451_capped_style',  candidate='hier20_4pillars_ensemble',  **c5_metrics(ho_hier20_cap, y_hold, w_hold)),
                dict(split='new451_raw',           candidate='dp20_hardft_s2027 (Cell 5)', **c5_metrics(ho_dp5_raw, y_hold, w_hold)),
                dict(split='new451_raw',           candidate='hier20_4pillars_ensemble',  **c5_metrics(ho_hier20_raw, y_hold, w_hold)),
                dict(split='all5383_capped_style', candidate='hier20_4pillars_ensemble',  **c5_metrics(p_all_cap, y_all_5383, w_all_5383)),
                dict(split='all5383_raw',          candidate='hier20_4pillars_ensemble',  **c5_metrics(p_all_raw, y_all_5383, w_all_5383)),
            ]
            print('\n================ BẢNG KIỂM ĐỊNH 22-MODEL DUAL-PEAK HARD-FT ENSEMBLE ================')
            print(pd.DataFrame(summary_oof).to_string(index=False))

            # 6. Hướng 1: Giải phẫu các mẫu thuộc 4 lớp Dư (17, 19, 12, 15) có ứng viên ở 4 lớp Thiếu (11, 18, 13, 14)
            base_c4_ult = pd.read_csv(v4_home / 'submission_v4_cell4_ultimate_master.csv')['label'].to_numpy(dtype=np.int64)
            preds_surg, preds_bal, audit_cands, surg_swaps, bal_swaps = c5_surgical_surplus_deficit_resolver(
                base_c4_ult, p_grand22, te_hier22, te_v12_ens, test_ids
            )
            preds_pure22 = te_hier22.argmax(axis=1) + 10

            print('\n================ BẢNG GIẢI PHẪU ỨNG VIÊN DƯ (17,19,12,15) -> THIẾU (11,18,13,14) ================')
            if audit_cands:
                print(pd.DataFrame(audit_cands[:20]).to_string(index=False))
            print(f'\n* Số dòng chuyển đổi ở bản Surgical Master (margin <= 0.14 & prob >= 0.28): {len(surg_swaps)} dòng -> {surg_swaps}')
            print(f'* Số dòng chuyển đổi ở bản Guarded Balanced50 (margin <= 0.26 & prob >= 0.22): {len(bal_swaps)} dòng -> {bal_swaps}')

            sub_surg_name = 'submission_v4_cell5_surgical_master.csv'
            sub_bal_name = 'submission_v4_cell5_guarded_balanced50.csv'
            sub_pure_name = 'submission_v4_cell5_pure_hier22_argmax.csv'

            for sname, slbls in [
                (sub_surg_name, preds_surg),
                (sub_bal_name, preds_bal),
                (sub_pure_name, preds_pure22),
            ]:
                sub_df = pd.DataFrame({'id': test_ids, 'label': slbls.astype(np.int64)})
                sub_df.to_csv(session / sname, index=False, lineterminator='\n')
                sub_df.to_csv(v4_home / sname, index=False, lineterminator='\n')
                print(f'* {sname}: diff_vs_Cell4_Ult(0.97272) = {int((slbls != base_c4_ult).sum())} dòng | '
                      f'class_counts = { {int(c): int((slbls == c).sum()) for c in range(10, 21)} }')

            # Lưu ma trận xác suất chi tiết của toàn bộ các trụ cột trên 550 mẫu Test để phục vụ tra cứu tức thì
            c5_save_npz(session / 'test_550_all_pillars_probs.npz', dict(
                test_ids=test_ids,
                te_s7_ens=te_s7_ens,
                te_s29_ens=te_s29_ens,
                te_u20_ens=te_u20_ens,
                te_dp5_ens=te_dp5_ens,
                te_u20_full_c4=te_u20_full_c4,
                te_dp5_full=te_dp5_full,
                te_v12_ens=te_v12_ens,
                te_hier22=te_hier22,
                p_grand22=p_grand22,
            ))

            report = dict(
                session_id=session.name,
                oof_summary=summary_oof,
                surplus_to_deficit_candidates=audit_cands[:25],
                surgical_swaps_vs_cell4=surg_swaps,
                balanced_swaps_vs_cell4=bal_swaps,
                class_counts_surgical={int(c): int((preds_surg == c).sum()) for c in range(10, 21)},
                class_counts_balanced50={int(c): int((preds_bal == c).sum()) for c in range(10, 21)},
                class_counts_pure22={int(c): int((preds_pure22 == c).sum()) for c in range(10, 21)},
            )
            c5_json(session / 'report.json', report)
            status.update(
                status='complete',
                old4932_capped_style_correct=int(summary_oof[1]['correct']),
                old4932_raw_correct=int(summary_oof[3]['correct']),
                all5383_capped_style_correct=int(summary_oof[8]['correct']),
                all5383_raw_correct=int(summary_oof[9]['correct']),
                surgical_swaps_count=len(surg_swaps),
                balanced_swaps_count=len(bal_swaps),
                elapsed_minutes=round((time.perf_counter() - started) / 60.0, 2),
            )
        except Exception as e:
            status.update(status='stopped_error', error_type=type(e).__name__, error=str(e))
            traceback.print_exc()
        finally:
            c5_json(session / 'status.json', status)
            print('\n=== GỬI LẠI LOG NÀY VÀ FILE ZIP BÁO CÁO ===')
            print(json.dumps(status, ensure_ascii=False, indent=2))
            sys.stdout, sys.stderr = oldout, olderr

    zip_path = session.parent / (session.name + '_v4_cell5_report.zip')
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


V4_CELL5_ZIP = run_v4_cell5()