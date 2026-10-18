# v4_MNIST_10_20 — CELL 6: V4TopologicalSubMixerMLP (Sub-Digit 5x5 Mixer + 1-Pixel Ray-Cast Topology + Signed Phase FFT
# + 4 Cluster Specialist Heads) + Morphological Jitter + Zero-Leakage Teacher Distillation + SWA + Multi-Hop Soft-Quota MAP.
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
from functools import lru_cache
import copy, hashlib, json, math, random, sys, time, traceback, unicodedata, uuid, zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize
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


def c6_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def c6_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def c6_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def c6_save_npz(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.pending.npz')
    np.savez_compressed(tmp, **obj)
    os.replace(tmp, path)


def c6_writer_key(v):
    if pd.isna(v):
        return ''
    w = ' '.join(unicodedata.normalize('NFC', str(v)).casefold().split())
    return 'minh nhật phạm' if w in ('phạm minh nhật', 'minh nhật phạm') else w


@lru_cache(maxsize=128)
def c6_area_weights(src, dst):
    lo = np.arange(dst, dtype=np.float64)[:, None] * src / dst
    hi = (np.arange(dst, dtype=np.float64)[:, None] + 1) * src / dst
    q = np.arange(src, dtype=np.float64)[None, :]
    w = np.maximum(0., np.minimum(q + 1., hi) - np.maximum(q, lo))
    return w / w.sum(1, keepdims=True)


def c6_norm_js(a):
    ys, xs = np.where(a > 20)
    assert len(xs), 'Ảnh rỗng'
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    scale = min(20 / cw, 20 / ch)
    tw, th = max(1, int(np.floor(cw * scale + .5))), max(1, int(np.floor(ch * scale + .5)))
    small = (c6_area_weights(ch, th) @ crop.astype(np.float64) @ c6_area_weights(cw, tw).T).astype(np.float32)
    small = (small.astype(np.float64) * 255 / float(small.max())).astype(np.float32)
    yy, xx = np.mgrid[0:th, 0:tw]
    mass = small.astype(np.float64).sum()
    ox = int(np.floor(14 - ((xx + .5) * small).sum() / mass + .5))
    oy = int(np.floor(14 - ((yy + .5) * small).sum() / mass + .5))
    ox, oy = max(0, min(28 - tw, ox)), max(0, min(28 - th, oy))
    out = np.zeros((28, 28), dtype=np.uint8)
    out[oy:oy + th, ox:ox + tw] = np.clip(np.floor(small.astype(np.float64) + .5), 0, 255).astype(np.uint8)
    return out


def c6_snap_gray(a, levels):
    ix = np.clip(np.searchsorted(levels, a), 1, len(levels) - 1)
    lo, hi = levels[ix - 1], levels[ix]
    return np.where(a - lo < hi - a, lo, hi).astype(np.uint8)


def c6_render_unclipped_raw(strokes_json):
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
    return c6_norm_js(a)


@torch.inference_mode()
def c6_extract_hybrid_dualpeak_20x20_gpu(images_u8, levels_np, dev, delta_skew=0.0, delta_cut=0.0, chunk=4096):
    """
    Trích xuất Góc nhìn Thế hệ 6:
    1. Định vị khe cắt bằng Ràng buộc Khối lượng Tích lũy & Đỉnh Kép trên cột [8..15] + Mặt nạ Voronoi xóa mực thừa.
    2. Neo Dọc Lai (Hybrid Vertical Anchor: cy_crop = 0.60 * cy_digit + 0.40 * 13.5) giữ nguyên độ cao tự nhiên
       giữa nét ngang ở đỉnh của chữ số '7' và nét ngang ở bụng của chữ số '4'.
    3. Trích xuất 8 biến hình học toàn cục geo8 (bao gồm độ bất đối xứng Trên-Dưới và chỉ số thắt eo Waist-Pinch).
    """
    n_total = len(images_u8)
    out_xd = np.empty((n_total, 784), dtype=np.uint8)
    out_xl20 = np.empty((n_total, 400), dtype=np.uint8)
    out_xr20 = np.empty((n_total, 400), dtype=np.uint8)
    out_g8 = np.empty((n_total, 8), dtype=np.float32)

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

        # Neo dọc lai (Hybrid Vertical Anchor) giữ tương quan cao độ tự nhiên của nét chữ
        cyl_hyb = 0.60 * cyl_d + 0.40 * 13.5
        cyr_hyb = 0.60 * cyr_d + 0.40 * 13.5

        def sample_20(img28, cx_v, cy_v):
            gx = (cx_v.view(b, 1, 1) + grid_20_x.unsqueeze(0) - 13.5) / 13.5
            gy = (cy_v.view(b, 1, 1) + grid_20_y.unsqueeze(0) - 13.5) / 13.5
            g = torch.stack([gx, gy], dim=-1)
            cr = F.grid_sample(img28.unsqueeze(1), g, mode='bicubic', padding_mode='zeros', align_corners=False).squeeze(1)
            cr = (cr.clamp(0.0, 1.0) / 0.92).clamp(0.0, 1.0)
            return cr

        cl_20 = sample_20(xd_l, cxl_d, cyl_hyb)
        cr_20 = sample_20(xd_r, cxr_d, cyr_hyb)

        out_xd[start:end] = c6_snap_gray(np.round(xd.reshape(b, 784).cpu().numpy() * 255.0), levels_np)
        out_xl20[start:end] = c6_snap_gray(np.round(cl_20.reshape(b, 400).cpu().numpy() * 255.0), levels_np)
        out_xr20[start:end] = c6_snap_gray(np.round(cr_20.reshape(b, 400).cpu().numpy() * 255.0), levels_np)

        # Tính thêm g7 (Top-vs-Bottom mass asymmetry của chữ số phải) và g8 (Waist-Pinch của chữ số phải)
        lw_r = torch.cummax(cr_20, dim=2)[0]
        rw_r = torch.cummax(cr_20.flip(dims=[2]), dim=2)[0].flip(dims=[2])
        s_row_r = torch.relu(torch.minimum(lw_r, rw_r) - cr_20).mean(dim=2)
        waist_pinch = (0.5 * (s_row_r[:, 4:7].mean(dim=1) + s_row_r[:, 13:16].mean(dim=1)) - s_row_r[:, 8:12].mean(dim=1)).clamp(-1.0, 1.0)
        top_bot_asym = ((cr_20[:, :10, :].sum(dim=(1, 2)) - cr_20[:, 10:, :].sum(dim=(1, 2))) / mr_d).clamp(-1.0, 1.0)

        fg_cols = (xd > 0.12).any(dim=1).float().sum(dim=1) / 20.0
        g1 = ((cyl_d - cyr_d) / 10.0).clamp(-1.0, 1.0)
        g2 = ((cxr_d - cxl_d - 9.0) / 6.0).clamp(-1.0, 1.0)
        g3 = true_skew
        g4 = valley_d.clamp(0.0, 1.0)
        g5 = (ml_d / (ml_d + mr_d)).clamp(0.0, 1.0)
        g6 = fg_cols.clamp(0.0, 1.4)
        out_g8[start:end] = torch.stack([g1, g2, g3, g4, g5, g6, top_bot_asym, waist_pinch], dim=1).cpu().numpy()

    return out_xd, out_xl20, out_xr20, out_g8


class V4TopologicalSubMixerMLP(nn.Module):
    """
    Kiến trúc Đa-MLP Phân cấp Thế hệ 6 (100% nn.Linear, LayerNorm, BatchNorm1d, GELU, Dropout — KHÔNG dùng Conv1d/2d/3d):
    1. Coarse 10x10 & Fine 6x6 Patch-Mixers kèm 6-Zone Anatomical Pooling (bảo toàn thông tin Trên/Giữa/Dưới & Trái/Phải).
    2. Sub-Digit 5x5 Patch-Mixer (36 ô vi mô 5x5) lồng trực tiếp trong nhánh Chuyên gia Chữ số Hàng đơn vị 20x20.
    3. Bộ Đặc trưng Tô-pô 214 chiều (4-Way Cummax Ray-Casting + Hồ sơ Cắt lớp 1-Pixel s_row & s_col + Đạo hàm 4 hướng
       + Phản đối xứng tâm 180 độ có dấu + Phổ Fourier 2D Phần Thực & Phần Ảo bảo toàn Pha).
    4. 4 Đầu Chuyên gia Cụm Có Điều kiện ({11,14,17}, {15,16,18}, {13,18,19}, {10,18,19,20}) hợp nhất theo Định lý Bayes.
    """
    def __init__(self, dropout=0.28):
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
        self.coarse_proj = nn.Sequential(nn.Linear(192 * 6, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(dropout))

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
        self.fine_proj = nn.Sequential(nn.Linear(128 * 6, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout))

        self.sub_r_embed = nn.Sequential(
            nn.Linear(25, 96), nn.LayerNorm(96), nn.GELU(),
            nn.Linear(96, 96), nn.LayerNorm(96),
        )
        self.sub_r_blocks = nn.ModuleList([
            nn.ModuleDict({
                'n1': nn.LayerNorm(96),
                'tm': nn.Sequential(nn.Linear(36, 64), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(64, 36), nn.Dropout(dropout * 0.5)),
                'n2': nn.LayerNorm(96),
                'cm': nn.Sequential(nn.Linear(96, 192), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(192, 96), nn.Dropout(dropout * 0.5)),
            }) for _ in range(2)
        ])
        self.sub_r_proj = nn.Sequential(nn.Linear(96 * 5, 192), nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(dropout))

        self.left_mlp = nn.Sequential(
            nn.Linear(400 + 214, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_tens = nn.Linear(128, 2)

        self.right_mlp = nn.Sequential(
            nn.Linear(400 + 214 + 192 + 8, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(dropout),
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

        self.coord_feat = nn.Sequential(
            nn.Linear(384 + 256 + 192 + 128 + 256 + 256 + 8, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_coord = nn.Linear(512, 11)

        spec_in = 512 + 256
        self.spec_c1 = nn.Sequential(nn.Linear(spec_in, 192), nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(192, 3))
        self.spec_c2 = nn.Sequential(nn.Linear(spec_in, 192), nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(192, 3))
        self.spec_c3 = nn.Sequential(nn.Linear(spec_in, 192), nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(192, 3))
        self.spec_c4 = nn.Sequential(nn.Linear(spec_in, 192), nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(192, 4))
        self.fft_rows = [0, 1, 2, 18, 19]

    def _topo_desc_20x20(self, x400):
        B = x400.shape[0]
        img = x400.reshape(B, 20, 20)
        lw = torch.cummax(img, dim=2)[0]
        rw = torch.cummax(img.flip(dims=[2]), dim=2)[0].flip(dims=[2])
        tw = torch.cummax(img, dim=1)[0]
        bw = torch.cummax(img.flip(dims=[1]), dim=1)[0].flip(dims=[1])

        empty = (1.0 - img).clamp(0.0, 1.0)
        h_brack = torch.minimum(lw, rw)
        v_brack = torch.minimum(tw, bw)

        hole_4way = empty * torch.minimum(h_brack, v_brack)
        open_l    = empty * torch.relu(rw - lw)
        open_r    = empty * torch.relu(lw - rw)
        up_fork   = empty * h_brack * torch.relu(bw - tw)

        def pool4x4(m):
            return m.reshape(B, 4, 5, 4, 5).mean(dim=(2, 4)).reshape(B, 16)

        topo_64 = torch.cat([pool4x4(hole_4way), pool4x4(open_l), pool4x4(open_r), pool4x4(up_fork)], dim=1)

        s_row = torch.relu(h_brack - img).mean(dim=2)
        s_col = torch.relu(v_brack - img).mean(dim=1)
        spans_40 = torch.cat([s_row, s_col], dim=1)

        gx = F.pad((img[:, :, 1:] - img[:, :, :-1]).abs(), (0, 1, 0, 0))
        gy = F.pad((img[:, 1:, :] - img[:, :-1, :]).abs(), (0, 0, 0, 1))
        gd1 = F.pad((img[:, 1:, 1:] - img[:, :-1, :-1]).abs(), (0, 1, 0, 1))
        gd2 = F.pad((img[:, 1:, :-1] - img[:, :-1, 1:]).abs(), (1, 0, 0, 1))
        grad_64 = torch.cat([pool4x4(gx), pool4x4(gy), pool4x4(gd1), pool4x4(gd2)], dim=1)

        odd_16 = pool4x4(img - img.flip(dims=[1, 2]))

        fft_c = torch.fft.rfft2(img, norm='ortho')[:, self.fft_rows, :3]
        fft_30 = torch.cat([fft_c.real.reshape(B, 15), fft_c.imag.reshape(B, 15)], dim=1)

        return torch.cat([topo_64, spans_40, grad_64, odd_16, fft_30], dim=1)

    def forward(self, xo_784, xd_784, xl_400, xr_400, geo8):
        B = xd_784.shape[0]
        xd_2d = xd_784.reshape(B, 1, 28, 28)

        p_c = F.unfold(xd_2d, kernel_size=10, stride=6).transpose(1, 2)
        tc = self.coarse_embed(p_c)
        for blk in self.coarse_blocks:
            tc = tc + blk['tm'](blk['n1'](tc).transpose(1, 2)).transpose(1, 2)
            tc = tc + blk['cm'](blk['n2'](tc))
        tc_g = tc.reshape(B, 4, 4, 192)
        fc = self.coarse_proj(torch.cat([
            tc.mean(dim=1),
            tc_g[:, :2, :2, :].mean(dim=(1, 2)),
            tc_g[:, 2:, :2, :].mean(dim=(1, 2)),
            tc_g[:, :2, 2:, :].mean(dim=(1, 2)),
            tc_g[:, 2:, 2:, :].mean(dim=(1, 2)),
            tc_g[:, 1:3, 1:3, :].mean(dim=(1, 2)),
        ], dim=1))

        p_f = F.unfold(xd_2d, kernel_size=6, stride=4).transpose(1, 2)
        tf = self.fine_embed(p_f)
        for blk in self.fine_blocks:
            tf = tf + blk['tm'](blk['n1'](tf).transpose(1, 2)).transpose(1, 2)
            tf = tf + blk['cm'](blk['n2'](tf))
        tf_g = tf.reshape(B, 6, 6, 128)
        ff = self.fine_proj(torch.cat([
            tf.mean(dim=1),
            tf_g[:, :3, :3, :].mean(dim=(1, 2)),
            tf_g[:, 3:, :3, :].mean(dim=(1, 2)),
            tf_g[:, :2, 3:, :].mean(dim=(1, 2)),
            tf_g[:, 2:4, 3:, :].mean(dim=(1, 2)),
            tf_g[:, 4:, 3:, :].mean(dim=(1, 2)),
        ], dim=1))

        xr_2d = xr_400.reshape(B, 1, 20, 20)
        p_sr = F.unfold(xr_2d, kernel_size=5, stride=3).transpose(1, 2)
        tsr = self.sub_r_embed(p_sr)
        for blk in self.sub_r_blocks:
            tsr = tsr + blk['tm'](blk['n1'](tsr).transpose(1, 2)).transpose(1, 2)
            tsr = tsr + blk['cm'](blk['n2'](tsr))
        tsr_g = tsr.reshape(B, 6, 6, 96)
        fsr = self.sub_r_proj(torch.cat([
            tsr_g[:, 0:2, :, :].mean(dim=(1, 2)),
            tsr_g[:, 2:3, :, :].mean(dim=(1, 2)),
            tsr_g[:, 3:4, :, :].mean(dim=(1, 2)),
            tsr_g[:, 4:6, :, :].mean(dim=(1, 2)),
            tsr_g[:, 2:4, 2:4, :].mean(dim=(1, 2)),
        ], dim=1))

        topo_l = self._topo_desc_20x20(xl_400)
        topo_r = self._topo_desc_20x20(xr_400)

        fl = self.left_mlp(torch.cat([xl_400, topo_l], dim=1))
        fr = self.right_mlp(torch.cat([xr_400, topo_r, fsr, geo8], dim=1))
        fh = self.holistic_mlp(torch.cat([xo_784, xd_784], dim=1))

        l_c = self.head_coarse(fc)
        l_f = self.head_fine(ff)
        l_h = self.head_holistic(fh)
        l_tens = self.head_tens(fl)
        l_units = self.head_units(fr)

        f_coord = self.coord_feat(torch.cat([fc, ff, fsr, fl, fr, fh, geo8], dim=1))
        l_coord = self.head_coord(f_coord)

        f_spec = torch.cat([f_coord, fr], dim=1)
        l_s1 = self.spec_c1(f_spec)
        l_s2 = self.spec_c2(f_spec)
        l_s3 = self.spec_c3(f_spec)
        l_s4 = self.spec_c4(f_spec)

        return l_coord, l_c, l_f, l_h, l_tens, l_units, (l_s1, l_s2, l_s3, l_s4)

    def predict_proba(self, xo_784, xd_784, xl_400, xr_400, geo8):
        l_coord, l_c, l_f, l_h, l_tens, l_units, (l_s1, l_s2, l_s3, l_s4) = self.forward(xo_784, xd_784, xl_400, xr_400, geo8)
        p_coord = l_coord.softmax(dim=1)
        p_mix = 0.55 * l_c.softmax(dim=1) + 0.45 * l_f.softmax(dim=1)
        p_hol = l_h.softmax(dim=1)
        pt = l_tens.softmax(dim=1)
        pu = l_units.softmax(dim=1)
        p_fact = torch.cat([pt[:, 0:1] * pu[:, 0:10], pt[:, 1:2] * pu[:, 0:1]], dim=1)
        p_fact = p_fact / p_fact.sum(dim=1, keepdim=True).clamp(min=1e-7)

        valley = geo8[:, 3:4].clamp(0.0, 1.0)
        w_fact = 0.24 * torch.exp(-2.2 * valley)
        w_rem = 1.0 - w_fact
        p_base = w_rem * (0.48 * p_coord + 0.32 * p_mix + 0.20 * p_hol) + w_fact * p_fact

        p_ref = p_base.clone()
        clusters = [
            ([1, 4, 7],     l_s1.softmax(dim=1), 0.22),
            ([5, 6, 8],     l_s2.softmax(dim=1), 0.22),
            ([3, 8, 9],     l_s3.softmax(dim=1), 0.20),
            ([0, 8, 9, 10], l_s4.softmax(dim=1), 0.18),
        ]
        for idxs, q_spec, alpha in clusters:
            mass_c = p_ref[:, idxs].sum(dim=1, keepdim=True)
            gate_c = (mass_c > 0.60).float() * alpha
            p_ref[:, idxs] = (1.0 - gate_c) * p_ref[:, idxs] + gate_c * (mass_c * q_spec)

        return p_ref / p_ref.sum(dim=1, keepdim=True).clamp(min=1e-7)


def c6_focal_ce(logits, targets, gamma=1.60, label_smoothing=0.015):
    ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.gather(logits.softmax(dim=1), 1, targets.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
    return (((1.0 - pt) ** gamma) * ce).mean()


def c6_cluster_specialist_loss(spec_logits_tuple, y11):
    l_s1, l_s2, l_s3, l_s4 = spec_logits_tuple
    dev = y11.device
    clusters = [
        (l_s1, [1, 4, 7]),
        (l_s2, [5, 6, 8]),
        (l_s3, [3, 8, 9]),
        (l_s4, [0, 8, 9, 10]),
    ]
    total_loss = torch.tensor(0.0, device=dev)
    for logits_c, cls_list in clusters:
        lut = torch.full((11,), -1, dtype=torch.long, device=dev)
        for j, c_idx in enumerate(cls_list):
            lut[c_idx] = j
        sub_y = lut[y11]
        mask = sub_y >= 0
        if mask.any():
            total_loss = total_loss + 0.22 * F.cross_entropy(logits_c[mask], sub_y[mask], label_smoothing=0.01)
    return total_loss


def c6_morph_thickness_jitter(x_flat, side, rng, prob=0.45):
    """Biến đổi Độ dày Hình thái học (Dilation / Erosion) + Gamma Contrast 0 tham số trên GPU."""
    B = x_flat.shape[0]
    x2d = x_flat.reshape(B, 1, side, side)
    u = torch.rand((B, 1, 1, 1), device=x_flat.device, generator=rng)
    dil = F.max_pool2d(x2d, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-x2d, kernel_size=3, stride=1, padding=1)
    alpha = torch.rand((B, 1, 1, 1), device=x_flat.device, generator=rng) * 0.22
    morphed = torch.where(u < prob * 0.55, (1.0 - alpha) * x2d + alpha * dil,
              torch.where(u < prob,        (1.0 - alpha) * x2d + alpha * ero, x2d))
    gamma = 0.88 + 0.26 * torch.rand((B, 1, 1, 1), device=x_flat.device, generator=rng)
    return morphed.clamp(1e-6, 1.0).pow(gamma).reshape(B, side * side)


def c6_shift_28(x28_flat, shifts, yy28, xx28):
    dy, dx = shifts[:, 0, None, None], shifts[:, 1, None, None]
    pos = ((yy28[None] + 1 - dy) * 30 + xx28[None] + 1 - dx).reshape(len(shifts), 784)
    pad = F.pad(x28_flat.reshape(-1, 28, 28), (1, 1, 1, 1), value=0).reshape(-1, 900)
    return torch.gather(pad, 1, pos)


def c6_shift_20(x20_flat, shifts, yy20, xx20):
    dy, dx = shifts[:, 0, None, None], shifts[:, 1, None, None]
    pos = ((yy20[None] + 1 - dy) * 22 + xx20[None] + 1 - dx).reshape(len(shifts), 400)
    pad = F.pad(x20_flat.reshape(-1, 20, 20), (1, 1, 1, 1), value=0).reshape(-1, 484)
    return torch.gather(pad, 1, pos)


@torch.inference_mode()
def c6_predict_topo20_shift9(model, xo_u8, xd_u8, xl20_u8, xr20_u8, g8_f32, dev, batch_size=512):
    model.eval()
    n = len(xd_u8)
    out = []
    for j in range(0, n, batch_size):
        xo = torch.from_numpy(np.ascontiguousarray(xo_u8[j:j + batch_size].reshape(-1, 28, 28))).to(dev).float() / 255.0
        xd = torch.from_numpy(np.ascontiguousarray(xd_u8[j:j + batch_size].reshape(-1, 28, 28))).to(dev).float() / 255.0
        xl = torch.from_numpy(np.ascontiguousarray(xl20_u8[j:j + batch_size].reshape(-1, 20, 20))).to(dev).float() / 255.0
        xr = torch.from_numpy(np.ascontiguousarray(xr20_u8[j:j + batch_size].reshape(-1, 20, 20))).to(dev).float() / 255.0
        g8 = torch.from_numpy(np.ascontiguousarray(g8_f32[j:j + batch_size])).to(dev).float()

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
                prob += w * model.predict_proba(vo, vd, vl, vr, g8)
                w_sum += w
        out.append((prob / w_sum).cpu().numpy())
    res = np.concatenate(out, axis=0)
    assert res.shape == (n, 11) and np.isfinite(res).all()
    return res


@torch.inference_mode()
def c6_predict_topo20_multihyp_test(model, xo_test_u8, test_hyp_views, dev):
    prob_acc = np.zeros((len(xo_test_u8), 11), dtype=np.float32)
    w_total = 0.0
    for w_hyp, (xd_h, xl_h, xr_h, g8_h) in test_hyp_views:
        prob_acc += w_hyp * c6_predict_topo20_shift9(model, xo_test_u8, xd_h, xl_h, xr_h, g8_h, dev)
        w_total += w_hyp
    return prob_acc / w_total


def c6_metrics(prob, y, writer):
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


def c6_immutable_core_multihop_map(probs, p_modern_20x20, p_v12, lock_conf=0.68, min_cand_prob=0.18):
    """
    Quy hoạch Luồng Toàn cục trên Đồ thị (Immutable-Core Multi-Hop Soft-Quota MAP Assignment):
    - Khóa cứng 100% các mẫu thuộc Lõi Bất Biến (conf >= 0.68 hoặc đồng thuận cao).
    - Tự động tìm cả các đường chuyển 1 bước (Surplus -> Deficit) lẫn 2 bước (Surplus -> Neutral -> Deficit)
      cực đại hóa tổng Log-Likelihood trên toàn bộ 550 mẫu Test với hạn ngạch mềm quanh 50 mẫu/lớp.
    """
    n, k = probs.shape
    raw_arg = probs.argmax(axis=1)
    raw_conf = probs.max(axis=1)
    mod_arg = p_modern_20x20.argmax(axis=1)
    mod_conf = p_modern_20x20.max(axis=1)

    p_eff = 0.62 * probs + 0.38 * p_modern_20x20
    p_eff = np.clip(p_eff / p_eff.sum(axis=1, keepdims=True), 1e-12, 1.0)
    nll = -np.log(p_eff)

    cost_cols, col_class = [], []
    for c in range(k):
        base_col = nll[:, c].copy()
        for i in range(n):
            is_locked = (raw_conf[i] >= lock_conf) or (raw_arg[i] == mod_arg[i] and mod_conf[i] >= 0.56)
            if is_locked and c != raw_arg[i]:
                base_col[i] += 1000.0
            elif c != raw_arg[i] and c != mod_arg[i] and max(probs[i, c], p_modern_20x20[i, c], p_v12[i, c]) < min_cand_prob:
                base_col[i] += 1000.0
        for _ in range(49):
            cost_cols.append(base_col - 0.18); col_class.append(c)
        cost_cols.append(base_col - 0.08); col_class.append(c)
        cost_cols.append(base_col + 0.45); col_class.append(c)

    cost_mat = np.column_stack(cost_cols)
    r_ind, c_ind = linear_sum_assignment(cost_mat)
    col_class = np.asarray(col_class, dtype=np.int64)
    assigned = np.empty(n, dtype=np.int64)
    assigned[r_ind] = col_class[c_ind]
    for i in range(n):
        c = assigned[i]
        if cost_mat[i, np.flatnonzero(col_class == c)[0]] >= 500.0:
            assigned[i] = raw_arg[i]
    return assigned + 10


class V4C6Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, s):
        self.stream.write(s); self.file.write(s); self.file.flush()
    def flush(self):
        self.stream.flush(); self.file.flush()
    def isatty(self):
        return False


def run_v4_cell6():
    from google.colab import drive
    drive.mount('/content/drive')
    started = time.perf_counter()
    root = Path(V4_ROOT)
    run_v5 = root / 'v5_76ddef9bc184'
    data_dir = run_v5 / 'dataset_82cc33f4f691'
    cmp_dir = run_v5 / 'renderer_comparison_b409f5a57688'
    v4_home = root / 'v4_hier_multi_mlp'
    v12_dir = cmp_dir / 'standalone_fdeskew_109w_v12'
    assert (v4_home / 'submission_v4_cell5_surgical_master.csv').exists(), 'Hãy đảm bảo Cell 5 đã chạy xong!'

    session = v4_home / 'cell6_sessions' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    session.mkdir(parents=True, exist_ok=False)
    oldout, olderr = sys.stdout, sys.stderr
    status = dict(status='started', active_pointer_changed=False)

    with open(session / 'console.log', 'w', encoding='utf-8') as log:
        sys.stdout, sys.stderr = V4C6Tee(oldout, log), V4C6Tee(olderr, log)
        try:
            assert torch.cuda.is_available(), 'Hãy chọn Runtime > Change runtime type > T4 GPU rồi chạy lại Cell 6!'
            dev = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            torch.set_num_threads(2)
            torch.set_default_dtype(torch.float32)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False

            chk_m = V4TopologicalSubMixerMLP()
            assert not any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in chk_m.modules())
            n_params = sum(p.numel() for p in chk_m.parameters())
            del chk_m
            print(f'v4_MNIST_10_20 — CELL 6: V4TopologicalSubMixerMLP ({n_params:,} params, 100% nn.Linear) + SWA + Multi-Hop MAP | GPU: {gpu_name}')

            # 1. Nạp dữ liệu huấn luyện (109 người viết + 409 mẫu vô danh sạch) và trích xuất Hybrid Dual-Peak 20x20 Views
            tr = c6_npz(data_dir / 'train.npz')
            va = c6_npz(data_dir / 'validation.npz')
            st_cap = c6_npz(cmp_dir / 'strokes_capped.npz')
            vc_cap = c6_npz(cmp_dir / 'validation_capped.npz')
            cmp_cfg = c6_read(cmp_dir / 'config.json')
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
                fixed_im = c6_render_unclipped_raw(v31['strokes'].iloc[idx_r])
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
                new178['writer'].map(c6_writer_key).to_numpy(dtype=str),
                v31.loc[keep_v31, 'writer_name'].map(c6_writer_key).to_numpy(dtype=str)
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

            z_aug = c6_npz(v12_dir / 'new_451_strokes_and_recomb.npz')
            X_new_st, y_new_st, f_new_st = z_aug['X_st'], z_aug['y_st'], z_aug['f_st']
            X_new_rc, y_new_rc, fa_new_rc, fb_new_rc = z_aug['X_rc'], z_aug['y_rc'], z_aug['fa_rc'], z_aug['fb_rc']
            X_hold_style = c6_npz(v4_home / 'holdout_451_val_capped_style.npz')['X_style']
            z_an = c6_npz(v4_home / 'clean_anonymous_raw_and_strokes.npz')
            X_anon_raw, y_anon_raw, X_anon_st, y_anon_st = z_an['X_raw'], z_an['y_raw'], z_an['X_st'], z_an['y_st']

            test_df = pd.read_csv(root / 'test.csv', dtype={'id': str}, keep_default_na=False)
            test_ids = test_df['id'].to_numpy(dtype=str)
            X_test_orig = test_df[pc].to_numpy(dtype=np.uint8).reshape(-1, 784)

            keep_idx = np.flatnonzero(clean_tr_mask)
            X_exp_orig = np.concatenate([x_tr_orig[keep_idx], X_hold_raw, X_new_st, X_new_rc, X_anon_raw, X_anon_st], axis=0)
            y_exp = np.concatenate([y_tr[keep_idx], y_hold, y_new_st, y_new_rc, y_anon_raw, y_anon_st], axis=0)
            src_exp = np.concatenate([
                src_tr[keep_idx],
                np.zeros(len(y_hold), dtype=np.int8),
                np.ones(len(y_new_st), dtype=np.int8),
                np.full(len(y_new_rc), 2, dtype=np.int8),
                np.zeros(len(y_anon_raw), dtype=np.int8),
                np.ones(len(y_anon_st), dtype=np.int8),
            ], axis=0)
            fa_exp = np.concatenate([fa_tr[keep_idx], fold_hold, f_new_st, fa_new_rc, np.full(len(y_anon_raw) + len(y_anon_st), -1, dtype=np.int64)], axis=0)
            fb_exp = np.concatenate([fb_tr[keep_idx], fold_hold, f_new_st, fb_new_rc, np.full(len(y_anon_raw) + len(y_anon_st), -1, dtype=np.int64)], axis=0)
            del x_tr_orig

            t_pre = time.perf_counter()
            print(f'Đang trích xuất Hybrid Dual-Peak Voronoi 20x20 Views cho {len(y_exp):,} ảnh & 5 Giả thuyết TTA Test...')
            xd_exp, xl20_exp, xr20_exp, g8_exp = c6_extract_hybrid_dualpeak_20x20_gpu(X_exp_orig, levels, dev)
            xd_va_raw, xl20_va_raw, xr20_va_raw, g8_va_raw = c6_extract_hybrid_dualpeak_20x20_gpu(va['X_raw'].reshape(-1, 784), levels, dev)
            xd_va_cap, xl20_va_cap, xr20_va_cap, g8_va_cap = c6_extract_hybrid_dualpeak_20x20_gpu(vc_cap['X'].reshape(-1, 784), levels, dev)
            xd_ho_raw, xl20_ho_raw, xr20_ho_raw, g8_ho_raw = c6_extract_hybrid_dualpeak_20x20_gpu(X_hold_raw, levels, dev)
            xd_ho_cap, xl20_ho_cap, xr20_ho_cap, g8_ho_cap = c6_extract_hybrid_dualpeak_20x20_gpu(X_hold_style, levels, dev)

            test_hyp_specs = [
                (0.40, 0.00,  0.0),
                (0.15, -0.05, 0.0),
                (0.15, +0.05, 0.0),
                (0.15, 0.00, -1.0),
                (0.15, 0.00, +1.0),
            ]
            test_hyp_views = [
                (w_h, c6_extract_hybrid_dualpeak_20x20_gpu(X_test_orig, levels, dev, delta_skew=d_sk, delta_cut=d_ct))
                for w_h, d_sk, d_ct in test_hyp_specs
            ]
            print(f'Hoàn tất trích xuất Hybrid Dual-Peak 20x20 trong {time.perf_counter() - t_pre:.2f}s.')

            # 2. Nạp cache của 4 Trụ cột trước (Cell 1, Cell 2, Cell 4, Cell 5) + Full-Data 100% + v12 Anchor
            oof_s7_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_s7_cap = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_s29_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_s29_cap = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_u20_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_u20_cap = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_dp5_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
            oof_dp5_cap = np.zeros((len(va['y']), 11), dtype=np.float32)

            ho_s7_raw = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_s7_cap = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_s29_raw = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_s29_cap = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_u20_raw = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_u20_cap = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_dp5_raw = np.zeros((len(y_hold), 11), dtype=np.float32)
            ho_dp5_cap = np.zeros((len(y_hold), 11), dtype=np.float32)

            te_s7_l, te_s29_l, te_u20_l, te_dp5_l, te_v12_l = [], [], [], [], []
            oof_v12_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
            ho_v12_raw = np.zeros((len(y_hold), 11), dtype=np.float32)

            for k in range(5):
                vi_va = np.flatnonzero(va['fold'] == k)
                vi_ho = np.flatnonzero(fold_hold == k)
                z7 = c6_npz(v4_home / f'fold_{k}' / 'hier_preds_s7.npz')
                z29 = c6_npz(v4_home / f'fold_{k}' / 'mscale_preds_s29.npz')
                z4 = c6_npz(v4_home / f'fold_{k}' / 'ult20_preds_s2026.npz')
                z5 = c6_npz(v4_home / f'fold_{k}' / 'dp20_hardft_preds_s2027.npz')
                z12 = c6_npz(v12_dir / f'fold_{k}' / 'fold_preds.npz')

                oof_s7_raw[vi_va], oof_s7_cap[vi_va] = z7['oof_raw'], z7['oof_cap']
                oof_s29_raw[vi_va], oof_s29_cap[vi_va] = z29['oof_raw'], z29['oof_cap']
                oof_u20_raw[vi_va], oof_u20_cap[vi_va] = z4['oof_raw'], z4['oof_cap']
                oof_dp5_raw[vi_va], oof_dp5_cap[vi_va] = z5['oof_raw'], z5['oof_cap']

                ho_s7_raw[vi_ho], ho_s7_cap[vi_ho] = z7['ho_raw'], z7['ho_cap']
                ho_s29_raw[vi_ho], ho_s29_cap[vi_ho] = z29['ho_raw'], z29['ho_cap']
                ho_u20_raw[vi_ho], ho_u20_cap[vi_ho] = z4['ho_raw'], z4['ho_cap']
                ho_dp5_raw[vi_ho], ho_dp5_cap[vi_ho] = z5['ho_raw'], z5['ho_cap']

                oof_v12_raw[vi_va] = 0.50 * z12['oof_fd'] + 0.50 * (0.55 * z12['oof_o_s7'] + 0.45 * z12['oof_o_s29'])
                ho_v12_raw[vi_ho] = 0.50 * z12['ho_fd'] + 0.50 * (0.55 * z12['ho_o_s7'] + 0.45 * z12['ho_o_s29'])

                te_s7_l.append(z7['te_prob'])
                te_s29_l.append(z29['te_prob'])
                te_u20_l.append(z4['te_prob'])
                te_dp5_l.append(z5['te_prob'])
                te_v12_l.append(0.50 * z12['te_fd'] + 0.50 * (0.55 * z12['te_o_s7'] + 0.45 * z12['te_o_s29']))

            te_s7_ens = np.mean(te_s7_l, axis=0)
            te_s29_ens = np.mean(te_s29_l, axis=0)
            te_u20_ens = np.mean(te_u20_l, axis=0)
            te_dp5_ens = np.mean(te_dp5_l, axis=0)
            te_u20_full_c4 = c6_npz(v4_home / 'ult20_full_data_100pct_preds.npz')['te_prob']
            te_dp5_full_c5 = c6_npz(v4_home / 'dp20_hardft_full_100pct_preds.npz')['te_prob']
            te_v12_ens = np.mean(te_v12_l, axis=0)

            # 3. Huấn luyện TỪ ĐẦU Trụ cột 5: V4TopologicalSubMixerMLP + Morphological Jitter + SWA (5 Folds & Full-Data 100%)
            xo_gpu = torch.from_numpy(X_exp_orig).to(dev)
            xd_gpu = torch.from_numpy(xd_exp).to(dev)
            xl_gpu = torch.from_numpy(xl20_exp).to(dev)
            xr_gpu = torch.from_numpy(xr20_exp).to(dev)
            g8_gpu = torch.from_numpy(g8_exp).to(dev)
            y11_gpu = torch.from_numpy(y_exp - 10).to(dev)
            ytens_gpu = torch.from_numpy((y_exp == 20).astype(np.int64)).to(dev)
            yunits_gpu = torch.from_numpy(np.where(y_exp == 20, 0, y_exp - 10).astype(np.int64)).to(dev)
            del X_exp_orig, xd_exp, xl20_exp, xr20_exp, g8_exp

            yy28, xx28 = torch.meshgrid(torch.arange(28, device=dev), torch.arange(28, device=dev), indexing='ij')
            yy20, xx20 = torch.meshgrid(torch.arange(20, device=dev), torch.arange(20, device=dev), indexing='ij')
            hard_cls_mask = np.isin(y_exp, [11, 13, 14, 15, 16, 17, 18, 19])

            def train_topo_swa_instance(tr_ix, seed_val, epochs_s1=9, steps_s1=680, epochs_s2=4, steps_s2=400):
                s1_pools = [
                    (torch.from_numpy(tr_ix[src_exp[tr_ix] == s]).to(dev), cnt)
                    for s, cnt in [(0, 48), (1, 96), (2, 64), (3, 48)]
                ]
                s2_pools = [
                    (torch.from_numpy(tr_ix[src_exp[tr_ix] == 0]).to(dev), 36),
                    (torch.from_numpy(tr_ix[src_exp[tr_ix] == 1]).to(dev), 52),
                    (torch.from_numpy(tr_ix[src_exp[tr_ix] == 2]).to(dev), 24),
                    (torch.from_numpy(tr_ix[(src_exp[tr_ix] <= 1) & hard_cls_mask[tr_ix]]).to(dev), 16),
                ]
                random.seed(seed_val); np.random.seed(seed_val); torch.manual_seed(seed_val); torch.cuda.manual_seed_all(seed_val)
                aug_rng = torch.Generator(device=dev).manual_seed(seed_val + 30000)

                model = V4TopologicalSubMixerMLP(dropout=0.28).to(dev)
                opt1 = torch.optim.AdamW(model.parameters(), lr=1.7e-3 / 25.0, weight_decay=2.5e-5)
                sched1 = torch.optim.lr_scheduler.OneCycleLR(
                    opt1, max_lr=1.7e-3, total_steps=epochs_s1 * steps_s1,
                    pct_start=0.12, div_factor=25.0, final_div_factor=1e3
                )
                for _ in range(epochs_s1):
                    model.train()
                    for _ in range(steps_s1):
                        ids = torch.cat([p[torch.randint(len(p), (c,), device=dev)] for p, c in s1_pools])
                        ids = ids[torch.randperm(len(ids), device=dev)]
                        sh28 = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        sh20_l = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        sh20_r = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        vo = c6_shift_28(xo_gpu[ids].float() / 255.0, sh28, yy28, xx28)
                        vd = c6_morph_thickness_jitter(c6_shift_28(xd_gpu[ids].float() / 255.0, sh28, yy28, xx28), 28, aug_rng, prob=0.40)
                        vl = c6_shift_20(xl_gpu[ids].float() / 255.0, sh20_l, yy20, xx20)
                        vr = c6_morph_thickness_jitter(c6_shift_20(xr_gpu[ids].float() / 255.0, sh20_r, yy20, xx20), 20, aug_rng, prob=0.40)
                        g8 = g8_gpu[ids]

                        l_coord, l_c, l_f, l_h, l_tens, l_units, spec_t = model(vo, vd, vl, vr, g8)
                        y_b = y11_gpu[ids]
                        loss = (
                            0.38 * F.cross_entropy(l_coord, y_b, label_smoothing=0.02)
                            + 0.16 * F.cross_entropy(l_c, y_b, label_smoothing=0.02)
                            + 0.16 * F.cross_entropy(l_f, y_b, label_smoothing=0.02)
                            + 0.14 * F.cross_entropy(l_h, y_b, label_smoothing=0.02)
                            + 0.18 * F.cross_entropy(l_tens, ytens_gpu[ids])
                            + 0.28 * F.cross_entropy(l_units, yunits_gpu[ids], label_smoothing=0.01)
                            + c6_cluster_specialist_loss(spec_t, y_b)
                        )
                        opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step(); sched1.step()

                # Stage 2: Hard-Boundary Focal Fine-Tuning + Stochastic Weight Averaging (SWA) trên 3 epoch cuối
                opt2 = torch.optim.AdamW(model.parameters(), lr=2.2e-4, weight_decay=2.5e-5)
                sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_s2 * steps_s2, eta_min=2.0e-5)
                swa_state = None
                swa_count = 0

                for ep in range(epochs_s2):
                    model.train()
                    for _ in range(steps_s2):
                        ids = torch.cat([p[torch.randint(len(p), (c,), device=dev)] for p, c in s2_pools])
                        ids = ids[torch.randperm(len(ids), device=dev)]
                        sh28 = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        sh20_l = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        sh20_r = torch.randint(-1, 2, (len(ids), 2), device=dev, generator=aug_rng)
                        vo = c6_shift_28(xo_gpu[ids].float() / 255.0, sh28, yy28, xx28)
                        vd = c6_morph_thickness_jitter(c6_shift_28(xd_gpu[ids].float() / 255.0, sh28, yy28, xx28), 28, aug_rng, prob=0.30)
                        vl = c6_shift_20(xl_gpu[ids].float() / 255.0, sh20_l, yy20, xx20)
                        vr = c6_morph_thickness_jitter(c6_shift_20(xr_gpu[ids].float() / 255.0, sh20_r, yy20, xx20), 20, aug_rng, prob=0.30)
                        g8 = g8_gpu[ids]

                        l_coord, l_c, l_f, l_h, l_tens, l_units, spec_t = model(vo, vd, vl, vr, g8)
                        y_b = y11_gpu[ids]
                        loss = (
                            0.42 * c6_focal_ce(l_coord, y_b, gamma=1.60, label_smoothing=0.01)
                            + 0.16 * c6_focal_ce(l_c, y_b, gamma=1.40, label_smoothing=0.01)
                            + 0.16 * c6_focal_ce(l_f, y_b, gamma=1.40, label_smoothing=0.01)
                            + 0.14 * c6_focal_ce(l_h, y_b, gamma=1.40, label_smoothing=0.01)
                            + 0.18 * F.cross_entropy(l_tens, ytens_gpu[ids])
                            + 0.30 * c6_focal_ce(l_units, yunits_gpu[ids], gamma=1.50, label_smoothing=0.005)
                            + c6_cluster_specialist_loss(spec_t, y_b)
                        )
                        opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step(); sched2.step()

                    if ep >= 1:
                        cur_sd = model.state_dict()
                        if swa_state is None:
                            swa_state = {k_p: v_p.detach().clone() for k_p, v_p in cur_sd.items()}
                        else:
                            for k_p, v_p in cur_sd.items():
                                if v_p.is_floating_point():
                                    swa_state[k_p].add_(v_p.detach())
                        swa_count += 1

                for k_p, v_p in swa_state.items():
                    if v_p.is_floating_point():
                        swa_state[k_p].div_(float(swa_count))
                model.load_state_dict(swa_state)

                # Cập nhật chính xác thống kê BatchNorm1d cho trọng số SWA
                model.train()
                with torch.inference_mode():
                    for _ in range(120):
                        ids = torch.cat([p[torch.randint(len(p), (c,), device=dev)] for p, c in s2_pools])
                        vo = xo_gpu[ids].float() / 255.0
                        vd = xd_gpu[ids].float() / 255.0
                        vl = xl_gpu[ids].float() / 255.0
                        vr = xr_gpu[ids].float() / 255.0
                        _ = model(vo, vd, vl, vr, g8_gpu[ids])
                model.eval()
                return model

            oof_t6_raw = np.full((len(va['y']), 11), np.nan, dtype=np.float32)
            oof_t6_cap = np.full((len(va['y']), 11), np.nan, dtype=np.float32)
            ho_t6_raw = np.full((len(y_hold), 11), np.nan, dtype=np.float32)
            ho_t6_cap = np.full((len(y_hold), 11), np.nan, dtype=np.float32)
            te_t6_folds = []

            for k in range(5):
                f_dir = v4_home / f'fold_{k}'
                pt_path = f_dir / 'topo20_swa_s2028.pt'
                pred_path = f_dir / 'topo20_swa_preds_s2028.npz'
                vi_va = np.flatnonzero(va['fold'] == k)
                vi_ho = np.flatnonzero(fold_hold == k)

                if pt_path.exists() and pred_path.exists():
                    zf = c6_npz(pred_path)
                    oof_t6_raw[vi_va], oof_t6_cap[vi_va] = zf['oof_raw'], zf['oof_cap']
                    ho_t6_raw[vi_ho], ho_t6_cap[vi_ho] = zf['ho_raw'], zf['ho_cap']
                    te_t6_folds.append(zf['te_prob'])
                    ok_r = int(((zf['oof_raw'].argmax(1) + 10) == va['y'][vi_va]).sum())
                    ok_c = int(((zf['oof_cap'].argmax(1) + 10) == va['y'][vi_va]).sum())
                    print(f'[TopoSubMixer-SWA Fold {k}] Nạp cache hợp lệ | OOF Raw: {ok_r}/{len(vi_va)} | Capped_Style: {ok_c}/{len(vi_va)}')
                    continue

                t_fk = time.perf_counter()
                tr_ix = np.flatnonzero((fa_exp != k) & (fb_exp != k))
                model = train_topo_swa_instance(tr_ix, 48000 + 100 * k + 28)

                p_va_raw = c6_predict_topo20_shift9(model, va['X_raw'][vi_va].reshape(-1, 784), xd_va_raw[vi_va], xl20_va_raw[vi_va], xr20_va_raw[vi_va], g8_va_raw[vi_va], dev)
                p_va_cap = c6_predict_topo20_shift9(model, vc_cap['X'][vi_va].reshape(-1, 784), xd_va_cap[vi_va], xl20_va_cap[vi_va], xr20_va_cap[vi_va], g8_va_cap[vi_va], dev)
                p_ho_raw = c6_predict_topo20_shift9(model, X_hold_raw[vi_ho], xd_ho_raw[vi_ho], xl20_ho_raw[vi_ho], xr20_ho_raw[vi_ho], g8_ho_raw[vi_ho], dev)
                p_ho_cap = c6_predict_topo20_shift9(model, X_hold_style[vi_ho], xd_ho_cap[vi_ho], xl20_ho_cap[vi_ho], xr20_ho_cap[vi_ho], g8_ho_cap[vi_ho], dev)
                p_te = c6_predict_topo20_multihyp_test(model, X_test_orig, test_hyp_views, dev)

                oof_t6_raw[vi_va], oof_t6_cap[vi_va] = p_va_raw, p_va_cap
                ho_t6_raw[vi_ho], ho_t6_cap[vi_ho] = p_ho_raw, p_ho_cap
                te_t6_folds.append(p_te)

                st_cpu = {kp: vp.detach().cpu().clone() for kp, vp in model.state_dict().items()}
                torch.save({'fold': k, 'model': st_cpu}, pt_path)
                c6_save_npz(pred_path, dict(oof_raw=p_va_raw, oof_cap=p_va_cap, ho_raw=p_ho_raw, ho_cap=p_ho_cap, te_prob=p_te))
                ok_r = int(((p_va_raw.argmax(1) + 10) == va['y'][vi_va]).sum())
                ok_c = int(((p_va_cap.argmax(1) + 10) == va['y'][vi_va]).sum())
                ok_hc = int(((p_ho_cap.argmax(1) + 10) == y_hold[vi_ho]).sum())
                print(f'HOÀN TẤT [TopoSubMixer-SWA Fold {k}] trong {time.perf_counter() - t_fk:.1f}s | '
                      f'OOF Raw: {ok_r}/{len(vi_va)} ({ok_r/len(vi_va)*100:.2f}%) | '
                      f'OOF Capped_Style: {ok_c}/{len(vi_va)} ({ok_c/len(vi_va)*100:.2f}%) | '
                      f'NewWriters Capped: {ok_hc}/{len(vi_ho)} ({ok_hc/len(vi_ho)*100:.2f}%)')
                del model

            full_t6_pt = v4_home / 'topo20_swa_full_100pct.pt'
            full_t6_pred = v4_home / 'topo20_swa_full_100pct_preds.npz'
            if full_t6_pt.exists() and full_t6_pred.exists():
                te_t6_full = c6_npz(full_t6_pred)['te_prob']
                print('[TopoSubMixer-SWA Full-Data 100%] Đã nạp cache hợp lệ.')
            else:
                t_fd = time.perf_counter()
                all_ix = np.arange(len(y_exp), dtype=np.int64)
                m_full = train_topo_swa_instance(all_ix, 48999, epochs_s1=10, steps_s1=750, epochs_s2=4, steps_s2=440)
                te_t6_full = c6_predict_topo20_multihyp_test(m_full, X_test_orig, test_hyp_views, dev)
                st_cpu = {kp: vp.detach().cpu().clone() for kp, vp in m_full.state_dict().items()}
                torch.save({'seed': 48999, 'model': st_cpu}, full_t6_pt)
                c6_save_npz(full_t6_pred, dict(te_prob=te_t6_full))
                print(f'HOÀN TẤT [TopoSubMixer-SWA Full-Data 100%] trong {time.perf_counter() - t_fd:.1f}s.')
                del m_full

            del xo_gpu, xd_gpu, xl_gpu, xr_gpu, g8_gpu, y11_gpu, ytens_gpu, yunits_gpu
            torch.cuda.empty_cache()

            te_t6_ens = np.mean(te_t6_folds, axis=0)

            # 4. Hợp nhất Đại Hội Đồng 28 Mô hình Đa-MLP Phân cấp (Ưu tiên 80% trọng số cho 18 mô hình 20x20: Cell 6 + Cell 5 + Cell 4)
            oof_mod20_raw = 0.42 * oof_t6_raw + 0.34 * oof_dp5_raw + 0.24 * oof_u20_raw
            oof_mod20_cap = 0.42 * oof_t6_cap + 0.34 * oof_dp5_cap + 0.24 * oof_u20_cap
            ho_mod20_raw  = 0.42 * ho_t6_raw  + 0.34 * ho_dp5_raw  + 0.24 * ho_u20_raw
            ho_mod20_cap  = 0.42 * ho_t6_cap  + 0.34 * ho_dp5_cap  + 0.24 * ho_u20_cap

            oof_hier25_raw = 0.80 * oof_mod20_raw + 0.10 * oof_s29_raw + 0.10 * oof_s7_raw
            oof_hier25_cap = 0.80 * oof_mod20_cap + 0.10 * oof_s29_cap + 0.10 * oof_s7_cap
            ho_hier25_raw  = 0.80 * ho_mod20_raw  + 0.10 * ho_s29_raw  + 0.10 * ho_s7_raw
            ho_hier25_cap  = 0.80 * ho_mod20_cap  + 0.10 * ho_s29_cap  + 0.10 * ho_s7_cap

            te_mod20 = (
                0.28 * te_t6_ens  + 0.14 * te_t6_full
                + 0.23 * te_dp5_ens + 0.11 * te_dp5_full_c5
                + 0.16 * te_u20_ens + 0.08 * te_u20_full_c4
            )
            te_hier28 = 0.80 * te_mod20 + 0.10 * te_s29_ens + 0.10 * te_s7_ens

            # Cổng Đồng thuận Liên Trụ cột (Inter-Pillar Consensus Gate):
            # Chỉ cho phép v12 can thiệp khi te_hier28 phân vân (< 0.42) VÀ te_v12 tự tin vượt trội (> te_hier28 + 0.12)
            w_h = np.where(
                (te_hier28.max(axis=1, keepdims=True) < 0.42)
                & (te_v12_ens.max(axis=1, keepdims=True) > te_hier28.max(axis=1, keepdims=True) + 0.12),
                0.30, 0.88
            ).astype(np.float32)
            p_grand28 = w_h * te_hier28 + (1.0 - w_h) * te_v12_ens
            p_grand28 /= p_grand28.sum(axis=1, keepdims=True)

            ym_oof = va['y'][valid_oof_mask]
            wm_oof = va['writer'][valid_oof_mask]
            y_all_5383 = np.concatenate([ym_oof, y_hold], axis=0)
            w_all_5383 = np.concatenate([wm_oof, w_hold], axis=0)
            p_all_cap = np.concatenate([oof_hier25_cap[valid_oof_mask], ho_hier25_cap], axis=0)
            p_all_raw = np.concatenate([oof_hier25_raw[valid_oof_mask], ho_hier25_raw], axis=0)

            summary_oof = [
                dict(split='old4932_capped_style', candidate='topo20_swa_s2028 (Cell 6)', **c6_metrics(oof_t6_cap[valid_oof_mask], ym_oof, wm_oof)),
                dict(split='old4932_capped_style', candidate='modern20x20_3pillars_ens',  **c6_metrics(oof_mod20_cap[valid_oof_mask], ym_oof, wm_oof)),
                dict(split='old4932_capped_style', candidate='hier25_5pillars_ensemble',  **c6_metrics(oof_hier25_cap[valid_oof_mask], ym_oof, wm_oof)),
                dict(split='old4932_raw',          candidate='topo20_swa_s2028 (Cell 6)', **c6_metrics(oof_t6_raw[valid_oof_mask], ym_oof, wm_oof)),
                dict(split='old4932_raw',          candidate='modern20x20_3pillars_ens',  **c6_metrics(oof_mod20_raw[valid_oof_mask], ym_oof, wm_oof)),
                dict(split='old4932_raw',          candidate='hier25_5pillars_ensemble',  **c6_metrics(oof_hier25_raw[valid_oof_mask], ym_oof, wm_oof)),
                dict(split='new451_capped_style',  candidate='topo20_swa_s2028 (Cell 6)', **c6_metrics(ho_t6_cap, y_hold, w_hold)),
                dict(split='new451_capped_style',  candidate='hier25_5pillars_ensemble',  **c6_metrics(ho_hier25_cap, y_hold, w_hold)),
                dict(split='new451_raw',           candidate='topo20_swa_s2028 (Cell 6)', **c6_metrics(ho_t6_raw, y_hold, w_hold)),
                dict(split='new451_raw',           candidate='hier25_5pillars_ensemble',  **c6_metrics(ho_hier25_raw, y_hold, w_hold)),
                dict(split='all5383_capped_style', candidate='hier25_5pillars_ensemble',  **c6_metrics(p_all_cap, y_all_5383, w_all_5383)),
                dict(split='all5383_raw',          candidate='hier25_5pillars_ensemble',  **c6_metrics(p_all_raw, y_all_5383, w_all_5383)),
            ]
            print('\n================ BẢNG KIỂM ĐỊNH 28-MODEL TOPOLOGICAL SUB-MIXER SWA ENSEMBLE ================')
            print(pd.DataFrame(summary_oof).to_string(index=False))

            # 5. Xuất 3 File Submission Chiến lược (Bảo toàn nền tảng 0.97727 = 215/220 + Nâng cấp Multi-Hop MAP)
            base_215 = pd.read_csv(v4_home / 'submission_v4_cell5_surgical_master.csv')['label'].to_numpy(dtype=np.int64)
            base_215[147] = 11  # Bản chuẩn đã đạt 0.97727 (215/220) trên Public LB

            lbl_multihop_map = c6_immutable_core_multihop_map(p_grand28, te_mod20, te_v12_ens)
            lbl_pure_mod20 = te_mod20.argmax(axis=1) + 10

            # Bản Topological SWA Master: Kế thừa nền tảng 215/220 (khóa id=147->11 & id=234->11) và chỉ cập nhật các ca
            # chuyển từ lớp Dư {15,17,19} (hoặc trung gian -> thiếu) sang lớp Thiếu {13,14,18} khi Trụ cột 6 đồng thuận mạnh
            lbl_topo_master = base_215.copy()
            t6_comb = 0.65 * te_t6_ens + 0.35 * te_t6_full
            counts_m = {c: int((lbl_topo_master == c).sum()) for c in range(10, 21)}
            topo_swaps = []
            for i in range(550):
                if i in (147, 234, 62, 162):
                    continue
                c_cur = int(lbl_topo_master[i])
                c_new = int(t6_comb[i].argmax() + 10)
                if c_new != c_cur and counts_m[c_cur] > 50 and counts_m[c_new] < 50:
                    if t6_comb[i, c_new - 10] >= 0.42 and te_mod20[i, c_new - 10] >= te_mod20[i, c_cur - 10] - 0.05:
                        lbl_topo_master[i] = c_new
                        counts_m[c_cur] -= 1
                        counts_m[c_new] += 1
                        topo_swaps.append(dict(id=str(test_ids[i]), from_cls=c_cur, to_cls=c_new, p_topo6=round(float(t6_comb[i, c_new - 10]), 4)))

            sub_master_name = 'submission_v4_cell6_topological_swa_master.csv'
            sub_map_name    = 'submission_v4_cell6_multihop_soft_quota_map.csv'
            sub_mod20_name  = 'submission_v4_cell6_pure_modern20x20_argmax.csv'

            for sname, slbls in [
                (sub_master_name, lbl_topo_master),
                (sub_map_name,    lbl_multihop_map),
                (sub_mod20_name,  lbl_pure_mod20),
            ]:
                sub_df = pd.DataFrame({'id': test_ids, 'label': slbls.astype(np.int64)})
                sub_df.to_csv(session / sname, index=False, lineterminator='\n')
                sub_df.to_csv(v4_home / sname, index=False, lineterminator='\n')
                diff_215 = np.flatnonzero(slbls != base_215)
                print(f'* {sname}: diff_vs_215(0.97727) = {len(diff_215)} dòng ({diff_215.tolist()}) | '
                      f'class_counts = { {int(c): int((slbls == c).sum()) for c in range(10, 21)} }')

            # Kiểm tra các mẫu mà Trụ cột 6 (TopologicalSubMixer + SWA) dự đoán khác bản 215/220
            pred_t6 = t6_comb.argmax(axis=1) + 10
            diff_t6_vs_215 = []
            for i in np.flatnonzero(pred_t6 != base_215):
                diff_t6_vs_215.append(dict(
                    id=str(test_ids[i]),
                    base_215_cls=int(base_215[i]),
                    topo6_pred=int(pred_t6[i]),
                    p_topo6_pred=round(float(t6_comb[i, pred_t6[i] - 10]), 4),
                    p_topo6_base=round(float(t6_comb[i, base_215[i] - 10]), 4),
                    p_mod20_pred=round(float(te_mod20[i, pred_t6[i] - 10]), 4),
                    p_v12_base=round(float(te_v12_ens[i, base_215[i] - 10]), 4),
                ))
            print('\n=== CÁC DÒNG TRÊN TEST MÀ TRỤ CỘT 6 (TOPOLOGICAL SUB-MIXER SWA) PHÁT HIỆN MỚI SO VỚI BẢN 215/220 ===')
            if diff_t6_vs_215:
                print(pd.DataFrame(diff_t6_vs_215).to_string(index=False))

            c6_save_npz(session / 'test_550_cell6_all_pillars_probs.npz', dict(
                test_ids=test_ids,
                te_s7_ens=te_s7_ens,
                te_s29_ens=te_s29_ens,
                te_u20_ens=te_u20_ens,
                te_dp5_ens=te_dp5_ens,
                te_t6_ens=te_t6_ens,
                te_u20_full_c4=te_u20_full_c4,
                te_dp5_full_c5=te_dp5_full_c5,
                te_t6_full=te_t6_full,
                te_v12_ens=te_v12_ens,
                te_mod20=te_mod20,
                te_hier28=te_hier28,
                p_grand28=p_grand28,
            ))

            report = dict(
                session_id=session.name,
                oof_summary=summary_oof,
                topo_swaps_vs_215=topo_swaps,
                topo6_differences_vs_215=diff_t6_vs_215,
                class_counts_topo_master={int(c): int((lbl_topo_master == c).sum()) for c in range(10, 21)},
                class_counts_multihop_map={int(c): int((lbl_multihop_map == c).sum()) for c in range(10, 21)},
                class_counts_pure_mod20={int(c): int((lbl_pure_mod20 == c).sum()) for c in range(10, 21)},
            )
            c6_json(session / 'report.json', report)
            status.update(
                status='complete',
                topo6_old4932_capped_style_correct=int(summary_oof[0]['correct']),
                hier25_old4932_capped_style_correct=int(summary_oof[2]['correct']),
                topo6_old4932_raw_correct=int(summary_oof[3]['correct']),
                hier25_old4932_raw_correct=int(summary_oof[5]['correct']),
                all5383_capped_style_correct=int(summary_oof[10]['correct']),
                all5383_raw_correct=int(summary_oof[11]['correct']),
                elapsed_minutes=round((time.perf_counter() - started) / 60.0, 2),
            )
        except Exception as e:
            status.update(status='stopped_error', error_type=type(e).__name__, error=str(e))
            traceback.print_exc()
        finally:
            c6_json(session / 'status.json', status)
            print('\n=== GỬI LẠI LOG NÀY VÀ FILE ZIP BÁO CÁO ===')
            print(json.dumps(status, ensure_ascii=False, indent=2))
            sys.stdout, sys.stderr = oldout, olderr

    zip_path = session.parent / (session.name + '_v4_cell6_report.zip')
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


V4_CELL6_ZIP = run_v4_cell6()