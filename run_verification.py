#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MNIST 10–20: verification of the supplied historical experiment.

Double-click KIEM_TRA_BTC.cmd on Windows, or run python run_verification.py.
fast replays caches; weights recomputes 28 v4 models but retains cached v12.
No test accuracy is inferred from agreement with a reference submission.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import html
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import types
import unicodedata
import uuid
import webbrowser
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / 'MNIST_data'
V5_NAME = 'v5_76ddef9bc184'
DATASET_NAME = 'dataset_82cc33f4f691'
CMP_NAME = 'renderer_comparison_b409f5a57688'
BUNDLE_NAME = 'test_550_cell6_all_pillars_probs.npz'
FOLD_PRED_NAMES = (
    'hier_preds_s7.npz', 'mscale_preds_s29.npz', 'ult20_preds_s2026.npz',
    'dp20_hardft_preds_s2027.npz', 'topo20_swa_preds_s2028.npz',
)
WEIGHT_NAMES = (
    'hier_mlp_from_scratch_s7.pt', 'mscale_mlp_from_scratch_s29.pt',
    'ult20_mlp_s2026.pt', 'dp20_hardft_s2027.pt', 'topo20_swa_s2028.pt',
)
FULL_WEIGHT_NAMES = ('ult20_full_data_100pct.pt', 'dp20_hardft_full_100pct.pt', 'topo20_swa_full_100pct.pt')
PILLAR_KEYS = {
    'te_s7_ens': 'te_s7_ens', 'te_s29_ens': 'te_s29_ens',
    'te_u20_ens': 'te_u20_ens', 'te_dp5_ens': 'te_dp5_ens', 'te_t6_ens': 'te_t6_ens',
    'te_u20_full': 'te_u20_full_c4', 'te_dp5_full': 'te_dp5_full_c5',
    'te_t6_full': 'te_t6_full', 'te_v12_ens': 'te_v12_ens',
}
V5_EXCLUSIONS = [458, 1767, 2213, 4355]
V31_EXCLUSIONS = [9, 15, 16, 25, 29, 78, 95, 222, 224, 231, 233, 234, 235, 238, 244, 251, 253, 256]
INPUT_FILES = set()
np = pd = None


class VerificationError(Exception):
    pass


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)
    def flush(self):
        self.stream.flush()
        self.log.flush()
    def isatty(self):
        return False


def check_file(path):
    path = Path(path)
    if not path.is_file():
        raise VerificationError(f'Thiếu file: {path}')
    with path.open('rb') as stream:
        prefix = stream.read(128)
    if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise VerificationError(f'Chỉ có con trỏ Git LFS: {path}\nChạy git lfs pull tại thư mục dự án rồi thử lại.')
    if not prefix:
        raise VerificationError(f'File rỗng: {path}')
    INPUT_FILES.add(path.resolve())
    return path


def legacy_status(zip_path):
    """Legacy cells catch errors internally; their status.json is authoritative."""
    with zipfile.ZipFile(zip_path) as archive:
        status = json.loads(archive.read('status.json').decode('utf-8'))
    if status.get('status') != 'complete':
        raise VerificationError(f'Cell gốc đã dừng với lỗi: {status.get("error", status)}. Báo cáo cell: {zip_path}')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dependencies(mode, no_install=False):
    """Install only when imports fail; never install into the user's global Python."""
    packages = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.1,<4'}
    if mode in ('weights', 'train'):
        packages.update({'scipy': 'scipy>=1.11,<2', 'torch': 'torch>=2.2,<3', 'PIL': 'Pillow>=10', 'tqdm': 'tqdm>=4.66'})
    missing = []
    for module, requirement in packages.items():
        try:
            importlib.import_module(module)
        except (ImportError, OSError, ValueError) as exc:
            print(f'[MÔI TRƯỜNG] {module}: {exc}')
            missing.append(requirement)
    if not missing:
        return None
    if no_install:
        raise VerificationError('Thiếu hoặc lỗi thư viện: ' + ', '.join(missing) + '. Chạy lại không có --no-install để tự cài.')
    env_dir = REPO_ROOT / '.venv-verification'
    env_python = env_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    inside_env = Path(sys.prefix).resolve() == env_dir.resolve()
    if not inside_env:
        if not env_python.is_file():
            print(f'[CÀI ĐẶT] Tạo môi trường riêng: {env_dir}', flush=True)
            subprocess.run([sys.executable, '-m', 'venv', str(env_dir)], check=True)
        # The child performs its own import checks and installs only missing packages.
        return subprocess.call([str(env_python), '-u', str(Path(__file__).resolve()), *sys.argv[1:]])
    print('[CÀI ĐẶT] Cần Internet ở lần cài đầu; quá trình này có thể mất vài phút.', flush=True)
    subprocess.run([str(env_python), '-m', 'ensurepip', '--upgrade'], check=True)
    subprocess.run([str(env_python), '-m', 'pip', 'install', '--disable-pip-version-check', *missing], check=True)
    # Restart to avoid partial imports from the failed dependency checks.
    child_args = sys.argv[1:]
    return subprocess.call([str(env_python), '-u', str(Path(__file__).resolve()), *child_args, '--no-install'])


def load_npz(path, keys=None):
    path = check_file(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            selected = archive.files if keys is None else keys
            absent = set(selected) - set(archive.files)
            if absent:
                raise VerificationError(f'{path.name}: thiếu khóa {sorted(absent)}')
            return {key: archive[key] for key in selected}
    except (OSError, ValueError, EOFError) as exc:
        raise VerificationError(f'Không đọc được NPZ {path}: {exc}') from exc


def validate_probs(value, rows, name):
    value = np.asarray(value)
    if value.shape != (rows, 11) or not np.issubdtype(value.dtype, np.number):
        raise VerificationError(f'{name}: cần ma trận ({rows}, 11), nhận {value.shape}, {value.dtype}')
    if not np.isfinite(value).all() or (value < -1e-6).any() or (value > 1.00001).any():
        raise VerificationError(f'{name}: xác suất không hợp lệ (NaN, Inf hoặc ngoài [0,1]).')
    if not np.allclose(value.sum(axis=1), 1.0, atol=2e-3, rtol=0):
        raise VerificationError(f'{name}: tổng xác suất mỗi dòng không bằng 1.')
    return value


def read_ids(path):
    frame = pd.read_csv(check_file(path), dtype={'id': str}, keep_default_na=False)
    if 'id' not in frame or frame['id'].eq('').any() or frame['id'].duplicated().any():
        raise VerificationError(f'{path}: cần cột id không rỗng, không trùng.')
    return frame


def require_columns(frame, columns, name):
    absent = set(columns) - set(frame.columns)
    if absent:
        raise VerificationError(f'{name}: thiếu cột {sorted(absent)}')


def validate_labels(values, name):
    try:
        labels = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise VerificationError(f'{name}: nhãn không phải số.') from exc
    if labels.ndim != 1 or not np.isfinite(labels).all() or not np.isin(labels, np.arange(10, 21)).all():
        raise VerificationError(f'{name}: nhãn phải là số nguyên 10..20.')
    return labels.astype(np.int64)


def writer_key(value):
    if pd.isna(value):
        return ''
    result = ' '.join(unicodedata.normalize('NFC', str(value)).casefold().split())
    return 'minh nhật phạm' if result in ('phạm minh nhật', 'minh nhật phạm') else result


def resolve_bundle(v4_home, override=None):
    if override:
        return Path(override).resolve()
    direct = v4_home / BUNDLE_NAME
    if direct.is_file():
        return direct
    candidates = []
    for path in sorted((v4_home / 'cell6_sessions').glob('*/' + BUNDLE_NAME), reverse=True):
        status_path = path.parent / 'status.json'
        if status_path.is_file():
            try:
                if json.loads(status_path.read_text(encoding='utf-8')).get('status') != 'complete':
                    continue
            except (ValueError, OSError):
                continue
        candidates.append(path)
    if candidates:
        print(f'[CACHE] Chọn session mới nhất: {candidates[0].parent.name} ({len(candidates)} ứng viên).')
        return candidates[0]
    return direct


def source_dir():
    for path in (REPO_ROOT / 'src', REPO_ROOT):
        if (path / 'cell_01.py').is_file():
            return path
    return REPO_ROOT / 'src'


def preflight(data_root, mode, bundle):
    v4 = data_root / 'v4_hier_multi_mlp'
    v5 = data_root / V5_NAME
    paths = [data_root / 'test.csv', data_root / 've_tay_10_20 - data.csv',
             data_root / 've_tay_10_20 - data_v31.csv', v5 / DATASET_NAME / 'validation.npz', bundle]
    paths.extend(v4 / f'fold_{fold}' / name for fold in range(5) for name in FOLD_PRED_NAMES)
    if mode == 'weights':
        paths += [source_dir() / f'cell_{index:02d}.py' for index in (1, 2, 4, 5, 6)]
        paths += [v5 / CMP_NAME / 'config.json']
        paths.extend(v4 / f'fold_{fold}' / name for fold in range(5) for name in WEIGHT_NAMES)
        paths += [v4 / name for name in FULL_WEIGHT_NAMES]
        paths.extend(v5 / CMP_NAME / 'standalone_fdeskew_109w_v12' / f'fold_{fold}' / 'fold_preds.npz' for fold in range(5))
    failures = []
    for path in paths:
        try:
            check_file(path)
        except VerificationError as exc:
            failures.append(str(exc))
    if failures:
        raise VerificationError('Bộ dữ liệu chưa đầy đủ:\n' + '\n'.join(failures))




# Numerical postprocessors copied verbatim from the supplied Cell 4 and Cell 5.
def c4_smooth_surplus_deficit_meta_gate(p_hier16, p_anchor, is_test=True):
    """
    Hợp nhất có Cổng Mượt + Ưu tiên Chiều Dư -> Thiếu (Smooth Margin & Surplus-to-Deficit Directional Meta-Gating):
    1. Trọng số cơ sở của Hier16 (16 mô hình Đa-MLP Phân cấp) đạt 0.85 khi tự tin, và giảm mượt theo Sigmoid
       khi p_hier16 mất tự tin so với p_anchor.
    2. Trên tập Test (550 mẫu = 50 mẫu/lớp): nếu p_hier16 và p_anchor bất đồng, nhưng nhãn của p_hier16
       chuyển mẫu từ một lớp đang DƯ (>50 mẫu) sang một lớp đang THIẾU (<50 mẫu) và cả 3 trụ cột Hier đều đồng thuận
       (prob >= 0.45), hệ thống giữ nguyên quyết định của Hier16 (khôi phục chính xác id=62 -> 20 và id=162 -> 18!).
    """
    conf_h = p_hier16.max(axis=1, keepdims=True)
    conf_a = p_anchor.max(axis=1, keepdims=True)
    diff = conf_h - conf_a
    # Sigmoid gate mượt từ 0.28 đến 0.88 quanh ngưỡng diff = -0.06
    sig = 1.0 / (1.0 + np.exp(-22.0 * (diff + 0.06)))
    w_h = (0.28 + 0.60 * sig).astype(np.float32)
    p_blend = w_h * p_hier16 + (1.0 - w_h) * p_anchor
    p_blend /= p_blend.sum(axis=1, keepdims=True)

    if not is_test:
        return p_blend

    # Hiệu chỉnh tiên nghiệm liên tục nhẹ (Damped Bayes Scaling)
    p_cal = np.clip(p_blend.astype(np.float64), 1e-12, 1.0)
    for _ in range(30):
        p_cal *= (50.0 / np.clip(p_cal.sum(axis=0, keepdims=True), 1e-6, None)) ** 0.50
        p_cal /= p_cal.sum(axis=1, keepdims=True)

    base_arg = p_blend.argmax(axis=1)
    cal_arg = p_cal.argmax(axis=1)
    preds = np.where((p_blend.max(axis=1) < 0.68) & (p_blend[np.arange(len(p_blend)), cal_arg] >= 0.36), cal_arg, base_arg)

    # Kiểm tra các mẫu mà Hier16 có xác suất tự nhiên >= 0.45 giúp chuyển từ lớp DƯ (>50) sang lớp THIẾU (<50)
    hier_arg = p_hier16.argmax(axis=1)
    hier_conf = p_hier16.max(axis=1)
    for _ in range(5):
        counts = {c: int((preds == c).sum()) for c in range(11)}
        changed = False
        for i in range(len(preds)):
            c_cur = int(preds[i])
            c_h = int(hier_arg[i])
            if c_cur != c_h and hier_conf[i] >= 0.45:
                if counts[c_cur] > 50 and counts[c_h] < 50:
                    preds[i] = c_h
                    counts[c_cur] -= 1
                    counts[c_h] += 1
                    changed = True
        if not changed:
            break

    return preds + 10, p_blend

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

def replay_historical_submissions(pillars):
    """Faithfully replay Cell 4, 5, 6, 7, 8, 9, including their explicit edits.

    Cell 6 uses a zero-based row assignment and excludes four row positions.
    Cell 7–9 edit IDs. These are historical choices, not learned inference rules.
    """
    ids = pillars['test_ids']
    if len(ids) != 550:
        raise VerificationError('Chỉ tái lập hậu xử lý lịch sử trên đúng 550 mẫu test gốc.')
    s7, s29, u20, dp5, t6 = (pillars[key] for key in ('te_s7_ens', 'te_s29_ens', 'te_u20_ens', 'te_dp5_ens', 'te_t6_ens'))
    ufull, dfull, tfull, anchor = (pillars[key] for key in ('te_u20_full', 'te_dp5_full', 'te_t6_full', 'te_v12_ens'))
    h16 = 0.28 * s7 + 0.28 * s29 + 0.28 * u20 + 0.16 * ufull
    c4, _ = c4_smooth_surplus_deficit_meta_gate(h16, anchor, is_test=True)
    h22 = 0.21 * s7 + 0.21 * s29 + 0.21 * u20 + 0.21 * dp5 + 0.08 * ufull + 0.08 * dfull
    sig = 1.0 / (1.0 + np.exp(-22.0 * ((h22.max(1, keepdims=True) - anchor.max(1, keepdims=True)) + 0.06)))
    weight = (0.30 + 0.58 * sig).astype(np.float32)
    grand22 = weight * h22 + (1.0 - weight) * anchor
    grand22 /= grand22.sum(1, keepdims=True)
    c5, _, _, _, _ = c5_surgical_surplus_deficit_resolver(c4, grand22, h22, anchor, ids)

    # EXACT Cell 6 behavior; do not replace these with invented probability gates.
    base215 = c5.copy()
    base215[147] = 11
    mod20 = 0.28 * t6 + 0.14 * tfull + 0.23 * dp5 + 0.11 * dfull + 0.16 * u20 + 0.08 * ufull
    combined = 0.65 * t6 + 0.35 * tfull
    master = base215.copy()
    counts = {label: int((master == label).sum()) for label in range(10, 21)}
    for index in range(550):
        if index in (147, 234, 62, 162):
            continue
        current = int(master[index])
        new = int(combined[index].argmax() + 10)
        if new != current and counts[current] > 50 and counts[new] < 50:
            if combined[index, new - 10] >= 0.42 and mod20[index, new - 10] >= mod20[index, current - 10] - 0.05:
                master[index] = new
                counts[current] -= 1
                counts[new] += 1

    def edit_by_id(labels, edits):
        result = labels.copy()
        for sample_id, label in edits.items():
            indices = np.flatnonzero(ids == sample_id)
            if len(indices) != 1:
                raise VerificationError(f'Không tìm được ID lịch sử duy nhất: {sample_id}')
            result[indices[0]] = label
        return result

    chain = edit_by_id(master, {'115': 13, '476': 13, '78': 18})
    plus481 = edit_by_id(chain, {'481': 18})
    plus_both = edit_by_id(plus481, {'234': 16})
    final = edit_by_id(chain, {'234': 16})
    return {
        'submission_v4_cell4_ultimate_master.csv': c4,
        'submission_v4_cell5_surgical_master.csv': c5,
        'submission_v4_cell6_topological_swa_master.csv': master,
        'submission_v4_cell6_2hop_chain_19_13_18.csv': chain,
        'submission_v4_cell6_2hop_plus_481_to_18.csv': plus481,
        'submission_v4_cell6_2hop_plus_481_and_234.csv': plus_both,
        'submission_v4_cell6_final_single_best.csv': final,
    }


def verify_oof_metrics(data_root):
    v4 = data_root / 'v4_hier_multi_mlp'
    va = load_npz(data_root / V5_NAME / DATASET_NAME / 'validation.npz', ['rows', 'y', 'writer', 'fold'])
    n_old = len(va['y'])
    if any(np.asarray(va[key]).shape != (n_old,) for key in va):
        raise VerificationError('validation.npz: các mảng rows/y/writer/fold không cùng độ dài.')
    if not np.isin(va['fold'], range(5)).all():
        raise VerificationError('validation.npz: fold phải thuộc 0..4.')
    valid = ~np.isin(va['rows'], V5_EXCLUSIONS)
    old_labels = validate_labels(va['y'], 'validation.y')
    if len(np.unique(va['rows'])) != n_old or int(valid.sum()) != 4932:
        raise VerificationError('Validation không khớp bộ lịch sử: cần rows duy nhất và 4932 mẫu sau loại trừ.')
    df1 = pd.read_csv(check_file(data_root / 've_tay_10_20 - data.csv'))
    df2 = pd.read_csv(check_file(data_root / 've_tay_10_20 - data_v31.csv'))
    require_columns(df1, ['label', 'writer'], 'data.csv')
    require_columns(df2, ['label', 'writer_name'], 'data_v31.csv')
    keep = ~df2.index.isin(V31_EXCLUSIONS)
    hold_labels = validate_labels(np.concatenate([df1.iloc[5452:]['label'].to_numpy(), df2.loc[keep, 'label'].to_numpy()]), 'holdout.y')
    hold_writers = np.concatenate([df1.iloc[5452:]['writer'].map(writer_key).to_numpy(dtype=str), df2.loc[keep, 'writer_name'].map(writer_key).to_numpy(dtype=str)])
    if len(hold_labels) != 451 or np.any(hold_writers == ''):
        raise VerificationError('Holdout không khớp bộ lịch sử: cần 451 mẫu có tên người viết.')
    old_writer_fold = {}
    # Use exactly the historical cached writer strings for fold reconstruction.
    for writer, fold in zip(va['writer'], va['fold']):
        writer = str(writer)
        if writer in old_writer_fold and old_writer_fold[writer] != int(fold):
            raise VerificationError(f'Người viết xuất hiện ở nhiều fold: {writer}')
        old_writer_fold[writer] = int(fold)
    new_writers = sorted([w for w in np.unique(hold_writers) if w not in old_writer_fold], key=lambda w: (-int((hold_writers == w).sum()), w))
    loads = np.zeros(5, dtype=np.int64)
    hold_fold_map = {}
    for writer in new_writers:
        fold = int(loads.argmin())
        hold_fold_map[writer] = fold
        loads[fold] += int((hold_writers == writer).sum())
    for writer in np.unique(hold_writers):
        if writer in old_writer_fold:
            hold_fold_map[writer] = old_writer_fold[writer]
    hold_folds = np.array([hold_fold_map[w] for w in hold_writers], dtype=np.int64)
    old_probs = [np.full((n_old, 11), np.nan, dtype=np.float32) for _ in range(2)]
    hold_probs = [np.full((451, 11), np.nan, dtype=np.float32) for _ in range(2)]
    for fold in range(5):
        old_indices = np.flatnonzero(va['fold'] == fold)
        hold_indices = np.flatnonzero(hold_folds == fold)
        archives = [load_npz(v4 / f'fold_{fold}' / name, ['oof_raw', 'oof_cap', 'ho_raw', 'ho_cap']) for name in FOLD_PRED_NAMES]
        for domain, (old_key, hold_key) in enumerate([('oof_raw', 'ho_raw'), ('oof_cap', 'ho_cap')]):
            old = [validate_probs(z[old_key], len(old_indices), f'fold{fold}/{name}/{old_key}') for name, z in zip(FOLD_PRED_NAMES, archives)]
            held = [validate_probs(z[hold_key], len(hold_indices), f'fold{fold}/{name}/{hold_key}') for name, z in zip(FOLD_PRED_NAMES, archives)]
            def ensemble(p):
                return 0.80 * (0.42 * p[4] + 0.34 * p[3] + 0.24 * p[2]) + 0.10 * p[1] + 0.10 * p[0]
            old_probs[domain][old_indices] = ensemble(old)
            hold_probs[domain][hold_indices] = ensemble(held)
    rows = []
    for domain, old, held in zip(('Raw', 'Capped_Style'), old_probs, hold_probs):
        validate_probs(old, n_old, f'OOF {domain}')
        validate_probs(held, 451, f'Holdout {domain}')
        for split, prob, labels in [('OOF cũ', old[valid], old_labels[valid]), ('Holdout bổ sung', held, hold_labels), ('Tổng hợp', np.concatenate([old[valid], held]), np.concatenate([old_labels[valid], hold_labels]))]:
            correct = int(((prob.argmax(1) + 10) == labels).sum())
            row = {'split': split, 'domain': domain, 'correct': correct, 'n': len(labels), 'accuracy_percent': 100.0 * correct / len(labels)}
            rows.append(row)
            print(f"  {split:18} {domain:14}: {correct}/{len(labels)} ({row['accuracy_percent']:.2f}%)")
    return {'metrics': rows, 'old_writers': len(np.unique(va['writer'][valid])), 'holdout_writers': len(np.unique(hold_writers)), 'strictly_new_writers': len(new_writers), 'overlapping_writers': sorted(set(hold_writers) & set(old_writer_fold)), 'scope': 'Tính lại chỉ số từ cache OOF/holdout; không tự chứng minh quá trình huấn luyện không rò rỉ.'}


def load_cached_pillars(bundle, test_ids):
    archive = load_npz(bundle, ['test_ids', *PILLAR_KEYS.values()])
    cached_ids = np.asarray(archive['test_ids']).astype(str)
    if cached_ids.shape != test_ids.shape or not np.array_equal(cached_ids, test_ids):
        raise VerificationError('ID/thứ tự test.csv khác cache 550 mẫu. Dừng để tránh áp hậu xử lý lên sai mẫu.')
    return {'test_ids': test_ids, **{key: validate_probs(archive[source], len(test_ids), source) for key, source in PILLAR_KEYS.items()}}


def load_inference_cell(index):
    """Load original definitions, removing only the known top-level training call."""
    path = check_file(source_dir() / f'cell_{index:02d}.py')
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    kept, removed = [], 0
    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == f'V4_CELL{index}_ZIP' for t in node.targets)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == f'run_v4_cell{index}'):
            removed += 1
        else:
            kept.append(node)
    if removed != 1:
        raise VerificationError(f'{path.name}: không nhận diện được lệnh train cuối file; từ chối import để tránh tự train.')
    tree.body = kept
    module = types.ModuleType(f'_verification_cell_{index:02d}')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def run_weights_inference_on_test(data_root, frame, report):
    import torch
    c1, c2, c4, c5, c6 = [load_inference_cell(i) for i in (1, 2, 4, 5, 6)]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    report['device'] = str(device)
    print(f'[WEIGHTS] 28 mô hình v4 trên {device}; phần v12 vẫn dùng cache, không train.')
    cmp_dir = data_root / V5_NAME / CMP_NAME
    config_path = check_file(cmp_dir / 'config.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    levels = np.asarray(config['renderer']['gray_levels'], dtype=np.float64)
    pixel_columns = [f'pixel{i}' for i in range(784)]
    require_columns(frame, pixel_columns, 'test.csv')
    pixels = frame[pixel_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 255).any() or not np.equal(pixels, np.floor(pixels)).all():
        raise VerificationError('test.csv: pixel phải là số nguyên trong [0,255].')
    pixels = pixels.astype(np.uint8)
    print('[WEIGHTS] Trích xuất các góc nhìn hình học...')
    xd, xl, xr, geometry = c1.v4_precompute_multi_mlp_views_gpu(pixels, levels, device)
    specs = [(0.40, 0., 0.), (0.15, -0.05, 0.), (0.15, 0.05, 0.), (0.15, 0., -1.), (0.15, 0., 1.)]
    extractors = [c4.c4_extract_voronoi_20x20_views_gpu, c5.c5_extract_dualpeak_voronoi_20x20_gpu, c6.c6_extract_hybrid_dualpeak_20x20_gpu]
    hypotheses = [[(w, fn(pixels, levels, device, delta_skew=skew, delta_cut=cut)) for w, skew, cut in specs] for fn in extractors]
    constructors = [c1.V4HierMultiMLP, c2.V4MultiScaleHierMLP, c4.V4Ultimate20x20HierMLP, c5.V4Ultimate20x20HierMLP, c6.V4TopologicalSubMixerMLP]
    def predict(model, architecture):
        if architecture == 0:
            return c1.v4_predict_hier_shift9(model, xd, xl, xr, geometry, device)
        if architecture == 1:
            return c2.c2_predict_mscale_shift9(model, pixels, xd, xl, xr, geometry, device)
        functions = [c4.c4_predict_ult20_multihyp_test, c5.c5_predict_ult20_multihyp_test, c6.c6_predict_topo20_multihyp_test]
        return functions[architecture - 2](model, pixels, hypotheses[architecture - 2], device)
    def from_checkpoint(path, architecture):
        model = constructors[architecture]().to(device)
        # Existing checkpoints also contain NumPy metadata; use only trusted project checkpoints.
        checkpoint = torch.load(check_file(path), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        result = validate_probs(predict(model, architecture), len(frame), path.name)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return result
    all_folds = [[] for _ in range(5)]
    anchor_folds = []
    v4 = data_root / 'v4_hier_multi_mlp'
    for fold in range(5):
        for architecture, filename in enumerate(WEIGHT_NAMES):
            print(f'[WEIGHTS] fold {fold}/4 — {filename}', flush=True)
            all_folds[architecture].append(from_checkpoint(v4 / f'fold_{fold}' / filename, architecture))
        archive = load_npz(cmp_dir / 'standalone_fdeskew_109w_v12' / f'fold_{fold}' / 'fold_preds.npz', ['te_fd', 'te_o_s7', 'te_o_s29'])
        for key in archive:
            validate_probs(archive[key], len(frame), f'v12 fold{fold}/{key}')
        anchor_folds.append(0.50 * archive['te_fd'] + 0.50 * (0.55 * archive['te_o_s7'] + 0.45 * archive['te_o_s29']))
    result = dict(test_ids=frame['id'].to_numpy(dtype=str))
    for key, values in zip(('te_s7_ens', 'te_s29_ens', 'te_u20_ens', 'te_dp5_ens', 'te_t6_ens'), all_folds):
        result[key] = np.mean(values, axis=0)
    for architecture, key, filename in zip((2, 3, 4), ('te_u20_full', 'te_dp5_full', 'te_t6_full'), FULL_WEIGHT_NAMES):
        print(f'[WEIGHTS] full data — {filename}', flush=True)
        result[key] = from_checkpoint(v4 / filename, architecture)
    result['te_v12_ens'] = np.mean(anchor_folds, axis=0)
    return result


def compare_submission(path, ids, labels, reference, out_dir):
    labels = validate_labels(labels, path.name)
    pd.DataFrame({'id': ids, 'label': labels}).to_csv(path, index=False, lineterminator='\n')
    result = {'file': path.name, 'sha256': sha256_file(path), 'n': len(ids), 'class_counts': {str(c): int((labels == c).sum()) for c in range(10, 21)}}
    if not reference.is_file():
        result.update(status='MISSING_REFERENCE', message='Đã tạo CSV; chưa có bản tham chiếu để đối chiếu.')
        return result
    try:
        ref = read_ids(reference)
        require_columns(ref, ['label'], reference.name)
        ref_labels = validate_labels(ref['label'], reference.name)
    except (VerificationError, ValueError, OSError) as exc:
        result.update(status='FAIL', message=str(exc))
        return result
    expected = dict(zip(ref['id'], ref_labels))
    actual = dict(zip(ids, labels))
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatches = [{'id': key, 'reference_label': int(expected[key]), 'generated_label': int(actual[key])} for key in ids if key in expected and expected[key] != actual[key]]
    same_order = list(ref['id']) == list(ids)
    matched = sum(key in expected and expected[key] == value for key, value in actual.items())
    result.update(reference=str(reference), reference_sha256=sha256_file(reference), matched=int(matched), reference_n=len(ref), same_id_order=same_order, missing_ids=missing, extra_ids=extra, mismatched_labels=len(mismatches), byte_identical=sha256_file(reference) == result['sha256'])
    result['status'] = 'PASS' if not missing and not extra and not mismatches and same_order else 'FAIL'
    result['message'] = f'Khớp ID + nhãn: {matched}/{len(ids)}; thứ tự ID: {"khớp" if same_order else "khác"}; byte: {"khớp" if result["byte_identical"] else "khác"}.'
    if mismatches:
        diff_path = out_dir / (path.stem + '_differences.csv')
        pd.DataFrame(mismatches).to_csv(diff_path, index=False)
        result['differences_file'] = diff_path.name
    return result


def run_legacy_training(data_root, out_dir):
    import torch
    if not torch.cuda.is_available():
        raise VerificationError('--mode train cần GPU CUDA theo các cell gốc. Dùng --mode fast để kiểm tra trên CPU.')
    paths = [check_file(source_dir() / f'cell_{i:02d}.py') for i in range(1, 10)]
    try:
        import google.colab  # noqa: F401
    except ImportError:
        google = sys.modules.setdefault('google', types.ModuleType('google'))
        colab = types.ModuleType('google.colab')
        colab.drive = types.SimpleNamespace(mount=lambda *a, **k: print(f'[LOCAL] Dữ liệu: {data_root}'))
        colab.files = types.SimpleNamespace(download=lambda p: print(f'[LOCAL] Đã xuất: {p}'))
        google.colab = colab
        sys.modules['google.colab'] = colab
    previous = Path.cwd()
    class LocalDataPaths(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.startswith('/content/drive/MyDrive/MNIST_data'):
                return ast.copy_location(ast.Constant(data_root.as_posix() + node.value[len('/content/drive/MyDrive/MNIST_data'):]), node)
            return node
    try:
        os.chdir(out_dir)
        for path in paths:
            print(f'[TRAIN] Đang chạy {path.name}', flush=True)
            # Map every path prefix, without breaking source quotes in Windows paths.
            tree = LocalDataPaths().visit(ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path)))
            namespace = {'__name__': '__main__', '__file__': str(path)}
            exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
            index = int(path.stem.split('_')[1])
            if index in (1, 2, 4, 5, 6):
                legacy_status(namespace[f'V4_CELL{index}_ZIP'])
    finally:
        os.chdir(previous)


def write_report(report, out_dir, latest_dir):
    """Write both per-run and latest summaries, using relative download links."""
    escaped = lambda value: html.escape(str(value), quote=True)
    json_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    for destination in (out_dir, latest_dir):
        prefix = '' if destination == out_dir else out_dir.name + '/'
        rows = ''.join(f'<tr><td>{escaped(m["split"])}</td><td>{escaped(m["domain"])}</td><td>{m["correct"]}/{m["n"]}</td><td>{m["accuracy_percent"]:.2f}%</td></tr>' for m in report.get('oof', {}).get('metrics', []))
        comparisons = ''.join(f'<tr><td><a href="{escaped(prefix + item["file"])}">{escaped(item["file"])}</a></td><td class="{item["status"]}">{escaped(item["status"])}</td><td>{escaped(item["message"])}</td></tr>' for item in report.get('submissions', []))
        deltas = ''.join(f'<tr><td>{escaped(key)}</td><td>{item["max_abs_delta"]:.7g}</td><td>{"PASS" if item["within_tolerance"] else "FAIL"}</td></tr>' for key, item in report.get('weights_vs_cache', {}).items())
        weights_table = '<h2>Suy luận trọng số so với cache</h2><table><tr><th>Nhánh</th><th>Độ lệch lớn nhất</th><th>Dung sai</th></tr>' + deltas + '</table>' if deltas else ''
        status = report['status']
        color = '#087f5b' if status == 'PASS' else '#a61e4d'
        notes = ''.join(f'<li>{escaped(note)}</li>' for note in report['notes'])
        body = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MNIST 10–20 — Kiểm tra BTC</title>
<style>body{{font:16px system-ui,sans-serif;background:#f2f5f9;color:#17243b;margin:0}}main{{max-width:1150px;margin:40px auto;background:white;border-radius:16px;padding:32px}}h1{{font-size:28px}}.status{{color:{color};font-size:26px;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:11px;text-align:left;border-bottom:1px solid #dde3ed;overflow-wrap:anywhere}}th{{background:#edf2f8}}a{{color:#1558a6}}pre{{white-space:pre-wrap;background:#fff0f3;padding:16px}}li{{margin:9px 0}}.PASS{{color:#087f5b;font-weight:700}}.FAIL,.MISSING_REFERENCE{{color:#a61e4d;font-weight:700}}small{{color:#58677d}}@media(max-width:700px){{main{{margin:10px;padding:16px}}table{{font-size:12px}}}}</style>
<main><h1>MNIST 10–20 · Kiểm tra BTC</h1><p class="status">{escaped(status)}</p><p>{escaped(report.get('summary', ''))}</p><small>{escaped(report['started_at'])} · Chế độ: {escaped(report['mode'])} · {report.get('elapsed_seconds', 0):.2f} giây</small>
<pre {"" if report.get('error') else "hidden"}>{escaped(report.get('error', ''))}</pre>
<h2>Độ chính xác từ cache validation</h2><table><tr><th>Tập</th><th>Miền</th><th>Đúng / tổng</th><th>Độ chính xác</th></tr>{rows}</table>
<h2>Đối chiếu submission</h2><table><tr><th>CSV được tạo</th><th>Trạng thái</th><th>Đối chiếu</th></tr>{comparisons}</table>
{weights_table}
<p><a href="{escaped(prefix + 'verification.log')}">Nhật ký</a> · <a href="{escaped(prefix + 'verification_report.json')}">Báo cáo JSON</a></p>
<h2>Phạm vi kiểm tra</h2><ul>{notes}</ul></main></html>'''
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'verification_report.json').write_text(json_text, encoding='utf-8')
        (destination / 'verification_report.html').write_text(body, encoding='utf-8')


def build_parser():
    parser = argparse.ArgumentParser(description='MNIST 10–20: tái lập và kiểm tra kết quả đã lưu.')
    parser.add_argument('--mode', choices=['fast', 'weights', 'train'], default='fast', help='fast: cache trên CPU; weights: 28 mô hình v4 + cache v12; train: chạy lại cell gốc, cần dữ liệu trung gian và CUDA.')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'verified_submissions')
    parser.add_argument('--cache', type=Path, help='Chọn chính xác bundle xác suất Cell 6 thay vì tự tìm session.')
    parser.add_argument('--no-install', action='store_true', help='Không tự cài thư viện thiếu (máy offline/đã chuẩn bị sẵn).')
    parser.add_argument('--open-report', action='store_true', help='Mở báo cáo HTML khi hoàn tất hoặc có lỗi.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if sys.version_info < (3, 10):
        print('Cần Python 3.10 trở lên. Có thể dùng Python 3.12 cho môi trường này.')
        return 2
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / ('run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6])
    run_dir.mkdir()
    report = {'status': 'RUNNING', 'mode': args.mode, 'started_at': datetime.now(timezone.utc).isoformat(), 'data_root': str(args.data_root), 'run_directory': str(run_dir), 'notes': [
        'fast tính lại chỉ số OOF/holdout và tái lập CSV từ cache; không chạy lại mô hình hoặc huấn luyện.',
        'weights chạy lại 28 mô hình v4; nhánh v12 và chỉ số OOF/holdout vẫn lấy cache. Thời gian tùy CPU/GPU.',
        'Hậu xử lý bám Cell 4–9 gốc, gồm sửa dòng 147 và các sửa theo ID. Đây là tái lập kết quả lịch sử trên 550 mẫu gốc, không phải bộ dự đoán cho test mới.',
        'Khớp submission tham chiếu không đồng nghĩa đúng nhãn test. Không tính độ chính xác test/Kaggle khi không có nhãn thật.',
        'Chỉ số OOF được tính từ cache; kiểm tra này không tự xác nhận toàn bộ quy trình train không rò rỉ dữ liệu.',
    ]}
    started = time.perf_counter()
    exit_code = 1
    child_code = None
    old_out, old_err = sys.stdout, sys.stderr
    with (run_dir / 'verification.log').open('w', encoding='utf-8') as log:
        sys.stdout, sys.stderr = Tee(old_out, log), Tee(old_err, log)
        try:
            print('MNIST 10–20 | KIỂM TRA KẾT QUẢ CHO BTC')
            print(f'Chế độ: {args.mode} | Dữ liệu: {args.data_root}')
            if args.mode != 'train':
                bundle = resolve_bundle(args.data_root / 'v4_hier_multi_mlp', args.cache)
                preflight(args.data_root, args.mode, bundle)
                report['probability_bundle'] = str(bundle)
            else:
                for index in range(1, 10):
                    check_file(source_dir() / f'cell_{index:02d}.py')
                if not args.data_root.is_dir():
                    raise VerificationError(f'Thiếu thư mục dữ liệu: {args.data_root}')
            child_code = ensure_dependencies(args.mode, args.no_install)
            if child_code is None:
                global np, pd
                np, pd = importlib.import_module('numpy'), importlib.import_module('pandas')
                report['environment'] = {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__}
                if args.mode == 'train':
                    run_legacy_training(args.data_root, run_dir)
                    report.update(status='TRAIN_COMPLETED', summary='Đã chạy cell gốc. Chạy chế độ fast để đối chiếu các cache/submission vừa tạo.')
                    exit_code = 0
                else:
                    frame = read_ids(args.data_root / 'test.csv')
                    if len(frame) != 550:
                        raise VerificationError('test.csv phải là bộ 550 mẫu lịch sử.')
                    ids = frame['id'].to_numpy(dtype=str)
                    cached = load_cached_pillars(bundle, ids)
                    print('\n[1/3] Tính lại chỉ số OOF / holdout từ cache...')
                    report['oof'] = verify_oof_metrics(args.data_root)
                    pillars = cached
                    if args.mode == 'weights':
                        pillars = run_weights_inference_on_test(args.data_root, frame, report)
                        deltas = {key: {'max_abs_delta': float(np.abs(pillars[key] - cached[key]).max()), 'within_tolerance': bool(np.allclose(pillars[key], cached[key], atol=1e-4, rtol=1e-4))} for key in PILLAR_KEYS}
                        report['weights_vs_cache'] = deltas
                    print('\n[2/3] Tái lập đúng các bước Cell 4–9 gốc (gồm sửa theo dòng/ID)...')
                    generated = replay_historical_submissions(pillars)
                    print('\n[3/3] Đối chiếu ID, nhãn, thứ tự và SHA256...')
                    comparisons = []
                    for filename, labels in generated.items():
                        reference = args.data_root / 'v4_hier_multi_mlp' / filename
                        # Cell 7–9 originally write to the notebook's working directory.
                        if not reference.is_file() and (REPO_ROOT / filename).is_file():
                            reference = REPO_ROOT / filename
                        result = compare_submission(run_dir / filename, ids, labels, reference, run_dir)
                        comparisons.append(result)
                        print(f'  {result["status"]:17} {filename}: {result["message"]}')
                    report['submissions'] = comparisons
                    failed = any(item['status'] == 'FAIL' for item in comparisons)
                    incomplete = any(item['status'] == 'MISSING_REFERENCE' for item in comparisons)
                    drift = any(not item['within_tolerance'] for item in report.get('weights_vs_cache', {}).values())
                    report['status'] = 'FAIL' if failed or drift else ('INCOMPLETE' if incomplete else 'PASS')
                    report['summary'] = {'PASS': 'Đã tính lại chỉ số validation và mọi CSV đều khớp ID, nhãn, thứ tự với bản tham chiếu.', 'FAIL': 'Có kết quả khác bản tham chiếu/cache. Xem bảng đối chiếu và báo cáo JSON.', 'INCOMPLETE': 'Đã tạo CSV và tính chỉ số; còn thiếu bản tham chiếu để xác nhận toàn bộ.'}[report['status']]
                    exit_code = 0 if report['status'] == 'PASS' else 1
        except KeyboardInterrupt:
            report.update(status='CANCELLED', error='Người dùng đã dừng kiểm tra.', summary='Kiểm tra bị dừng.')
            exit_code = 130
        except Exception as exc:
            report.update(status='ERROR', error=str(exc), summary='Chưa hoàn tất kiểm tra. Chi tiết lỗi được lưu bên dưới và trong nhật ký.')
            print(f'\n[LỖI] {exc}')
            traceback.print_exc()
            exit_code = 2
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    # A dependency child writes its own authoritative report; do not overwrite it.
    if child_code is not None:
        return child_code
    report['elapsed_seconds'] = time.perf_counter() - started
    report['inputs'] = [{'path': str(path), 'sha256': sha256_file(path)} for path in sorted(INPUT_FILES) if path.is_file()]
    write_report(report, run_dir, args.output_dir)
    print(f'\n[{report["status"]}] {report["summary"]}\nBáo cáo: {args.output_dir / "verification_report.html"}')
    if args.open_report:
        webbrowser.open((args.output_dir / 'verification_report.html').as_uri())
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
