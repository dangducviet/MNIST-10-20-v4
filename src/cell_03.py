# v4_MNIST_10_20 — CELL 3 (5 giây, không train): Kiểm chứng Confidence-Advantage Gated Fusion trên 4 Cổng OOF & Xuất File Nộp Cuối Cùng.
from pathlib import Path
import json
import numpy as np
import pandas as pd
from google.colab import drive, files

drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/MNIST_data')
run_v5 = root / 'v5_76ddef9bc184'
data_dir = run_v5 / 'dataset_82cc33f4f691'
cmp_dir = run_v5 / 'renderer_comparison_b409f5a57688'
v4_home = root / 'v4_hier_multi_mlp'
v12_dir = cmp_dir / 'standalone_fdeskew_109w_v12'

def load_z(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

va = load_z(data_dir / 'validation.npz')
valid_oof_mask = ~np.isin(va['rows'], [458, 1767, 2213, 4355])
ym_oof = va['y'][valid_oof_mask]

# Nạp xác suất OOF, Holdout (451) và Test (550) đã lưu cache của Hier10 (Cell 1 + Cell 2) và Orig/v12/v5
oof_s7_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
oof_s7_cap = np.zeros((len(va['y']), 11), dtype=np.float32)
oof_s29_raw = np.zeros((len(va['y']), 11), dtype=np.float32)
oof_s29_cap = np.zeros((len(va['y']), 11), dtype=np.float32)
oof_v12_raw = np.zeros((len(va['y']), 11), dtype=np.float32)

ho_s7_raw_l, ho_s7_cap_l, ho_s29_raw_l, ho_s29_cap_l, ho_v12_l, y_ho_l = [], [], [], [], [], []
te_s7_l, te_s29_l, te_v12_l = [], [], []

for k in range(5):
    vi = np.flatnonzero(va['fold'] == k)
    z7 = load_z(v4_home / f'fold_{k}' / 'hier_preds_s7.npz')
    z29 = load_z(v4_home / f'fold_{k}' / 'mscale_preds_s29.npz')
    z12 = load_z(v12_dir / f'fold_{k}' / 'fold_preds.npz')

    oof_s7_raw[vi], oof_s7_cap[vi] = z7['oof_raw'], z7['oof_cap']
    oof_s29_raw[vi], oof_s29_cap[vi] = z29['oof_raw'], z29['oof_cap']
    oof_v12_raw[vi] = 0.50 * z12['oof_fd'] + 0.50 * (0.55 * z12['oof_o_s7'] + 0.45 * z12['oof_o_s29'])

    ho_s7_raw_l.append(z7['ho_raw']); ho_s7_cap_l.append(z7['ho_cap'])
    ho_s29_raw_l.append(z29['ho_raw']); ho_s29_cap_l.append(z29['ho_cap'])
    ho_v12_l.append(0.50 * z12['ho_fd'] + 0.50 * (0.55 * z12['ho_o_s7'] + 0.45 * z12['ho_o_s29']))

    te_s7_l.append(z7['te_prob'])
    te_s29_l.append(z29['te_prob'])
    te_v12_l.append(0.50 * z12['te_fd'] + 0.50 * (0.55 * z12['te_o_s7'] + 0.45 * z12['te_o_s29']))

z_inc = load_z(cmp_dir / 'ft_cv_12861935de9d' / 'capped' / 'oof_all_folds.npz')
z_s29_v5 = load_z(cmp_dir / 'seed29_ensemble_cv_dd71a4292e16' / 'oof_all_folds.npz')
oof_v5_raw = 0.65 * z_inc['shift9_raw_prob'] + 0.35 * z_s29_v5['seed29_shift9_raw_prob']
oof_v5_cap = 0.65 * z_inc['shift9_capped_style_prob'] + 0.35 * z_s29_v5['seed29_shift9_capped_style_prob']

oof_hier10_raw = 0.50 * oof_s7_raw + 0.50 * oof_s29_raw
oof_hier10_cap = 0.50 * oof_s7_cap + 0.50 * oof_s29_cap
oof_anchor_raw = 0.65 * oof_v12_raw + 0.35 * oof_v5_raw

def confidence_gated_blend(p_hier, p_anchor, base_w_hier=0.82, fallback_w_hier=0.30, margin=-0.05):
    """
    Khi Hier10 tự tin (conf_hier >= conf_anchor + margin), dùng 82% Hier10 + 18% Anchor.
    Khi Hier10 mất tự tin do cắt khung 18x18 (conf_hier < conf_anchor - 0.05), hạ trọng số Hier10 xuống 30% và dùng 70% Anchor toàn ảnh!
    """
    conf_h = p_hier.max(axis=1, keepdims=True)
    conf_a = p_anchor.max(axis=1, keepdims=True)
    w_h = np.where(conf_h >= conf_a + margin, base_w_hier, fallback_w_hier).astype(np.float32)
    p_out = w_h * p_hier + (1.0 - w_h) * p_anchor
    return p_out / p_out.sum(axis=1, keepdims=True)

oof_gated_cap = confidence_gated_blend(oof_hier10_cap, oof_v5_cap)[valid_oof_mask]
oof_gated_raw = confidence_gated_blend(oof_hier10_raw, oof_anchor_raw)[valid_oof_mask]
inc_cap_ok = (z_inc['shift9_capped_style_prob'][valid_oof_mask].argmax(1) + 10) == ym_oof
gated_cap_ok = (oof_gated_cap.argmax(1) + 10) == ym_oof
gated_raw_ok = (oof_gated_raw.argmax(1) + 10) == ym_oof

print('=== KIỂM CHỨNG CONFIDENCE-GATED FUSION TRÊN 4.932 MẪU OOF ===')
print(f"1. OOF Capped_Style: {int(gated_cap_ok.sum())}/4932 ({gated_cap_ok.mean()*100:.4f}%) | "
      f"vs Incumbent (4738): fixed={int((~inc_cap_ok & gated_cap_ok).sum())}, broken={int((inc_cap_ok & ~gated_cap_ok).sum())}, net=+{int(gated_cap_ok.sum() - inc_cap_ok.sum())}")
print(f"2. OOF Raw:          {int(gated_raw_ok.sum())}/4932 ({gated_raw_ok.mean()*100:.4f}%)")

# Áp dụng chính xác cùng hàm Confidence-Gated Fusion lên 550 mẫu Test
te_hier10 = 0.50 * np.mean(te_s7_l, axis=0) + 0.50 * np.mean(te_s29_l, axis=0)
te_anchor = np.mean(te_v12_l, axis=0)
te_gated_prob = confidence_gated_blend(te_hier10, te_anchor, base_w_hier=0.82, fallback_w_hier=0.30, margin=-0.05)

# Hiệu chỉnh tiên nghiệm nhẹ (Continuous Damped Scaling) trên xác suất đã qua cổng bảo vệ
p_cal = np.clip(te_gated_prob.astype(np.float64), 1e-12, 1.0)
for _ in range(30):
    p_cal *= (50.0 / np.clip(p_cal.sum(axis=0, keepdims=True), 1e-6, None)) ** 0.50
    p_cal /= p_cal.sum(axis=1, keepdims=True)

raw_arg = te_gated_prob.argmax(axis=1)
cal_arg = p_cal.argmax(axis=1)
final_c = np.where((te_gated_prob.max(axis=1) < 0.68) & (te_gated_prob[np.arange(550), cal_arg] >= 0.36), cal_arg, raw_arg) + 10

test_ids = pd.read_csv(root / 'test.csv', dtype={'id': str})['id'].to_numpy()
out_sub = pd.DataFrame({'id': test_ids, 'label': final_c.astype(np.int64)})
out_path = v4_home / 'submission_v4_confidence_gated_master.csv'
out_sub.to_csv(out_path, index=False, lineterminator='\n')

c2_cont = pd.read_csv(v4_home / 'submission_v4_hier10_continuous_prior.csv')['label'].to_numpy()
diff_vs_c2 = np.flatnonzero(final_c != c2_cont)
print('\n=== SO SÁNH VỚI BẢN CELL 2 (submission_v4_hier10_continuous_prior.csv) ===')
print(f'Số dòng được cổng bảo vệ khôi phục khỏi lỗi cắt khung 18x18: {len(diff_vs_c2)} dòng -> IDs: {diff_vs_c2.tolist()}')
for idx in diff_vs_c2:
    print(f'  id={idx:3d}: Cell2_Hier10={int(c2_cont[idx])} (conf={te_hier10[idx].max():.3f}) -> Gated_Master={int(final_c[idx])} (anchor_conf={te_anchor[idx].max():.3f})')
print('Phân bố 11 lớp của Gated Master:', {int(c): int((final_c == c).sum()) for c in range(10, 21)})
print('Đã lưu và tải về:', out_path)
files.download(str(out_path))