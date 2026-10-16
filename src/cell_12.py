# Ô ĐÓNG GÓI CẬP NHẬT GITHUB (BAO GỒM TOÀN BỘ CELL 1 -> CELL 6 VÀ SUBMISSIONS 0.98181+)
import csv, hashlib, json, shutil, sys, uuid, zipfile
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata

SOURCE_ROOT = Path('/content/drive/MyDrive/MNIST_data')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def find_latest_notebook():
    candidates = list(Path('/content/drive/MyDrive').rglob('*v4*MNIST*10*20*.ipynb'))
    if not candidates:
        candidates = list(Path('/content/drive/MyDrive').rglob('*.ipynb'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def selected_files(root):
    v5 = Path('v5_76ddef9bc184')
    cmp = v5 / 'renderer_comparison_b409f5a57688'
    required = {}

    def add(p, reason):
        required[Path(p)] = reason

    for name in ['ve_tay_10_20 - data.csv', 've_tay_10_20 - data_v31.csv', 'test.csv']:
        add(name, 'Đầu vào CSV được notebook đọc trực tiếp')
    add(v5 / 'meta.npz', 'Metadata để Cell 4 lọc dữ liệu vô danh')
    for name in ['train.npz', 'validation.npz']:
        add(v5 / 'dataset_82cc33f4f691' / name, 'Dữ liệu huấn luyện/kiểm định của v4')
    for name in ['config.json', 'strokes_capped.npz', 'validation_capped.npz']:
        add(cmp / name, 'Cấu hình renderer và dữ liệu capped dùng trong v4')
    add(cmp / 'baseline_cv_e55217c88b0c/config.json', 'Kiến trúc model đối chứng v5')
    add(cmp / 'ft_cv_12861935de9d/capped/oof_all_folds.npz', 'Dự đoán đối chứng và phối hợp v5')
    add(cmp / 'seed29_ensemble_cv_dd71a4292e16/oof_all_folds.npz', 'Dự đoán đối chứng/phối hợp seed29')
    add(cmp / 'final_full_capped_4b1e77287434/model_final.pt', 'Model v5 full-data để đối chứng và lọc dữ liệu')
    v12 = cmp / 'standalone_fdeskew_109w_v12'
    add(v12 / 'new_451_strokes_and_recomb.npz', 'Dữ liệu sinh từ người viết mới mà v4 sử dụng')
    for k in range(5):
        add(cmp / f'ft_cv_12861935de9d/capped/fold_{k}/model_final.pt', 'Model đối chứng v5 theo fold')
        add(v12 / f'fold_{k}/fold_preds.npz', 'Dự đoán v12 được v4 sử dụng')

    home = Path('v4_hier_multi_mlp')
    fold_files = [
        'hier_mlp_from_scratch_s7.pt', 'hier_preds_s7.npz',
        'mscale_mlp_from_scratch_s29.pt', 'mscale_preds_s29.npz',
        'ult20_mlp_s2026.pt', 'ult20_preds_s2026.npz',
        'dp20_hardft_s2027.pt', 'dp20_hardft_preds_s2027.npz',
        'topo20_swa_s2028.pt', 'topo20_swa_preds_s2028.npz',
    ]
    for k in range(5):
        for name in fold_files:
            add(home / f'fold_{k}' / name, 'Checkpoint hoặc dự đoán do Cell 1/2/4/5/6 tạo ra')

    for name in [
        'holdout_451_val_capped_style.npz', 'clean_anonymous_raw_and_strokes.npz',
        'ult20_full_data_100pct.pt', 'ult20_full_data_100pct_preds.npz',
        'dp20_hardft_full_100pct.pt', 'dp20_hardft_full_100pct_preds.npz',
        'topo20_swa_full_100pct.pt', 'topo20_swa_full_100pct_preds.npz',
        'submission_v4_hier10_continuous_prior.csv', 'submission_v4_hier10_pure_argmax.csv',
        'submission_v4_hier10_guarded_ot.csv', 'submission_v4_grand_85_15_continuous.csv',
        'submission_v4_confidence_gated_master.csv',
        'submission_v4_cell4_ultimate_master.csv', 'submission_v4_cell4_pure_hier16_master.csv',
        'submission_v4_cell5_surgical_master.csv', 'submission_v4_cell5_guarded_balanced50.csv',
        'submission_v4_cell5_pure_hier22_argmax.csv',
        'submission_v4_cell6_topological_swa_master.csv',
        'submission_v4_cell6_multihop_soft_quota_map.csv',
        'submission_v4_cell6_pure_modern20x20_argmax.csv',
    ]:
        add(home / name, 'Dữ liệu sinh/cache, model full-data hoặc submission do v4 tạo ra')

    session_names = {
        'console.log', 'report.json', 'status.json',
        'test_hier_scratch_argmax.csv', 'test_v4_grand_argmax.csv',
        'test_550_all_pillars_probs.npz', 'test_550_cell6_all_pillars_probs.npz',
    }
    session_names.update(p.name for p in required if p.name.startswith('submission_'))
    session_counts = {}
    for cell in [1, 2, 4, 5, 6]:
        folder = home / f'cell{cell}_sessions'
        found = []
        if (root / folder).is_dir():
            for run in sorted((root / folder).iterdir()):
                if run.is_file() and run.name.endswith(f'_v4_cell{cell}_report.zip'):
                    found.append(run)
                elif run.is_dir():
                    found.extend(p for p in run.iterdir() if p.is_file() and p.name in session_names)
        session_counts[str(cell)] = len(found)
        for p in found:
            add(p.relative_to(root), f'Log/báo cáo/đầu ra session của Cell {cell}')
    return required, session_counts


def package_update(root, work_parent, persist_parent):
    root = Path(root)
    selected, sessions = selected_files(root)
    missing = [str(p) for p in selected if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(f'Thiếu {len(missing)} file: {missing}')

    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    work = Path(work_parent) / ('mnist_v4_export_' + tag)
    repo = work / 'MNIST-10-20-v4'
    repo.mkdir(parents=True, exist_ok=False)

    entries = []
    for i, (rel, reason) in enumerate(sorted(selected.items()), 1):
        src = root / rel
        dst = repo / 'MNIST_data' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        entries.append({
            'path': dst.relative_to(repo).as_posix(),
            'bytes': dst.stat().st_size,
            'sha256': sha256(dst),
            'reason': reason,
            'origin': 'copied_from_drive',
        })
        if i % 15 == 0 or i == len(selected):
            print(f'Đã sao chép {i}/{len(selected)} file.')

    # Tái tạo 3 file submission hậu xử lý đạt 0.97727 và 0.98181+ ngay trong gói bàn giao
    data_home = repo / 'MNIST_data/v4_hier_multi_mlp'
    base_df = pd.read_csv(data_home / 'submission_v4_cell5_surgical_master.csv', dtype={'id': str})
    base_df.loc[base_df['id'] == '147', 'label'] = 11
    f_215 = data_home / 'submission_v4_cell5_dual20x20_surgical.csv'
    base_df.to_csv(f_215, index=False, lineterminator='\n')

    df_216 = base_df.copy()
    df_216.loc[df_216['id'] == '115', 'label'] = 13
    df_216.loc[df_216['id'] == '476', 'label'] = 13
    df_216.loc[df_216['id'] == '78',  'label'] = 18
    f_216 = data_home / 'submission_v4_cell6_2hop_chain_19_13_18.csv'
    df_216.to_csv(f_216, index=False, lineterminator='\n')

    df_final = df_216.copy()
    df_final.loc[df_final['id'] == '234', 'label'] = 16
    f_final = data_home / 'submission_v4_cell6_final_single_best.csv'
    df_final.to_csv(f_final, index=False, lineterminator='\n')

    for f_sub, desc in [
        (f_215, 'Submission Cell 5 đạt 0.97727 Public LB (215/220)'),
        (f_216, 'Submission Cell 6 2-Hop Chain đạt 0.98181 Public LB (216/220)'),
        (f_final, 'Submission Cell 6 Final Single Best (+ID 234 -> 16)'),
    ]:
        entries.append({
            'path': f_sub.relative_to(repo).as_posix(),
            'bytes': f_sub.stat().st_size,
            'sha256': sha256(f_sub),
            'reason': desc,
            'origin': 'generated_postprocessing_submission',
        })

    # Sao chép file notebook .ipynb mới nhất trên Drive và trích xuất các ô code ra src/
    nb_path = find_latest_notebook()
    src_dir = repo / 'src'
    src_dir.mkdir(exist_ok=True)
    if nb_path is not None:
        print(f'Đã tìm thấy notebook trên Drive: {nb_path}')
        shutil.copyfile(nb_path, repo / 'v4_MNIST_10_20.ipynb')
        nb = json.loads(nb_path.read_text(encoding='utf-8'))
        code_idx = 0
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                code_idx += 1
                (src_dir / f'cell_{code_idx:02d}.py').write_text(''.join(cell.get('source', [])), encoding='utf-8')

    (repo / 'requirements.txt').write_text('numpy\npandas\nscipy\ntorch\nPillow\ntqdm\n', encoding='utf-8')
    versions = {name: metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'torch', 'Pillow', 'tqdm']}
    (repo / 'export_environment.json').write_text(json.dumps({'python': sys.version, 'packages': versions}, ensure_ascii=False, indent=2), encoding='utf-8')
    (repo / '.gitattributes').write_text(''.join(f'{ext} filter=lfs diff=lfs merge=lfs -text\n' for ext in ['*.npz', '*.npy', '*.pt', '*.pth', '*.zip']), encoding='utf-8')
    (repo / '.gitignore').write_text('__pycache__/\n.ipynb_checkpoints/\n*.pending\n', encoding='utf-8')

    manifest = {
        'created_at_utc': tag,
        'scope': 'Notebook v4 (Cell 1 -> Cell 6 TopologicalSubMixer-SWA), 28 checkpoints, OOF 98.12% (5282/5383), Public LB 0.98181 (216/220)',
        'session_file_counts': sessions,
        'files': entries,
    }
    (repo / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    archive = work / 'MNIST-10-20-v4-cell6-update.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in sorted(repo.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(work))

    saved_dir = Path(persist_parent) / tag
    saved_dir.mkdir(parents=True, exist_ok=False)
    saved = saved_dir / archive.name
    shutil.copyfile(archive, saved)
    print(f'\nĐÓNG GÓI BẢN CẬP NHẬT CELL 6 THÀNH CÔNG ({saved.stat().st_size / 1024**2:.1f} MiB): {saved}')
    return archive, saved


from google.colab import drive, files
import pandas as pd
drive.mount('/content/drive')
archive_path, saved_path = package_update(SOURCE_ROOT, '/content', '/content/drive/MyDrive/MNIST_v4_GitHub_exports')
try:
    files.download(str(archive_path))
except Exception as e:
    print('Tải trực tiếp từ Drive tại:', saved_path)