"""实验 S：加载三轮已有 V1 权重，仅替换深度×类型×方向推理抽样。"""

import argparse
import copy
import gc
import glob
import hashlib
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
if sys.platform == 'win32' and (PROJECT / '.runtime_deps').is_dir():
    sys.path.insert(1, str(PROJECT / '.runtime_deps'))

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

import deepspatial as ds
from deepspatial.data_utils.sampling import depth_type_source_schedule


def file_sha256(path):
    """记录使用的原始权重身份，不复制或重新保存模型。"""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_sampling(observed, config, seeds, bin_width_um, grid_origin_um):
    """只生成来源调度，统计重复抽样的计数方差；不求解 ODE 或读取隐藏真值。"""
    classes = sorted({str(c) for a in observed for c in a.obs[config['data']['label_key']]})
    codes = [pd.Categorical(a.obs[config['data']['label_key']].astype(str), categories=classes).codes for a in observed]
    thickness = config['reconstruction']['thickness']
    records = []
    for gap in range(len(observed) - 1):
        z0, z1 = (float(a.obs[config['data']['z_key']].iloc[0]) for a in observed[gap:gap + 2])
        total = max(1, int((len(codes[gap]) + len(codes[gap + 1])) / 2 * (z1 - z0) / thickness))
        samples, expected, keys = [], None, None
        for seed in range(seeds):
            schedule = depth_type_source_schedule(codes[gap], codes[gap + 1], z0, z1, total,
                                                  bin_width_um, grid_origin_um, rng=np.random.default_rng(seed))
            frame = pd.DataFrame(schedule['budget'])
            # 将两个方向加总，直接检查每个固定薄层的来源类型计数。
            grouped = frame.groupby(['depth_bin_index', 'source_class_code'])[['expected_n', 'allocated_n']].sum()
            if keys is None:
                keys, expected = grouped.index, grouped.expected_n.to_numpy()
            samples.append(grouped.allocated_n.to_numpy())
        values = np.asarray(samples)
        for j, (bin_id, code) in enumerate(keys):
            records.append(dict(gap_start_um=z0, gap_end_um=z1, depth_bin_index=int(bin_id),
                                bin_center_um=float(grid_origin_um + bin_id * bin_width_um),
                                source_cell_class=classes[code], expected_n=float(expected[j]),
                                audit_seeds=seeds, mean_n=float(values[:, j].mean()),
                                variance_n=float(values[:, j].var()), zero_fraction=float((values[:, j] == 0).mean()),
                                minimum_n=int(values[:, j].min()), maximum_n=int(values[:, j].max())))
    return pd.DataFrame(records)


def run_experiment(source_root, output_root, holdouts=(1, 2, 3), sampling_seed=42,
                   bin_width_um=10.0, grid_origin_um=0.0, device='auto', audit_seeds=100,
                   audit_only=False):
    """逐轮加载已完成实验，先排除隐藏文件；从不调用 setup_data/build_model/fit。"""
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if not output_root.is_relative_to(PROJECT) or output_root == PROJECT:
        raise ValueError('Output must be a new subdirectory inside HM_DeepSpital.')
    if (not holdouts or len(set(holdouts)) != len(holdouts)
            or any(type(i) is not int or i not in (1, 2, 3) for i in holdouts)):
        raise ValueError('holdouts must be a nonempty, unique subset of [1, 2, 3].')
    if (not np.isfinite([bin_width_um, grid_origin_um]).all() or bin_width_um <= 0
            or type(audit_seeds) is not int or audit_seeds < 0 or (audit_only and audit_seeds == 0)):
        raise ValueError('Use a positive finite bin width, finite origin and nonnegative audit_seeds; audit-only needs seeds.')
    plans = []
    for heldout in holdouts:
        folder, output = source_root / f'holdout_{heldout}', output_root / f'holdout_{heldout}'
        if output.exists():
            raise FileExistsError(f'Output already exists: {output}; select a new output root.')
        config = json.loads((folder / 'run_config.json').read_text(encoding='utf-8-sig'))
        if config['run']['heldout_indices'] != [heldout]:
            raise ValueError(f'Checkpoint fold and heldout configuration disagree: {folder}')
        checkpoint = folder / 'model/model.ckpt'
        if not checkpoint.is_file() or not checkpoint.with_name('config.json').is_file():
            raise FileNotFoundError(f'Saved model and metadata are required: {checkpoint}')
        paths = [Path(p) for p in glob.glob(str(PROJECT / config['input_glob']), recursive=True)]
        paths.sort(key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', p.name)])
        if len(paths) != len(config['slice_z_um']) or heldout >= len(paths) or len(paths) < 3:
            raise ValueError('Input file count, depths and heldout index must match the original run.')
        visible = [path for i, path in enumerate(paths) if i != heldout]
        if sorted(p.name for p in visible) != sorted(Path(p).name for p in config['run']['visible_files']):
            raise ValueError('Visible input filenames differ from the completed training run.')
        plans.append((heldout, folder, output, config, checkpoint, paths))
    completed = []
    for heldout, folder, output, config, checkpoint, paths in plans:
        print(f'Experiment S: holdout {heldout}; checkpoint {checkpoint}', flush=True)
        model = result = None
        observed = []
        try:
            for i, (path, depth) in enumerate(zip(paths, config['slice_z_um'])):
                if i == heldout:
                    continue
                a = ad.read_h5ad(path)
                a.obs[config['data']['z_key']] = float(depth)
                observed.append(a)
            del a
            run_config = copy.deepcopy(config)
            run_config['reconstruction'].update(source_sampling='depth_type_stratified', device=device,
                                                 depth_bin_width_um=bin_width_um, depth_grid_origin_um=grid_origin_um)
            run_config['experiment_s'] = dict(
                source_checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                sampling_implementation_sha256=file_sha256(PROJECT / 'deepspatial/data_utils/sampling.py'),
                core_implementation_sha256=file_sha256(PROJECT / 'deepspatial/core.py'),
                checkpoint_weights='EMA', sampling_seed=sampling_seed, audit_seeds=audit_seeds,
                audit_only=audit_only, training_performed=False,
                schedule_storage='generated.h5ad:obs (z_coord, depth_bin_index, parent_id, source_z, source_cell_index, source_direction, source_cell_class)',
            )
            output.mkdir(parents=True, exist_ok=False)
            (output / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
            if audit_seeds:
                print(f'Holdout {heldout}: auditing {audit_seeds} sampling seeds without ODE', flush=True)
                audit = audit_sampling(observed, config, audit_seeds, bin_width_um, grid_origin_um)
                audit.to_csv(output / 'sampling_audit.csv', index=False)
            if not audit_only:
                model = ds.DeepSpatial()
                model.load_checkpoint(str(checkpoint))
                # 单独重置推理随机数；模型恢复和审计不影响本次来源调度。
                pl.seed_everything(sampling_seed, workers=True)
                result = model.reconstruct_full_volume(observed, **run_config['reconstruction'])
                result.uns['experiment_s'] = run_config['experiment_s']
                result.write_h5ad(output / 'generated.h5ad')
                result.uns['depth_type_sampling']['budget'].to_csv(output / 'sampling_budget.csv', index=False)
                counts = result.obs.groupby(['depth_bin_index', 'depth_bin_center_um', 'source_direction',
                                             'source_cell_class', 'cell_class'], observed=True).size().rename('generated_n').reset_index()
                counts.to_csv(output / 'sampling_counts.csv', index=False)
                summary = dict(heldout_id=heldout, generated_cells=result.n_obs,
                               class_switches=int((result.obs.cell_class.astype(str) != result.obs.source_cell_class.astype(str)).sum()),
                               output=str(output))
            else:
                summary = dict(heldout_id=heldout, audit_only=True, output=str(output))
            completed.append(summary)
            print(f'Experiment S completed: {summary}', flush=True)
        finally:
            del result, model, observed
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return completed


def main():
    """支持服务器命令行；默认复用三个已完成 V1 模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=PROJECT / 'outputs/v1_holdouts_1_2_3')
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'outputs/s_holdouts_1_2_3')
    parser.add_argument('--holdouts', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--sampling-seed', type=int, default=42)
    parser.add_argument('--bin-width-um', type=float, default=10.0)
    parser.add_argument('--grid-origin-um', type=float, default=0.0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--audit-seeds', type=int, default=100)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    run_experiment(args.source_root, args.output_root, tuple(args.holdouts), args.sampling_seed,
                   args.bin_width_um, args.grid_origin_um, args.device, args.audit_seeds, args.audit_only)


if __name__ == '__main__':
    main()
