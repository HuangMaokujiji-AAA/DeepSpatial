"""G 续跑：检查完成状态，复用原配对缓存和 S 调度，从 last.ckpt 恢复训练。"""
import argparse
import copy
import gc
import glob
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import pickle

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
if sys.platform == 'win32' and (PROJECT / '.runtime_deps').is_dir():
    sys.path.insert(1, str(PROJECT / '.runtime_deps'))

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from scipy import sparse
import torch
from torch.utils.data import DataLoader, Dataset

from deepspatial import DeepSpatial
from deepspatial.bio_prior import NicheFeaturizer
from deepspatial.data_utils.local_matching import matching_parameters
from deepspatial.data_utils.sampling import replay_source_schedule
from examples.run_g import file_sha256, schedule_sha256, set_condition, state_sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_schedule(obs, z_key, expected):
    """兼容原 run_g 以操作系统默认 CSV 换行计算的历史哈希，不改原实验实现。"""
    digest = schedule_sha256(obs, z_key)
    if digest == expected:
        return digest
    fields = ['parent_id', 'source_cell_id', 'source_z', 'source_cell_index', 'source_direction',
              'source_cell_class', 'depth_bin_index', z_key]
    frame = obs[fields].copy()
    for field in ('parent_id', 'source_cell_id', 'source_direction', 'source_cell_class'):
        frame[field] = frame[field].astype(str)
    for newline in ('\n', '\r\n'):
        digest = hashlib.sha256(frame.to_csv(index=False, float_format='%.17g', lineterminator=newline).encode()).hexdigest()
        if digest == expected:
            return digest
    raise ValueError('Generated source/type/depth schedule differs from the original S schedule.')


def inspect_results(output_root=PROJECT / 'outputs/g_holdouts_1_2_3', holdouts=(1, 2, 3),
                    modes=('real', 'zero', 'shuffle')):
    """只读检查；completed 需要满轮数的检查点、最终权重及可读且调度一致的 H5AD。"""
    rows = []
    for heldout in holdouts:
        fold = Path(output_root) / f'holdout_{heldout}'
        shared = read_json(fold / 'shared/run_config.json')
        target = shared['training']['max_epochs']
        for mode in modes:
            folder = fold / mode
            row = dict(holdout=heldout, mode=mode, completed_epochs=0, target_epochs=target,
                       action='train_new', generated_cells=None, reason='尚未开始')
            last = folder / 'checkpoints/last.ckpt'
            config = read_json(folder / 'run_config.json') if (folder / 'run_config.json').is_file() else None
            if last.is_file():
                checkpoint = torch.load(last, map_location='cpu', weights_only=False)
                if not checkpoint.get('state_dict') or any(not np.isfinite(t.cpu().numpy()).all() for t in checkpoint['state_dict'].values()):
                    raise ValueError(f'Checkpoint contains invalid weights: {last}')
                row['completed_epochs'] = int(checkpoint['epoch']) + 1
                row['global_step'] = int(checkpoint['global_step'])
                row.update(action='resume_training', reason='恢复最后一个完整 epoch 的检查点')
                if row['completed_epochs'] >= target:
                    row.update(action='reconstruct', reason='训练已完成，补齐模型导出或生成结果')
                    required = ['model/model.ckpt', 'model/config.json', 'generated.h5ad',
                                'sampling_budget.csv', 'generation_support_counts.csv']
                    if config and all((folder / p).is_file() for p in required):
                        try:
                            saved = torch.load(folder / 'model/model.ckpt', map_location='cpu', weights_only=False)['state_dict']
                            if set(saved) != set(checkpoint['state_dict']) or any(
                                    not torch.equal(saved[k], checkpoint['state_dict'][k]) for k in saved):
                                raise ValueError('Final model differs from the last training checkpoint.')
                            result = ad.read_h5ad(folder / 'generated.h5ad')
                            expected = shared['experiment_g']['source_schedule_sha256']
                            verify_schedule(result.obs, shared['data']['z_key'], expected)
                            if (config['experiment_g'].get('generated_schedule_sha256') != expected
                                    or result.uns['experiment_g'].get('generated_schedule_sha256') != expected):
                                raise ValueError('Generated schedule identity is missing or differs from S.')
                            if (config['experiment_g']['niche_mode'] != mode
                                    or result.uns['experiment_g']['niche_mode'] != mode
                                    or config['run']['heldout_indices'] != [heldout]):
                                raise ValueError('Fold/condition identity is inconsistent.')
                            arrays = [result.X, *result.obsm.values(), *result.layers.values()]
                            if any(not np.isfinite(a.data if sparse.issparse(a) else np.asarray(a)).all() for a in arrays):
                                raise ValueError('Generated arrays contain NaN/Inf.')
                            budget = pd.read_csv(folder / 'sampling_budget.csv')
                            counts = pd.read_csv(folder / 'generation_support_counts.csv')
                            if int(budget.allocated_n.sum()) != result.n_obs or int(counts.generated_n.sum()) != result.n_obs:
                                raise ValueError('Budget/support totals differ from generated cells.')
                            row.update(action='skip', generated_cells=int(result.n_obs), reason='训练及结果完整，跳过')
                            del result, saved
                        except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
                            row['reason'] = f'训练完成，但结果需修复：{exc}'
                del checkpoint
            elif folder.exists() and any(folder.rglob('*.ckpt')):
                raise ValueError(f'Checkpoint files exist but last.ckpt is missing: {folder}; inspect before restarting.')
            elif config and config['experiment_g'].get('generated_schedule_sha256'):
                raise ValueError(f'Completed result claims exist without last.ckpt: {folder}; do not retrain automatically.')
            rows.append(row)
            gc.collect()
    return pd.DataFrame(rows)


class CachedPairs(Dataset):
    """与原训练相同的字典式轨迹，只从已保存端点索引构造。"""
    def __init__(self, tensors, stats):
        self.tensors, self.sampling_stats = tensors, stats

    def __len__(self):
        return len(self.tensors['cell0'])

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.tensors.items()}


def prepare_cached_data(config, observed, shared):
    """按原 setup_data 的变换和 DataLoader 初始化顺序重建，完全不重新求 UOT。"""
    data = config['data']
    model = DeepSpatial()
    model.spatial_key, model.z_key, model.label_key = (data[k] for k in ('spatial_key', 'z_key', 'label_key'))
    model.source_sampling = data['source_sampling']
    model.local_matching_config = copy.deepcopy(data['local_matching_config'])
    model.data_config = {k: data[k] for k in ('balanced_fraction', 'source_sampling')}
    model._validate_slices(observed, require_order=True)
    model.gene_names = observed[0].var_names.tolist()
    model.niche_featurizer = NicheFeaturizer(**data['niche_config']).fit(observed, model.label_key)
    model.categories = pd.Index(model.niche_featurizer.categories)
    model._normalize_spatial(observed)
    with np.load(shared / 'pair_indices.npz', allow_pickle=False) as cache:
        indices = {key: cache[key].copy() for key in ('slice0', 'cell0', 'slice1', 'cell1')}
    n = len(indices['cell0'])
    if not n or any(v.shape != (n,) or not np.issubdtype(v.dtype, np.integer) for v in indices.values()):
        raise ValueError('Invalid cached pair arrays.')
    if (not np.array_equal(indices['slice1'], indices['slice0'] + 1)
            or (indices['slice0'] < 0).any() or (indices['slice1'] >= len(observed)).any()):
        raise ValueError('Cached pairs must join adjacent visible slices.')
    tensors = {key: torch.from_numpy(value) for key, value in indices.items()}
    endpoints = []
    for suffix in ('0', '1'):
        arrays = dict(x=np.empty((n, 2), np.float32), g=np.empty((n, len(model.gene_names)), np.float32),
                      c=np.empty((n, len(model.categories)), np.float32), z=np.empty((n, 1), np.float32))
        physical = np.empty((n, 2))
        for i, a in enumerate(observed):
            mask = indices['slice' + suffix] == i
            parents = indices['cell' + suffix][mask]
            if (parents < 0).any() or (parents >= a.n_obs).any():
                raise ValueError('Cached parent index is out of range.')
            arrays['x'][mask] = a.obsm['spatial_norm'][parents]
            genes = a.X[parents]
            arrays['g'][mask] = genes.toarray() if sparse.issparse(genes) else genes
            codes = pd.Categorical(a.obs[model.label_key], categories=model.categories).codes
            arrays['c'][mask] = np.eye(len(model.categories), dtype=np.float32)[codes[parents]]
            arrays['z'][mask] = float(a.obs.z_norm.iloc[0])
            physical[mask] = a.obsm[model.spatial_key][parents]
        tensors.update({key + suffix: torch.from_numpy(value) for key, value in arrays.items()})
        endpoints.append(physical)
    tensors['delta_z'] = tensors['z1'] - tensors['z0']
    # 原实现先以 float64 深度作差后写入 float32。
    for i in range(len(observed) - 1):
        mask = indices['slice0'] == i
        tensors['delta_z'][mask] = float(observed[i + 1].obs.z_norm.iloc[0] - observed[i].obs.z_norm.iloc[0])
        gap = float(observed[i + 1].obs[model.z_key].iloc[0] - observed[i].obs[model.z_key].iloc[0])
        radius, _ = matching_parameters(model.local_matching_config, gap)
        if (np.linalg.norm(endpoints[1][mask] - endpoints[0][mask], axis=1) > radius + 1e-8).any():
            raise ValueError('Cached pairs violate the original local radius.')
    if not torch.equal(tensors['c0'], tensors['c1']):
        raise ValueError('Cached pairs contain cross-type matches.')
    model.dataset = CachedPairs(tensors, read_json(shared / 'trajectory_sampling.json'))
    model.train_loader = DataLoader(model.dataset, batch_size=data['batch_size'], shuffle=True,
                                   num_workers=data['num_workers'])
    batch = next(iter(model.train_loader))  # 保留原 setup_data 消费 Torch 随机数的步骤。
    model.gene_dim, model.num_classes = batch['g0'].shape[1], batch['c0'].shape[1]
    return model


def resume_experiment(source_root=PROJECT / 'outputs/s_holdouts_1_2_3',
                      output_root=PROJECT / 'outputs/g_holdouts_1_2_3', holdouts=(1, 2, 3),
                      modes=('real', 'zero', 'shuffle'), device='auto'):
    """只处理未完成组；max_epochs 始终表示原计划的总轮数，而非追加轮数。"""
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if not output_root.is_relative_to(PROJECT) or output_root == PROJECT:
        raise ValueError('Output must remain inside HM_DeepSpital.')
    if (not holdouts or len(set(holdouts)) != len(holdouts) or not set(holdouts) <= {1, 2, 3}
            or not modes or len(set(modes)) != len(modes) or not set(modes) <= {'real', 'zero', 'shuffle'}):
        raise ValueError('Use unique holdouts from [1,2,3] and modes from real/zero/shuffle.')
    plan = inspect_results(output_root, holdouts, modes)
    print(plan.to_string(index=False), flush=True)
    for heldout in holdouts:
        pending = plan[(plan.holdout == heldout) & (plan.action != 'skip')]
        if pending.empty:
            continue
        fold, source = output_root / f'holdout_{heldout}', source_root / f'holdout_{heldout}'
        shared = fold / 'shared'
        config = read_json(shared / 'run_config.json')
        protocol = config['experiment_g']
        if config['run']['heldout_indices'] != [heldout] or config['training']['devices'] != 1:
            raise ValueError('Fold identity or single-device setting is inconsistent.')
        if file_sha256(shared / 'pair_indices.npz') != protocol['shared_pair_sha256']:
            raise ValueError('Original pairing cache checksum differs.')
        for name, expected in protocol['implementation_sha256'].items():
            if file_sha256(PROJECT / name.replace('__', '/')) != expected:
                raise ValueError(f'Original scientific implementation changed: {name}; sync the original code before resuming.')
        if file_sha256(source / 'generated.h5ad') != protocol['source_s_h5ad_sha256']:
            raise ValueError('Source S file differs from the original G run.')
        paths = sorted((Path(p) for p in glob.glob(str(PROJECT / config['input_glob']), recursive=True)),
                       key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', p.name)])
        if len(paths) != len(config['slice_z_um']):
            raise ValueError('Input count differs from the saved run.')
        if sorted(p.name for i, p in enumerate(paths) if i != heldout) != sorted(Path(p).name for p in config['run']['visible_files']):
            raise ValueError('Visible filenames differ from the saved run.')
        observed = []
        for i, (path, depth) in enumerate(zip(paths, config['slice_z_um'])):
            if i == heldout:
                continue
            a = ad.read_h5ad(path)
            a.obs[config['data']['z_key']] = float(depth)
            observed.append(a)
        backed = ad.read_h5ad(source / 'generated.h5ad', backed='r')
        try:
            reference = ad.AnnData(obs=backed.obs.copy(), uns={'depth_type_sampling': copy.deepcopy(backed.uns['depth_type_sampling'])})
        finally:
            backed.file.close()
        verify_schedule(reference.obs, config['data']['z_key'], protocol['source_schedule_sha256'])
        pl.seed_everything(config['seed'], workers=True)
        prepared = prepare_cached_data(config, observed, shared)
        fitted = prepared.niche_featurizer.to_dict()
        metadata_paths = list(fold.glob('*/model/config.json')) + list(fold.glob('*/checkpoints/config.json'))
        if metadata_paths:
            metadata = read_json(metadata_paths[0])
            fitted_before = copy.deepcopy(metadata['niche_featurizer'])
            fitted_before['mode'] = fitted['mode']
            if (metadata['spatial_stats'] != prepared.spatial_stats or metadata['gene_names'] != prepared.gene_names
                    or metadata['categories'] != prepared.categories.tolist() or fitted_before != fitted):
                raise ValueError('Visible data normalization, genes, labels or niche statistics differ from the saved model.')
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
        # 原组配置提供初始状态身份；新组也必须与其完全相同。
        initial_hashes = {read_json(p)['experiment_g']['initial_state_sha256'] for p in fold.glob('*/run_config.json')
                          if p.parent.name != 'shared'}
        if len(initial_hashes) != 1:
            raise ValueError('Existing condition initial-state identities are missing or inconsistent.')
        for item in pending.itertuples():
            folder = fold / item.mode
            model = copy.copy(prepared)
            condition_hash = set_condition(model, observed, item.mode, fitted)
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state(rng[3])
            model.build_model(**config['model'])
            initial_hash = state_sha256(model.module.state_dict())
            if initial_hash not in initial_hashes:
                raise ValueError('Rebuilt initialization differs from the original run; check Torch version and seed.')
            current = read_json(folder / 'run_config.json') if (folder / 'run_config.json').is_file() else copy.deepcopy(config)
            if item.mode == current.get('experiment_g', {}).get('niche_mode') and current['experiment_g']['paired_condition_sha256'] != condition_hash:
                raise ValueError('Cached pair conditions differ from the original run.')
            current['data']['niche_config']['mode'] = item.mode
            current['experiment_g'].update(niche_mode=item.mode, initial_state_sha256=initial_hash,
                                           paired_condition_sha256=condition_hash)
            current['reconstruction']['device'] = device
            folder.mkdir(exist_ok=True)
            (folder / 'run_config.json').write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')
            last = folder / 'checkpoints/last.ckpt'
            history = dict(action=item.action, from_completed_epochs=int(item.completed_epochs),
                           target_epochs=int(item.target_epochs), checkpoint_sha256=file_sha256(last) if last.is_file() else None)
            with (folder / 'resume_history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(history, ensure_ascii=False) + '\n')
            print(f'G holdout {heldout} {item.mode}: {item.action}; {item.completed_epochs}/{item.target_epochs} epochs', flush=True)
            if item.action == 'reconstruct':
                model.load_checkpoint(last, config_path=folder / 'checkpoints/config.json')
            else:
                model.fit(save_dir=str(folder / 'checkpoints'), save_ckpt=True,
                          resume_ckpt_path=str(last) if item.action == 'resume_training' else None,
                          **config['training'])
            model.save_model(folder / 'model')
            pl.seed_everything(config['seed'], workers=True)
            result = model.reconstruct_full_volume(observed, sampling_reference=reference, **current['reconstruction'])
            digest = verify_schedule(result.obs, model.z_key, protocol['source_schedule_sha256'])
            current['experiment_g']['generated_schedule_sha256'] = digest
            result.uns['experiment_g'] = current['experiment_g']
            temporary = folder / 'generated.resume.tmp.h5ad'
            result.write_h5ad(temporary)
            temporary.replace(folder / 'generated.h5ad')
            result.uns['depth_type_sampling']['budget'].to_csv(folder / 'sampling_budget.csv', index=False)
            result.obs.groupby(['source_z', 'source_cell_class', 'match_status', 'generation_mode'], observed=True).size().rename('generated_n').reset_index().to_csv(folder / 'generation_support_counts.csv', index=False)
            (folder / 'run_config.json').write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')
            del model, result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del prepared, observed, reference
        gc.collect()
    return inspect_results(output_root, holdouts, modes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/g_holdouts_1_2_3')
    parser.add_argument('--source-root', type=Path, default=PROJECT / 'outputs/s_holdouts_1_2_3')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    status = inspect_results(args.output) if args.check_only else resume_experiment(args.source_root, args.output)
    print(status.to_string(index=False))


if __name__ == '__main__':
    main()
