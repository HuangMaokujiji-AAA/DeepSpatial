"""实验 G：局部几何 UOT + 固定 S 调度，独立训练 real/zero/shuffle 条件对照。"""
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
if sys.platform == 'win32' and (PROJECT / '.runtime_deps').is_dir():
    sys.path.insert(1, str(PROJECT / '.runtime_deps'))

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from deepspatial import DeepSpatial
from deepspatial.bio_prior import NicheFeaturizer
from deepspatial.data_utils.local_matching import matching_parameters
from deepspatial.data_utils.sampling import replay_source_schedule
from examples.run_s import file_sha256


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def schedule_sha256(obs, z_key):
    fields = ['parent_id', 'source_cell_id', 'source_z', 'source_cell_index', 'source_direction',
              'source_cell_class', 'depth_bin_index', z_key]
    frame = obs[fields].copy()
    for field in ('parent_id', 'source_cell_id', 'source_direction', 'source_cell_class'):
        frame[field] = frame[field].astype(str)
    return hashlib.sha256(frame.to_csv(index=False, float_format='%.17g').encode()).hexdigest()


def set_condition(model, observed, mode, fitted_state):
    """仅替换条件；拟合统计、所有配对端点和 DataLoader 均保持不变。"""
    state = copy.deepcopy(fitted_state)
    state['mode'] = mode
    model.niche_featurizer = NicheFeaturizer.from_dict(state)
    features = []
    for a in observed:
        a.obsm['niche_cond'] = model.niche_featurizer.transform(a, model.spatial_key, model.label_key)
        features.append(a.obsm['niche_cond'])
    for suffix in ('0', '1'):
        slices = model.dataset.tensors['slice' + suffix].numpy()
        cells = model.dataset.tensors['cell' + suffix].numpy()
        h = np.empty((len(cells), len(model.niche_featurizer.feature_names)), dtype=np.float32)
        for i, values in enumerate(features):
            mask = slices == i
            h[mask] = values[cells[mask]]
        model.dataset.tensors['h' + suffix] = torch.from_numpy(h)
    return state_sha256({k: model.dataset.tensors[k] for k in ('h0', 'h1')})


def run_experiment(source_root=PROJECT / 'outputs/s_holdouts_1_2_3',
                   output_root=PROJECT / 'outputs/g_holdouts_1_2_3',
                   config_path=PROJECT / 'configs/merfish_g.json', holdouts=(1, 2, 3),
                   modes=('real', 'zero', 'shuffle'), device='auto', pairing_only=False):
    """先排除隐藏文件，每折只配对一次；每组从相同初始化重新训练。"""
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    if not output_root.is_relative_to(PROJECT) or output_root == PROJECT:
        raise ValueError('Output must be a new subdirectory inside HM_DeepSpital.')
    if not holdouts or len(set(holdouts)) != len(holdouts) or any(i not in (1, 2, 3) for i in holdouts):
        raise ValueError('holdouts must be a unique nonempty subset of [1,2,3].')
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= {'real', 'zero', 'shuffle'}:
        raise ValueError('modes must be a unique nonempty subset of real/zero/shuffle.')
    if config['training'].get('devices') != 1:
        raise ValueError('G runs sequentially on one device; use training.devices=1.')
    if config['data']['niche_config']['mode'] != 'real':
        raise ValueError('Shared setup must fit the real niche features.')
    matching_parameters(config['data']['local_matching_config'], 100.)
    paths = [Path(p) for p in glob.glob(str(PROJECT / config['input_glob']), recursive=True)]
    paths.sort(key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', p.name)])
    if len(paths) != len(config['slice_z_um']) or len(paths) != 5:
        raise ValueError('G expects five input files and five explicit physical depths.')
    plans = []
    for heldout in holdouts:
        source, output = source_root / f'holdout_{heldout}', output_root / f'holdout_{heldout}'
        if output.exists():
            raise FileExistsError(f'Output exists: {output}; choose a new output root.')
        previous = json.loads((source / 'run_config.json').read_text(encoding='utf-8-sig'))
        if previous['run']['heldout_indices'] != [heldout] or not (source / 'generated.h5ad').is_file():
            raise ValueError(f'A completed S fold is required: {source}')
        original_data = {k: v for k, v in config['data'].items() if k != 'local_matching_config'}
        if (original_data != previous['data'] or any(config[k] != previous[k] for k in ('seed', 'model', 'training', 'slice_z_um'))):
            raise ValueError('G must preserve the S/V1 data, model, training, seed and depth settings.')
        visible_names = sorted(p.name for i, p in enumerate(paths) if i != heldout)
        if visible_names != sorted(Path(p).name for p in previous['run']['visible_files']):
            raise ValueError('Visible input filenames differ from S.')
        for key in ('thickness', 'steps', 'chunk_size', 'save_intermediates', 'source_sampling', 'depth_bin_width_um', 'depth_grid_origin_um'):
            if config['reconstruction'].get(key) != previous['reconstruction'].get(key):
                raise ValueError(f'Reconstruction parameter differs from S: {key}')
        plans.append((heldout, source, output, previous))
    completed = []
    for heldout, source, output, previous in plans:
        print(f'Experiment G: holdout {heldout}; shared local matching', flush=True)
        pl.seed_everything(config['seed'], workers=True)
        observed = []
        for i, (path, depth) in enumerate(zip(paths, config['slice_z_um'])):
            if i == heldout:
                continue  # 必须在 read_h5ad 之前排除隐藏文件。
            a = ad.read_h5ad(path)
            a.obs[config['data']['z_key']] = float(depth)
            observed.append(a)
        # 只读取 S 的来源调度元数据；不把生成表达/坐标用于训练或条件构建。
        backed = ad.read_h5ad(source / 'generated.h5ad', backed='r')
        try:
            reference = ad.AnnData(obs=backed.obs.copy(), uns={'depth_type_sampling': copy.deepcopy(backed.uns['depth_type_sampling'])})
        finally:
            backed.file.close()
        z_key, label_key = config['data']['z_key'], config['data']['label_key']
        categories = sorted({str(c) for a in observed for c in a.obs[label_key]})
        reconstruction = config['reconstruction']
        for a0, a1 in zip(observed[:-1], observed[1:]):
            total = max(1, int((a0.n_obs + a1.n_obs) / 2 *
                              (float(a1.obs[z_key].iloc[0]) - float(a0.obs[z_key].iloc[0])) / reconstruction['thickness']))
            replay_source_schedule(reference, a0, a1, categories, total, z_key, label_key,
                                   reconstruction['depth_bin_width_um'], reconstruction['depth_grid_origin_um'])
        prepared = DeepSpatial()
        prepared.setup_data(observed, **config['data'])
        fitted_state = prepared.niche_featurizer.to_dict()
        initial_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                       torch.cuda.get_rng_state() if not pairing_only and torch.cuda.is_available() else None)
        output.mkdir(parents=True, exist_ok=False)
        shared = output / 'shared'
        shared.mkdir()
        np.savez_compressed(shared / 'pair_indices.npz', **{k: prepared.dataset.tensors[k].numpy()
                            for k in ('slice0', 'cell0', 'slice1', 'cell1')})
        endpoints = []
        for suffix in ('0', '1'):
            cells = prepared.dataset.tensors['cell' + suffix].numpy()
            slices = prepared.dataset.tensors['slice' + suffix].numpy()
            xy = np.empty((len(cells), 2))
            for i, a in enumerate(observed):
                mask = slices == i
                xy[mask] = a.obsm[prepared.spatial_key][cells[mask]]
            endpoints.append(xy)
        pair_frame = pd.DataFrame(dict(
            slice_pair=prepared.dataset.tensors['slice0'].numpy(),
            cell_class=np.asarray(prepared.categories)[prepared.dataset.tensors['c0'].argmax(1).numpy()],
            paired_distance_um=np.linalg.norm(endpoints[1] - endpoints[0], axis=1),
        ))
        pair_frame.groupby(['slice_pair', 'cell_class'], observed=True).paired_distance_um.agg(
            sampled_pairs='size', mean_um='mean', median_um='median',
            p95_um=lambda s: s.quantile(.95), maximum_um='max',
        ).reset_index().to_csv(shared / 'paired_distance_by_type.csv', index=False)
        diagnostics = pd.concat([pd.DataFrame(d) for d in prepared.dataset.matching_diagnostics], ignore_index=True)
        diagnostics['match_status'] = np.where(diagnostics.candidate_count > 0, 'matched', 'unsupported')
        diagnostics.to_csv(shared / 'matching_support.csv', index=False)
        diagnostics.groupby(['slice_pair', 'source_z_um', 'target_z_um', 'cell_class'], observed=True).agg(
            cells=('cell_id', 'size'), supported_cells=('candidate_count', lambda s: int((s > 0).sum())),
            mean_candidates=('candidate_count', 'mean'), nearest_same_type_median_um=('nearest_same_type_um', 'median'),
            transport_mass=('transport_mass', 'sum'), mean_conditional_entropy=('conditional_entropy', 'mean'),
        ).reset_index().to_csv(shared / 'matching_by_type.csv', index=False)
        (shared / 'trajectory_sampling.json').write_text(json.dumps(prepared.dataset.sampling_stats, indent=2), encoding='utf-8')
        expected_schedule_hash = schedule_sha256(reference.obs, prepared.z_key)
        protocol = dict(
            experiment='G', heldout_id=heldout, training_performed=not pairing_only,
            shared_pair_sha256=file_sha256(shared / 'pair_indices.npz'),
            source_s_h5ad_sha256=file_sha256(source / 'generated.h5ad'),
            source_s_config_sha256=file_sha256(source / 'run_config.json'),
            source_schedule_sha256=expected_schedule_hash,
            matching_cost='alpha*(physical_xy_distance/scaled_spatial_scale)^2 + (1-alpha)*global_normalized_cosine',
            matching_support='same cell type and scaled physical radius; no niche matching cost',
            unsupported_policy='no training pair; inference copies source xy/expression/type at target depth; included in evaluation',
            checkpoint_weights='EMA',
            shuffle_policy='within cell type and visible slice, fixed permutation for training and inference',
            rarity='visible training abundance <= 0.05; evaluation only',
            implementation_sha256={p.replace('/', '__'): file_sha256(PROJECT / p) for p in (
                'examples/run_g.py', 'deepspatial/core.py', 'deepspatial/data_utils/local_matching.py',
                'deepspatial/data_utils/dataset.py', 'deepspatial/data_utils/sampling.py', 'deepspatial/bio_prior/niche.py')},
        )
        run_config = copy.deepcopy(config)
        run_config['run'] = copy.deepcopy(previous['run'])
        run_config['experiment_g'] = protocol
        (shared / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
        initial_hash = None
        for mode in (() if pairing_only else modes):
            folder = output / mode
            folder.mkdir()
            model = copy.copy(prepared)
            feature_hash = set_condition(model, observed, mode, fitted_state)
            random.setstate(initial_rng[0]); np.random.set_state(initial_rng[1]); torch.set_rng_state(initial_rng[2])
            if initial_rng[3] is not None:
                torch.cuda.set_rng_state(initial_rng[3])
            model.build_model(**config['model'])
            identity = state_sha256(model.module.state_dict())
            if initial_hash is not None and identity != initial_hash:
                raise RuntimeError('Condition models do not share identical initial weights.')
            initial_hash = identity
            current = copy.deepcopy(run_config)
            current['data']['niche_config']['mode'] = mode
            current['experiment_g'].update(niche_mode=mode, initial_state_sha256=identity, paired_condition_sha256=feature_hash)
            current['reconstruction']['device'] = device
            (folder / 'run_config.json').write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'G holdout {heldout}: training {mode}', flush=True)
            model.fit(save_dir=str(folder / 'checkpoints'), save_ckpt=True, **config['training'])
            model.save_model(folder / 'model')
            pl.seed_everything(config['seed'], workers=True)
            result = model.reconstruct_full_volume(observed, sampling_reference=reference, **current['reconstruction'])
            actual_hash = schedule_sha256(result.obs, prepared.z_key)
            if actual_hash != expected_schedule_hash:
                raise RuntimeError('Generated parent/type/depth schedule differs from S.')
            current['experiment_g']['generated_schedule_sha256'] = actual_hash
            result.uns['experiment_g'] = current['experiment_g']
            result.write_h5ad(folder / 'generated.h5ad')
            result.uns['depth_type_sampling']['budget'].to_csv(folder / 'sampling_budget.csv', index=False)
            result.obs.groupby(['source_z', 'source_cell_class', 'match_status', 'generation_mode'], observed=True).size().rename('generated_n').reset_index().to_csv(folder / 'generation_support_counts.csv', index=False)
            (folder / 'run_config.json').write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')
            completed.append(dict(heldout_id=heldout, mode=mode, generated_cells=result.n_obs,
                                  source_anchor_cells=int((result.obs.generation_mode == 'source_anchor').sum()), output=str(folder)))
            del result, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if pairing_only:
            completed.append(dict(heldout_id=heldout, pairing_only=True, output=str(shared)))
        del prepared, observed, reference, diagnostics
        gc.collect()
        print(f'Experiment G completed: holdout {heldout}', flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=PROJECT / 'outputs/s_holdouts_1_2_3')
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/g_holdouts_1_2_3')
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/merfish_g.json')
    parser.add_argument('--holdouts', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--modes', nargs='+', choices=['real', 'zero', 'shuffle'], default=['real', 'zero', 'shuffle'])
    parser.add_argument('--device', default='auto', help='Inference device; training follows config accelerator.')
    parser.add_argument('--pairing-only', action='store_true', help='Only cache local pairing and support audits; no training/ODE.')
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.source_root, args.output, args.config, args.holdouts,
                                    args.modes, args.device, args.pairing_only), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
