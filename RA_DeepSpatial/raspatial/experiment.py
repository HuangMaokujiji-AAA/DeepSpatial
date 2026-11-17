"""Fixed train/validation/test protocol, eight arms, checkpoint freeze and staged execution."""
import gc
from pathlib import Path
import platform

import anndata as ad
import numpy as np
import pandas as pd
import torch

from .data import prepare, SliceStore, read_section, bracket
from .dataset import PairDataset
from .evaluation import evaluate, fit_expression_probe, load_expression_probe
from .io import digest, identity, read_json, source_identity, write_json
from .markers import load_markers, MarkerTasks
from .model import make_model
from .reconstruction import schedule, add_targets, reconstruct, write_h5ad
from .training import fit, seed_all
from .transport import Transports


ARMS = {
    'baseline': dict(base=True, baseline=True, raw=True, A=False, B=False, C=False, D=False),
    'control': dict(base=True, A=False, B=False, C=False, D=False),
    'R0': dict(A=False, B=False, C=False, D=False),
    'full': dict(A=True, B=True, C=True, D=True),
    'no_A': dict(raw=True, A=False, B=True, C=True, D=True),
    'no_B': dict(A=True, B=False, C=True, D=True),
    'no_C': dict(A=True, B=True, C=False, D=True),
    'no_D': dict(A=True, B=True, C=True, D=False),
}


def load_model(folder, name, state, config, device):
    path = Path(folder)/name/'last.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = make_model(state, config, ARMS[name])
    model.load_state_dict(saved['ema'])
    return model.to(device).eval()


def validate_config(config):
    if config['seed'] != 42:
        raise ValueError('This protocol fixes seed=42. Create a separate protocol for additional seeds.')
    if not 0 < config['evaluation']['rare_fraction'] <= 1:
        raise ValueError('Invalid rarity threshold.')
    for section, keys in (('training', ('epochs', 'batch_size', 'lr', 'gradient_clip')),
                          ('C', ('every_steps', 'cells', 'rollout_steps', 'bandwidth_um', 'grid_um')),
                          ('pairing', ('n_samples', 'local_reg', 'raw_reg', 'tau', 'radius_um', 'reference_gap_um', 'block_size')),
                          ('reconstruction', ('steps', 'chunk_size', 'export_step_um')),
                          ('evaluation', ('spatial_bin_um', 'niche_radius_um', 'matching_cap_um', 'max_assignment_entries'))):
        if any(config[section][key] <= 0 for key in keys):
            raise ValueError(f'Nonpositive configuration value in {section}.')
    optional_eval_pos = ('top_hvg_n', 'max_w2_entries', 'sliced_w2_projections', 'ssim_sigma',
                         'nhood_z_threshold', 'probe_hidden_size', 'probe_epochs', 'probe_batch_size', 'probe_lr')
    if any(config['evaluation'].get(key, 1) <= 0 for key in optional_eval_pos):
        raise ValueError('Nonpositive optional configuration value in evaluation.')
    if config['reconstruction']['steps'] < 2 or config['data']['cache_slices'] < 3:
        raise ValueError('Need at least two ODE steps and three cached sections.')


def run(project, config, cache, output, stage='prepare', arms=None, device='cuda:0', resume=False):
    project, cache, output = Path(project).resolve(), Path(cache).resolve(), Path(output).resolve()
    validate_config(config)
    if not cache.is_relative_to(project) or not output.is_relative_to(project) or cache == output:
        raise ValueError('Cache and output must be separate directories inside this independent project.')
    arms = list(ARMS) if arms is None else list(arms)
    if len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError('Unknown or duplicate arm.')
    marker_path = (project/config['markers']['path']).resolve()
    if stage not in ('prepare', 'summary') and (any(ARMS[a]['C'] for a in arms) or stage in ('freeze', 'test', 'all')) and not marker_path.is_file():
        raise FileNotFoundError(f'Download/import external markers first: {marker_path}. See resources/README.md.')
    state, splits, paths = prepare(project, config, cache)
    if stage == 'prepare':
        return state, splits
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu only for engineering checks.')
    if marker_path.is_file():
        markers, independent, coverage = load_markers(marker_path, state, config)
        counts = np.asarray(state['counts'])
        rare_codes = np.flatnonzero(counts/counts.sum() <= config['evaluation']['rare_fraction'])
        covered = sum(bool(markers[int(c)]) for c in rare_codes)
        print(f'External loss markers cover {covered}/{len(rare_codes)} training-defined rare types.', flush=True)
        if any(ARMS[name]['C'] for name in arms) and covered == 0:
            raise ValueError('No external measured markers cover rare training types. Inspect taxonomy IDs and marker_coverage.')
    elif any(ARMS[name]['C'] for name in arms) or stage in ('freeze', 'test', 'all'):
        raise FileNotFoundError(f'Download/import external markers first: {marker_path}. See resources/README.md.')
    else:
        markers = independent = {i: [] for i in range(len(state['classes']))}
        coverage = pd.DataFrame()
    experiment_identity = dict(config=config, split=splits, source_sha256=source_identity(project),
                               prepared_identity=digest(cache/'identity.json'), state_sha256=digest(cache/'state.json'),
                               marker_sha256=digest(marker_path) if marker_path.is_file() else None)
    identity(output, experiment_identity)
    coverage.to_csv(output/'marker_coverage.csv', index=False)
    environment = dict(python=platform.python_version(), torch=torch.__version__, device=str(device), cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
    if not (output/'environment.json').exists():
        write_json(output/'environment.json', environment)
    write_json(output/f'environment_{stage}.json', environment)
    store = SliceStore(cache, state, config['data']['cache_slices'])
    transports = Transports(cache, store, config)
    if stage == 'all' and resume and (output/'frozen_test_protocol.json').exists():
        stage = 'test'
    if stage in ('train', 'all'):
        if (output/'frozen_test_protocol.json').exists():
            raise ValueError('This experiment is frozen for testing; use a new output for further training.')
        requested = list(arms)
        if any(not ARMS[a].get('base', False) for a in requested):
            requested = ['control']+[a for a in requested if a != 'control']
        for name in requested:
            arm = ARMS[name]; folder = output/name
            metadata = dict(experiment_identity, arm=name, options=arm)
            if not arm.get('base', False):
                metadata['control_checkpoint_sha256'] = digest(output/'control/last.pt')
            identity(folder, metadata)
            completed = folder/'trained.json'
            if completed.exists():
                if read_json(completed)['checkpoint_sha256'] != digest(folder/'last.pt'):
                    raise ValueError(f'Completed checkpoint changed: {name}')
                if not resume:
                    raise FileExistsError(f'{name} already trained; use --resume.')
                continue
            dataset = PairDataset(store, transports, state, config, arm)
            seed_all(config['seed'])
            base_state = None
            if not arm.get('base', False):
                base_state = torch.load(output/'control/last.pt', map_location='cpu', weights_only=False)['ema']
            model = make_model(state, config, arm, base_state)
            anchors = PairDataset(store, transports, state, config, ARMS['control']) if arm['D'] else None
            marker_tasks = MarkerTasks(store, transports, state, config, markers, config['seed']+431,
                                       transport_kind='raw' if arm.get('raw') else 'local') if arm['C'] else None
            fit(model, dataset, config, folder, state, device, metadata, marker_tasks, anchors, resume)
            write_json(completed, dict(checkpoint_sha256=digest(folder/'last.pt'), fixed_epochs=config['training']['epochs'],
                                       backbone_epochs=config['training']['epochs'] if not arm.get('base') else 0,
                                       checkpoint_selection='last_epoch_predeclared'))
            del model, dataset, anchors, marker_tasks
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        if stage == 'train':
            return
    if stage in ('validate', 'all'):
        assess('validation', output, state, config, store, transports, splits, paths, markers, independent, arms, device)
        summarize(output, splits)
        if stage == 'validate':
            return
    if stage in ('freeze', 'all'):
        freeze(output, splits)
        if stage == 'freeze':
            return
    if stage in ('test', 'all'):
        verify_freeze(output, splits)
        assess('test', output, state, config, store, transports, splits, paths, markers, independent, arms, device)
        summarize(output, splits)
        if stage == 'test':
            return
    if stage == 'summary':
        summarize(output, splits)
    elif stage == 'export':
        verify_freeze(output, splits)
        export_volume(output, state, config, store, transports, arms, device)


MAIN_TABLE_COLUMNS = (
    'rare_macro_ospa1_um',
    'rare_macro_one_to_one_f1_50um',
    'rare_macro_verified_one_to_one_f1_50um',
    'rare_macro_w2_distance_um',
    'ultra_rare_macro_w2_distance_um',
    'rare_macro_count_relative_error',
    'rare_macro_missing_rate',
    'rare_heldout_C_marker_macro_spatial_pcc',
    'rare_heldout_C_marker_macro_spatial_ssim',
    'rare_heldout_C_marker_macro_morans_i_abs_error',
    'rare_macro_probe_concordance',
    'rare_macro_probe_concordance_pre_sparsify',
    'rare_macro_unbiased_niche_js',
    'rare_nhood_zscore_matrix_pcc',
    'global_top_hvg_grid_pcc',
    'global_top_hvg_grid_ssim',
    'common_macro_ospa1_um',
)


def ensure_expression_probe(output, store, state, config, device):
    """Fit or load the train-only frozen expression probe; never touches validation/test sections."""
    output = Path(output)
    probe_path = output / 'expression_probe.pt'
    meta_path = output / 'expression_probe.json'
    state_hash = digest(store.cache / 'state.json')
    if probe_path.is_file() and meta_path.is_file():
        meta = read_json(meta_path)
        current_hash = digest(probe_path)
        if meta.get('probe_sha256') != current_hash or meta.get('state_sha256') != state_hash:
            raise ValueError('Frozen expression probe changed or does not match prepared train state.')
        return load_expression_probe(probe_path, state, device=device), current_hash
    if (output / 'frozen_test_protocol.json').exists():
        raise ValueError('Cannot fit expression probe after test protocol is frozen.')
    probe, info = fit_expression_probe(store, state, config, device, save_path=probe_path)
    probe_hash = digest(probe_path)
    write_json(meta_path, dict(probe_sha256=probe_hash, state_sha256=state_hash, **info))
    return probe, probe_hash


def assess(role, output, state, config, store, transports, splits, paths, markers, independent, arms, device):
    for name in arms:
        if not (output/name/'trained.json').exists():
            raise FileNotFoundError(f'Train {name} before generation/evaluation.')
    probe, probe_hash = ensure_expression_probe(output, store, state, config, device)
    targets = [r for r in splits if r['role'] == role]
    for target in targets:
        left, right = store[target['context_left']], store[target['context_right']]
        natural = schedule(left, right, target['z_um'], state, config)
        local = add_targets(natural, left, right, transports, 'local', config['seed'])
        raw = None
        for name in arms:
            folder = output/name/role/target['section']; folder.mkdir(parents=True, exist_ok=True)
            checkpoint_hash = digest(output/name/'last.pt')
            prediction_path = folder/'prediction.h5ad'; stamp = folder/'generated.json'
            if stamp.exists():
                saved = read_json(stamp)
                if saved['checkpoint_sha256'] != checkpoint_hash or saved['prediction_sha256'] != digest(prediction_path):
                    raise ValueError(f'Prediction/checkpoint changed: {folder}')
                continue
            arm = ARMS[name]
            if arm.get('baseline'):
                current = schedule(left, right, target['z_um'], state, config, stratified=False)
                current = dict(current, target=np.full(len(current['parent']), -1, np.int64))
            elif arm.get('raw'):
                if raw is None:
                    raw = add_targets(natural, left, right, transports, 'raw', config['seed'])
                current = raw
            else:
                current = local
            model = load_model(output, name, state, config, device)
            prediction = reconstruct(model, left, right, current, state, config, device, bool(arm.get('baseline')))
            write_h5ad(prediction_path, prediction)
            np.savez_compressed(folder/'schedule.npz', **current)
            write_json(stamp, dict(checkpoint_sha256=checkpoint_hash, prediction_sha256=digest(prediction_path),
                                   schedule_sha256=digest(folder/'schedule.npz'), target_depth_um=target['z_um']))
            print(f'Generated {name}/{role}/{target["section"]}: {prediction.n_obs:,}', flush=True)
            del model, prediction
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        # All requested methods generate this target before its expression/labels are opened.
        truth = read_section(paths[target['section']], config)
        truth_hash = digest(paths[target['section']])
        for name in arms:
            folder = output/name/role/target['section']
            metric_path = folder/'metrics.json'
            if metric_path.exists():
                saved = read_json(metric_path)
                if (saved['truth_sha256'] != truth_hash or saved['prediction_sha256'] != digest(folder/'prediction.h5ad')
                        or saved.get('probe_sha256') != probe_hash):
                    raise ValueError('Evaluation inputs changed.')
                continue
            prediction = ad.read_h5ad(folder/'prediction.h5ad')
            metrics, per_type, per_marker = evaluate(truth, prediction, state, config, markers, independent, probe=probe)
            per_type.to_csv(folder/'per_type.csv', index=False)
            per_marker.to_csv(folder/'per_marker.csv', index=False)
            write_json(metric_path, dict(arm=name, role=role, target=target['section'], seed=config['seed'],
                                         truth_sha256=truth_hash, prediction_sha256=digest(folder/'prediction.h5ad'),
                                         probe_sha256=probe_hash, **metrics))


def freeze(output, splits):
    checkpoint_hashes, validation_hashes = {}, {}
    probe_path = output/'expression_probe.pt'
    if not probe_path.is_file():
        raise ValueError('Validation expression probe missing before test freeze.')
    for name in ARMS:
        if not (output/name/'trained.json').exists():
            raise ValueError(f'All eight arms must finish before test freeze: {name}')
        checkpoint_hashes[name] = digest(output/name/'last.pt')
        for row in splits:
            if row['role'] == 'validation':
                path = output/name/'validation'/row['section']/'metrics.json'
                if not path.is_file():
                    raise ValueError(f'Validation missing: {path}')
                validation_hashes[str(path.relative_to(output))] = digest(path)
    value = dict(experiment_identity_sha256=digest(output/'identity.json'),
                 expression_probe_sha256=digest(probe_path),
                 checkpoints=checkpoint_hashes,
                 validation=validation_hashes, selection='predeclared_final_epoch_no_test_selection',
                 test_targets=[r['section'] for r in splits if r['role'] == 'test'])
    path = output/'frozen_test_protocol.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('Test protocol already frozen with different artifacts.')
    write_json(path, value)


def verify_freeze(output, splits):
    path = output/'frozen_test_protocol.json'
    if not path.is_file():
        raise ValueError('Run validation and freeze before opening test expression/labels.')
    frozen = read_json(path)
    if frozen['experiment_identity_sha256'] != digest(output/'identity.json'):
        raise ValueError('Frozen protocol identity changed.')
    if 'expression_probe_sha256' in frozen:
        probe_path = output/'expression_probe.pt'
        if not probe_path.is_file() or digest(probe_path) != frozen['expression_probe_sha256']:
            raise ValueError('Frozen expression probe changed.')
    for name, expected in frozen['checkpoints'].items():
        if digest(output/name/'last.pt') != expected:
            raise ValueError(f'Frozen checkpoint changed: {name}')
    for name, expected in frozen['validation'].items():
        if digest(output/name) != expected:
            raise ValueError('Validation evidence changed after freeze.')
    if frozen['test_targets'] != [r['section'] for r in splits if r['role'] == 'test']:
        raise ValueError('Frozen test targets changed.')


def summarize(output, splits):
    statuses, rows = [], []
    for name in ARMS:
        for target in (r for r in splits if r['role'] != 'train'):
            folder = output/name/target['role']/target['section']
            complete = (folder/'metrics.json').is_file()
            statuses.append(dict(arm=name, role=target['role'], target=target['section'], evaluated=complete))
            if complete:
                rows.append(read_json(folder/'metrics.json'))
    pd.DataFrame(statuses).to_csv(output/'completion.csv', index=False)
    if not rows:
        return
    table = pd.DataFrame(rows); table.to_csv(output/'all_metrics.csv', index=False)
    for role in ('validation', 'test'):
        if not all(r['evaluated'] for r in statuses if r['role'] == role):
            continue
        part = table[table.role == role]
        numeric = [c for c in part.select_dtypes(include=np.number) if c != 'seed']
        part.groupby('arm')[numeric].agg(['mean', 'std']).to_csv(output/f'{role}_summary.csv')
        headline = [c for c in MAIN_TABLE_COLUMNS if c in part.columns]
        if headline:
            main_df = part.groupby('arm', sort=False)[headline].agg(['mean', 'std'])
            main_df.columns = [f'{col}_{stat}' for col, stat in main_df.columns]
            ordered = [a for a in ARMS if a in main_df.index]
            main_df.reindex(ordered).reset_index().to_csv(output/f'{role}_main_table.csv', index=False)
        deltas = []
        for reference in ('baseline', 'control', 'R0', 'full'):
            ref = part[part.arm == reference].set_index('target')
            for _, row in part.iterrows():
                if row.arm == reference:
                    continue
                delta = dict(arm=row.arm, reference=reference, target=row.target)
                for key in numeric:
                    delta[key] = row[key]-ref.loc[row.target, key]
                deltas.append(delta)
        pd.DataFrame(deltas).to_csv(output/f'{role}_paired_deltas.csv', index=False)


def export_volume(output, state, config, store, transports, arms, device):
    """Stream arbitrary-depth virtual planes; never concatenate the million-cell volume in RAM."""
    rows = []
    depths = np.arange(state['train'][0]['z_um'], state['train'][-1]['z_um'], config['reconstruction']['export_step_um'])[1:]
    for z in depths:
        if any(abs(z-r['z_um']) < 1e-6 for r in state['train']):
            continue
        lrow, rrow = bracket(state['train'], z)
        left, right = store[lrow['section']], store[rrow['section']]
        natural = schedule(left, right, z, state, config)
        local = add_targets(natural, left, right, transports, 'local', config['seed'])
        for name in arms:
            path = output/name/'volume'/f'z_{z:012.3f}um.h5ad'
            if path.exists():
                rows.append(dict(arm=name, z_um=z, path=str(path.relative_to(output)), status='existing')); continue
            arm = ARMS[name]
            if arm.get('baseline'):
                current = schedule(left, right, z, state, config, False)
                current['target'] = np.full(len(current['parent']), -1, np.int64)
            else:
                current = add_targets(natural, left, right, transports, 'raw', config['seed']) if arm.get('raw') else local
            model = load_model(output, name, state, config, device)
            result = reconstruct(model, left, right, current, state, config, device, bool(arm.get('baseline')))
            write_h5ad(path, result)
            rows.append(dict(arm=name, z_um=z, path=str(path.relative_to(output)), generated_n=result.n_obs, status='generated'))
            del model, result
    pd.DataFrame(rows).to_csv(output/'volume_manifest.csv', index=False)
