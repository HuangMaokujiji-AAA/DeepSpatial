"""Train three stages once; evaluate explicit interventions on a fixed generated point set."""
import hashlib
import json
import os
from pathlib import Path
import platform

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .data import (discover, prepare, validate, spatial_inputs, histogram, neighborhood,
                   endpoints, grid)
from .evaluation import fit_probe, evaluate
from .models import (SpatialDensity, PositionType, ConditionalExpression,
                     fit, seed_all, batches_predict)

PROJECT = Path(__file__).resolve().parents[1]
VARIANTS = ('full', 'spatial_linear', 'type_prior', 'expression_no_type', 'expression_no_niche')
INTERVENTIONS = ('type_position_shuffle', 'expression_type_shuffle', 'expression_niche_shuffle')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path = Path(path); tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def training_data(slices, state, config):
    spatial = []; supervision = []; contexts = []; genes = []; labels = []; rows = []
    for i, target in enumerate(slices):
        z = target['z_um']; others = [s for s in slices if s['index'] != target['index']]
        left, right = endpoints(others, z)
        if 0 < i < len(slices)-1:
            f, baseline = spatial_inputs(left, right, z, state, config)
            spatial.append((f, baseline, histogram(target, state, config)))
            supervision.append(dict(target_index=target['index'], left_index=left['index'],
                                    right_index=right['index'], target_n=len(target['g'])))
        ctx = neighborhood(target['physical'], z, others, state, config)
        contexts.append(ctx); genes.append(target['g']); labels.append(target['y'])
        rows.extend(dict(target_index=target['index'], cell_id=cell,
                         left_index=left['index'], right_index=right['index']) for cell in target['ids'])
    merged = {k: np.concatenate([c[k] for c in contexts])
              for k in ('features', 'gene_features', 'prior', 'mean', 'positive', 'empty')}
    return ([np.concatenate([s[i] for s in spatial]) for i in range(3)], merged,
            np.concatenate(labels), np.vstack(genes), supervision, pd.DataFrame(rows))


def type_probabilities(model, ctx, config, device, prior_only=False):
    if prior_only:
        p = np.maximum(ctx['prior'], 1e-5)
        return (p/p.sum(1, keepdims=True)).astype(np.float32)
    logits = batches_predict(model, [ctx['features'], ctx['prior']], device,
                             config['training']['batch_size'])
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return (p/p.sum(1, keepdims=True)).astype(np.float32)


def expression_arrays(ctx, types):
    return [ctx['features'], types, ctx['gene_features'], ctx['mean'], ctx['positive']]


def sample_types(probabilities, rng):
    codes = np.minimum((rng.random(len(probabilities))[:, None] > probabilities.cumsum(1)).sum(1),
                       probabilities.shape[1]-1)
    return codes, np.eye(probabilities.shape[1], dtype=np.float32)[codes]


def expression_predict(model, ctx, types, config, device, variant='full', details=False):
    return batches_predict(model, expression_arrays(ctx, types), device, config['training']['batch_size'],
                           no_type=variant == 'expression_no_type', no_niche=variant == 'expression_no_niche',
                           all_outputs=details)


def stage_losses(state, config, variant):
    def spatial_loss(model, f, baseline, target):
        rate, ratio = model(f, baseline)
        poisson = (rate-target*rate.clamp_min(1e-8).log()).mean()
        penalty = config['spatial']['residual_penalty']*ratio.square().mean()
        return poisson+penalty, dict(poisson=poisson, residual_penalty=penalty)
    def type_loss(model, f, prior, target):
        logits = model(f, prior)
        ce = F.cross_entropy(logits, target, reduction='none')
        pi = torch.as_tensor(state['counts'], device=f.device, dtype=f.dtype); pi /= pi.sum()
        eta = config['type']['balanced_risk']
        weight = ((1-eta)+eta/(len(pi)*pi)).clamp(max=config['type']['weight_cap'])
        return (ce*weight[target]).mean(), dict(natural_ce=ce.mean())
    def gene_loss(model, f, p, gf, mean, positive, target):
        expected, logits, magnitude = model(f, p, gf, mean, positive,
                                           no_type=variant == 'expression_no_type',
                                           no_niche=variant == 'expression_no_niche')
        log_loss = (torch.log1p(expected)-torch.log1p(target)).square().mean()
        scaled_loss = F.smooth_l1_loss(expected/model.scale, target/model.scale)
        presence = F.binary_cross_entropy_with_logits(logits, (target > 0).float())
        mask = (target > 0).float()
        positive_loss = ((magnitude-torch.log1p(target)).square()*mask).sum()/mask.sum().clamp_min(1)
        loss = log_loss+scaled_loss+config['expression']['presence_weight']*presence
        loss += config['expression']['positive_weight']*positive_loss
        return loss, dict(log_mse=log_loss, scaled_huber=scaled_loss,
                          presence_bce=presence, positive_log_mse=positive_loss)
    return spatial_loss, type_loss, gene_loss


def train(slices, state, config, folder, seed, device, variants, resume):
    spatial_arrays, ctx, y, g, supervision, row_info = training_data(slices, state, config)
    write_json(folder/'spatial_supervision.json', supervision)
    row_info.to_csv(folder/'conditional_supervision.csv', index=False)
    pd.DataFrame(dict(cell_class=state['classes'], train_n=state['counts'])).to_csv(folder/'train_counts.csv', index=False)
    losses = stage_losses(state, config, 'full')
    seed_all(seed)
    density = fit(SpatialDensity(config), spatial_arrays, losses[0], config, folder/'spatial',
                  'spatial', seed+1, device, resume)
    seed_all(seed+2)
    classifier = fit(PositionType(len(state['classes']), config), [ctx['features'], ctx['prior'], y],
                     losses[1], config, folder/'type', 'type', seed+2, device, resume)
    predicted = type_probabilities(classifier, ctx, config, device)
    np.savez_compressed(folder/'training_type_predictions.npz', probabilities=predicted, target=y)
    # Stage 3 sees sampled frozen stage-2 predictions, never true target labels as input.
    for variant in variants:
        if variant == 'spatial_linear':
            continue  # Same conditional models; only stage-1 intensity is replaced.
        p = type_probabilities(classifier, ctx, config, device, prior_only=variant == 'type_prior')
        codes, condition = sample_types(p, np.random.default_rng(seed+404))
        np.savez_compressed(folder/f'training_expression_types_{variant}.npz', probabilities=p, sampled_codes=codes)
        seed_all(seed+3)
        fit(ConditionalExpression(state, config), [*expression_arrays(ctx, condition), g],
            stage_losses(state, config, variant)[2], config, folder/f'expression_{variant}',
            'expression', seed+3, device, resume)
    return density, classifier


def load_models(state, config, folder, device, variant='full'):
    density = SpatialDensity(config)
    classifier = PositionType(len(state['classes']), config)
    expression = ConditionalExpression(state, config)
    gene_variant = 'full' if variant in ('spatial_linear', *INTERVENTIONS) else variant
    for model, name in ((density, 'spatial'), (classifier, 'type'), (expression, f'expression_{gene_variant}')):
        saved = torch.load(folder/name/'last.pt', map_location='cpu', weights_only=False)
        if saved['epoch'] != config['training']['epochs']['expression' if name.startswith('expression') else name]:
            raise ValueError(f'Incomplete training stage: {folder/name}')
        model.load_state_dict(saved['model']); model.to(device).eval()
    return density, classifier, expression


def generate_points(density, slices, state, config, heldout, seed, device, linear=False):
    center = config['slice_z_um'][heldout]; left, right = endpoints(slices, center)
    width = config['reconstruction']['slab_width_um']
    # Centers use a fixed physical grid, with the actual hidden depth included by protocol.
    if not np.isclose(center/width, round(center/width)):
        raise ValueError('Hidden depth must lie on the configured slab-center grid.')
    depths = np.arange(np.floor(left['z_um']/width)+1, np.ceil(right['z_um']/width))*width
    rng = np.random.default_rng(seed+101); locations = []; all_rates = []; counts = []
    xy_grid = grid(state, config)
    for z in depths:
        f, base = spatial_inputs(left, right, z, state, config)
        rates = base if linear else batches_predict(density, [f, base], device, config['training']['batch_size'])
        n = max(1, int(rng.poisson(float(rates.sum()))))
        ix = rng.choice(len(rates), size=n, p=rates.astype(float)/rates.sum(dtype=float))
        xy = xy_grid[ix]+rng.uniform(-.5, .5, size=(n, 2))*config['spatial']['bin_um']
        locations.append(np.column_stack((xy, np.full(n, z))))
        all_rates.append(rates)
        counts.append(dict(z_um=float(z), expected_n=float(rates.sum()), generated_n=n))
    return np.vstack(locations), dict(depths_um=depths, grid_xy=xy_grid, intensity=np.asarray(all_rates)), counts


def generate(density, classifier, expression, points, slices, state, config, device, variant, seed):
    """Forward-only dependencies: changing genes cannot move points or change types."""
    n = len(points); nc = len(state['classes']); ng = len(state['genes'])
    types = np.empty((n, nc), np.float32); genes = np.empty((n, ng), np.float32)
    codes = np.empty(n, np.int64)
    empty = np.empty(n, bool); presence = np.empty((n, ng), np.float32)
    rng = np.random.default_rng(seed+202)
    for z in np.unique(points[:, 2]):
        ix = np.flatnonzero(points[:, 2] == z)
        ctx = neighborhood(points[ix, :2], z, slices, state, config)
        query = ctx
        if variant == 'type_position_shuffle':
            perm = rng.permutation(len(ix))
            query = dict(ctx, features=ctx['features'][perm], prior=ctx['prior'][perm])
        p = type_probabilities(classifier, query, config, device, prior_only=variant == 'type_prior')
        types[ix] = p
        # Separate RNG: expression interventions cannot alter reported type draws.
        local_seed = np.random.SeedSequence([seed, int(round(z*1000)), 205])
        codes[ix], condition_p = sample_types(p, np.random.default_rng(local_seed))
        if variant == 'expression_type_shuffle':
            condition_p = condition_p[rng.permutation(len(ix))]
        if variant == 'expression_niche_shuffle':
            perm = rng.permutation(len(ix))
            ctx = dict(ctx, **{k: ctx[k][perm] for k in ('gene_features', 'mean', 'positive')})
        genes[ix], logits, _ = expression_predict(expression, ctx, condition_p, config, device, variant, details=True)
        presence[ix] = torch.from_numpy(logits).sigmoid().numpy()
        empty[ix] = ctx['empty']
    # Sample categorical labels to represent the posterior, rather than erase small
    # populations through argmax. Probabilities and argmax are both retained for audit.
    obs = pd.DataFrame(dict(cell_class=np.asarray(state['classes'])[codes],
                            cell_class_argmax=np.asarray(state['classes'])[types.argmax(1)],
                            z_coord=points[:, 2], parent_id=[f'point_{i}' for i in range(n)],
                            generation_mode='spatial_intensity', empty_neighborhood=empty,
                            annotation_origin='position_conditioned_posterior_sample', is_observed=False),
                       index=[f'generated_{i}' for i in range(n)])
    result = ad.AnnData(X=genes, obs=obs, var=pd.DataFrame(index=state['genes']),
                        obsm={'spatial':points[:, :2], 'spatial_3d':points,
                              'cell_type_probabilities':types},
                        layers={'expression_before_sparsification':genes.copy(),
                                'expression_presence_probability':presence})
    result.uns['staged_reconstruction'] = dict(variant=variant, expression_output='hurdle_conditional_mean',
                                              type_condition='sampled_predicted_class_onehot', source_labels_inherited=False)
    return result


def conditional_diagnostics(truth, slices, state, config, folder, device, seed):
    """Evaluation-only oracle position/type probes; never used for training/selection."""
    _, classifier, expression = load_models(state, config, folder, device)
    z = config['slice_z_um'][state['heldout']]
    ctx = neighborhood(np.asarray(truth.obsm[config['data']['spatial_key']]), z, slices, state, config)
    p = type_probabilities(classifier, ctx, config, device)
    labels = truth.obs[config['data']['label_key']].astype(str)
    y = pd.Categorical(labels, categories=state['classes']).codes
    valid = y >= 0
    rng = np.random.default_rng(seed+303)
    prior = type_probabilities(classifier, ctx, config, device, prior_only=True)
    rows, scores = [], []
    for intervention, q in (('learned', p), ('local_prior', prior), ('shuffled_position', p[rng.permutation(len(p))])):
        class_nll = []
        for code, name in enumerate(state['classes']):
            mask = y == code
            nll = float(-np.log(q[mask, code].clip(1e-8)).mean()) if mask.any() else np.nan
            if mask.any(): class_nll.append(nll)
            rows.append(dict(intervention=intervention, cell_class=name, truth_n=int(mask.sum()),
                             mean_correct_probability=float(q[mask, code].mean()) if mask.any() else np.nan,
                             nll=nll, argmax_recall=float((q[mask].argmax(1) == code).mean()) if mask.any() else np.nan))
        if valid.any():
            onehot = np.eye(len(state['classes']))[y[valid]]
            scores.append(dict(intervention=intervention, known_n=int(valid.sum()), unseen_n=int((~valid).sum()),
                               natural_nll=float(-np.log(q[valid, y[valid]].clip(1e-8)).mean()),
                               macro_class_nll=float(np.mean(class_nll)),
                               brier=float(((q[valid]-onehot)**2).sum(1).mean())))
    pd.DataFrame(rows).to_csv(folder/'oracle_position_type_by_class.csv', index=False)
    pd.DataFrame(scores).to_csv(folder/'oracle_position_type_scores.csv', index=False)
    _, predicted_condition = sample_types(p, np.random.default_rng(seed+304))
    oracle_p = predicted_condition.copy(); oracle_p[valid] = np.eye(len(state['classes']), dtype=np.float32)[y[valid]]
    actual = np.log1p(validate(truth, config['data'], state['genes']))
    experiments = [('predicted_type', ctx, predicted_condition), ('oracle_type', ctx, oracle_p),
                   ('shuffled_type', ctx, predicted_condition[rng.permutation(len(p))])]
    perm = rng.permutation(len(p))
    shuffled = dict(ctx, **{k: ctx[k][perm] for k in ('gene_features', 'mean', 'positive')})
    experiments.append(('shuffled_niche_expression', shuffled, predicted_condition))
    records = []
    for name, context, condition in experiments:
        prediction, presence_logits, _ = expression_predict(expression, context, condition, config, device, details=True)
        errors = (np.log1p(prediction)-actual)**2
        positive = (actual > 0).astype(float)
        bce = np.logaddexp(0, presence_logits)-positive*presence_logits
        for code, cell_class in enumerate(state['classes']):
            mask = y == code
            records.append(dict(intervention=name, cell_class=cell_class, n=int(mask.sum()),
                                per_cell_log_mse=float(errors[mask].mean()) if mask.any() else np.nan,
                                gene_presence_bce=float(bce[mask].mean()) if mask.any() else np.nan))
    pd.DataFrame(records).to_csv(folder/'oracle_position_expression_interventions.csv', index=False)
    write_json(folder/'oracle_diagnostics.json', dict(evaluation_only=True, unseen_truth_n=int((~valid).sum()),
                                                     oracle_type_applies_to_known_classes_only=True))


def spatial_diagnostics(truth, slices, state, config, folder, device):
    """Evaluate held-out density without feeding truth counts back to the generator."""
    density, _, _ = load_models(state, config, folder, device)
    z = config['slice_z_um'][state['heldout']]; left, right = endpoints(slices, z)
    f, base = spatial_inputs(left, right, z, state, config)
    learned = batches_predict(density, [f, base], device, config['training']['batch_size'])
    xy = np.asarray(truth.obsm[config['data']['spatial_key']])
    ij = np.floor((xy-state['grid_origin'])/config['spatial']['bin_um']).astype(int)
    inside = ((ij >= 0) & (ij < state['grid_shape'])).all(1)
    flat = np.ravel_multi_index(ij[inside].T, state['grid_shape'])
    target = np.bincount(flat, minlength=len(base))
    records = []
    for name, rate in (('learned', learned), ('linear', base)):
        rate = rate.astype(float)
        records.append(dict(model=name, predicted_expected_n=float(rate.sum()), truth_n=int(truth.n_obs),
                            truth_outside_grid_n=int((~inside).sum()),
                            absolute_count_error=float(abs(rate.sum()-truth.n_obs)),
                            poisson_nll_without_constant=float(np.mean(rate-target*np.log(rate.clip(1e-12)))),
                            binned_density_tv=float(np.abs(rate/rate.sum()-target/max(1, target.sum())).sum()/2)))
    pd.DataFrame(records).to_csv(folder/'stage1_density_diagnostics.csv', index=False)


def run(config, output, holdouts=(1, 2, 3), seeds=(42,), variants=('full',), stage='all',
        device='auto', resume=False, excluded=(), role='development', interventions=False):
    if stage not in ('all', 'train', 'generate', 'evaluate') or role not in ('development', 'validation', 'test'):
        raise ValueError('Unknown stage or evaluation role.')
    if not variants or any(v not in VARIANTS for v in variants):
        raise ValueError('Unknown variant.')
    if config['training']['batch_size'] < 1 or min(config['training']['epochs'].values()) < 1:
        raise ValueError('Batch size and all epoch counts must be positive.')
    if config['neighborhood']['neighbors'] < 1 or config['reconstruction']['slab_width_um'] <= 0:
        raise ValueError('Require positive neighbor count and slab width.')
    for section, keys in (('spatial', ('bin_um', 'smooth_sigma_bins', 'intensity_floor', 'max_log_ratio')),
                          ('neighborhood', ('radius_um', 'sigma_xy_um'))):
        if any(not np.isfinite(config[section][k]) or config[section][k] <= 0 for k in keys):
            raise ValueError(f'Require finite positive {section} scales.')
    if not 0 <= config['type']['balanced_risk'] <= 1 or config['type']['weight_cap'] < 1:
        raise ValueError('Type risk mixture must be in [0,1], and weight cap >=1.')
    if any(h <= 0 or h >= len(config['slice_z_um'])-1 for h in holdouts):
        raise ValueError('Holdouts must be interior slice indices.')
    if any(not np.isclose(config['slice_z_um'][h]/config['reconstruction']['slab_width_um'],
                          round(config['slice_z_um'][h]/config['reconstruction']['slab_width_um'])) for h in holdouts):
        raise ValueError('Holdout depths must lie on the slab-center grid.')
    variants = list(dict.fromkeys(['full', *variants]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    paths = discover(config, PROJECT)
    sources = {str(p.relative_to(PROJECT)).replace('\\', '/'): digest(p)
               for root in ('stspatial', 'examples') for p in sorted((PROJECT/root).rglob('*.py'))}
    root_identity = dict(config=config, source_sha256=sources, excluded=list(excluded), role=role,
                         variants=variants, interventions=bool(interventions))
    manifest = output/'experiment.json'
    if manifest.exists():
        if not resume or json.loads(manifest.read_text(encoding='utf-8')) != root_identity:
            raise ValueError('Output exists or source/config/protocol changed. Choose a new root or identical --resume.')
    elif any(output.iterdir()):
        raise ValueError('Refusing a nonempty output folder without a valid experiment manifest.')
    else:
        write_json(manifest, root_identity)
    records = []
    for heldout in holdouts:
        slices, state = prepare(paths, config, heldout, excluded)
        visible_hashes = {str(paths[i]): digest(paths[i]) for i in state['train_indices']}
        for seed in seeds:
            folder = output/f'holdout_{heldout}'/f'seed_{seed}'; folder.mkdir(parents=True, exist_ok=True)
            identity = dict(root_identity, heldout=int(heldout), seed=int(seed), visible_sha256=visible_hashes)
            stamp = folder/'manifest.json'
            if stamp.exists() and json.loads(stamp.read_text(encoding='utf-8')) != identity:
                raise ValueError('Visible input bytes changed; resume refused.')
            write_json(stamp, identity); write_json(folder/'state.json', state)
            write_json(folder/'environment.json', dict(python=platform.python_version(), torch=torch.__version__,
                       device=str(device), cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None))
            print(f'holdout={heldout}, seed={seed}, variants={variants}, stage={stage}', flush=True)
            completed = folder/'trained.json'
            if completed.exists():
                saved_hashes = json.loads(completed.read_text(encoding='utf-8'))
                if any(digest(folder/p) != checksum for p, checksum in saved_hashes.items()):
                    raise ValueError('A completed checkpoint changed; resume refused.')
            if stage in ('all', 'train') and not completed.exists():
                train(slices, state, config, folder, seed, device, variants, resume)
                write_json(folder/'trained.json', {str(p.relative_to(folder)): digest(p) for p in folder.glob('*/last.pt')})
            if stage in ('all', 'generate', 'evaluate'):
                if not (folder/'trained.json').exists():
                    raise FileNotFoundError('Complete all training stages before generation/evaluation.')
                trained = json.loads((folder/'trained.json').read_text(encoding='utf-8'))
                if any(digest(folder/p) != checksum for p, checksum in trained.items()):
                    raise ValueError('A completed checkpoint changed after training.')
            generated_names = [*variants, *(INTERVENTIONS if interventions else ())]
            if stage in ('all', 'generate'):
                density, classifier, expression = load_models(state, config, folder, device)
                points, rates, counts = generate_points(density, slices, state, config, heldout, seed, device)
                np.savez_compressed(folder/'stage1_spatial_distribution.npz', **rates)
                np.save(folder/'stage1_points.npy', points)
                pd.DataFrame(counts).to_csv(folder/'stage1_counts.csv', index=False)
                for variant in generated_names:
                    current = folder/variant; current.mkdir(exist_ok=True)
                    density, classifier, expression = load_models(state, config, folder, device, variant)
                    current_points = points
                    if variant == 'spatial_linear':
                        current_points, _, _ = generate_points(density, slices, state, config, heldout, seed, device, linear=True)
                    result = generate(density, classifier, expression, current_points, slices, state, config, device, variant, seed)
                    result.write_h5ad(current/'predicted_gap.h5ad', compression='gzip')
                    mask = np.isclose(result.obs.z_coord.to_numpy(), config['slice_z_um'][heldout])
                    result[mask].copy().write_h5ad(current/'predicted_holdout.h5ad', compression='gzip')
                    write_json(current/'generated.json', dict(gap_sha256=digest(current/'predicted_gap.h5ad'),
                                                              slab_sha256=digest(current/'predicted_holdout.h5ad')))
            # First access to hidden X/labels/coordinates occurs after generation.
            if stage in ('all', 'evaluate'):
                truth = ad.read_h5ad(paths[heldout]); validate(truth, config['data'], state['genes'])
                write_json(folder/'evaluation_truth.json', dict(file=str(paths[heldout]), sha256=digest(paths[heldout]), role=role))
                probe, markers = fit_probe(slices, state, seed); joblib.dump(probe, folder/'probe.joblib')
                for variant in generated_names:
                    current = folder/variant
                    stamp = json.loads((current/'generated.json').read_text(encoding='utf-8'))
                    if digest(current/'predicted_holdout.h5ad') != stamp['slab_sha256']:
                        raise ValueError('Saved prediction changed before evaluation.')
                    prediction = ad.read_h5ad(current/'predicted_holdout.h5ad')
                    table, metrics, marker_table = evaluate(truth, prediction, state, config, probe, markers)
                    # Same point cloud, deterministic argmax view: expose posterior-sampling effects.
                    argmax_prediction = prediction.copy()
                    argmax_prediction.obs['cell_class'] = prediction.obs.cell_class_argmax.to_numpy()
                    _, argmax_metrics, _ = evaluate(truth, argmax_prediction, state, config, probe, markers)
                    metrics.update({'argmax_'+k: v for k, v in argmax_metrics.items() if k.startswith('model_class_')})
                    metrics.update(arm=variant, heldout=int(heldout), seed=int(seed), role=role)
                    table.to_csv(current/'per_type.csv', index=False)
                    marker_table.to_csv(current/'marker_sparsification.csv', index=False)
                    write_json(current/'metrics.json', metrics); records.append(metrics)
                conditional_diagnostics(truth, slices, state, config, folder, device, seed)
                spatial_diagnostics(truth, slices, state, config, folder, device)
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(output.glob('holdout_*/seed_*/*/metrics.json'))]
    if rows:
        pd.DataFrame(rows).to_csv(output/'all_runs.csv', index=False)
    return records
