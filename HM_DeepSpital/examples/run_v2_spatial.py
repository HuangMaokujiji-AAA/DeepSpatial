"""V2 空间分支首轮：复用 G zero，比较 OT 重心和 OT 配对采样，不重新训练。"""
import argparse
import copy
import gc
import glob
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
from scipy import sparse
from deepspatial.data_utils.local_matching import compute_local_uot
from deepspatial.data_utils.spatial_transport import spatial_transport_volume
from examples.resume_g import inspect_results, verify_schedule
from examples.run_s import file_sha256


def run_experiment(source_root=PROJECT / 'outputs/g_holdouts_1_2_3',
                   output_root=PROJECT / 'outputs/v2_spatial_holdouts_1_2_3', holdouts=(1, 2, 3),
                   modes=('source_xy', 'ot_barycentric', 'ot_sampled'), seed=42):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if not output_root.is_relative_to(PROJECT) or output_root == PROJECT:
        raise ValueError('Output must be a new subdirectory inside HM_DeepSpital.')
    if (not holdouts or len(set(holdouts)) != len(holdouts) or not set(holdouts) <= {1, 2, 3}
            or not modes or len(set(modes)) != len(modes) or not set(modes) <= {'source_xy', 'ot_barycentric', 'ot_sampled'}):
        raise ValueError('Use unique holdouts [1,2,3] and valid spatial modes.')
    for fold in holdouts:
        if (output_root / f'holdout_{fold}').exists():
            raise FileExistsError('Output exists; select a new output root.')
    status = inspect_results(source_root, holdouts, ('zero',))
    if not (status.action == 'skip').all():
        raise ValueError('Complete G zero checkpoints and generated volumes are required.')
    completed = []
    for fold in holdouts:
        source = source_root / f'holdout_{fold}/zero'
        config = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
        if config['run']['heldout_indices'] != [fold] or config['data']['niche_config']['mode'] != 'zero':
            raise ValueError('Input must be the correct frozen G zero fold.')
        paths = sorted((Path(p) for p in glob.glob(str(PROJECT / config['input_glob']), recursive=True)),
                       key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p.name)])
        if len(paths) != len(config['slice_z_um']):
            raise ValueError('Visible input paths and explicit depths differ from the original run.')
        if sorted(p.name for i, p in enumerate(paths) if i != fold) != sorted(Path(p).name for p in config['run']['visible_files']):
            raise ValueError('Visible filenames differ from the original run.')
        observed = []
        for i, (path, depth) in enumerate(zip(paths, config['slice_z_um'])):
            if i == fold:
                continue
            a = ad.read_h5ad(path)
            a.obs[config['data']['z_key']] = float(depth)
            observed.append(a)
        base = ad.read_h5ad(source / 'generated.h5ad')
        expected = config['experiment_g']['source_schedule_sha256']
        verify_schedule(base.obs, config['data']['z_key'], expected)
        output = output_root / f'holdout_{fold}'
        shared = output / 'shared'
        shared.mkdir(parents=True, exist_ok=False)
        couplings, diagnostics = [], []
        data = config['data']
        for gap, (left, right) in enumerate(zip(observed[:-1], observed[1:])):
            def genes(a):
                return (a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)).astype(np.float32)
            pi, support = compute_local_uot(
                left.obsm[data['spatial_key']], genes(left), left.obs[data['label_key']].astype(str).to_numpy(),
                right.obsm[data['spatial_key']], genes(right), right.obs[data['label_key']].astype(str).to_numpy(),
                float(right.obs[data['z_key']].iloc[0] - left.obs[data['z_key']].iloc[0]), data['local_matching_config'],
                data['alpha_spatial'], data['uot_reg'], data['uot_tau'])
            couplings.append(pi)
            sparse.save_npz(shared / f'coupling_{gap}.npz', pi)
            for side, a in enumerate((left, right)):
                diagnostics.append(pd.DataFrame(dict(gap=gap, side=side, cell_id=a.obs_names.to_numpy(),
                                                     cell_class=a.obs[data['label_key']].astype(str).to_numpy(), **support[side])))
        pd.concat(diagnostics, ignore_index=True).to_csv(shared / 'matching_support.csv', index=False)
        provenance = dict(source='G zero frozen output', source_h5ad_sha256=file_sha256(source / 'generated.h5ad'),
                          source_model_sha256=file_sha256(source / 'model/model.ckpt'), training_performed=False,
                          modified_fields='XY and spatial provenance only; expression/type/depth/schedule frozen',
                          target_sampling_seed=int(seed), evaluation_rare_threshold=.05,
                          experimental_status='spatial mechanism candidate; no performance claim',
                          implementation_sha256={name: file_sha256(PROJECT / path) for name, path in (
                              ('spatial_transport', 'deepspatial/data_utils/spatial_transport.py'),
                              ('local_matching', 'deepspatial/data_utils/local_matching.py'),
                              ('runner', 'examples/run_v2_spatial.py'))})
        for mode in modes:
            folder = output / mode
            folder.mkdir()
            result = spatial_transport_volume(base, observed, couplings, mode, seed,
                                               data['spatial_key'], data['z_key'], data['label_key'])
            verify_schedule(result.obs, data['z_key'], expected)
            current = copy.deepcopy(config)
            current['experiment_v2_spatial'] = dict(**provenance, mode=mode, spatial_metadata=result.uns['v2_spatial'])
            (folder / 'run_config.json').write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')
            result.uns['experiment_v2_spatial'] = current['experiment_v2_spatial']
            result.write_h5ad(folder / 'generated.h5ad')
            result.uns['depth_type_sampling']['budget'].to_csv(folder / 'sampling_budget.csv', index=False)
            result.obs.groupby(['source_cell_class', 'generation_mode'], observed=True).agg(
                cells=('source_cell_index', 'size'), median_displacement_um=('v2_source_displacement_um', 'median'),
                mean_displacement_um=('v2_source_displacement_um', 'mean')).reset_index().to_csv(folder / 'spatial_displacement.csv', index=False)
            completed.append(dict(holdout=fold, mode=mode, generated_cells=result.n_obs, output=str(folder)))
            print(completed[-1], flush=True)
            del result
        del base, observed, couplings
        gc.collect()
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=PROJECT / 'outputs/g_holdouts_1_2_3')
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/v2_spatial_holdouts_1_2_3')
    parser.add_argument('--holdouts', type=int, nargs='+', default=[1, 2, 3])
    args = parser.parse_args()
    run_experiment(args.source_root, args.output, args.holdouts)
