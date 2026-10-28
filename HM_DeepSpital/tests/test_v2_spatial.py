"""V2 工程检查：显式双向路径、表达冻结、无支持保留及可运行入口。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(PROJECT), str(PROJECT / '.runtime_deps')]
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from deepspatial.data_utils.spatial_transport import spatial_transport_volume
from deepspatial.data_utils.sampling import depth_type_source_schedule
from examples.run_g import schedule_sha256
from examples.run_v2_spatial import inspect_sources, run_experiment
from test_pipeline import slices


class SpatialV2Tests(unittest.TestCase):
    def test_analytic_paths_frozen_values_and_unsupported(self):
        left, right = [], []
        observed = []
        for k, x in enumerate(([0., 100., 1000.], [20., 120., 2000.])):
            observed.append(ad.AnnData(X=np.ones((3, 2)),
                obs=pd.DataFrame({'cell_class': ['A', 'A', 'B'], 'z_coord': k * 100.}, index=[f'{k}_{j}' for j in range(3)]),
                var=pd.DataFrame(index=['g1', 'g2']), obsm={'spatial': np.column_stack([x, [0., 0., 0.]])}))
        obs = pd.DataFrame(dict(source_z=[0., 0., 100., 0.], source_cell_index=[0, 0, 0, 2],
                               source_cell_id=['0_0', '0_0', '1_0', '0_2'], source_cell_class=['A', 'A', 'A', 'B'],
                               source_direction=['forward', 'forward', 'backward', 'forward'],
                               z_coord=[25., 75., 25., 50.], cell_class=['A', 'A', 'A', 'B'],
                               is_observed=False, generation_mode='ode'), index=['a', 'b', 'c', 'd'])
        base = ad.AnnData(X=sparse.csr_matrix(np.arange(8).reshape(4, 2)), obs=obs,
                         var=observed[0].var.copy(), obsm={'spatial': np.ones((4, 2))},
                         layers={'frozen': np.arange(8).reshape(4, 2)})
        pi = sparse.csr_matrix([[.25, .75, 0], [0, 1, 0], [0, 0, 0]])
        result = spatial_transport_volume(base, observed, [pi], 'ot_barycentric')
        np.testing.assert_allclose(result.obsm['spatial'][:, 0], [23.75, 71.25, 5., 1000.])
        np.testing.assert_array_equal(result.X.toarray(), base.X.toarray())
        np.testing.assert_array_equal(result.layers['frozen'], base.layers['frozen'])
        pd.testing.assert_series_equal(result.obs.cell_class, base.obs.cell_class)
        np.testing.assert_array_equal(base.obsm['spatial'], np.ones((4, 2)))
        self.assertEqual(result.obs.generation_mode.iloc[-1], 'source_anchor')
        draw = spatial_transport_volume(base, observed, [pi], 'ot_sampled', seed=7)
        again = spatial_transport_volume(base, observed, [pi], 'ot_sampled', seed=7)
        np.testing.assert_array_equal(draw.obsm['spatial'], again.obsm['spatial'])
        self.assertAlmostEqual(draw.obsm['spatial'][1, 0], 3 * draw.obsm['spatial'][0, 0])
        self.assertEqual(draw.obs.v2_target_cell_index.iloc[0], draw.obs.v2_target_cell_index.iloc[1])
        copied = spatial_transport_volume(base, observed, [pi], 'source_xy')
        np.testing.assert_array_equal(copied.obsm['spatial'][:, 0], [0, 0, 20, 1000])
        with self.assertRaises(ValueError):
            spatial_transport_volume(base, observed, [sparse.eye(3, 2)], 'ot_sampled')

    def test_runner_frozen_input_and_hidden_file_exclusion(self):
        # Minimal frozen-output fixture: no shared, model, checkpoints, logs or S directory.
        with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            observed = []
            for i in range(5):
                a = slices()[0]
                a.obs_names = [f'slice{i}_{j}' for j in range(a.n_obs)]
                a.obs.z_coord = float(i * 50)
                a.obsm['spatial'] += i
                a.write_h5ad(data / f'merfish_{i}.h5ad')
                if i != 2:
                    observed.append(a)
            segments = []
            categories = ['common', 'rare']
            for a0, a1 in zip(observed[:-1], observed[1:]):
                z0, z1 = a0.obs.z_coord.iloc[0], a1.obs.z_coord.iloc[0]
                s = depth_type_source_schedule(np.array([0, 0, 0, 0, 1, 1]), np.array([0, 0, 0, 0, 1, 1]), z0, z1, 30)
                rows, genes, xy = [], [], []
                for t, fwd, idx, bin_id in zip(s['t'], s['is_forward'], s['source_index'], s['depth_bin_index']):
                    a = a0 if fwd else a1
                    rows.append(dict(parent_id=f'{a.obs.z_coord.iloc[0]:.12g}:{a.obs_names[idx]}', source_z=a.obs.z_coord.iloc[0],
                        source_cell_index=int(idx), source_cell_id=a.obs_names[idx], source_cell_class=str(a.obs.cell_class.iloc[idx]),
                        source_direction='forward' if fwd else 'backward', z_coord=z0 + t * (z1 - z0), depth_bin_index=int(bin_id),
                        cell_class=str(a.obs.cell_class.iloc[idx]), is_observed=False, generation_mode='ode'))
                    genes.append(a.X[idx]); xy.append(a.obsm['spatial'][idx])
                seg = ad.AnnData(X=np.asarray(genes), obs=pd.DataFrame(rows), var=a0.var.copy(), obsm={'spatial': np.asarray(xy)})
                budget = pd.DataFrame(s['budget'])
                budget['source_cell_class'] = [categories[i] for i in budget.source_class_code]
                seg.uns['depth_type_sampling'] = dict(bin_width_um=10., grid_origin_um=0., budget=budget)
                segments.append(seg)
            base = ad.concat(segments, index_unique='-', uns_merge='first')
            base.uns['depth_type_sampling']['budget'] = pd.concat([s.uns['depth_type_sampling']['budget'] for s in segments], ignore_index=True)
            config = json.loads((PROJECT / 'configs/merfish_g.json').read_text())
            config.update(input_glob=str(data / 'merfish_[0-4].h5ad'), run=dict(heldout_indices=[2], visible_files=[f'merfish_{i}.h5ad' for i in [0, 1, 3, 4]]))
            config['data']['niche_config']['mode'] = 'zero'
            digest = schedule_sha256(base.obs, 'z_coord')
            config['experiment_g'] = dict(source_schedule_sha256=digest, generated_schedule_sha256=digest,
                                         niche_mode='zero', heldout_id=2)
            base.uns['experiment_g'] = copy.deepcopy(config['experiment_g'])
            source = root / 'G/holdout_2/zero'
            source.mkdir(parents=True)
            (source / 'run_config.json').write_text(json.dumps(config), encoding='utf-8')
            base.write_h5ad(source / 'generated.h5ad')
            real_read = ad.read_h5ad
            def read_visible(path, *args, **kwargs):
                self.assertNotEqual(Path(path).name, 'merfish_2.h5ad')
                return real_read(path, *args, **kwargs)
            with patch('examples.run_v2_spatial.ad.read_h5ad', side_effect=read_visible):
                self.assertEqual(inspect_sources(root / 'G', (2,)).status.iloc[0], 'ready')
                completed = run_experiment(root / 'G', root / 'V2', holdouts=(2,))
            self.assertEqual(len(completed), 3)
            for item in completed:
                result = real_read(Path(item['output']) / 'generated.h5ad')
                np.testing.assert_array_equal(result.X, base.X)
                self.assertEqual(schedule_sha256(result.obs, 'z_coord'), config['experiment_g']['source_schedule_sha256'])
                self.assertFalse(result.uns['experiment_v2_spatial']['training_performed'])
                self.assertFalse(result.uns['experiment_v2_spatial']['source_model_available'])
            (source / 'generated.h5ad').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'SOURCE_ROOT'):
                run_experiment(root / 'G', root / 'missing_output', holdouts=(2,))
            self.assertFalse((root / 'missing_output').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
