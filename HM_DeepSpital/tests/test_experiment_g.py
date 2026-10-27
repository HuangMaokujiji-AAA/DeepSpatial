"""G 工程检查：局部支持、拒绝伪配对、三组独立训练及精确 S 调度重放。"""
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
import torch
from deepspatial import DeepSpatial
from deepspatial.module import DeepSpatialModule
from deepspatial.data_utils.local_matching import compute_local_uot, local_support, matching_parameters, sample_local_pairs
from examples.run_g import run_experiment, schedule_sha256
from test_pipeline import slices

torch.set_num_threads(1)


class ExperimentGTests(unittest.TestCase):
    def test_local_support_transport_and_balanced_sampling(self):
        geometry = dict(max_distance_um=10., spatial_scale_um=10., reference_gap_um=100.)
        self.assertEqual(matching_parameters(geometry, 50), (5., 5.))
        x = np.array([[0, 0], [5, 0], [1000, 0], [0, 10.], [0, 0]])
        y = np.array([[0, 0], [12, 0], [2000, 0], [0, 11.], [0, 0]])
        c0, c1 = np.array([0, 0, 0, 1, 2]), np.array([0, 0, 0, 1, 3])
        pi, support = compute_local_uot(x, np.ones((5, 3)), c0, y, np.ones((5, 3)), c1, 100., geometry)
        np.testing.assert_array_equal(support[0]['candidate_count'], [1, 2, 0, 1, 0])
        self.assertGreater(pi[0, 0], 0)  # 精确零距离是有效边。
        self.assertEqual(pi[2].sum(), 0)
        self.assertEqual(pi[4].sum(), 0)  # 缺类不能强配为其他类型。
        self.assertTrue(np.isinf(support[0]['nearest_same_type_um'][4]))
        for fraction in (0, 1):
            r, c, _ = sample_local_pairs(pi, c0, c1, 200, fraction)
            self.assertTrue((np.linalg.norm(x[r] - y[c], axis=1) <= 10).all())
            np.testing.assert_array_equal(c0[r], c1[c])
            self.assertFalse(np.isin(r, [2, 4]).any())
        # 隔离无候选端点仍保留原始全切片边际；一个零成本边有解析解。
        one, _ = compute_local_uot(x[:1], [[1, 1]], [0], y[:1], [[1, 1]], [0], 100., geometry)
        self.assertAlmostEqual(float(one.sum()), 1., places=3)
        isolated, _ = compute_local_uot(x[[0, 2]], np.ones((2, 2)), [0, 2], y[[0, 2]], np.ones((2, 2)), [0, 3], 100., geometry)
        self.assertAlmostEqual(float(isolated.sum()), .25 ** (2 / 3), places=3)
        empty, _ = compute_local_uot(x[:1], [[1, 1]], [2], y[:1], [[1, 1]], [3], 100., geometry)
        self.assertEqual(empty.nnz, 0)
        with self.assertRaises(ValueError):
            sample_local_pairs(empty, [2], [3], 2, 1)

    def test_three_controls_replay_and_unsupported_source_anchor(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            original = []
            for i in range(5):
                a = slices()[0]
                a.obs_names = [f'slice{i}_{j}' for j in range(a.n_obs)]
                a.obs['z_coord'] = float(i * 50)
                a.obsm['spatial'] = a.obsm['spatial'].astype(float) + i
                # 每张可见切片留一个相距遥远的稀有端点，不能被其他类救回。
                a.obsm['spatial'][-1] = [10000 * (i + 1) + .123456789, 4000]
                a.write_h5ad(data / f'merfish_{i}.h5ad')
                original.append(a)
            visible = [a.copy() for i, a in enumerate(original) if i != 2]
            config = dict(seed=42, input_glob=str(data / 'merfish_[0-4].h5ad'), slice_z_um=[0, 50, 100, 150, 200],
                          data=dict(spatial_key='spatial', z_key='z_coord', label_key='cell_class',
                                    batch_size=32, num_workers=0, n_samples_base=64,
                                    uot_reg=.05, uot_tau=.05, alpha_spatial=.5,
                                    balanced_fraction=.2, source_sampling='stratified',
                                    niche_config=dict(radius=10, programs={'p': ['Mbp']}, mode='real')),
                          model=dict(hidden_size=32, depth=1, num_heads=4, patch_size=2,
                                     sampling_method='euler', bidirectional_training=True),
                          training=dict(max_epochs=1, accelerator='cpu', devices=1),
                          reconstruction=dict(thickness=10., steps=4, chunk_size=4, save_intermediates=True,
                                              source_sampling='depth_type_stratified', depth_bin_width_um=10., depth_grid_origin_um=0.),
                          run=dict(heldout_indices=[2], visible_files=[f'data/merfish_{i}.h5ad' for i in (0, 1, 3, 4)]))
            model = DeepSpatial()
            model.setup_data(visible, **config['data'])
            model.build_model(**config['model'])
            reference = model.reconstruct_full_volume(visible, device='cpu', **config['reconstruction'])
            source = root / 'S/holdout_2'
            source.mkdir(parents=True)
            reference.write_h5ad(source / 'generated.h5ad')
            (source / 'run_config.json').write_text(json.dumps(config), encoding='utf-8')
            config['data']['local_matching_config'] = dict(max_distance_um=20., spatial_scale_um=10., reference_gap_um=100.)
            config_path = root / 'g.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            real_read = ad.read_h5ad
            real_setup = DeepSpatial.setup_data
            real_fit = DeepSpatial.fit
            real_sample = DeepSpatialModule.sample
            conditions = {}
            setups = []
            sampled_parents = []

            def read_visible(path, *args, **kwargs):
                self.assertNotEqual(Path(path).name, 'merfish_2.h5ad')
                return real_read(path, *args, **kwargs)

            def setup_once(instance, *args, **kwargs):
                setups.append(1)
                return real_setup(instance, *args, **kwargs)

            def fit_control(instance, *args, **kwargs):
                mode = instance.niche_featurizer.mode
                conditions[mode] = instance.dataset.tensors['h0'].clone()
                return real_fit(instance, *args, **kwargs)

            def sample_supported(instance, batch, *args, **kwargs):
                sampled_parents.extend(batch['x0'][:, 0].cpu().numpy())
                return real_sample(instance, batch, *args, **kwargs)

            with patch('examples.run_g.ad.read_h5ad', side_effect=read_visible), \
                    patch.object(DeepSpatial, 'setup_data', new=setup_once), \
                    patch.object(DeepSpatial, 'fit', new=fit_control), \
                    patch.object(DeepSpatialModule, 'sample', new=sample_supported):
                results = run_experiment(root / 'S', root / 'G', config_path, holdouts=(2,), device='cpu')
            self.assertEqual(len(setups), 1)
            self.assertEqual(len(results), 3)
            self.assertTrue(sampled_parents)
            self.assertTrue((np.asarray(sampled_parents) < .01).all())
            self.assertEqual(set(conditions), {'real', 'zero', 'shuffle'})
            self.assertEqual(int(torch.count_nonzero(conditions['zero'])), 0)
            self.assertFalse(torch.equal(conditions['real'], conditions['shuffle']))
            initial_hashes = set()
            schedule_hash = schedule_sha256(reference.obs, 'z_coord')
            for mode in ('real', 'zero', 'shuffle'):
                folder = root / 'G/holdout_2' / mode
                metadata = json.loads((folder / 'run_config.json').read_text())
                initial_hashes.add(metadata['experiment_g']['initial_state_sha256'])
                self.assertEqual(metadata['experiment_g']['generated_schedule_sha256'], schedule_hash)
                result = real_read(folder / 'generated.h5ad')
                self.assertEqual(result.n_obs, reference.n_obs)
                unsupported = result.obs.generation_mode == 'source_anchor'
                self.assertGreater(unsupported.sum(), 0)
                self.assertTrue((result.obs.loc[unsupported, 'matching_candidate_count'] == 0).all())
                for row in np.flatnonzero(unsupported):
                    entry = result.obs.iloc[row]
                    a = original[int(entry.source_z / 50)]
                    parent = int(entry.source_cell_index)
                    np.testing.assert_array_equal(result.obsm['spatial'][row], a.obsm['spatial'][parent])
                    np.testing.assert_array_equal(result.X[row].toarray().ravel(), a.X[parent])
                    self.assertEqual(entry.cell_class, a.obs.cell_class.iloc[parent])
                restored = DeepSpatial()
                restored.load_checkpoint(folder / 'model/model.ckpt')
                self.assertEqual(restored.local_matching_config, config['data']['local_matching_config'])
                self.assertEqual(restored.niche_featurizer.mode, mode)
            self.assertEqual(len(initial_hashes), 1)
            with self.assertRaises(FileExistsError):
                run_experiment(root / 'S', root / 'G', config_path, holdouts=(2,), device='cpu')


if __name__ == '__main__':
    unittest.main(verbosity=2)
