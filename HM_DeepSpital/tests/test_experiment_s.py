"""实验 S：检查联合抽样分布、来源预算及只加载已有权重的推理路径。"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))
if sys.platform == 'win32' and (PROJECT / '.runtime_deps').is_dir():
    sys.path.insert(1, str(PROJECT / '.runtime_deps'))

import anndata as ad
import numpy as np
import pandas as pd
import torch
from deepspatial import DeepSpatial
from deepspatial.data_utils.sampling import depth_type_source_schedule
from examples.run_s import run_experiment, file_sha256
from test_pipeline import slices

torch.set_num_threads(1)


class ExperimentSTests(unittest.TestCase):
    def test_budget_parent_support_boundaries_and_reproducibility(self):
        left, right = np.array([0, 0, 0, 0, 1]), np.array([0] * 9 + [2])
        s = depth_type_source_schedule(left, right, -7, 63, 137, 10, 4, rng=np.random.default_rng(7))
        same = depth_type_source_schedule(left, right, -7, 63, 137, 10, 4, rng=np.random.default_rng(7))
        for key in ('t', 'source_index', 'source_class', 'depth_bin_index', 'is_forward'):
            np.testing.assert_array_equal(s[key], same[key])
        self.assertEqual(len(s['t']), 137)
        z = -7 + s['t'] * 70
        np.testing.assert_array_equal(np.floor((z - 4 + 5) / 10).astype(int), s['depth_bin_index'])
        self.assertTrue(((s['t'] >= 0) & (s['t'] < 1)).all())
        budget = pd.DataFrame(s['budget'])
        self.assertEqual(int(budget.allocated_n.sum()), 137)
        self.assertAlmostEqual(budget.expected_n.sum(), 137, places=10)
        for row in budget.itertuples():
            mask = ((s['depth_bin_index'] == row.depth_bin_index) & (s['source_class'] == row.source_class_code)
                    & (s['is_forward'] == (row.source_direction == 'forward')))
            self.assertEqual(int(mask.sum()), row.allocated_n)
            self.assertLessEqual(abs(row.allocated_n - row.conditional_expected_n), 1 + 1e-10)
        for forward, labels in ((True, left), (False, right)):
            m = s['is_forward'] == forward
            np.testing.assert_array_equal(labels[s['source_index'][m]], s['source_class'][m])
            for code in np.unique(labels):
                parents = np.flatnonzero(labels == code)
                counts = np.bincount(s['source_index'][m & (s['source_class'] == code)], minlength=len(labels))[parents]
                self.assertLessEqual(int(np.ptp(counts)), 1)

    def test_type_variance_and_original_conditional_time_density(self):
        labels = np.array([0] * 99 + [1])
        # 等密度时中心10µm层期望正好一个稀有类：联合方向舍入不能让其变成0或2。
        for seed in range(30):
            s = depth_type_source_schedule(labels, labels, 0, 100, 1000, rng=np.random.default_rng(seed))
            rare = (s['depth_bin_index'] == 5) & (s['source_class'] == 1)
            self.assertEqual(int(rare.sum()), 1)
        # 单层覆盖整个区间时，原正向条件深度密度为2(1-t)，反向为2t。
        s = depth_type_source_schedule(labels, labels, 0, 100, 20000, 200, 50, rng=np.random.default_rng(4))
        self.assertAlmostEqual(s['t'][s['is_forward']].mean(), 1 / 3, delta=.015)
        self.assertAlmostEqual(s['t'][~s['is_forward']].mean(), 2 / 3, delta=.015)
        # 不等端点密度下，E[t]与方向总量仍保持原解析分布。
        s = depth_type_source_schedule(np.zeros(20, int), np.zeros(80, int), 0, 100, 20000,
                                       200, 50, rng=np.random.default_rng(2))
        expected_t = (20 / 2 + 60 / 3) / 50
        self.assertAlmostEqual(s['t'].mean(), expected_t, delta=.015)
        self.assertAlmostEqual((~s['is_forward']).mean(), expected_t, delta=1 / 20000 + 1e-10)
        # 小于1的期望允许零；零总数也能返回可核验的空调度。
        s = depth_type_source_schedule(np.array([0]), np.array([0, 1]), 0, 10, 0)
        self.assertEqual(len(s['t']), 0)
        self.assertEqual(sum(r['allocated_n'] for r in s['budget']), 0)
        with self.assertRaises(ValueError):
            depth_type_source_schedule(labels, labels, 0, 100, 10, 0)

    def test_saved_checkpoint_inference_without_training_or_hidden_read(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            original = []
            for i in range(5):
                a = slices()[0]
                a.obs_names = [f'slice{i}_{j}' for j in range(a.n_obs)]
                a.obs['z_coord'] = float(i * 50)
                a.obsm['spatial'] = a.obsm['spatial'] + i
                a.write_h5ad(data / f'merfish_{i}.h5ad')
                original.append(a)
            visible = [a.copy() for i, a in enumerate(original) if i != 2]
            model = DeepSpatial()
            model.setup_data(visible, num_workers=0, batch_size=32, n_samples_base=64,
                             source_sampling='stratified', niche_config={'radius': 10, 'programs': {'p': ['Mbp']}})
            model.build_model(hidden_size=32, depth=1, num_heads=4, patch_size=2, sampling_method='euler')
            source = root / 'completed/holdout_2'
            model.save_model(source / 'model')
            checkpoint = source / 'model/model.ckpt'
            before = file_sha256(checkpoint)
            config = dict(seed=42, input_glob=str(data / 'merfish_[0-4].h5ad'), slice_z_um=[0, 50, 100, 150, 200],
                          data={'label_key': 'cell_class', 'z_key': 'z_coord'},
                          reconstruction={'thickness': 10, 'steps': 4, 'chunk_size': 4, 'save_intermediates': True},
                          run={'heldout_indices': [2], 'visible_files': [f'data/merfish_{i}.h5ad' for i in (0, 1, 3, 4)]})
            (source / 'run_config.json').write_text(json.dumps(config), encoding='utf-8')
            read_h5ad = ad.read_h5ad
            reads = []

            def read_visible(path, *args, **kwargs):
                self.assertNotEqual(Path(path).name, 'merfish_2.h5ad')
                reads.append(Path(path).name)
                return read_h5ad(path, *args, **kwargs)

            with patch('examples.run_s.ad.read_h5ad', side_effect=read_visible), \
                    patch.object(DeepSpatial, 'fit', side_effect=AssertionError('S must not train')), \
                    patch.object(DeepSpatial, 'setup_data', side_effect=AssertionError('S must not rebuild UOT')):
                runs = run_experiment(root / 'completed', root / 'S', (2,), device='cpu', audit_seeds=2)
            self.assertEqual(reads, ['merfish_0.h5ad', 'merfish_1.h5ad', 'merfish_3.h5ad', 'merfish_4.h5ad'])
            self.assertEqual(file_sha256(checkpoint), before)
            output = root / 'S/holdout_2'
            result = read_h5ad(output / 'generated.h5ad')
            self.assertEqual(runs[0]['generated_cells'], 120)
            self.assertEqual(result.uns['hm_reconstruction']['generated_cells'], result.n_obs)
            self.assertEqual(result.uns['hm_reconstruction']['unique_parents'], result.obs.parent_id.nunique())
            budget = pd.read_csv(output / 'sampling_budget.csv')
            self.assertEqual(int(budget.allocated_n.sum()), result.n_obs)
            self.assertEqual(len(budget.groupby(['gap_start_um', 'gap_end_um'])), 3)
            np.testing.assert_array_equal(np.floor((result.obs.z_coord + 5) / 10).astype(int), result.obs.depth_bin_index)
            np.testing.assert_allclose(result.obs.z_norm, result.obs.z_coord / 200)
            self.assertTrue((output / 'sampling_audit.csv').is_file())
            self.assertTrue((output / 'sampling_counts.csv').is_file())
            with self.assertRaises(FileExistsError):
                run_experiment(root / 'completed', root / 'S', (2,), device='cpu', audit_seeds=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
