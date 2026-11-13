"""Runnable engineering tests, including all eight arms; not biological validation."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import ot
import torch

from raspatial.data import make_split, prepare, read_section, SliceStore
from raspatial.dataset import PairDataset, interpolate
from raspatial.evaluation import assignment
from raspatial.experiment import ARMS, run, verify_freeze
from raspatial.io import read_json
from raspatial.markers import load_markers, MarkerTasks, import_table
from raspatial.model import BaseFlow, ResidualFlow
from raspatial.training import fit, seed_all
from raspatial.transport import corrected, sinkhorn, global_uot_samples, local_uot, Transports, _raw_cost

torch.set_num_threads(1)


def fixture(root):
    cfg = read_json(PROJECT/'configs/abca2.json')
    cfg['input_glob'] = str(root/'input/sec_*.h5ad')
    cfg['split'] = dict(validation=['sec_003', 'sec_007'], test=['sec_005', 'sec_009'])
    cfg['data'].update(coordinate_to_um=1., niche_genes=3, cache_slices=4)
    cfg['markers']['path'] = str(root/'markers.csv')
    cfg['niche'].update(radius_um=25., sigma_um=12.)
    cfg['pairing'].update(n_samples=32, radius_um=80., spatial_scale_um=40., reference_gap_um=40., block_size=32,
                          max_iter=1000, tolerance=1e-8)
    cfg['model'].update(hidden_size=16, depth=1, num_heads=2, patch_size=2)
    cfg['residual']['hidden_size'] = 8
    cfg['training'].update(epochs=1, batch_size=8)
    cfg['C'].update(every_steps=1, cells=4, rollout_steps=2, bandwidth_um=20., grid_um=20.)
    cfg['reconstruction'].update(steps=3, chunk_size=32, export_step_um=30.)
    cfg['evaluation'].update(spatial_bin_um=20., niche_radius_um=25., matching_cap_um=100.)
    (root/'input').mkdir()
    rng = np.random.default_rng(8)
    for k in range(12):
        xy = rng.uniform(0, 100, (100, 2)); xy[-1] = [50+k*.3, 50]
        g = rng.poisson(2, (100, 6)).astype(np.float32)+1
        g[-1, :5] += 5
        obs = pd.DataFrame(dict(subclass=['001 common']*99+['002 rare'], cell_label=[f's{k}c{i}' for i in range(100)], z=float(20*k)),
                           index=[f's{k}c{i}' for i in range(100)])
        a = ad.AnnData(X=g, obs=obs, var=pd.DataFrame(dict(gene_symbol=[f'g{i}' for i in range(6)]), index=[f'id{i}' for i in range(6)]), obsm={'spatial': xy})
        a.write_h5ad(root/'input'/f'sec_{k:03d}.h5ad')
    pd.DataFrame(dict(subclass=['002 rare']*6, gene=[f'g{i}' for i in range(6)], source='synthetic_fixture')).to_csv(root/'markers.csv', index=False)
    return cfg


class PipelineTests(unittest.TestCase):
    def test_uot_equivalence_and_zero_support(self):
        cost = np.array([[.1, .2, .8], [.8, .3, .1]])
        reg, tau = .8, .05
        kernel = np.exp(-cost/reg)/6
        expected = ot.unbalanced.sinkhorn_unbalanced(np.ones(2)/2, np.ones(3)/3, cost,
                                                     reg=reg, reg_m=tau, reg_type='kl', numItermax=1000, stopThr=1e-12)
        for matrix in (kernel, sparse.csr_matrix(kernel)):
            u, v, _ = sinkhorn(matrix, tau, reg, 1e-12, 1000)
            np.testing.assert_allclose(u[:, None]*kernel*v, expected, rtol=1e-8, atol=1e-12)
        p = sparse.csr_matrix([[9., 1., 0.], [0., 1., 1.], [0., 0., 0.]])
        q = corrected(p, 1.)
        np.testing.assert_allclose(np.asarray(q.sum(1)).ravel(), [.5, .5, 0.])
        self.assertTrue(np.all(q.toarray()[p.toarray() == 0] == 0))
        with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
            cfg = read_json(PROJECT/'configs/abca2.json')
            cfg['pairing'].update(tolerance=1e-12, max_iter=1000, block_size=1)
            left = dict(x=np.array([[0., 0.], [.8, 1.]]), g=np.array([[1., 2.], [3., 1.]]), y=np.array([0, 1]))
            right = dict(x=np.array([[.1, 0.], [.9, 1.], [.5, .4]]), g=np.array([[2., 1.], [1., 4.], [3., 2.]]), y=np.array([0, 1, 0]))
            row, col, _ = global_uot_samples(left, right, cfg, Path(temp), 32, np.random.default_rng(42))
            with np.load(Path(temp)/'raw_factors.npz') as archive:
                factors = {k: archive[k] for k in archive.files}
            costs = _raw_cost(left, right, 0, 2, factors, .5)
            expected_cost = .5*ot.dist(left['x'], right['x'], metric='euclidean')/ot.dist(left['x'], right['x'], metric='euclidean').max()
            dg = ot.dist(left['g'], right['g'], metric='cosine')
            expected_cost += .5*dg/dg.max()+10*(left['y'][:, None] != right['y'][None, :])
            np.testing.assert_allclose(costs, expected_cost, atol=1e-12)
            expected = ot.unbalanced.sinkhorn_unbalanced(np.ones(2)/2, np.ones(3)/3, costs, reg=.8, reg_m=.05,
                                                         reg_type='kl', numItermax=1000, stopThr=1e-12)
            actual = factors['u'][:, None]*np.exp(-costs/.8)/6*factors['v']
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
            self.assertEqual(len(row), 32); self.assertEqual(len(col), 32)
            self.assertFalse((Path(temp)/'global_kernel.tmp.npy').exists())

    def test_capped_assignment_missing_and_exact(self):
        a = np.array([[0., 0.], [2., 0.]])
        b = np.array([[0., 0.], [100., 0.], [200., 0.]])
        value, i, j = assignment(a, b, 10.)
        self.assertAlmostEqual(value, 20/3)
        self.assertEqual(len(i), 1)
        self.assertEqual(assignment(a, np.empty((0, 2)), 10.)[0], 10.)
        self.assertEqual(assignment(a, a, 10.)[0], 0.)

    def test_split_isolation_marker_partition_and_frozen_zero_residual(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp); cfg = fixture(root)
            original = read_section
            seen = []
            def guarded(path, config, expression=True):
                if expression:
                    seen.append(Path(path).stem)
                    self.assertNotIn(Path(path).stem, cfg['split']['validation']+cfg['split']['test'])
                return original(path, config, expression)
            with patch('raspatial.data.read_section', side_effect=guarded):
                state, rows, _ = prepare(PROJECT, cfg, root/'cache')
            self.assertEqual(len(seen), 8)
            self.assertEqual(sum(state['counts']), 800)
            self.assertAlmostEqual(state['counts'][1]/sum(state['counts']), .01)
            store = SliceStore(root/'cache', state, 4)
            with self.assertRaises(ValueError):
                store['sec_005']
            markers, held, _ = load_markers(root/'markers.csv', state, cfg)
            self.assertTrue(markers[1]); self.assertTrue(held[1]); self.assertFalse(set(markers[1]) & set(held[1]))
            transports = Transports(root/'cache', store, cfg)
            ds = PairDataset(store, transports, state, cfg, ARMS['full'])
            batch = {k: torch.as_tensor(np.stack([ds[i][k] for i in range(4)])) for k in ds[0]}
            inputs, target, _ = interpolate(batch, state, torch.device('cpu'), torch.full((4,), .3))
            base = BaseFlow(state, cfg).eval()
            model = ResidualFlow(state, cfg, ARMS['full'], base.state_dict()).eval()
            for a, b in zip(base(**inputs), model(**inputs)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            tasks = MarkerTasks(store, transports, state, cfg, markers, 42)
            loss, log = tasks.loss(model, torch.device('cpu'))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(float(model.head[-1].weight.grad.abs().sum()), 0.)
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            self.assertNotIn(log['target'], cfg['split']['validation']+cfg['split']['test'])
            class NegativeGenes(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.velocity = torch.nn.Parameter(torch.tensor(-1000.))
                def forward(self, xt, gt, ct, **unused):
                    return xt*0+self.velocity*0, self.velocity.expand_as(gt), ct*0
            negative = NegativeGenes()
            zero_field_loss, _ = tasks.loss(negative, torch.device('cpu'))
            zero_field_loss.backward()
            self.assertTrue(torch.isfinite(negative.velocity.grad))

    def test_all_eight_arms_and_test_lock(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp); cfg = fixture(root); out = root/'out'; cache = root/'cache'
            reader = read_section
            def guarded(path, config, expression=True):
                name = Path(path).stem
                if expression and name in cfg['split']['test']:
                    self.assertTrue((out/'frozen_test_protocol.json').is_file())
                if expression and name in cfg['split']['validation']:
                    self.assertTrue(all((out/arm/'trained.json').is_file() for arm in ARMS))
                return reader(path, config, expression)
            with patch('raspatial.data.read_section', side_effect=guarded), patch('raspatial.experiment.read_section', side_effect=guarded):
                run(PROJECT, cfg, cache, out, stage='all', device='cpu')
            frozen = read_json(out/'frozen_test_protocol.json')
            self.assertEqual(set(frozen['checkpoints']), set(ARMS))
            self.assertTrue((out/'test_summary.csv').exists())
            completion = pd.read_csv(out/'completion.csv')
            self.assertEqual(len(completion), 32)
            self.assertTrue(completion.evaluated.all())
            for arm in ['R0', 'full', 'no_B', 'no_C', 'no_D']:
                a = read_json(out/arm/'test/sec_005/generated.json')
                b = read_json(out/'control/test/sec_005/generated.json')
                self.assertEqual(a['schedule_sha256'], b['schedule_sha256'])
            pred = ad.read_h5ad(out/'full/test/sec_005/prediction.h5ad')
            self.assertEqual(pred.uns['generation']['coordinate_units'], 'um')
            self.assertIn('expression_before_sparsification', pred.layers)
            run(PROJECT, cfg, cache, out, stage='test', device='cpu', resume=True)
            run(PROJECT, cfg, cache, out, stage='all', device='cpu', resume=True)
            with (out/'full/last.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                verify_freeze(out, read_json(cache/'split.json'))

    def test_marker_import_does_not_invent_columns(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            source = root/'table.xlsx'
            pd.DataFrame({'subclass': ['002 rare', '002 rare'], 'marker_genes': ['A, B;C', 'D E']}).to_excel(source, index=False)
            result = import_table(source, root/'markers.csv')
            self.assertEqual(set(result.gene), {'A', 'B', 'C', 'D', 'E'})
            with self.assertRaises(FileExistsError):
                import_table(source, root/'markers.csv')

    def test_epoch_resume_restores_marker_and_anchor_random_streams(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp); cfg = fixture(root); cfg['training']['epochs'] = 2
            state, _, _ = prepare(PROJECT, cfg, root/'cache')
            store = SliceStore(root/'cache', state, 4); transports = Transports(root/'cache', store, cfg)
            markers, _, _ = load_markers(root/'markers.csv', state, cfg)
            dataset = PairDataset(store, transports, state, cfg, ARMS['full'])
            anchors = PairDataset(store, transports, state, cfg, ARMS['control'])
            seed_all(42); base_state = BaseFlow(state, cfg).state_dict()
            def model():
                seed_all(42)
                return ResidualFlow(state, cfg, ARMS['full'], base_state)
            def tasks():
                return MarkerTasks(store, transports, state, cfg, markers, 473)
            expected = fit(model(), dataset, cfg, root/'continuous', state, torch.device('cpu'), {'test': 'resume'}, tasks(), anchors)
            from raspatial.training import save_checkpoint
            def crash_after_checkpoint(path, value):
                save_checkpoint(path, value)
                if value['epoch'] == 1:
                    raise RuntimeError('simulated interruption after epoch checkpoint')
            with patch('raspatial.training.save_checkpoint', side_effect=crash_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    fit(model(), dataset, cfg, root/'interrupted', state, torch.device('cpu'), {'test': 'resume'}, tasks(), anchors)
            actual = fit(model(), dataset, cfg, root/'interrupted', state, torch.device('cpu'), {'test': 'resume'}, tasks(), anchors, resume=True)
            for key in expected.state_dict():
                torch.testing.assert_close(actual.state_dict()[key], expected.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
