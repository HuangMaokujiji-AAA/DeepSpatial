"""Run with: python -m unittest discover -s tests -v (from MMDeepSpatial)."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import geoopt
import numpy as np
import pandas as pd
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmdeepspatial.compare import compare_runs
from mmdeepspatial.data import Preprocessor
from mmdeepspatial.evaluation import classification_table, evaluate, fit_probe
from mmdeepspatial.experiment import DEFAULT_CONFIG, encode_adata, load_encoder, run_experiment
from mmdeepspatial.representation import ExpressionAE, Geometry, pairwise_components, set_seed


def toy_slices(folder):
    rng = np.random.default_rng(53)
    for i in range(3):
        labels = np.array(["common"]*34 + ["other"]*4 + ["rare"]*2)
        # Exactly 5% rare; other is common (>5%). Include a held-out-only type.
        if i == 1:
            labels[0] = "unseen"
        x = rng.poisson(2, size=(40, 8)).astype("float32")
        x[labels == "common", :2] += 8
        x[labels == "other", 2:4] += 10
        x[labels == "rare", 4:6] += 12
        a = ad.AnnData(sparse.csr_matrix(x),
                       obs=pd.DataFrame({"cell_class": labels}, index=[f"s{i}_c{j}" for j in range(40)]),
                       var=pd.DataFrame(index=[f"g{j}" for j in range(8)]))
        a.obsm["spatial"] = rng.normal(size=(40, 2))
        a.write_h5ad(folder/f"merfish_{i}.h5ad")


class GeometryChecks(unittest.TestCase):
    def test_capacity_and_manifold_gradients(self):
        counts = []
        for arm in ("E0", "E1", "E2"):
            set_seed(0)
            model = ExpressionAE(8, 4, 12, arm, .7)
            counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
            recon, z = model(torch.randn(12, 8))
            recon.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(float(model.encoder[0].weight.grad.norm()), 0)
            if arm != "E0":
                self.assertLess(float((z[:, model.e_dim:].detach().norm(dim=1)*.7**.5).max()), .981)
        self.assertEqual(len(set(counts)), 1)

    def test_distance_identity_symmetry_boundary_geoopt(self):
        for c in (.1, 1., 3.):
            ball = geoopt.PoincareBall(c=c).double()
            # geoopt initializes its softplus curvature parameter in float32;
            # use the effective stored c for a strict float64 formula comparison.
            effective_c = float(ball.c.detach())
            torch.manual_seed(4)
            u = ball.expmap0(torch.randn(9, 4, dtype=torch.float64)*.3)
            u[-1] = torch.tensor([.98/effective_c**.5, 0, 0, 0], dtype=torch.float64)
            _, d = pairwise_components(u, u, "E1", effective_c)
            expected = ball.dist(u[:, None, :], u[None, :, :])
            np.testing.assert_allclose(d.numpy(), expected.detach().numpy(), rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(d, d.T, atol=1e-12)
            np.testing.assert_allclose(d.diag(), 0., atol=1e-12)
            self.assertTrue(torch.isfinite(d).all() and (d >= 0).all())

    def test_near_zero_curvature_recovers_euclidean_autoencoder(self):
        set_seed(7)
        euclidean = ExpressionAE(8, 4, 12, "E0", 1e-6)
        set_seed(7)
        hyperbolic = ExpressionAE(8, 4, 12, "E1", 1e-6)
        x = torch.randn(9, 8)*.1
        np.testing.assert_allclose(euclidean(x)[0].detach(), hyperbolic(x)[0].detach(), atol=1e-6)

    def test_chunked_knn_matches_full_and_excludes_same_slice(self):
        rng = np.random.default_rng(0)
        for arm in ("E0", "E1", "E2"):
            raw = rng.normal(size=(27, 4)).astype("float32")*.1
            geom = Geometry(arm, 1., .3, scales=(2., 3.))
            full = geom.distances(torch.from_numpy(raw[:7]), torch.from_numpy(raw[7:])).numpy()
            ds, ids = geom.neighbors(raw[:7], raw[7:], 3, torch.device("cpu"), 2, 4)
            np.testing.assert_array_equal(ids, np.argsort(full, axis=1)[:, :3])
            np.testing.assert_allclose(ds, np.take_along_axis(full, ids, axis=1))
            slices = np.repeat([0, 1, 2], 9)
            _, masked = geom.neighbors(raw, raw, 1, torch.device("cpu"), 4, 5,
                                        exclude_same_slice=(slices, slices))
            self.assertTrue((slices[masked[:, 0]] != slices).all())

    def test_fusion_endpoints(self):
        x, y = torch.randn(5, 4)*.1, torch.randn(7, 4)*.1
        e, h = pairwise_components(x, y, "E2", 1.)
        np.testing.assert_allclose(Geometry("E2", 1., 1.).distances(x, y), e)
        np.testing.assert_allclose(Geometry("E2", 1., 0.).distances(x, y), h)


class DataAndMetricChecks(unittest.TestCase):
    def test_training_only_stats_genes_and_roundtrip(self):
        a = ad.AnnData(np.array([[1., 1, 3], [2, 1, 2], [3, 1, 1]], dtype="float32"),
                       var=pd.DataFrame(index=["variable", "constant", "variable2"]))
        prep = Preprocessor.fit([a, a], transform="identity")
        self.assertEqual(set(prep.genes), {"variable", "variable2"})
        mean = prep.mean.copy()
        test = a.copy()
        test.X[:] = 999
        test.var_names = ["variable", "test_only", "variable2"]
        prep.apply(test)
        np.testing.assert_array_equal(mean, prep.mean)
        self.assertNotIn("test_only", prep.genes)
        with tempfile.TemporaryDirectory() as d:
            prep.save(Path(d)/"prep.npz")
            loaded = Preprocessor.load(Path(d)/"prep.npz")
            np.testing.assert_allclose(prep.apply(a), loaded.apply(a))
        with self.assertRaises(ValueError):
            prep.apply(test[:, ["test_only"]])

    def test_absent_unseen_no_rare_and_permuted_probe(self):
        cfg = deepcopy(DEFAULT_CONFIG)
        x = np.array([[-3., 0], [-2, 0], [2, 0], [3, 0]], dtype="float32")
        train_y = np.array(["a", "a", "b", "b"])
        probe = fit_probe(x, train_y, cfg, 0)
        table, _, _ = classification_table(probe, x[:2], train_y, np.array(["a", "unseen"]), .05)
        self.assertFalse(table.is_rare.any())
        self.assertEqual(table.set_index("cell_type").loc["unseen", "recall"], 0)
        self.assertFalse(table.set_index("cell_type").loc["b", "test_present"])
        self.assertTrue(np.isnan(table.set_index("cell_type").loc["b", "auprc"]))
        # A deterministic shuffled-label control destroys an otherwise separable signal.
        xx = np.concatenate((np.ones((200, 1))*-4, np.ones((200, 1))*4)).astype("float32")
        yy = np.array(["a"]*200+["b"]*200)
        clean = fit_probe(xx, yy, cfg, 0)
        shuffle = fit_probe(xx, np.random.default_rng(0).permutation(yy), cfg, 0)
        self.assertGreater(clean.score(xx, yy), shuffle.score(xx, yy)+.4)

    def test_notebook_structure_and_syntax(self):
        import nbformat
        notebooks = sorted(ROOT.glob("E*.ipynb"))
        self.assertEqual(len(notebooks), 3)
        for notebook in notebooks:
            nb = nbformat.read(notebook, as_version=4)
            nbformat.validate(nb)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    ast.parse(cell.source)
                    self.assertIsNone(cell.execution_count)
                    self.assertFalse(cell.outputs)

    def test_coverage_missing_support_and_empty_rare_set(self):
        cfg = deepcopy(DEFAULT_CONFIG)
        cfg.update(retrieval_ks=[1], query_chunk=2, reference_chunk=2)
        train_z = np.array([[-3., 0], [-2, 0], [2, 0], [3, 0]], dtype="float32")
        meta = pd.DataFrame(dict(cell_type=["a", "a", "b", "b"], slice_index=[0, 2, 0, 0]))
        test_z = np.array([[-2., 0], [2., 0], [0., 0]], dtype="float32")
        test_meta = pd.DataFrame(dict(cell_type=["a", "b", "unseen"], slice_index=[1, 1, 1]))
        probe = fit_probe(train_z, meta.cell_type.to_numpy(), cfg, 0)
        table, summary, _, _ = evaluate(probe, test_z, train_z, test_z, meta, test_meta,
                                        Geometry("E0", 1.), cfg, 0, torch.device("cpu"))
        table = table.set_index("cell_type")
        self.assertEqual(table.loc["b", "coverage_status"], "single_train_slice")
        self.assertTrue(np.isnan(table.loc["b", "coverage"]))
        self.assertEqual(table.loc["unseen", "coverage_status"], "no_train_support")
        self.assertEqual(summary.rare_evaluated_types.iloc[0], 0)
        self.assertTrue(np.isnan(summary.rare_macro_auprc.iloc[0]))


class EndToEndChecks(unittest.TestCase):
    def test_all_arms_freeze_load_resume_and_compare(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            toy_slices(folder)
            config = dict(input_glob=str(folder/"merfish_*.h5ad"), output_root=str(folder/"outputs"),
                          holdouts=[1], seeds=[0], epochs=2, dim=4, hidden=12, batch_size=16,
                          device="cpu", retrieval_ks=[1, 5, 15], coverage_calibration_max=12,
                          query_chunk=7, reference_chunk=13, log_every=1, cpu_threads=1)
            # Spy on file parsing and fit order: no held-out AnnData before ALL fitting.
            import mmdeepspatial.experiment as runner
            actual_read, actual_fit, actual_probe = runner.read_slice, runner.fit_ae, runner.fit_probe
            for arm in ("E0", "E1", "E2"):
                state = dict(ae=False, probe=False)
                def train(*args, **kwargs):
                    model, timing = actual_fit(*args, **kwargs)
                    state["ae"] = True
                    return model, timing
                def probe(*args, **kwargs):
                    self.assertTrue(state["ae"])
                    result = actual_probe(*args, **kwargs)
                    state["probe"] = True
                    return result
                def read(path, key):
                    if path.name == "merfish_1.h5ad":
                        self.assertTrue(state["ae"] and state["probe"])
                    return actual_read(path, key)
                with patch.object(runner, "read_slice", side_effect=read), \
                     patch.object(runner, "fit_ae", side_effect=train), \
                     patch.object(runner, "fit_probe", side_effect=probe):
                    result = run_experiment(arm, config)
                self.assertEqual(len(result), 1)
                run = folder/f"outputs/{arm}_ae/holdout_1/seed_0"
                table = pd.read_csv(run/"per_type.csv").set_index("cell_type")
                self.assertTrue(table.loc["rare", "is_rare"])
                self.assertEqual(table.loc["rare", "train_fraction"], .05)
                self.assertFalse(table.loc["other", "is_rare"])
                self.assertFalse(table.loc["unseen", "train_supported"])
                self.assertEqual(result.unknown_test_n.iloc[0], 1)
                loaded, prep, geometry = load_encoder(run)
                test = ad.read_h5ad(folder/"merfish_1.h5ad")
                z = encode_adata(test, loaded, prep)
                with np.load(run/"embeddings.npz") as emb:
                    np.testing.assert_allclose(z, emb["test"], atol=1e-6)
                self.assertTrue((run/"completed.json").is_file())
                # Resume must not retrain or touch the test file.
                with patch.object(runner, "fit_ae", side_effect=AssertionError("retrained")):
                    run_experiment(arm, config)
                with self.assertRaises(FileExistsError):
                    run_experiment(arm, dict(config, epochs=3))
            all_runs, aggregate, types = compare_runs(folder/"outputs")
            self.assertEqual(len(all_runs), 3)
            self.assertEqual(all_runs.trainable_parameters.nunique(), 1)
            self.assertEqual(set(types.arm), {"E0", "E1", "E2"})
            # PCA is a separate completed E0 baseline, not an overwritten AE run.
            result = run_experiment("E0", dict(config, e0_method="pca", shuffled_probe=True))
            self.assertEqual(result.trainable_parameters.iloc[0], 0)
            pca_run = folder/"outputs/E0_pca/holdout_1/seed_0"
            pca, prep, _ = load_encoder(pca_run)
            self.assertEqual(encode_adata(ad.read_h5ad(folder/"merfish_1.h5ad"), pca, prep).shape, (40, 4))
            self.assertTrue((pca_run/"shuffled_probe_per_type.csv").exists())
            # Incomplete directories cannot be mistaken for completed runs.
            (folder/"outputs/E1_ae/holdout_1/seed_0/completed.json").unlink()
            with self.assertRaises(FileExistsError):
                run_experiment("E1", config)
            with self.assertRaises(ValueError):
                compare_runs(folder/"outputs")


if __name__ == "__main__":
    unittest.main()
