"""Changed-path checks for the A0-A5 experiment package."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import torch

from test_experiments import ROOT, toy_slices
from mmdeepspatial.euclidean import ARMS, EUCLIDEAN_CONFIG, compare_arms, reconstruction_weights, run_arm
from mmdeepspatial.experiment import encode_adata, load_encoder, run_experiment
from mmdeepspatial.graph import InductiveGCNAE, training_neighbors
from mmdeepspatial.representation import ExpressionAE, Geometry, fit_ae, set_seed


class EuclideanChecks(unittest.TestCase):
    def test_cosine_and_weighted_encoder_objective(self):
        x = np.array([[1., 2., 0], [0, 0, 0], [-1, 0, 1]], dtype="float32")
        actual = Geometry("E0", 1., metric="cosine").distances(torch.from_numpy(x), torch.from_numpy(x[:2])).numpy()
        np.testing.assert_allclose(actual, cosine_distances(x.astype("float64"), x[:2].astype("float64")), atol=1e-12)
        labels = np.array(["a"]*9+["b"])
        weights = reconstruction_weights(labels)
        self.assertAlmostEqual(float(weights.mean()), 1., places=6)
        self.assertAlmostEqual(float(weights[-1]/weights[0]), 3., places=6)
        self.assertAlmostEqual(float(weights[:9].sum()/weights[-1]), 3., places=6)
        cfg = deepcopy(EUCLIDEAN_CONFIG)
        cfg.update(dim=4, hidden=12, epochs=1, batch_size=64, log_every=1)
        values = np.random.default_rng(9).normal(size=(10, 8)).astype("float32")
        set_seed(7)
        initial = ExpressionAE(8, 4, 12, "E0", 1.)
        with torch.no_grad():
            errors = (initial(torch.from_numpy(values))[0]-torch.from_numpy(values)).square()
            expected_mse = float(errors.mean())
            expected_loss = float((errors.mean(1)*torch.from_numpy(weights)).mean())
        with tempfile.TemporaryDirectory() as temp:
            fit_ae(values, cfg, "E0", 7, torch.device("cpu"), Path(temp), sample_weights=weights)
            history = pd.read_csv(Path(temp)/"training.csv").iloc[0]
            self.assertAlmostEqual(history.train_mse, expected_mse, places=6)
            self.assertAlmostEqual(history.train_objective, expected_loss, places=6)

    def test_gcn_matches_graph_formula_and_queries_are_independent(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(24, 8)).astype("float32")
        slices = np.repeat([0, 2], 12)
        neighbors = training_neighbors(x, slices, 3, query_chunk=4)
        np.testing.assert_array_equal(neighbors[:, 0], np.arange(len(x)))
        self.assertTrue((slices[neighbors] == slices[:, None]).all())
        self.assertTrue(all(len(set(row)) == 4 for row in neighbors))
        set_seed(4)
        model = InductiveGCNAE(8, 4, 12, "E0", 1.)
        matrix = torch.zeros(len(x), len(x))
        matrix[torch.arange(len(x))[:, None], torch.from_numpy(neighbors)] = 1/4
        with torch.no_grad():
            first = model.encoder[1](model.encoder[0](matrix@torch.from_numpy(x)))
            expected = model.encoder[2](matrix@first).numpy()
        np.testing.assert_allclose(model.encode_training(x, neighbors, torch.device("cpu"), 5), expected, atol=1e-6)
        model.bind_reference(x, neighbors)
        query = rng.normal(size=(7, 8)).astype("float32")
        original = model.encode_queries(query, torch.device("cpu"), batch_size=2)
        reverse = model.encode_queries(query[::-1].copy(), torch.device("cpu"), batch_size=5)
        np.testing.assert_allclose(original, reverse[::-1], atol=1e-6)
        changed = query.copy()
        changed[0] = 100
        np.testing.assert_allclose(original[1:], model.encode_queries(changed, torch.device("cpu"), 3)[1:], atol=1e-6)
        with self.assertRaises(ValueError):
            training_neighbors(x, slices, 12)

    def test_six_arm_fit_freeze_roundtrip_pairing_and_reuse(self):
        import mmdeepspatial.euclidean as runner
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            toy_slices(folder)
            cfg = dict(input_glob=str(folder/"merfish_*.h5ad"), output_root=str(folder/"out"),
                       holdouts=[1], seeds=[0, 1], dim=4, hidden=12, epochs=2, batch_size=16,
                       device="cpu", cpu_threads=1, graph_k=3, graph_query_chunk=7,
                       retrieval_ks=[1, 5], query_chunk=7, reference_chunk=13,
                       coverage_calibration_max=12, log_every=1)
            actual_read, actual_probe = runner.read_slice, runner.fit_probe
            for arm in ARMS:
                state = dict(frozen=False)
                def probe(*args, **kwargs):
                    result = actual_probe(*args, **kwargs)
                    state["frozen"] = True
                    return result
                def read(path, key):
                    if path.name == "merfish_1.h5ad":
                        self.assertTrue(state["frozen"])
                        state["frozen"] = False
                    return actual_read(path, key)
                with patch.object(runner, "read_slice", side_effect=read), patch.object(runner, "fit_probe", side_effect=probe):
                    result = run_arm(arm, cfg)
                self.assertEqual(len(result), 2)
                for seed in cfg["seeds"]:
                    run = folder/f"out/{arm}/holdout_1/seed_{seed}"
                    model, prep, geometry = load_encoder(run)
                    test = ad.read_h5ad(folder/"merfish_1.h5ad")
                    encoded = encode_adata(test, model, prep)
                    with np.load(run/"embeddings.npz") as embeddings:
                        np.testing.assert_allclose(encoded, embeddings["test"], atol=1e-6)
                    test.obsm["spatial"][:] = 1e9
                    test.obs["cell_class"] = "changed_label"
                    np.testing.assert_allclose(encode_adata(test, model, prep), encoded, atol=1e-6)
                    weights = pd.read_csv(run/"training_weights.csv").reconstruction_weight
                    if arm in {"A3", "A5"}:
                        self.assertGreater(weights.max(), weights.min())
                    else:
                        np.testing.assert_array_equal(weights, 1.)
                    self.assertEqual(geometry.metric, "cosine" if arm == "A0" else "euclidean")
                with patch.object(runner, "fit_ae", side_effect=AssertionError("retrained")):
                    run_arm(arm, cfg)
                with self.assertRaises(FileExistsError):
                    run_arm(arm, dict(cfg, epochs=3))
            runs, aggregate, types = compare_arms(folder/"out")
            self.assertEqual(len(runs), 12)
            self.assertEqual(len(aggregate), 6)
            self.assertTrue(runs[runs.arm == "A0"].test_reconstruction_mse.isna().all())
            self.assertEqual(runs[runs.arm.isin(["A2", "A3", "A4", "A5"])].trainable_parameters.nunique(), 1)
            self.assertEqual(len(pd.read_csv(folder/"out/comparison/paired_deltas.csv")), 12)
            for ordinary, balanced in (("A2", "A3"), ("A4", "A5")):
                state0 = torch.load(folder/f"out/{ordinary}/holdout_1/seed_0/encoder.pt", weights_only=True)["state_dict"]
                state1 = torch.load(folder/f"out/{balanced}/holdout_1/seed_0/encoder.pt", weights_only=True)["state_dict"]
                self.assertFalse(torch.equal(state0["encoder.0.weight"], state1["encoder.0.weight"]))
            # Reuse an actual E0 pipeline run, keeping both provenance and weights.
            legacy_cfg = {k: v for k, v in dict(cfg, output_root=str(folder/"legacy"), diagnostics=False).items()
                          if k not in {"graph_k", "graph_query_chunk"}}
            run_experiment("E0", legacy_cfg)
            reuse_cfg = dict(cfg, output_root=str(folder/"reuse"), reuse_a2_root=str(folder/"legacy/E0_ae"))
            reused = run_arm("A2", reuse_cfg)
            self.assertTrue(reused.encoder_reused.all())
            self.assertTrue((reused.train_seconds == 0).all())
            with patch.object(runner, "fit_ae", side_effect=AssertionError("retrained")):
                run_arm("A2", reuse_cfg)
            with self.assertRaises(ValueError):
                run_arm("A2", dict(reuse_cfg, epochs=3, output_root=str(folder/"bad_reuse")))
            # Partial runs cannot pass comparison or be overwritten on resume.
            (folder/"out/A5/holdout_1/seed_1/completed.json").unlink()
            with self.assertRaises(ValueError):
                compare_arms(folder/"out")
            with self.assertRaises(FileExistsError):
                run_arm("A5", cfg)

    def test_notebooks_have_clean_shared_configuration(self):
        notebooks = sorted(ROOT.glob("A*.ipynb"))
        self.assertEqual(len(notebooks), 3)
        all_arms = []
        for path in notebooks:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    ast.parse(cell.source)
                    self.assertIsNone(cell.execution_count)
                    self.assertFalse(cell.outputs)
                    for node in ast.parse(cell.source).body:
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ARMS_TO_RUN" for t in node.targets):
                            all_arms.extend(ast.literal_eval(node.value))
        self.assertEqual(sorted(all_arms), list(ARMS))


if __name__ == "__main__":
    unittest.main()
