"""Run with numpy/h5py/anndata/scipy/pandas: python tests/test_train_openst.py."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_openst as run


def main():
    # Defaults must match the actual original implementation, not a tutorial override.
    tree = ast.parse((run.ROOT / "deepspatial/core.py").read_text())
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name, expected in (("setup_data", run.DATA_KWARGS), ("build_model", run.MODEL_KWARGS)):
        func = functions[name]
        defaults = dict(zip([a.arg for a in func.args.args][-len(func.args.defaults):],
                            [ast.literal_eval(d) for d in func.args.defaults]))
        for key, value in expected.items():
            if key not in ("label_key",):  # Only the dataset's label column differs.
                assert defaults[key] == value, (name, key, defaults[key], value)
    assert ast.literal_eval(functions["fit"].args.defaults[0]) == 100
    assert len(run.TRAIN) == 13 and set(run.TRAIN).isdisjoint(run.TEST + run.VALIDATION)
    assert set(run.TRAIN + run.TEST + run.VALIDATION) == set(run.SECTIONS)
    assert [run.reference_pair(s) for s in run.TEST] == [(3, 5), (17, 19), (23, 25), (7, 11)]
    assert [run.reference_pair(s) for s in run.VALIDATION] == [(5, 7), (33, 36)]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ids = np.repeat(run.SECTIONS, 3)
        xyz = np.column_stack((np.arange(len(ids), dtype=float), np.ones(len(ids)), ids * 100.0))
        original = ad.AnnData(
            X=sparse.csr_matrix(np.tile([1., 2., 3.], (len(ids), 1)), dtype=np.float32),
            obs=pd.DataFrame({"n_section": ids, "annotation": pd.Categorical(["A", "B", "A"] * 19)},
                             index=[f"cell_{i}" for i in range(len(ids))]),
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
            obsm={"spatial": xyz[:, :2] + 999, "spatial_3d_aligned": xyz},
            layers={"raw": sparse.csr_matrix(np.ones((len(ids), 3)) * 100)},
        )
        path = root / "input.h5ad"
        original.write_h5ad(path)
        manifest = run.inspect_input(path)
        assert manifest["n_cells"] == 57 and manifest["n_genes"] == 3
        assert len(manifest["gene_names_sha256"]) == 64
        stats = dict(x_min=0., x_range=60., y_min=0., y_range=2., z_min=200., z_range=3400.)
        calls = []

        class Generator:
            spatial_stats = stats.copy()

            def reconstruct_between_slices(self, left, right, **kwargs):
                a, b = int(left.obs.n_section.iloc[0]), int(right.obs.n_section.iloc[0])
                assert a in run.TRAIN and b in run.TRAIN
                assert "spatial_norm" in left.obsm and "z_norm" in right.obs
                assert kwargs["steps"] == 20 and kwargs["thickness"] == 10
                calls.append((a, b))
                z = np.linspace(left.obs.z_coord.iloc[0], right.obs.z_coord.iloc[0], 101)
                return ad.AnnData(
                    X=sparse.csr_matrix(np.tile([1., 2., 3.], (len(z), 1))),
                    obs=pd.DataFrame({"annotation": pd.Categorical(["A"] * len(z)), "z_coord": z,
                                      "n_section": a, "z_norm": 999.},
                                     index=[f"generated_{i}" for i in range(len(z))]),
                    var=left.var.copy(), obsm={"spatial": np.ones((len(z), 2))},
                )

        with run.open_input(path) as source:
            assert not hasattr(source, "layers")
            train = {s: run.read_slice(source, s) for s in run.TRAIN}
            assert all(a.n_obs == 3 for a in train.values())
            first = train[2]
            assert np.array_equal(first.obsm["spatial"], xyz[:3, :2])
            assert np.array_equal(first.X.toarray(), [[1., 2., 3.]] * 3)
            model = Generator()
            args = SimpleNamespace(seed=42, slab_half_width=5., thickness=10., ode_steps=20,
                                   chunk_size=4, device="cpu")
            # No network is trained here: test the complete real H5AD export/assessment flow.
            with patch.dict(sys.modules, {"pytorch_lightning": SimpleNamespace(seed_everything=lambda *a, **k: None)}):
                with redirect_stdout(io.StringIO()):
                    run.reconstruct(model, source, train, manifest, root, args)
            assert model.spatial_stats == stats
            for split, sections in (("test", run.TEST), ("validation", run.VALIDATION)):
                for section in sections:
                    prediction = ad.read_h5ad(root / split / f"reconstructed_section_{section:02d}.h5ad")
                    assert prediction.n_obs > 0 and prediction.var_names.equals(original.var_names)
                    assert set(prediction.obs) == {"annotation", "z_coord", "n_section", "split"}
                    assert set(prediction.obs.n_section) == {section}
                    assert np.all(np.abs(prediction.obs.z_coord.to_numpy() - section * 100) <= 5)
                    assert np.array_equal(prediction.obsm["spatial_3d_aligned"][:, 2], prediction.obs.z_coord)
                    metrics = json.loads((root / split / f"section_{section:02d}_metrics.json").read_text())
                    assert np.isclose(metrics["gene_mean_mse"], 0) and np.isclose(metrics["gene_mean_pcc"], 1)
            assert calls == [(5, 7), (33, 36), (3, 5), (17, 19), (23, 25), (7, 11)]
            assert len(pd.read_csv(root / "reconstruction_metrics.csv")) == 6
            first.X[0, :] = 0
            # Test the read boundary against zero-expression cells (undefined cosine distance).
            bad_source = SimpleNamespace(obs=first.obs, var=first.var, obsm=first.obsm, X=first.X)
            try:
                run.read_slice(bad_source, 2)
            except ValueError as error:
                assert "zero-expression" in str(error)
            else:
                raise AssertionError("Zero-expression cells must fail before UOT.")
    print("Open-ST split, original defaults, aligned data loading, six H5AD exports and metric checks passed.")


if __name__ == "__main__":
    main()
