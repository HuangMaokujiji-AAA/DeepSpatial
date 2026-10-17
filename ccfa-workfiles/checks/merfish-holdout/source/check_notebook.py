"""Validate the notebook protocol and metrics without starting model training.

Run from the project root using an environment with anndata and nbformat:
    python ccfa-workfiles/checks/merfish-holdout/source/check_notebook.py
"""
import ast
import json
import tempfile
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK = ROOT / "merfish_leave_one_slice_out.ipynb"


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    cells = {cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"}
    for name, source in cells.items():
        ast.parse(source, filename=f"{NOTEBOOK.name}:{name}")
    assert notebook.cells[1].id == "holdout-ids"
    assert all(not c.outputs and c.execution_count is None for c in notebook.cells if c.cell_type == "code")

    scope = {}
    for name in ("holdout-ids", "configuration", "analysis-imports", "data-helpers", "metric-helpers"):
        exec(compile(cells[name], f"notebook:{name}", "exec"), scope)
    np, pd, ad, sp = (scope[name] for name in ("np", "pd", "ad", "sp"))

    # Real input files: all folds have four independent training slices and a bracketing pair.
    real_slices = {i: scope["read_slice"](i) for i in range(5)}
    for heldout in (1, 2, 3):
        ids = scope["training_ids"](heldout)
        assert len(ids) == 4 and heldout not in ids
        assert heldout - 1 in ids and heldout + 1 in ids
        assert all(real_slices[0].var_names.equals(a.var_names) for a in real_slices.values())
        train_counts = pd.concat([real_slices[i].obs[scope["LABEL_KEY"]] for i in ids]).value_counts()
        train_counts = train_counts[train_counts > 0]
        perfect = real_slices[heldout]
        _, summary = scope["evaluate_holdout"](perfect, perfect.copy(), train_counts)
        assert summary["rare_missing_classes"] == 0
        assert summary["symmetric_mean_nn_um"] == 0
        assert summary["rare_macro_coverage_recall_50um"] == 1
        assert np.isclose(summary["global_mean_log_expression_pcc"], 1)
        print(f"Input fold {heldout}: train={ids}, heldout_cells={perfect.n_obs}, classes={len(train_counts)}")

    def make_adata(labels, xy, sparse=False):
        expression = np.tile([1.0, 3.0, 7.0], (len(labels), 1))
        return ad.AnnData(
            X=sp.csr_matrix(expression) if sparse else expression,
            obs=pd.DataFrame({scope["LABEL_KEY"]: labels}, index=[str(i) for i in range(len(labels))]),
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
            obsm={"spatial": np.asarray(xy, dtype=float)},
        )

    # A missing rare class and a class never seen in training must remain in the output table.
    truth = make_adata(["common", "rare", "unseen"], [[0, 0], [10, 0], [20, 0]])
    prediction = make_adata(["common"], [[0, 0]], sparse=True)
    counts = pd.Series({"common": 1000, "rare": 1})
    evaluate = scope["evaluate_holdout"]
    table, summary = evaluate(truth, prediction, counts)
    labels = table.set_index("cell_class")
    assert labels.loc["rare", "missing_in_prediction"]
    assert labels.loc["rare", "coverage_recall_50um"] == 0
    assert labels.loc["unseen", "unseen_in_training"]
    assert not labels.loc["unseen", "is_rare_train"]
    assert summary["rare_missing_classes"] == 1
    assert summary["unseen_training_classes"] == 1
    assert summary["rare_macro_coverage_recall_50um"] == 0
    assert np.isnan(labels.loc["rare", "mean_log_expression_pcc"])

    # Class counts alone cannot identify a spatial hole: keep class rare but move it 200 um.
    displaced = make_adata(["common", "rare", "unseen"], [[0, 0], [210, 0], [20, 0]])
    table, summary = evaluate(truth, displaced, counts)
    labels = table.set_index("cell_class")
    assert summary["rare_missing_classes"] == 0
    assert labels.loc["rare", "coverage_recall_100um"] == 0
    assert labels.loc["rare", "truth_to_pred_nn_mean_um"] == 200

    try:
        evaluate(truth, prediction[:, ::-1].copy(), counts)
    except ValueError:
        pass
    else:
        raise AssertionError("Gene-order mismatch must fail")

    build_dir = Path(__file__).resolve().parents[1] / "build"
    build_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="io-check-", dir=build_dir) as temp_dir:
        temp = Path(temp_dir)
        assert temp.resolve().is_relative_to(ROOT.resolve())
        scope["write_json"](temp / "metrics.json", {"undefined": float("nan"), "ok": 1})
        assert json.loads((temp / "metrics.json").read_text(encoding="utf-8")) == {"undefined": None, "ok": 1}
        a = scope["add_spatial_3d"](real_slices[2].copy())
        a.write_h5ad(temp / "truth.h5ad", compression="gzip")
        b = ad.read_h5ad(temp / "truth.h5ad")
        assert b.obsm["spatial_3d"].shape == (real_slices[2].n_obs, 3)
        assert np.all(b.obsm["spatial_3d"][:, 2] == 100.0)

    print("PASS: notebook schema/syntax, real-data splits, missing-rare/unseen labels, spatial hole, gene order, JSON/h5ad I/O")
    print("Training was not executed by this check.")


if __name__ == "__main__":
    main()
