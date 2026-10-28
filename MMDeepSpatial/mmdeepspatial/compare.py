"""Paired descriptive comparison, with protocol and split consistency checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare_runs(output_root, e0_method="ae"):
    root = Path(output_root)
    summaries, types, protocols, environments = [], [], [], []
    keys = ["holdout", "seed"]
    for arm in ("E0", "E1", "E2"):
        method = e0_method if arm == "E0" else "ae"
        folder = root/f"{arm}_{method}"
        if not (folder/"summary.csv").is_file():
            raise FileNotFoundError(f"Wait for all three notebooks to finish: {folder}/summary.csv")
        summary = pd.read_csv(folder/"summary.csv")
        if summary.duplicated(keys).any():
            raise ValueError(f"Duplicate fold/seed rows in {folder}.")
        expected = None
        for row in summary.itertuples():
            run = folder/f"holdout_{row.holdout}"/f"seed_{row.seed}"
            if not (run/"completed.json").exists():
                raise ValueError(f"Incomplete run: {run}")
            manifest = json.loads((run/"manifest.json").read_text())
            cfg = manifest["config"]
            expected = {(h, s) for h in cfg["holdouts"] for s in cfg["seeds"]}
            # Runtime/output choices do not change the scientific protocol.
            protocol = {k: v for k, v in cfg.items() if k not in
                        {"output_root", "device", "cpu_threads", "log_every", "e0_method"}}
            protocols.append(protocol)
            environments.append(manifest["source_sha256"])
            types.append(pd.read_csv(run/"per_type.csv"))
            if arm != "E0":
                baseline = root/f"E0_{e0_method}"/f"holdout_{row.holdout}"/f"seed_{row.seed}"
                base_manifest = json.loads((baseline/"manifest.json").read_text())
                for field in ("inputs", "train_indices", "test_index"):
                    if manifest[field] != base_manifest[field]:
                        raise ValueError(f"Different {field}; cannot compare arms.")
                with np.load(run/"preprocessing.npz") as a, np.load(baseline/"preprocessing.npz") as b:
                    if any(not np.array_equal(a[k], b[k]) for k in a.files):
                        raise ValueError("Preprocessing differs between arms.")
        if expected != set(summary[keys].itertuples(index=False, name=None)):
            raise ValueError(f"Not all configured folds/seeds completed in {folder}.")
        summaries.append(summary)
    if any(p != protocols[0] for p in protocols) or any(p != environments[0] for p in environments):
        raise ValueError("Scientific configuration or implementation differs between runs.")
    if any(set(s[keys].itertuples(index=False, name=None)) !=
           set(summaries[0][keys].itertuples(index=False, name=None)) for s in summaries[1:]):
        raise ValueError("Arms have different folds/seeds.")
    all_runs = pd.concat(summaries, ignore_index=True)
    per_type = pd.concat(types, ignore_index=True)
    metrics = [c for c in all_runs if c.startswith(("rare_macro_", "common_macro_"))]
    metrics += ["overall_accuracy", "train_seconds", "peak_train_gpu_mb", "test_reconstruction_mse"]
    # Average seeds within folds before descriptive fold statistics. These folds from
    # one tissue are correlated; this is NOT a biological confidence interval.
    by_fold = all_runs.groupby(["arm", "method", "holdout"], observed=True)[metrics].mean().reset_index()
    aggregate = by_fold.groupby(["arm", "method"], observed=True)[metrics].agg(["mean", "std"])
    aggregate.columns = [f"{m}_{stat}" for m, stat in aggregate.columns]
    baseline = summaries[0].set_index(keys)
    deltas = []
    for current in summaries[1:]:
        d = current.set_index(keys)[metrics] - baseline[metrics]
        d.columns = [f"delta_vs_E0_{c}" for c in d.columns]
        d["arm"] = current.arm.iloc[0]
        deltas.append(d.reset_index())
    destination = root/"comparison"
    destination.mkdir(exist_ok=True)
    all_runs.to_csv(destination/"all_runs.csv", index=False)
    per_type.to_csv(destination/"all_cell_types.csv", index=False)
    by_fold.to_csv(destination/"seed_averaged_by_fold.csv", index=False)
    aggregate.reset_index().to_csv(destination/"fold_mean_std.csv", index=False)
    pd.concat(deltas, ignore_index=True).to_csv(destination/"paired_deltas_vs_E0.csv", index=False)
    counts = all_runs.groupby("arm", observed=True).trainable_parameters.unique().to_dict()
    if e0_method == "ae" and len({tuple(v) for v in counts.values()}) != 1:
        raise ValueError("Unexpected AE parameter-count mismatch.")
    return all_runs, aggregate.reset_index(), per_type


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--e0-method", choices=["ae", "pca"], default="ae")
    args = parser.parse_args()
    _, table, _ = compare_runs(args.output_root, args.e0_method)
    print(table.to_string(index=False))
