"""A0-A5: raw/PCA/AE/GCN controls with paired encoder-loss balancing."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
import torch

from .data import Preprocessor, cell_metadata, input_paths, read_slice
from .evaluation import evaluate, fit_probe
from .experiment import DEFAULT_CONFIG, ROOT, environment, file_sha256, load_encoder, validate_config, write_json
from .graph import aggregate_features, training_neighbors
from .representation import Geometry, encode, fit_ae, set_seed


ARMS = {"A0": ("raw", False), "A1": ("pca", False), "A2": ("ae", False),
        "A3": ("ae", True), "A4": ("gcn", False), "A5": ("gcn", True)}
EUCLIDEAN_CONFIG = dict(deepcopy(DEFAULT_CONFIG), output_root=str(ROOT/"outputs/euclidean_v1"),
                        diagnostics=False, graph_k=10, graph_query_chunk=512,
                        balance_power=.5, reuse_a2_root=None)


def reconstruction_weights(labels, power=.5):
    """Inverse-sqrt training frequency, normalized to per-cell mean 1.

    Uniform minibatch sampling and number of updates stay unchanged. This is
    training-label-assisted reconstruction, not a change to the linear probe.
    """
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = counts[inverse].astype("float64")**(-power)
    return (weights/weights.mean()).astype("float32")


def add_rarity_strata(table, summary):
    metrics = ["auprc", "f1", "recall", "coverage"]
    metrics += [c for c in table if c.startswith(("retrieval_hit_at_", "knn_purity_at_"))]
    for name, rarity in (("ultra_rare", "ultra_rare"), ("low_rare", "rare")):
        part = table[(table.rarity == rarity) & table.train_supported & table.test_present]
        summary[f"{name}_evaluated_types"] = len(part)
        for metric in metrics:
            summary[f"{name}_macro_{metric}"] = float(part[metric].mean())


def reusable_a2(root, holdout, seed, cfg, inputs, prep, train_indices):
    """Verify the scientific configuration and retain original model provenance."""
    folder = Path(root)/f"holdout_{holdout}"/f"seed_{seed}"
    if not (folder/"completed.json").is_file():
        raise ValueError(f"A2 reuse source is incomplete: {folder}")
    old = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if (old["arm"], old["method"], old["holdout"], old["seed"]) != ("E0", "ae", holdout, seed):
        raise ValueError(f"A2 reuse source is not the matching E0 AE: {folder}")
    fields = ["label_key", "layer", "transform", "n_genes", "dim", "hidden", "epochs",
              "batch_size", "lr", "weight_decay", "grad_clip"]
    if any(old["config"][k] != cfg[k] for k in fields):
        raise ValueError(f"A2 reuse training configuration differs: {folder}")
    identities = lambda records: [(v["index"], v["sha256"]) for v in records]
    if identities(old["inputs"]) != identities(inputs) or old["train_indices"] != train_indices or old["test_index"] != holdout:
        raise ValueError(f"A2 reuse data/split differs: {folder}")
    original_prep = Preprocessor.load(folder/"preprocessing.npz")
    if any(not np.array_equal(getattr(prep, k), getattr(original_prep, k)) for k in ("genes", "mean", "scale")):
        raise ValueError(f"A2 reuse preprocessing differs: {folder}")
    return dict(folder=str(folder.resolve()), encoder_sha256=file_sha256(folder/"encoder.pt"),
                training_log_sha256=file_sha256(folder/"training.csv"),
                source_sha256=old["source_sha256"])


def run_arm(arm, config=None):
    if arm not in ARMS:
        raise ValueError("arm must be A0-A5.")
    cfg = deepcopy(EUCLIDEAN_CONFIG)
    if config:
        if set(config)-set(cfg):
            raise ValueError(f"Unknown config keys: {sorted(set(config)-set(cfg))}")
        cfg.update(deepcopy(config))
    paths = input_paths(cfg["input_glob"])
    validate_config(cfg, paths, "E0")
    for key in ("graph_k", "graph_query_chunk"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if not np.isfinite(cfg["balance_power"]) or not 0 < cfg["balance_power"] <= 1:
        raise ValueError("balance_power must be in (0, 1].")
    if cfg["diagnostics"]:
        raise ValueError("E0/E1/E2 diagnostics are separate; set diagnostics=False for A0-A5.")
    cfg["input_glob"] = str(Path(cfg["input_glob"]).expanduser().resolve())
    cfg["output_root"] = str(Path(cfg["output_root"]).expanduser().resolve())
    if cfg["reuse_a2_root"] is not None:
        cfg["reuse_a2_root"] = str(Path(cfg["reuse_a2_root"]).expanduser().resolve())
    device = torch.device(cfg["device"])
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("Use an available cpu/cuda device.")
    torch.set_num_threads(cfg["cpu_threads"])
    set_seed(cfg["seeds"][0])
    info = environment()
    info.update(device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    # Byte hashes provide provenance only; the held-out AnnData is not parsed here.
    inputs = [dict(index=i, path=str(p), sha256=file_sha256(p)) for i, p in enumerate(paths)]
    method, balanced = ARMS[arm]
    destination = Path(cfg["output_root"])/arm
    destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    for holdout in cfg["holdouts"]:
        train_indices = [i for i in range(len(paths)) if i != holdout]
        training = [read_slice(paths[i], cfg["label_key"]) for i in train_indices]
        prep = Preprocessor.fit(training, cfg["layer"], cfg["transform"], cfg["n_genes"])
        x = np.concatenate([prep.apply(a) for a in training])
        meta = cell_metadata(training, train_indices, cfg["label_key"])
        if method != "raw" and cfg["dim"] > min(x.shape):
            raise ValueError("dim exceeds available training genes/cells.")
        weights = reconstruction_weights(meta.cell_type.to_numpy(), cfg["balance_power"]) if balanced else np.ones(len(x), dtype="float32")
        graph_start = time.perf_counter()
        neighbors = training_neighbors(x, meta.slice_index.to_numpy(), cfg["graph_k"],
                                       cfg["graph_query_chunk"]) if method == "gcn" else None
        graph = dict(indices=neighbors, inputs=aggregate_features(x, neighbors)) if neighbors is not None else None
        graph_seconds = time.perf_counter()-graph_start if graph is not None else 0.
        for seed in cfg["seeds"]:
            folder = destination/f"holdout_{holdout}"/f"seed_{seed}"
            reuse = reusable_a2(cfg["reuse_a2_root"], holdout, seed, cfg, inputs, prep, train_indices) if arm == "A2" and cfg["reuse_a2_root"] else None
            manifest = dict(arm=arm, method=method, config=cfg, holdout=holdout, seed=seed,
                            train_indices=train_indices, test_index=holdout, inputs=inputs,
                            source_sha256=info["source_sha256"], reused_a2=reuse,
                            encoder_uses_labels=balanced)
            if folder.exists():
                if (folder/"completed.json").is_file() and json.loads((folder/"manifest.json").read_text(encoding="utf-8")) == manifest:
                    print(f"Skip completed {arm} holdout={holdout} seed={seed}", flush=True)
                    summaries.append(pd.read_csv(folder/"summary.csv"))
                    continue
                raise FileExistsError(f"{folder}: incomplete/different run; use a new output_root.")
            folder.mkdir(parents=True)
            write_json(folder/"manifest.json", manifest)
            write_json(folder/"environment.json", info)
            prep.save(folder/"preprocessing.npz")
            meta.to_csv(folder/"train_cells.csv", index=False)
            meta.assign(reconstruction_weight=weights).to_csv(folder/"training_weights.csv", index=False)
            meta.groupby(["slice_index", "cell_type"], observed=True).size().unstack(fill_value=0).to_csv(folder/"train_type_counts_by_slice.csv")
            print(f"Start {arm} holdout={holdout} seed={seed} train={x.shape} method={method} balanced={balanced}", flush=True)
            try:
                set_seed(seed)
                start = time.perf_counter()
                if method in {"raw", "pca"}:
                    model = FunctionTransformer() if method == "raw" else PCA(n_components=cfg["dim"], svd_solver="full", random_state=seed)
                    model.fit(x)
                    train_z = model.transform(x).astype("float32")
                    timing = dict(train_seconds=time.perf_counter()-start, peak_train_gpu_mb=0., trainable_parameters=0)
                    joblib.dump(model, folder/f"{method}.joblib")
                    if method == "pca":
                        pd.DataFrame(dict(component=np.arange(1, cfg["dim"]+1), explained_variance_ratio=model.explained_variance_ratio_)).to_csv(folder/"pca_variance.csv", index=False)
                elif reuse:
                    source = Path(reuse["folder"])
                    model, _, _ = load_encoder(source, device=device)
                    for filename in ("encoder.pt", "training.csv"):
                        shutil.copyfile(source/filename, folder/filename)
                    original = pd.read_csv(source/"summary.csv").iloc[0]
                    timing = dict(train_seconds=0., peak_train_gpu_mb=0., source_train_seconds=float(original.train_seconds),
                                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
                    train_z = encode(model, x, device)
                else:
                    model, timing = fit_ae(x, cfg, "E0", seed, device, folder,
                                           sample_weights=weights if balanced else None, graph=graph)
                    if method == "gcn":
                        train_z = model.encode_training(x, neighbors, device, cfg["batch_size"])
                        model.bind_reference(x, neighbors)
                        np.savez_compressed(folder/"graph_reference.npz", x=x, neighbors=neighbors,
                                            slice_indices=meta.slice_index.to_numpy())
                    else:
                        train_z = encode(model, x, device)
                timing["graph_build_seconds_per_fold"] = graph_seconds
                metric = "cosine" if method == "raw" else "euclidean"
                geometry = Geometry("E0", 1., metric=metric).calibrate(train_z, meta.slice_index.to_numpy(), seed, device)
                write_json(folder/"geometry.json", dict(arm="E0", curvature=1., metric=metric, scales=geometry.scales))
                start = time.perf_counter()
                probe = fit_probe(train_z, meta.cell_type.to_numpy(), cfg, seed)
                timing["probe_fit_seconds"] = time.perf_counter()-start
                joblib.dump(probe, folder/"probe.joblib")
                # Every fitted object, including the reference kNN and graph states,
                # depends exclusively on visible slices before this read.
                test = read_slice(paths[holdout], cfg["label_key"])
                test_x = prep.apply(test)
                test_meta = cell_metadata([test], [holdout], cfg["label_key"])
                start = time.perf_counter()
                if method in {"raw", "pca"}:
                    test_z = model.transform(test_x).astype("float32")
                elif method == "gcn":
                    test_z = model.encode_queries(test_x, device, cfg["batch_size"])
                else:
                    test_z = encode(model, test_x, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timing["test_encode_seconds"] = time.perf_counter()-start
                start = time.perf_counter()
                table, summary, predictions, probabilities = evaluate(
                    probe, test_z, train_z, test_z, meta, test_meta, geometry, cfg, seed, device)
                timing["evaluation_seconds"] = time.perf_counter()-start
                if method == "raw":
                    timing["test_reconstruction_mse"] = np.nan  # no compression/reconstruction task
                else:
                    with torch.no_grad():
                        recon = model.inverse_transform(test_z) if method == "pca" else np.concatenate([
                            model.decoder(torch.as_tensor(test_z[i:i+cfg["batch_size"]], device=device)).cpu().numpy()
                            for i in range(0, len(test_z), cfg["batch_size"])])
                    timing["test_reconstruction_mse"] = float(np.mean((recon-test_x)**2))
                add_rarity_strata(table, summary)
                identity = dict(arm=arm, method=method, holdout=holdout, seed=seed,
                                encoder_uses_labels=balanced, encoder_reused=bool(reuse),
                                latent_dim=train_z.shape[1], train_n=len(x), n_genes=len(prep.genes))
                summary = summary.assign(**identity, **timing)
                table.assign(**identity).to_csv(folder/"per_type.csv", index=False)
                summary.to_csv(folder/"summary.csv", index=False)
                predictions.to_csv(folder/"test_predictions.csv", index=False)
                test_meta.to_csv(folder/"test_cells.csv", index=False)
                test_meta.groupby(["slice_index", "cell_type"], observed=True).size().rename("n").to_csv(folder/"test_type_counts.csv")
                np.savez_compressed(folder/"embeddings.npz", train=train_z, test=test_z)
                np.savez_compressed(folder/"probe_probabilities.npz", probabilities=probabilities, classes=probe.classes_.astype(str))
                write_json(folder/"completed.json", dict(status="completed", arm=arm, holdout=holdout, seed=seed))
                summaries.append(summary)
                pd.concat(summaries, ignore_index=True).to_csv(destination/"summary.csv", index=False)
                print(summary[["rare_macro_auprc", "ultra_rare_macro_auprc", "rare_macro_f1", "overall_accuracy"]].to_string(index=False), flush=True)
            except Exception as error:
                write_json(folder/"failed.json", dict(error_type=type(error).__name__, message=str(error)))
                raise
            finally:
                if "model" in locals():
                    del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    result = pd.concat(summaries, ignore_index=True)
    result.to_csv(destination/"summary.csv", index=False)
    return result


def compare_arms(output_root):
    """Validate six completed arms and save descriptive, within-split contrasts."""
    root = Path(output_root)
    summaries, types, protocols, sources = [], [], [], []
    for arm in ARMS:
        summary = pd.read_csv(root/arm/"summary.csv")
        if summary.duplicated(["holdout", "seed"]).any():
            raise ValueError(f"Duplicate run keys in {arm}.")
        for row in summary.itertuples():
            run = root/arm/f"holdout_{row.holdout}"/f"seed_{row.seed}"
            if not (run/"completed.json").is_file() or (run/"failed.json").exists():
                raise ValueError(f"Incomplete run: {run}")
            m = json.loads((run/"manifest.json").read_text(encoding="utf-8"))
            if (m["arm"], m["method"], m["holdout"], m["seed"]) != (arm, ARMS[arm][0], row.holdout, row.seed):
                raise ValueError(f"Run identity mismatch: {run}")
            cfg = m["config"]
            expected = {(h, s) for h in cfg["holdouts"] for s in cfg["seeds"]}
            if expected != set(summary[["holdout", "seed"]].itertuples(index=False, name=None)):
                raise ValueError(f"Incomplete configured grid: {arm}")
            pd.testing.assert_frame_equal(pd.read_csv(run/"summary.csv"),
                                          summary[(summary.holdout == row.holdout) & (summary.seed == row.seed)].reset_index(drop=True), check_dtype=False)
            protocols.append({k: v for k, v in cfg.items() if k not in {"output_root", "device", "cpu_threads", "log_every", "reuse_a2_root"}})
            sources.append(m["source_sha256"])
            baseline = root/"A0"/f"holdout_{row.holdout}"/f"seed_{row.seed}"
            bm = json.loads((baseline/"manifest.json").read_text(encoding="utf-8"))
            for key in ("inputs", "train_indices", "test_index"):
                if m[key] != bm[key]:
                    raise ValueError(f"Different {key}: {run}")
            with np.load(run/"preprocessing.npz") as x, np.load(baseline/"preprocessing.npz") as y:
                if any(not np.array_equal(x[k], y[k]) for k in x.files):
                    raise ValueError(f"Different preprocessing: {run}")
            types.append(pd.read_csv(run/"per_type.csv"))
        summaries.append(summary)
    if any(p != protocols[0] for p in protocols) or any(s != sources[0] for s in sources):
        raise ValueError("Scientific configurations or current implementations differ between arms.")
    runs, per_type = pd.concat(summaries, ignore_index=True), pd.concat(types, ignore_index=True)
    if runs[runs.arm.isin(["A2", "A3", "A4", "A5"])].trainable_parameters.nunique() != 1:
        raise ValueError("A2-A5 encoder/decoder parameter counts differ.")
    for holdout, seed in runs[["holdout", "seed"]].drop_duplicates().itertuples(index=False, name=None):
        with np.load(root/"A4"/f"holdout_{holdout}/seed_{seed}/graph_reference.npz") as x, np.load(root/"A5"/f"holdout_{holdout}/seed_{seed}/graph_reference.npz") as y:
            if any(not np.array_equal(x[k], y[k]) for k in x.files):
                raise ValueError("A4/A5 reference graph differs.")
    metrics = [c for c in runs if "_macro_" in c] + ["overall_accuracy", "test_reconstruction_mse", "train_seconds", "peak_train_gpu_mb"]
    by_fold = runs.groupby(["arm", "holdout"], observed=True)[metrics].mean()
    aggregate = by_fold.groupby("arm", observed=True).agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    contrasts = [("A1", "A0"), ("A2", "A0"), ("A3", "A2"), ("A4", "A2"), ("A5", "A4"), ("A5", "A3")]
    deltas = []
    for arm, reference in contrasts:
        current = runs[runs.arm == arm].set_index(["holdout", "seed"])
        baseline = runs[runs.arm == reference].set_index(["holdout", "seed"])
        delta = current[metrics]-baseline[metrics]
        delta.columns = [f"delta_{c}" for c in delta]
        deltas.append(delta.assign(arm=arm, reference=reference).reset_index())
    destination = root/"comparison"
    destination.mkdir(exist_ok=True)
    runs.to_csv(destination/"all_runs.csv", index=False)
    per_type.to_csv(destination/"all_cell_types.csv", index=False)
    by_fold.reset_index().to_csv(destination/"seed_averaged_by_fold.csv", index=False)
    aggregate.reset_index().to_csv(destination/"fold_mean_std.csv", index=False)
    pd.concat(deltas, ignore_index=True).to_csv(destination/"paired_deltas.csv", index=False)
    return runs, aggregate.reset_index(), per_type


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--config", type=Path, help="Optional JSON configuration overrides.")
    parser.add_argument("--compare", type=Path, metavar="OUTPUT_ROOT", help="Validate and summarize an existing six-arm result set.")
    args = parser.parse_args()
    if args.compare:
        print(compare_arms(args.compare)[1].to_string(index=False))
    else:
        cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
        for arm in args.arms:
            run_arm(arm, cfg)
