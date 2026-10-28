"""Run E0/E1/E2 in separate processes/notebook kernels, one GPU per arm."""
from copy import deepcopy
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

# Required by deterministic CUDA matrix multiplication, including the CLI path.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch

from .data import Preprocessor, cell_metadata, input_paths, read_slice
from .evaluation import evaluate, fit_probe
from .representation import ExpressionAE, Geometry, encode, fit_ae, probe_coordinates, set_seed

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = dict(
    input_glob=str(ROOT.parent / "data/merfish_mouse_hypothalamus/**/merfish_[0-4].h5ad"),
    output_root=str(ROOT / "outputs/representations_v1"),
    holdouts=[1, 2, 3], seeds=[0, 1, 2, 3, 4], label_key="cell_class",
    layer=None, transform="log1p", n_genes=0, rare_fraction=.05,
    dim=32, hidden=128, epochs=100, batch_size=512, lr=1e-3, weight_decay=1e-5,
    grad_clip=5., curvature=1., fusion_alpha=.5, log_every=10,
    e0_method="ae", device="cuda", cpu_threads=4,
    retrieval_ks=[1, 5, 15], query_chunk=128, reference_chunk=2048,
    coverage_quantile=.95, coverage_calibration_max=2048,
    probe_c=1., probe_class_weight="balanced", probe_max_iter=2000,
    shuffled_probe=False,
)


def file_sha256(path):
    digest = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    # JSON null means unavailable, rather than a non-standard NaN literal.
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
            return None
        if isinstance(obj, np.generic):
            return obj.item()
        return obj
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def validate_config(cfg, paths, arm):
    if arm not in {"E0", "E1", "E2"}:
        raise ValueError("arm must be E0/E1/E2.")
    for key in ("holdouts", "seeds", "retrieval_ks"):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]) or any(type(v) is not int for v in cfg[key]):
            raise ValueError(f"{key} must contain unique integers.")
    if not set(cfg["holdouts"]) <= set(range(1, len(paths)-1)):
        raise ValueError("Holdouts must be interior slice indices (natural file order, zero based).")
    if min(cfg["seeds"]) < 0 or min(cfg["retrieval_ks"]) < 1:
        raise ValueError("Seeds must be nonnegative and k positive.")
    for key in ("dim", "hidden", "epochs", "batch_size", "log_every", "query_chunk", "reference_chunk",
                "coverage_calibration_max", "probe_max_iter", "cpu_threads"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if cfg["dim"] < 2 or cfg["dim"] % 2:
        raise ValueError("Use even dim >=2 in ALL arms to match E2's split.")
    for key in ("lr", "grad_clip", "curvature", "probe_c"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if not np.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    if not (0 < cfg["rare_fraction"] < 1 and 0 < cfg["coverage_quantile"] < 1
            and 0 <= cfg["fusion_alpha"] <= 1):
        raise ValueError("Invalid rarity, coverage quantile or fusion alpha.")
    if type(cfg["n_genes"]) is not int or cfg["n_genes"] < 0:
        raise ValueError("n_genes must be nonnegative; zero uses all variable training genes.")
    if cfg["transform"] not in {"identity", "log1p"} or cfg["e0_method"] not in {"ae", "pca"}:
        raise ValueError("Use identity/log1p transform and ae/pca E0 method.")
    if cfg["probe_class_weight"] not in {None, "balanced"}:
        raise ValueError("probe_class_weight must be null or balanced.")


def environment():
    versions = {}
    for name in ("torch", "geoopt", "numpy", "scipy", "pandas", "scikit-learn", "anndata", "h5py"):
        versions[name] = importlib.metadata.version(name)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                capture_output=True, check=False).stdout.strip() or None
    except FileNotFoundError:
        commit = None
    return dict(python=platform.python_version(), versions=versions, git_commit=commit,
                cuda_runtime=torch.version.cuda, cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                source_sha256={p.name: file_sha256(p) for p in sorted((ROOT/"mmdeepspatial").glob("*.py"))})


def run_experiment(arm, config=None):
    cfg = deepcopy(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(deepcopy(config))
    paths = input_paths(cfg["input_glob"])
    validate_config(cfg, paths, arm)
    cfg["input_glob"] = str(Path(cfg["input_glob"]).expanduser().resolve())
    cfg["output_root"] = str(Path(cfg["output_root"]).expanduser().resolve())
    device = torch.device(cfg["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu or cuda; float64 distance evaluation requires these devices.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check the kernel/GPU assignment, or set device='cpu' for smoke tests.")
    torch.set_num_threads(cfg["cpu_threads"])
    set_seed(cfg["seeds"][0])
    info = environment()
    info["device"] = str(device)
    info["gpu_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    # Hash inputs without parsing held-out AnnData. These bytes never affect model fitting.
    inputs = [dict(index=i, path=str(p), sha256=file_sha256(p)) for i, p in enumerate(paths)]
    method = "pca" if arm == "E0" and cfg["e0_method"] == "pca" else "ae"
    destination = Path(cfg["output_root"]) / f"{arm}_{method}"
    destination.mkdir(parents=True, exist_ok=True)
    print(f"{arm} | {device} | {len(paths)} slices | output={destination}", flush=True)
    summaries = []
    for holdout in cfg["holdouts"]:
        train_indices = [i for i in range(len(paths)) if i != holdout]
        # No access to held-out expression/labels here.
        training = [read_slice(paths[i], cfg["label_key"]) for i in train_indices]
        prep = Preprocessor.fit(training, cfg["layer"], cfg["transform"], cfg["n_genes"])
        x = np.concatenate([prep.apply(a) for a in training])
        meta = cell_metadata(training, train_indices, cfg["label_key"])
        if cfg["dim"] > min(x.shape):
            raise ValueError("dim exceeds training gene/cell count; reduce dim identically in all arms.")
        if meta.cell_type.nunique() < 2:
            raise ValueError("Need at least two annotated cell types in visible training slices.")
        counts = meta.groupby(["slice_index", "cell_type"], observed=True).size().unstack(fill_value=0)
        for seed in cfg["seeds"]:
            folder = destination / f"holdout_{holdout}" / f"seed_{seed}"
            manifest = dict(arm=arm, method=method, config=cfg, holdout=holdout, seed=seed,
                            train_indices=train_indices, test_index=holdout, inputs=inputs,
                            source_sha256=info["source_sha256"])
            if folder.exists():
                if ((folder/"completed.json").is_file() and (folder/"manifest.json").is_file()
                        and json.loads((folder/"manifest.json").read_text()) == manifest):
                    print(f"Skip completed {arm} holdout={holdout} seed={seed}", flush=True)
                    summaries.append(pd.read_csv(folder/"summary.csv"))
                    continue
                raise FileExistsError(f"{folder}: incomplete or different run. Use a new output_root; no files overwritten.")
            folder.mkdir(parents=True)
            write_json(folder/"manifest.json", manifest)
            write_json(folder/"environment.json", info)
            prep.save(folder/"preprocessing.npz")
            meta.to_csv(folder/"train_cells.csv", index=False)
            counts.to_csv(folder/"train_type_counts_by_slice.csv")
            print(f"Start {arm} holdout={holdout} seed={seed} train={x.shape}", flush=True)
            try:
                set_seed(seed)
                if method == "pca":
                    start = time.perf_counter()
                    model = PCA(n_components=cfg["dim"], svd_solver="full", random_state=seed)
                    model.fit(x)
                    train_z = model.transform(x).astype("float32")
                    timing = dict(train_seconds=time.perf_counter()-start, peak_train_gpu_mb=0.,
                                  trainable_parameters=0)
                    joblib.dump(model, folder/"pca.joblib")
                    pd.DataFrame([dict(component=i+1, explained_variance_ratio=v) for i, v
                                  in enumerate(model.explained_variance_ratio_)]).to_csv(folder/"pca_variance.csv", index=False)
                else:
                    model, timing = fit_ae(x, cfg, arm, seed, device, folder)
                    train_z = encode(model, x, device)
                geometry = Geometry(arm, cfg["curvature"], cfg["fusion_alpha"]).calibrate(
                    train_z, meta.slice_index.to_numpy(), seed, device)
                write_json(folder/"geometry.json", dict(arm=arm, curvature=cfg["curvature"],
                                                       alpha=cfg["fusion_alpha"], scales=geometry.scales))
                train_probe_x = probe_coordinates(train_z, arm, cfg["curvature"])
                probe = fit_probe(train_probe_x, meta.cell_type.to_numpy(), cfg, seed)
                joblib.dump(probe, folder/"probe.joblib")
                # Encoder, normalization, geometry scales and probe are now all frozen.
                test = read_slice(paths[holdout], cfg["label_key"])
                test_x = prep.apply(test)
                test_meta = cell_metadata([test], [holdout], cfg["label_key"])
                start = time.perf_counter()
                test_z = model.transform(test_x).astype("float32") if method == "pca" else encode(model, test_x, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timing["test_encode_seconds"] = time.perf_counter()-start
                test_probe_x = probe_coordinates(test_z, arm, cfg["curvature"])
                np.savez_compressed(folder/"embeddings.npz", train=train_z, test=test_z)
                test_meta.to_csv(folder/"test_cells.csv", index=False)
                start = time.perf_counter()
                table, summary, cell_results, probabilities = evaluate(
                    probe, test_probe_x, train_z, test_z, meta, test_meta, geometry, cfg, seed, device)
                timing["evaluation_seconds"] = time.perf_counter()-start
                recon = model.inverse_transform(test_z) if method == "pca" else None
                if recon is None:
                    with torch.no_grad():
                        recon = np.concatenate([model(torch.as_tensor(test_x[i:i+cfg["batch_size"]], device=device))[0]
                                                .cpu().numpy() for i in range(0, len(test_x), cfg["batch_size"])])
                timing["test_reconstruction_mse"] = float(np.mean((recon-test_x)**2))
                if arm != "E0":
                    split = cfg["dim"]//2 if arm == "E2" else 0
                    for name, z in (("train", train_z), ("test", test_z)):
                        norm = np.linalg.norm(z[:, split:], axis=1)*cfg["curvature"]**.5
                        timing[f"{name}_max_ball_radius_fraction"] = float(norm.max())
                        timing[f"{name}_near_boundary_fraction"] = float(np.mean(norm >= .979))
                        if not np.isfinite(norm).all() or norm.max() >= 1:
                            raise FloatingPointError("Invalid Poincare embedding.")
                for key, value in dict(arm=arm, method=method, holdout=holdout, seed=seed, **timing).items():
                    summary[key] = value
                table.assign(arm=arm, method=method, holdout=holdout, seed=seed).to_csv(folder/"per_type.csv", index=False)
                summary.to_csv(folder/"summary.csv", index=False)
                cell_results.to_csv(folder/"test_predictions.csv", index=False)
                np.savez_compressed(folder/"probe_probabilities.npz", probabilities=probabilities,
                                    classes=probe.classes_.astype(str))
                test_meta.groupby(["slice_index", "cell_type"], observed=True).size().rename("n").to_csv(
                    folder/"test_type_counts.csv")
                if cfg["shuffled_probe"]:
                    labels = np.random.default_rng(seed).permutation(meta.cell_type.to_numpy())
                    control = fit_probe(train_probe_x, labels, cfg, seed)
                    from .evaluation import classification_table
                    shuffled, _, _ = classification_table(control, test_probe_x, labels,
                                                            test_meta.cell_type.to_numpy(), cfg["rare_fraction"])
                    shuffled.to_csv(folder/"shuffled_probe_per_type.csv", index=False)
                write_json(folder/"completed.json", dict(status="completed", arm=arm, holdout=holdout, seed=seed))
                summaries.append(summary)
                pd.concat(summaries, ignore_index=True).to_csv(destination/"summary.csv", index=False)
                print(summary[["rare_macro_auprc", "rare_macro_f1", "overall_accuracy"]].to_string(index=False), flush=True)
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


def load_encoder(folder, device="cpu"):
    """Load our own trusted run artifacts for later frozen UOT integration."""
    folder = Path(folder)
    manifest = json.loads((folder/"manifest.json").read_text())
    prep = Preprocessor.load(folder/"preprocessing.npz")
    if manifest["method"] == "pca":
        model = joblib.load(folder/"pca.joblib")
    else:
        checkpoint = torch.load(folder/"encoder.pt", map_location="cpu", weights_only=True)
        model = ExpressionAE(**{k: checkpoint[k] for k in ("n_genes", "dim", "hidden", "arm", "curvature")})
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device).eval()
    geometry = Geometry(**json.loads((folder/"geometry.json").read_text()))
    return model, prep, geometry


def encode_adata(adata, model, prep, device="cpu"):
    x = prep.apply(adata)
    return model.transform(x).astype("float32") if isinstance(model, PCA) else encode(model, x, torch.device(device))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["E0", "E1", "E2"])
    parser.add_argument("--config", type=Path, help="Optional JSON configuration overrides.")
    args = parser.parse_args()
    run_experiment(args.arm, json.loads(args.config.read_text()) if args.config else None)
