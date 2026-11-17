"""Original DeepSpatial on Open-ST GSE251926. See docs/openst_reproduction.md."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import inspect
import json
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SECTIONS = (2, 3, 4, 5, 6, 7, 9, 11, 17, 18, 19, 23, 24, 25, 26, 28, 33, 34, 36)
TEST = (4, 18, 24, 9)
VALIDATION = (6, 34)
TRAIN = tuple(s for s in SECTIONS if s not in TEST + VALIDATION)
DATA_KWARGS = dict(
    spatial_key="spatial", z_key="z_coord", label_key="annotation",
    batch_size=128, num_workers=4, n_samples_base=50000,
    alpha_spatial=0.5, uot_reg=0.8, uot_tau=0.05, mode="fit",
)
MODEL_KWARGS = dict(
    patch_size=8, hidden_size=256, depth=6, num_heads=8, mlp_ratio=4.0,
    path_type="Linear", lr=2e-4, weight_decay=1e-5,
    lambda_g=0.1, lambda_c=10.0, sampling_method="dopri5", atol=1e-5, rtol=1e-5,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def inspect_input(path):
    """Read metadata only; do not load the 8 GB expression matrix or raw layer."""
    with h5py.File(path, "r") as f:
        ids = f["obs/n_section"][:]
        xyz = f["obsm/spatial_3d_aligned"][:]
        genes = f["var/_index"].asstr()[:]
        if xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all():
            raise ValueError("Expected finite registered XYZ coordinates.")
        if set(np.unique(ids)) != set(SECTIONS):
            raise ValueError(f"Unexpected physical section IDs: {np.unique(ids)}")
        if len(set(genes)) != len(genes):
            raise ValueError("Gene names must be unique; no implicit gene renaming is performed.")
        if np.any(f["obs/annotation/codes"][:] < 0):
            raise ValueError("Missing annotation labels.")
        records = []
        for section in SECTIONS:
            z = np.unique(xyz[ids == section, 2])
            if len(z) != 1:
                raise ValueError(f"Section {section} does not have a uniform physical Z.")
            records.append(dict(section=section, n_cells=int(np.sum(ids == section)),
                                z_um=float(z[0]), split="test" if section in TEST else
                                "validation" if section in VALIDATION else "train"))
        if np.any(np.diff([r["z_um"] for r in records]) <= 0):
            raise ValueError("Physical section order and registered Z disagree.")
        if f["X"].attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("This runner expects the supplied CSR expression matrix.")
    by_id = {r["section"]: r for r in records}
    largest = max((by_id[a]["n_cells"] * by_id[b]["n_cells"], a, b)
                  for a, b in zip(TRAIN, TRAIN[1:]))
    return dict(
        input=str(path), n_cells=len(ids), n_genes=len(genes), sections=records,
        input_size_bytes=Path(path).stat().st_size, input_mtime_ns=Path(path).stat().st_mtime_ns,
        gene_names_sha256=hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        train_sections=list(TRAIN), validation_sections=list(VALIDATION), test_sections=list(TEST),
        coordinate_source="obsm/spatial_3d_aligned; micrometers; no rescaling of physical Z",
        expression_source="X (already log-transformed in the supplied file); no additional preprocessing",
        resources=dict(largest_uot_pair=list(largest[1:]),
                       one_float64_uot_matrix_gib=largest[0] * 8 / 2**30,
                       sampled_gene_endpoints_gib=2 * 50000 * len(genes) * 4 / 2**30,
                       gene_tokens=int(np.ceil(len(genes) / 8)),
                       note="UOT holds multiple dense matrices; tensor assembly also makes copies. "
                            "Inference densifies both complete endpoint slices on the selected device. "
                            "These figures are component sizes, not peak memory estimates."),
    )


def reference_pair(section):
    return max(s for s in TRAIN if s < section), min(s for s in TRAIN if s > section)


@contextmanager
def open_input(path):
    # read_h5ad(backed='r') still loads non-X layers. Open only the needed groups.
    try:
        from anndata.io import read_elem, sparse_dataset
    except ImportError:  # anndata 0.10.x, used by the original project environment
        from anndata.experimental import read_elem, sparse_dataset
    with h5py.File(path, "r") as f:
        yield SimpleNamespace(obs=read_elem(f["obs"]), var=read_elem(f["var"]),
                              obsm={"spatial_3d_aligned": f["obsm/spatial_3d_aligned"][:]},
                              X=sparse_dataset(f["X"]))


def read_slice(source, section):
    import anndata as ad
    import scipy.sparse as sp

    rows = np.flatnonzero(source.obs["n_section"].to_numpy() == section)
    xyz = np.asarray(source.obsm["spatial_3d_aligned"])[rows]
    # Load only X, annotation and registered coordinates; leave layers/raw on disk.
    x = source.X[rows].astype(np.float32)
    values = x.data if sp.issparse(x) else x
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Section {section}: invalid or negative input expression.")
    if np.any(np.asarray(x.sum(axis=1)).ravel() <= 0):
        raise ValueError(f"Section {section}: zero-expression cells cause undefined cosine UOT cost.")
    obs = source.obs.iloc[rows][["annotation", "n_section"]].copy()
    obs["z_coord"] = xyz[:, 2]
    return ad.AnnData(X=x, obs=obs, var=source.var.copy(),
                      obsm={"spatial": xyz[:, :2].copy(), "spatial_3d_aligned": xyz.copy()})


def normalize_with_saved_stats(model, slices):
    """Apply training-only statistics; never recompute them on held-out sections."""
    s = model.spatial_stats
    for a in slices:
        xy = a.obsm["spatial"].copy()
        xy[:, 0] = (xy[:, 0] - s["x_min"]) / s["x_range"]
        xy[:, 1] = (xy[:, 1] - s["y_min"]) / s["y_range"]
        a.obsm["spatial_norm"] = xy
        a.obs["z_norm"] = (a.obs["z_coord"] - s["z_min"]) / s["z_range"]


def load_model(checkpoint_dir):
    import pandas as pd
    import torch
    from deepspatial import DeepSpatial

    config = json.loads((checkpoint_dir / "config.json").read_text())
    model = DeepSpatial()
    for key in ("gene_dim", "num_classes", "spatial_stats", "spatial_key", "z_key", "label_key"):
        setattr(model, key, config[key])
    model.categories = pd.Index(config["categories"])
    # Original load_checkpoint forwards internal transport keys to build_model,
    # whose signature rejects them. Restore locally without changing library code.
    accepted = inspect.signature(model.build_model).parameters
    kwargs = {k: v for k, v in {**config["model_config"], **config["train_config"]}.items()
              if k in accepted}
    model.build_model(**kwargs)
    checkpoint = torch.load(checkpoint_dir / "last.ckpt", map_location="cpu", weights_only=False)
    model.module.load_state_dict(checkpoint["state_dict"])
    return model


def evaluate(truth, prediction):
    """Unpaired gene-mean errors and registered XY distances, on the supplied X scale."""
    from scipy.spatial import cKDTree

    if not truth.var_names.equals(prediction.var_names) or prediction.n_obs == 0:
        raise ValueError("Empty reconstruction or mismatched gene order.")
    a = np.asarray(truth.X.mean(axis=0)).ravel().astype(float)
    b = np.asarray(prediction.X.mean(axis=0)).ravel().astype(float)
    xy_t, xy_p = truth.obsm["spatial"], prediction.obsm["spatial"]
    recall_dist = cKDTree(xy_p).query(xy_t)[0]
    precision_dist = cKDTree(xy_t).query(xy_p)[0]
    return dict(truth_n=truth.n_obs, predicted_n=prediction.n_obs,
                gene_mean_mse=float(np.mean((a - b)**2)), gene_mean_mae=float(np.mean(np.abs(a - b))),
                gene_mean_pcc=float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None,
                symmetric_xy_nn_um=float((recall_dist.mean() + precision_dist.mean()) / 2),
                xy_coverage_recall_50um=float(np.mean(recall_dist <= 50)),
                xy_coverage_precision_50um=float(np.mean(precision_dist <= 50)))


def reconstruct(model, source, train_by_id, manifest, run_dir, args):
    import pandas as pd
    import pytorch_lightning as pl

    records = {r["section"]: r for r in manifest["sections"]}
    normalize_with_saved_stats(model, list(train_by_id.values()))
    summaries = []
    for split, sections in (("validation", VALIDATION), ("test", TEST)):
        for section in sections:
            pl.seed_everything(args.seed + section, workers=True)
            left, right = reference_pair(section)
            center = records[section]["z_um"]
            if not (records[left]["z_um"] < center - args.slab_half_width and
                    center + args.slab_half_width < records[right]["z_um"]):
                raise ValueError(f"Section {section}: slab must lie strictly inside training endpoints.")
            print(f"Reconstructing {split} #{section} from training #{left}, #{right}", flush=True)
            # Use original bidirectional volume generation, then select a fixed physical slab.
            gap = model.reconstruct_between_slices(
                train_by_id[left], train_by_id[right], thickness=args.thickness,
                steps=args.ode_steps, chunk_size=args.chunk_size, device=args.device,
            )
            mask = np.abs(gap.obs["z_coord"].to_numpy() - center) <= args.slab_half_width
            prediction = gap[mask].copy()
            del gap
            if prediction.n_obs == 0:
                raise RuntimeError(f"Section {section}: no generated cells in the target slab.")
            # Original assembly inherits source obs columns. Do not export those as predictions.
            prediction.obs = prediction.obs[["annotation", "z_coord"]].copy()
            prediction.obs["n_section"] = section
            prediction.obs["split"] = split
            prediction.obsm["spatial_3d_aligned"] = np.column_stack(
                [prediction.obsm["spatial"], prediction.obs["z_coord"].to_numpy()])
            prediction.uns["openst_protocol"] = dict(
                target_section=section, target_z_um=center, reference_sections=[left, right],
                slab_half_width_um=args.slab_half_width, thickness_um=args.thickness,
                ode_steps=args.ode_steps, checkpoint="checkpoints/last.ckpt", weights="EMA",
                expression_scale=manifest["expression_source"], train_sections=list(TRAIN),
            )
            output = run_dir / split
            output.mkdir(exist_ok=True)
            path = output / f"reconstructed_section_{section:02d}.h5ad"
            prediction.write_h5ad(path, compression="gzip")
            # Read held-out expression only after reconstruction; no truth count or label guides sampling.
            truth = read_slice(source, section)
            summary = dict(section=section, split=split, reference_left=left, reference_right=right,
                           file=str(path), **evaluate(truth, prediction))
            summaries.append(summary)
            labels = pd.concat([
                truth.obs.annotation.astype(str).value_counts().rename("truth_n"),
                prediction.obs.annotation.astype(str).value_counts().rename("predicted_n"),
            ], axis=1).fillna(0).astype(int)
            labels.index.name = "annotation"
            labels.to_csv(output / f"section_{section:02d}_class_counts.csv")
            pd.DataFrame(summaries).to_csv(run_dir / "reconstruction_metrics.csv", index=False)
            write_json(output / f"section_{section:02d}_metrics.json", summary)
            print(f"Saved {path} ({prediction.n_obs} cells)", flush=True)
            del truth, prediction
            gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/Open-ST/GSE251926_metastatic_lymph_node_3d.h5ad")
    parser.add_argument("--output-dir", type=Path, help="New run directory; defaults to a timestamped directory.")
    parser.add_argument("--inspect", action="store_true", help="Only inspect splits and memory component sizes (no training imports).")
    parser.add_argument("--reconstruct-only", type=Path, metavar="RUN_DIR", help="Reload final checkpoint and finish/repeat reconstruction.")
    parser.add_argument("--accelerator", choices=("auto", "gpu", "cpu", "mps"), default="auto")
    parser.add_argument("--device", default="auto", help="Reconstruction device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--thickness", type=float, default=10.0)
    parser.add_argument("--slab-half-width", type=float, default=5.0)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()
    if min(args.epochs, args.thickness, args.slab_half_width, args.chunk_size) <= 0 or args.ode_steps < 2 or args.num_workers < 0:
        parser.error("Require positive epochs/thickness/slab width/chunk size, ODE steps >= 2 and workers >= 0.")
    args.input = args.input.resolve()
    manifest = inspect_input(args.input)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if args.inspect:
        return
    import anndata as ad
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from deepspatial import DeepSpatial

    if args.reconstruct_only:
        run_dir = args.reconstruct_only.resolve()
        saved = json.loads((run_dir / "run_parameters.json").read_text())
        if manifest != saved["data"]:
            raise ValueError("Input metadata differs from the original run.")
        for key in ("seed", "thickness", "slab_half_width", "ode_steps"):
            setattr(args, key, saved["arguments"][key])
    else:
        run_dir = (args.output_dir or ROOT / "output/openst" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        write_json(run_dir / "run_parameters.json", dict(
            data=manifest, arguments=manifest_args, data_kwargs={**DATA_KWARGS, "num_workers": args.num_workers},
            model_kwargs=MODEL_KWARGS, inference_checkpoint="checkpoints/last.ckpt", inference_weights="EMA",
            validation_protocol="Reserved reconstruction assessment; original fixed-epoch training has no validation loop or early stopping.",
            versions=dict(torch=str(torch.__version__), lightning=pl.__version__, anndata=ad.__version__),
        ))
        import pandas as pd
        pd.DataFrame(manifest["sections"]).to_csv(run_dir / "split_manifest.csv", index=False)
    print(f"Run directory: {run_dir}", flush=True)
    with open_input(args.input) as source:
        pl.seed_everything(args.seed, workers=True)
        train_by_id = {section: read_slice(source, section) for section in TRAIN}
        if not args.reconstruct_only:
            model = DeepSpatial()
            model.setup_data(list(train_by_id.values()), **{**DATA_KWARGS, "num_workers": args.num_workers})
            model.build_model(**MODEL_KWARGS)
            checkpoint_dir = run_dir / "checkpoints"
            model._save_config(str(checkpoint_dir))
            # Gene identities are needed in addition to the original dimension-only metadata.
            train_by_id[TRAIN[0]].var.to_csv(checkpoint_dir / "genes.csv")
            callback = ModelCheckpoint(dirpath=str(checkpoint_dir), monitor="loss", mode="min",
                                       filename="deepspatial-{epoch:02d}-{loss:.4f}", save_top_k=1, save_last=True)
            trainer = pl.Trainer(max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
                                 callbacks=[callback], logger=CSVLogger(str(run_dir), name="logs", version=0),
                                 default_root_dir=str(run_dir))
            trainer.fit(model.module, model.train_loader)
            trainer.save_checkpoint(str(checkpoint_dir / "last.ckpt"))
            write_json(run_dir / "training_result.json", dict(
                status="completed", epochs=trainer.current_epoch, global_step=trainer.global_step,
                final_checkpoint="checkpoints/last.ckpt", best_training_loss_checkpoint=callback.best_model_path,
                best_training_loss=float(callback.best_model_score) if callback.best_model_score is not None else None))
            # Release large trajectory tensors and optimizer state before inference.
            del model, trainer, callback
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model = load_model(run_dir / "checkpoints")
        if args.device == "auto":
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        reconstruct(model, source, train_by_id, manifest, run_dir, args)
        write_json(run_dir / "completed.json", dict(status="completed", test_sections=list(TEST),
                                                    validation_sections=list(VALIDATION)))


if __name__ == "__main__":
    main()
