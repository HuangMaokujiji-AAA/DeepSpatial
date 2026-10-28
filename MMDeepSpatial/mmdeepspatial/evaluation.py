"""Identical frozen probes and geometry-aware cross-slice evaluation in all arms."""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_probe(x, labels, cfg, seed):
    if len(np.unique(labels)) < 2:
        raise ValueError("Linear probe requires at least two training cell types.")
    probe = make_pipeline(StandardScaler(), LogisticRegression(
        C=cfg["probe_c"], class_weight=cfg["probe_class_weight"],
        max_iter=cfg["probe_max_iter"], random_state=seed, solver="lbfgs"))
    # Never silently treat an unconverged probe as a completed experiment.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        probe.fit(x, labels)
    return probe


def classification_table(probe, test_x, train_labels, test_labels, rare_fraction):
    probabilities = probe.predict_proba(test_x)
    predicted = probe.classes_[probabilities.argmax(1)]
    train_counts = pd.Series(train_labels).value_counts()
    classes = sorted(set(train_labels) | set(test_labels))
    rows = []
    for label in classes:
        truth, guess = test_labels == label, predicted == label
        n_train, n_test = int(train_counts.get(label, 0)), int(truth.sum())
        fraction = n_train/len(train_labels)
        tp = int((truth & guess).sum())
        precision = tp/int(guess.sum()) if guess.any() else 0.
        recall = tp/n_test if n_test else np.nan
        f1 = 2*tp/(n_test+int(guess.sum())) if n_test+int(guess.sum()) else np.nan
        index = np.flatnonzero(probe.classes_ == label)
        ap = float(average_precision_score(truth, probabilities[:, index[0]])) if n_test and len(index) else np.nan
        rows.append(dict(cell_type=label, train_n=n_train, train_fraction=fraction,
                         test_n=n_test, test_prevalence=n_test/len(test_labels), predicted_n=int(guess.sum()),
                         is_rare=bool(n_train and fraction <= rare_fraction),
                         rarity="unseen" if not n_train else "ultra_rare" if fraction < .01
                         else "rare" if fraction <= rare_fraction else "common",
                         train_supported=n_train > 0, test_present=n_test > 0,
                         auprc=ap, precision=precision, recall=recall, f1=f1))
    return pd.DataFrame(rows), predicted, probabilities


def evaluate(probe, test_probe_x, train_z, test_z, train_meta, test_meta, geometry, cfg, seed, device):
    train_y, test_y = train_meta.cell_type.to_numpy(), test_meta.cell_type.to_numpy()
    table, predicted, probabilities = classification_table(
        probe, test_probe_x, train_y, test_y, cfg["rare_fraction"])
    ks = cfg["retrieval_ks"]
    k = max(ks)
    if len(train_z) < k:
        raise ValueError("Training reference has fewer cells than max retrieval k.")
    distances, indices = geometry.neighbors(test_z, train_z, k, device,
                                           cfg["query_chunk"], cfg["reference_chunk"])
    matches = train_y[indices] == test_y[:, None]
    cell_results = test_meta.copy()
    cell_results["probe_prediction"] = predicted
    cell_results["nearest_distance"] = distances[:, 0]
    for value in ks:
        cell_results[f"retrieval_hit_at_{value}"] = matches[:, :value].any(1)
        cell_results[f"knn_purity_at_{value}"] = matches[:, :value].mean(1)
    grouped = cell_results.groupby("cell_type", observed=True)
    for value in ks:
        for name in (f"retrieval_hit_at_{value}", f"knn_purity_at_{value}"):
            table[name] = table.cell_type.map(grouped[name].mean())

    # Type-conditional support radius: train-only 95th percentile of nearest same-type
    # distances in OTHER visible slices. A type seen in only one slice is uncalibratable.
    rng = np.random.default_rng(seed)
    calibration = []
    cell_results["covered_same_type"] = np.nan
    slices = train_meta.slice_index.to_numpy()
    for label in table.cell_type:
        ref = np.flatnonzero(train_y == label)
        query = np.flatnonzero(test_y == label)
        eligible = ref if len(np.unique(slices[ref])) >= 2 else np.array([], dtype=int)
        if len(eligible) > cfg["coverage_calibration_max"]:
            eligible = np.sort(rng.choice(eligible, cfg["coverage_calibration_max"], replace=False))
        threshold = np.nan
        if len(eligible):
            ds, _ = geometry.neighbors(train_z[eligible], train_z[ref], 1, device,
                                       cfg["query_chunk"], cfg["reference_chunk"],
                                       exclude_same_slice=(slices[eligible], slices[ref]))
            threshold = float(np.quantile(ds[:, 0], cfg["coverage_quantile"]))
        coverage = np.nan
        if len(query) and np.isfinite(threshold):
            ds, _ = geometry.neighbors(test_z[query], train_z[ref], 1, device,
                                       cfg["query_chunk"], cfg["reference_chunk"])
            covered = ds[:, 0] <= threshold
            coverage = float(covered.mean())
            cell_results.loc[query, "covered_same_type"] = covered.astype(float)
        calibration.append(dict(cell_type=label, coverage_threshold=threshold,
                                calibration_n=len(eligible), coverage=coverage,
                                coverage_status="ok" if len(eligible) and len(query)
                                else "absent_test" if not len(query)
                                else "no_train_support" if not len(ref) else "single_train_slice"))
    table = table.merge(pd.DataFrame(calibration), on="cell_type", validate="one_to_one")
    summary = dict(overall_accuracy=float((predicted == test_y).mean()),
                   test_n=len(test_y), unknown_test_n=int((~np.isin(test_y, probe.classes_)).sum()),
                   absent_test_types=int((~table.test_present).sum()),
                   rare_types_defined=int(table.is_rare.sum()))
    for name, mask in (("all", table.train_supported & table.test_present),
                       ("rare", table.is_rare & table.test_present),
                       ("common", table.train_supported & ~table.is_rare & table.test_present)):
        part = table.loc[mask]
        summary[f"{name}_evaluated_types"] = len(part)
        for metric in ("auprc", "f1", "recall", "coverage",
                       *[f"retrieval_hit_at_{v}" for v in ks], *[f"knn_purity_at_{v}" for v in ks]):
            summary[f"{name}_macro_{metric}"] = float(part[metric].mean())
        summary[f"{name}_coverage_evaluable_types"] = int(part.coverage.notna().sum())
    return table, pd.DataFrame([summary]), cell_results, probabilities
