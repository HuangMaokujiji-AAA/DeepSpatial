"""训练配对混合抽样与保持来源边际的低方差推理抽样。"""

import numpy as np


def systematic_counts(probabilities, total):
    """用随机起点系统抽样分配整数预算；零概率类别不会被强行补齐。"""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if total < 0 or probabilities.ndim != 1 or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Sampling probabilities must be finite and nonnegative; total must be nonnegative.")
    if total == 0:
        return np.zeros(len(probabilities), dtype=np.int64)
    if probabilities.sum() <= 0:
        raise ValueError("Positive sampling mass is required.")
    cumulative = np.cumsum(probabilities / probabilities.sum())
    cumulative[-1] = 1.0
    positions = (np.random.random() + np.arange(total)) / total
    draws = np.searchsorted(cumulative, positions, side="right")
    return np.bincount(draws, minlength=len(probabilities))


def stratified_source_indices(labels, total):
    """先按实测类型占比分配，再均衡同类 parent；不增加稀有类期望比例。"""
    labels = np.asarray(labels)
    if not len(labels):
        raise ValueError("Cannot sample from an empty source slice.")
    classes, class_sizes = np.unique(labels, return_counts=True)
    class_counts = systematic_counts(class_sizes, total)
    draws = []
    for label, count in zip(classes, class_counts):
        parents = np.flatnonzero(labels == label)
        # 打乱循环次序，使不同源细胞获得相同的期望抽样次数。
        parents = np.random.permutation(parents)
        full, remainder = divmod(int(count), len(parents))
        draws.append(np.concatenate([np.tile(parents, full), parents[:remainder]]))
    result = np.concatenate(draws).astype(np.int64)
    np.random.shuffle(result)  # 避免来源类型与目标深度的排序发生关联。
    return result


def sample_coupling_indices(coupling, labels0, labels1, total, balanced_fraction=0.0):
    """混合原 UOT 分布与有效同类子块，保持子块内部运输权重。"""
    if not 0 <= balanced_fraction <= 1:
        raise ValueError("balanced_fraction must lie in [0, 1].")
    if not np.isfinite(coupling).all() or (coupling < 0).any() or coupling.sum() <= 0:
        raise ValueError("UOT coupling must contain finite, nonnegative, positive mass.")
    labels0, labels1 = np.asarray(labels0), np.asarray(labels1)
    blocks = []
    if balanced_fraction:
        for label in np.intersect1d(labels0, labels1):
            rows, cols = np.flatnonzero(labels0 == label), np.flatnonzero(labels1 == label)
            block = coupling[np.ix_(rows, cols)]
            if block.sum() > 0:
                blocks.append((int(label), rows, cols))
    balanced_n = int(round(total * balanced_fraction)) if blocks else 0
    flat = coupling.ravel()
    draws = np.random.choice(len(flat), size=total - balanced_n, p=flat / flat.sum(), replace=True)
    idx0, idx1 = np.unravel_index(draws, coupling.shape)
    mixed0, mixed1 = [idx0], [idx1]
    for index, count in enumerate(np.bincount(
        np.random.randint(len(blocks), size=balanced_n) if balanced_n else np.empty(0, dtype=int),
        minlength=len(blocks),
    )):
        _, rows, cols = blocks[index]
        block = coupling[np.ix_(rows, cols)]
        draws = np.random.choice(block.size, size=count, p=(block / block.sum()).ravel(), replace=True)
        local0, local1 = np.unravel_index(draws, block.shape)
        mixed0.append(rows[local0])
        mixed1.append(cols[local1])
    idx0, idx1 = np.concatenate(mixed0), np.concatenate(mixed1)
    if balanced_n:
        order = np.random.permutation(total)
        idx0, idx1 = idx0[order], idx1[order]
    return idx0, idx1, {"balanced_pairs": balanced_n, "eligible_classes": [b[0] for b in blocks]}
