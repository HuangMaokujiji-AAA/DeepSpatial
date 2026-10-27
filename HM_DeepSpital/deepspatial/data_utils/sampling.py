"""训练配对混合抽样与保持来源边际的低方差推理抽样。"""

import numpy as np


def systematic_counts(probabilities, total, rng=None):
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
    rng = np.random if rng is None else rng
    positions = (rng.random() + np.arange(total)) / total
    draws = np.searchsorted(cumulative, positions, side="right")
    return np.bincount(draws, minlength=len(probabilities))


def _direction_times(lower, upper, n0, n1, forward, total, rng):
    """在薄层内按原 f(t)P(direction|t) 拒绝抽样，不改为均匀深度。"""
    result = np.empty(total, dtype=np.float64)
    left, right = n0 + (n1 - n0) * lower, n0 + (n1 - n0) * upper
    maximum = 1 - lower if forward else upper
    filled = 0
    while filled < total:
        u = rng.random(total - filled)
        # 线性密度 CDF 的稳定逆式；端点细胞数相同时自然退化为均匀分布。
        local_t = u * (left + right) / (left + np.sqrt(left**2 + u * (right**2 - left**2)))
        t = lower + (upper - lower) * local_t
        t = np.minimum(t, np.nextafter(upper, lower))
        accepted = rng.random(len(t)) * maximum < (1 - t if forward else t)
        values = t[accepted]
        result[filled:filled + len(values)] = values
        filled += len(values)
    return result


def depth_type_source_schedule(labels0, labels1, z0_um, z1_um, total,
                               bin_width_um=10.0, grid_origin_um=0.0, rng=None):
    """按薄层×方向×类型分配来源，保持原深度/方向/类别的联合期望。

    预算只使用两侧可见标签；小于 1 的期望仍可分到零，不强补类别。
    返回全体细胞的深度、方向、来源索引及可核验的期望/整数预算。
    """
    labels0, labels1 = np.asarray(labels0), np.asarray(labels1)
    if any(x.ndim != 1 or not len(x) or x.dtype.kind not in 'iu' or (x < 0).any()
           for x in (labels0, labels1)):
        raise ValueError('Source labels must be nonempty one-dimensional nonnegative integer arrays.')
    if not isinstance(total, (int, np.integer)) or total < 0:
        raise ValueError('total must be a nonnegative integer.')
    if (not np.isfinite([z0_um, z1_um, bin_width_um, grid_origin_um]).all()
            or z1_um <= z0_um or bin_width_um <= 0):
        raise ValueError('Use finite ordered physical depths, a positive bin width and finite grid origin.')
    rng = np.random if rng is None else rng
    classes = np.union1d(labels0, labels1)
    probabilities = [np.array([(labels == c).sum() for c in classes], dtype=float) / len(labels)
                     for labels in (labels0, labels1)]
    first = int(np.floor((z0_um - grid_origin_um + bin_width_um / 2) / bin_width_um))
    stop = int(np.ceil((z1_um - grid_origin_um + bin_width_um / 2) / bin_width_um))
    bins = np.arange(first, stop, dtype=np.int64)
    physical_left = np.maximum(z0_um, grid_origin_um + (bins - .5) * bin_width_um)
    physical_right = np.minimum(z1_um, grid_origin_um + (bins + .5) * bin_width_um)
    lower, upper = (physical_left - z0_um) / (z1_um - z0_um), (physical_right - z0_um) / (z1_um - z0_um)
    # 两点 Gauss 积分精确积分 f(t)*(1-t) 和 f(t)*t 这两个二次多项式。
    midpoint, half = (lower + upper) / 2, (upper - lower) / 2
    points = midpoint[:, None] + half[:, None] * np.array([-1, 1]) / np.sqrt(3)
    n0, n1 = len(labels0), len(labels1)
    density = (n0 + (n1 - n0) * points) / ((n0 + n1) / 2)
    direction_mass = np.column_stack([half * (density * (1 - points)).sum(1),
                                      half * (density * points).sum(1)])
    joint_mass = direction_mass[:, :, None] * np.asarray(probabilities)[None, :, :]
    layer_mass = joint_mass.sum(axis=(1, 2))
    layer_counts = systematic_counts(layer_mass, total, rng)
    times, directions, source_classes, bin_ids, budget = [], [], [], [], []
    for b, count in enumerate(layer_counts):
        # 同一类型的两个方向必须相邻，保证“层×类型”总数也只作 floor/ceil 舍入。
        counts = systematic_counts(joint_mass[b].T.ravel(), int(count), rng).reshape(-1, 2).T
        for d in (0, 1):
            size = int(counts[d].sum())
            times.append(_direction_times(lower[b], upper[b], n0, n1, d == 0, size, rng))
            directions.append(np.full(size, d == 0, dtype=bool))
            source_classes.append(np.repeat(classes, counts[d]))
            bin_ids.append(np.full(size, bins[b], dtype=np.int64))
            for c, label in enumerate(classes):
                budget.append(dict(depth_bin_index=int(bins[b]), bin_center_um=float(grid_origin_um + bins[b] * bin_width_um),
                                   bin_left_um=float(physical_left[b]), bin_right_um=float(physical_right[b]),
                                   gap_start_um=float(z0_um), gap_end_um=float(z1_um),
                                   source_direction='forward' if d == 0 else 'backward', source_class_code=int(label),
                                   expected_n=float(total * joint_mass[b, d, c] / layer_mass.sum()),
                                   conditional_expected_n=float(count * joint_mass[b, d, c] / layer_mass[b]),
                                   allocated_n=int(counts[d, c])))
    t = np.concatenate(times)
    forward = np.concatenate(directions)
    codes = np.concatenate(source_classes)
    depth_bin = np.concatenate(bin_ids)
    parent = np.empty(total, dtype=np.int64)
    for is_forward, labels in ((True, labels0), (False, labels1)):
        for label in np.unique(labels):
            children = np.flatnonzero((forward == is_forward) & (codes == label))
            parents = rng.permutation(np.flatnonzero(labels == label))
            full, remainder = divmod(len(children), len(parents))
            draws = np.concatenate([np.tile(parents, full), parents[:remainder]])
            parent[rng.permutation(children)] = draws
    order = rng.permutation(total)
    return dict(t=t[order], is_forward=forward[order], source_index=parent[order],
                source_class=codes[order], depth_bin_index=depth_bin[order], budget=budget,
                bin_width_um=float(bin_width_um), grid_origin_um=float(grid_origin_um))


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
