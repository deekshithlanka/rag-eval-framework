"""Small, dependency-light statistics helpers: bootstrap CIs and agreement metrics."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap confidence interval."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a, b, n: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float, float]:
    """Mean of (b - a) over paired items, with a bootstrap CI. Positive means b is higher."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    return bootstrap_ci(b - a, n=n, alpha=alpha, seed=seed)


def cohen_kappa(a, b, weights: str | None = None, labels=None) -> float:
    """Cohen's kappa. weights=None for nominal labels, 'quadratic' for ordinal scores."""
    a = list(a)
    b = list(b)
    labels = sorted(set(a) | set(b)) if labels is None else list(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    if k < 2:
        return 1.0
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((k, k))
    if weights == "quadratic":
        w = ((i - j) ** 2) / ((k - 1) ** 2)
    else:
        w = (i != j).astype(float)
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)


def agreement(a, b, tolerance: int = 0) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float((np.abs(a - b) <= tolerance).mean())
