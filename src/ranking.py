"""Ranking agreement between two models (proposal H3).

Under SCAR the PU model mainly rescales probabilities, so its ranking of pairs should stay
close to the naive one; differences come from the change in functional form (the PU risk
is not a monotone transform of the naive fit) or from SCAR violations. Measures:
  - Spearman correlation of the two models' scores over the same pairs;
  - overlap of the top-k lists (share of the first model's top k that is in the second's);
  - Spearman correlation of the coefficients, signed and by magnitude.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def top_k(scores, k: int) -> np.ndarray:
    """Indices of the k highest scores (ties broken by position, stable)."""
    if not 0 < k <= len(scores):
        raise ValueError("k must lie in [1, len(scores)]")
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:k]


def top_k_overlap(a, b, k: int) -> float:
    """|top_k(a) ∩ top_k(b)| / k."""
    return len(np.intersect1d(top_k(a, k), top_k(b, k))) / k


def ranking_agreement(scores_a, scores_b, coef_a=None, coef_b=None, ks=(100, 1000)) -> dict:
    """H3 statistics for two models scored on the same pairs (and fitted on the same design)."""
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("scores must cover the same pairs")
    out = {"n": int(len(a)), "spearman": float(spearmanr(a, b).statistic)}
    for k in ks:
        if k <= len(a):
            out[f"top{k}_overlap"] = top_k_overlap(a, b, k)
    if coef_a is not None:
        ca, cb = np.asarray(coef_a, dtype=float), np.asarray(coef_b, dtype=float)
        out["coef_spearman"] = float(spearmanr(ca, cb).statistic)
        out["coef_abs_spearman"] = float(spearmanr(np.abs(ca), np.abs(cb)).statistic)
    return out
