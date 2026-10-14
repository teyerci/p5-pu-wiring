"""Dropout benchmark, first part: thinning the recorded region and the realised c.

The benchmark (proposal §5) treats the recorded region as truth and deletes data:

- synapse dropout: each synapse is kept with probability q, so a pair with n
  synapses keeps Binomial(n, q). Weak connections fall below τ first, so this
  violates SCAR the way real detection does;
- connection dropout: each connected pair keeps all or none of its synapses with
  probability q. This satisfies SCAR (c = q for every positive) and isolates the
  cost of the violation.

Relabelling at τ after thinning gives the benchmark's s; the true label y is the
unthinned pair at τ. The realised label frequency c = P(s=1 | y=1) is reported by
stratum of the true synapse count, exactly (binomial tail) and by simulation.
Features are rebuilt from the thinned graph in the full benchmark (weeks 7–9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

STRATA = [(10, 19, "10–19"), (20, 49, "20–49"), (50, np.inf, "50+")]
MODES = ("synapse", "connection")


def thin(pair_syn: pd.Series, q: float, mode: str = "synapse", seed: int = 0) -> pd.Series:
    """Thinned synapse counts per pair (pairs left with 0 are dropped). Deterministic given seed."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not 0 < q <= 1:
        raise ValueError("q must lie in (0, 1]")
    rng = np.random.default_rng(seed)
    n = pair_syn.to_numpy().astype(np.int64)
    if mode == "synapse":
        kept = rng.binomial(n, q)
    else:
        kept = np.where(rng.random(len(n)) < q, n, 0)
    out = pd.Series(kept, index=pair_syn.index, name=pair_syn.name)
    return out[out > 0]


def stratum(n: pd.Series) -> pd.Series:
    labels = [s[2] for s in STRATA]
    edges = [s[0] - 0.5 for s in STRATA] + [np.inf]
    return pd.cut(n, bins=edges, labels=labels)


def realised_c(pair_syn: pd.Series, q: float, tau: int = 10, mode: str = "synapse", seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Realised c = P(still ≥ τ after thinning | ≥ τ before), by stratum of the true count and overall:
    exact expectation and the mean and range over seeds."""
    pos = pair_syn[pair_syn >= tau]
    st = stratum(pos)
    if mode == "synapse":
        exact = pd.Series(binom.sf(tau - 1, pos.to_numpy(), q), index=pos.index)
    else:
        exact = pd.Series(q, index=pos.index, dtype=float)
    sims = []
    for seed in seeds:
        t = thin(pos, q, mode, seed).reindex(pos.index, fill_value=0)
        sims.append((t >= tau).astype(float))
    sim = pd.concat(sims, axis=1)
    rows = {}
    for label in [s[2] for s in STRATA] + ["all positives"]:
        m = (st == label).to_numpy() if label != "all positives" else np.ones(len(pos), bool)
        per_seed = sim[m].mean(axis=0)
        rows[label] = {"positives": int(m.sum()), "exact": float(exact[m].mean()),
                       "sim_mean": float(per_seed.mean()), "sim_min": float(per_seed.min()), "sim_max": float(per_seed.max())}
    return pd.DataFrame(rows).T.assign(q=q, mode=mode, tau=tau)
