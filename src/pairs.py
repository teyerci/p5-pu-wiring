"""Pair sampling for P5 (course step 15): recorded positives plus a documented sample
of unrecorded pairs, for one region.

Label (proposal, course threshold policy): a pair (i, j) of roster neurons is
recorded as connected, s = 1, when i makes at least τ synapses onto j summed over
the region's neuropils. Every other ordered pair of distinct roster neurons is
unlabelled, s = 0, including pairs recorded with 1 to τ - 1 synapses (stratum
"below_tau"); a pair with no recorded synapse in the region is "unrecorded".

Sampling scheme (the only one so far, "uniform"): all positives are kept with
weight 1; every unlabelled pair is included independently with probability ρ and
gets weight 1/ρ (inverse inclusion), so weighted sums over the sample estimate
population totals. ρ is given directly (`rate`) or as a target number of
unlabelled pairs per positive (`per_positive`, ρ = k · P / U). The candidate set
is all ordered pairs of distinct roster neurons, whatever the dropout condition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eda_shared.eda_helpers import ALIASES

CX_NEUROPILS = frozenset({"PB", "FB", "EB", "NO"})
SCHEMES = ("uniform",)


def cx_roster(neurons: pd.DataFrame) -> pd.Index:
    """The P5 roster: neurons annotated cell_class == "CX" (a stated deviation from the
    course's region rule, so that the synapses being predicted do not pick the roster)."""
    return pd.Index(np.sort(neurons.loc[neurons["cell_class"] == "CX", "neuron_id"].to_numpy()), name="neuron_id")


def in_region(neuropil: pd.Series, region=CX_NEUROPILS) -> np.ndarray:
    """True where a neuropil name (FlyWire ME_L or neuPrint ME(L) spelling) is in `region` (base names)."""
    base = neuropil.astype("string").str.replace(r"(_[LR]|\([LR]\))$", "", regex=True).replace(ALIASES)
    return base.isin(region).to_numpy()


def region_pair_counts(edgeroi: pd.DataFrame, ids, region=CX_NEUROPILS) -> pd.Series:
    """Synapses from i onto j summed over the region's neuropils, for roster pairs with any."""
    ids = pd.Index(ids)
    er = edgeroi[edgeroi["pre_id"].isin(ids) & edgeroi["post_id"].isin(ids)]
    er = er[in_region(er["neuropil"], region)]
    s = er.groupby(["pre_id", "post_id"])["syn_count"].sum().astype("int64")
    return s[s.index.get_level_values(0) != s.index.get_level_values(1)].rename("syn")


def _pair_from_index(idx: np.ndarray, n: int):
    """Ordered pair (i, j), i != j, from an index in [0, n(n-1))."""
    i = idx // (n - 1)
    r = idx % (n - 1)
    return i, r + (r >= i)


def sample_pairs(ids, pair_syn: pd.Series, tau: int = 10, scheme: str = "uniform",
                 rate: float | None = None, per_positive: float | None = None, seed: int = 0) -> pd.DataFrame:
    """Positives (syn >= tau) plus a sample of the unlabelled ordered pairs.

    Returns pre_id, post_id, syn (recorded synapses in the region, 0 if none), s,
    stratum (positive / below_tau / unrecorded) and weight (1 or 1/ρ). Deterministic
    given `seed`. Exactly one of `rate` and `per_positive` must be given.

    `ids` must be unique. `pair_syn` is restricted to pairs of distinct roster
    neurons here, so a table built for a wider population cannot change P or U.
    With no unlabelled pairs the result is the positives alone; `per_positive`
    with no positives is refused (give `rate`).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if (rate is None) == (per_positive is None):
        raise ValueError("give exactly one of rate and per_positive")
    ids = np.sort(np.asarray(ids, dtype="int64"))
    if len(np.unique(ids)) != len(ids):
        raise ValueError("roster ids must be unique")
    n = len(ids)
    total = n * (n - 1)
    pre, post = pair_syn.index.get_level_values(0), pair_syn.index.get_level_values(1)
    pair_syn = pair_syn[pre.isin(ids) & post.isin(ids) & (pre != post)]
    if pair_syn.index.duplicated().any():
        raise ValueError("pair_syn has duplicate pairs")
    pos = pair_syn[pair_syn >= tau]
    unlabelled = total - len(pos)
    if rate is not None:
        rho = rate
    elif unlabelled == 0:
        rho = 1.0                                        # nothing to sample
    elif len(pos) == 0:
        raise ValueError("no positives at this τ: per_positive is undefined, give rate")
    else:
        rho = min(1.0, per_positive * len(pos) / unlabelled)
    if not 0 < rho <= 1:
        raise ValueError(f"sampling rate must be in (0, 1], got {rho}")

    rng = np.random.default_rng(seed)
    m = rng.binomial(total, rho)
    idx = np.sort(rng.choice(total, size=m, replace=False))
    i, j = _pair_from_index(idx, n)
    sample = pd.MultiIndex.from_arrays([ids[i], ids[j]], names=["pre_id", "post_id"])
    sample = sample.difference(pos.index)                    # Bernoulli(ρ) over the unlabelled pairs only

    syn = pair_syn.reindex(sample, fill_value=0).to_numpy()
    neg = pd.DataFrame({"pre_id": sample.get_level_values(0), "post_id": sample.get_level_values(1),
                        "syn": syn, "s": 0,
                        "stratum": np.where(syn > 0, "below_tau", "unrecorded"), "weight": 1.0 / rho})
    posdf = pd.DataFrame({"pre_id": pos.index.get_level_values(0), "post_id": pos.index.get_level_values(1),
                          "syn": pos.to_numpy(), "s": 1, "stratum": "positive", "weight": 1.0})
    out = pd.concat([posdf, neg], ignore_index=True)
    out.attrs.update({"scheme": scheme, "rate": rho, "tau": tau, "seed": seed, "roster": n,
                      "candidate_pairs": total, "positives": len(pos), "unlabelled": unlabelled})
    return out.astype({"pre_id": "int64", "post_id": "int64", "syn": "int64", "s": "int8", "weight": "float64"})
