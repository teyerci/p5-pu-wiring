"""Pilot: naive vs PU logistic models on the fafb central-complex roster (τ = 10).

Not the final design (no tuned λ, one split, no bootstrap): a first look that the
pipeline runs end to end and what it shows. Writes reports/pilot_pu.json.

- Sample: all 32,125 positives plus 10 unlabelled pairs per positive (src.pairs).
- Features: leave-pair-out pair features (src.pair_features).
- Split: 20% of the roster's cell types are held-out sources (pairs whose pre neuron has
  such a type are the test set). Training features are built on a graph without any
  held-out candidate pair's synapses; test features on the full graph, leave-pair-out.
  Target neurons are shared between sets; a pilot simplification.
- Models: naive (c = 1) and PU at c ∈ {0.9, 0.7, 0.5}; λ = 1e-2; inverse-inclusion weights.
- Metrics on the test set (weighted): AUC for s, Brier score for s with the model's
  P(s=1|x) (naive g, PU c·f), mean P(y=1|x) on unlabelled pairs (s = 0), and the share of
  below-τ pairs (1–9 synapses) among the top-scored unlabelled pairs (synapse counts are
  never a feature, so this is an internal check).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import fetch_fafb                                                # noqa: E402
from src.pair_features import design, feature_columns, split_features    # noqa: E402
from src.pairs import cx_roster, region_pair_counts, sample_pairs         # noqa: E402
from src.pu import PULogistic                                             # noqa: E402

TAU, LAM, SEED = 10, 1e-2, 0
neurons, edges, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edges(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
pair_syn = region_pair_counts(edgeroi, ids)
sample = sample_pairs(ids, pair_syn, tau=TAU, per_positive=10, seed=SEED)
# split first: 20% of the roster's cell types are held-out sources; training features are built
# on a graph without any held-out candidate pair's synapses (src.pair_features.split_features)
rng = np.random.default_rng(SEED)
roster = neurons[neurons["neuron_id"].isin(ids)]
types = np.sort(roster["cell_type"].fillna("untyped").unique())
test_types = set(rng.choice(types, size=int(0.2 * len(types)), replace=False))
test_sources = roster.loc[roster["cell_type"].fillna("untyped").isin(test_types), "neuron_id"]
f_train, f_test, is_test = split_features(neurons, edges, edgeroi, sample, ids, test_sources, tau=TAU)
s, a, stratum = sample["s"].to_numpy(float), sample["weight"].to_numpy(), sample["stratum"].to_numpy()

out = {"sample": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in sample.attrs.items()},
       "split": {"test_types": len(test_types), "train_pairs": int((~is_test).sum()), "test_pairs": int(is_test.sum()),
                 "test_positives": int(s[is_test].sum())}, "lambda": LAM, "ablation": {}, "pu": {}}


def evaluate(model, X, s_, a_, strat):
    g = model.predict_recorded(X)
    f = model.predict_proba(X)
    unl = s_ == 0
    top = np.argsort(-f[unl])[: max(1, int(0.01 * unl.sum()))]
    return {"auc_s": float(roc_auc_score(s_, g, sample_weight=a_)),
            "brier_s": float(np.average((g - s_) ** 2, weights=a_)),
            "mean_p_s_vs_recorded": [float(np.average(g, weights=a_)), float(np.average(s_, weights=a_))],
            "mean_f_unrecorded": float(np.average(f[unl], weights=a_[unl])),
            "below_tau_share_top1pct_unlabelled": float((strat[unl][top] == "below_tau").mean()),
            "below_tau_share_all_unlabelled": float(np.average(strat[unl] == "below_tau", weights=a_[unl]))}


# ablation: feature groups added one at a time (naive model)
groups_seq = [["identity"], ["identity", "cell_type"], ["identity", "cell_type", "region"],
              ["identity", "cell_type", "region", "degree"], ["identity", "cell_type", "region", "degree", "spatial"]]
for groups in groups_seq:
    cols = feature_columns(f_train, groups)
    Xtr, Xte, names = design(f_train, f_test, cols)
    m = PULogistic(c=1.0, lam=LAM).fit(Xtr, s[~is_test], a[~is_test])
    out["ablation"]["+".join(groups)] = {"n_features": len(names), "converged": m.converged_,
                                         **evaluate(m, Xte, s[is_test], a[is_test], stratum[is_test])}
    print("+".join(groups), out["ablation"]["+".join(groups)], flush=True)

# naive vs PU on the full feature set
cols = feature_columns(f_train, groups_seq[-1])
Xtr, Xte, names = design(f_train, f_test, cols)
for c in (1.0, 0.9, 0.7, 0.5):
    m = PULogistic(c=c, lam=LAM).fit(Xtr, s[~is_test], a[~is_test])
    out["pu"][str(c)] = {"pi": m.pi_, "converged": m.converged_, "iterations": m.n_iter_,
                         **evaluate(m, Xte, s[is_test], a[is_test], stratum[is_test])}
    print(c, out["pu"][str(c)], flush=True)

(ROOT / "reports" / "pilot_pu.json").write_text(json.dumps(out, indent=1, default=float))
