"""H3 on the real fafb central-complex data: does the PU model keep the naive ranking?

Real-data fit, as at prediction time: no split; all 32k positives plus a uniform sample of
10 unlabelled pairs per positive (src.pairs), leave-pair-out features on the full graph
(src.pair_features), λ = 1e-4 (inner CV did not beat it, reports/benchmark_cv.md).
c is unknown here, so PU is fitted over a sensitivity range c ∈ {0.9, 0.8, 0.7, 0.6, 0.5}.

Over the sampled unlabelled pairs (the pairs a ranked list is drawn from) it compares the
naive score with each PU score (src.ranking): Spearman correlation, top-100 and top-1000
overlap, and coefficient rank agreement; also the share of recorded below-τ pairs
(1-9 synapses) among the top 1000 (synapse counts are never a feature). It lists the
largest coefficients of the naive and the c = 0.7 model, and writes the top 500 unlabelled
pairs of the c = 0.7 model to reports/h3_top_unrecorded.csv.
The unlabelled pairs are a 1-in-ρ sample, so this list is a sample of the full ranking,
not the full ranked list (which would score every roster pair).
Writes reports/h3_real.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import fetch_fafb                                                    # noqa: E402
from src.benchmark import GROUPS                                              # noqa: E402
from src.pair_features import build_pair_features, design, feature_columns    # noqa: E402
from src.pairs import cx_roster, region_pair_counts, sample_pairs             # noqa: E402
from src.pu import PULogistic                                                 # noqa: E402
from src.ranking import ranking_agreement, top_k                              # noqa: E402

TAU, LAM, SEED, CS, LIST_C = 10, 1e-4, 0, (0.9, 0.8, 0.7, 0.6, 0.5), 0.7
neurons, edges, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edges(), fetch_fafb.load_edgeroi()
ids = cx_roster(neurons)
sample = sample_pairs(ids, region_pair_counts(edgeroi, ids), tau=TAU, per_positive=10, seed=SEED)
feats = build_pair_features(neurons, edges, edgeroi, sample, ids, tau=TAU)
X, _, names = design(feats, None, feature_columns(feats, GROUPS))
s, a = sample["s"].to_numpy(float), sample["weight"].to_numpy()
unl = s == 0
below = sample["stratum"].to_numpy()[unl] == "below_tau"

models = {1.0: PULogistic(c=1.0, lam=LAM).fit(X, s, a)}
for c in CS:
    models[c] = PULogistic(c=c, lam=LAM).fit(X, s, a)
scores = {c: m.predict_proba(X[unl]) for c, m in models.items()}
naive = scores[1.0]

out = {"lambda": LAM, "seed": SEED, "pairs": int(len(s)), "positives": int(s.sum()), "unlabelled_scored": int(unl.sum()),
       "below_tau_share_unlabelled": float(below.mean()), "p_s": float(s @ a / a.sum()),
       "converged": {str(c): m.converged_ for c, m in models.items()},
       "naive": {"mean_p_unlabelled": float(naive.mean()), "below_tau_share_top1000": float(below[top_k(naive, 1000)].mean())},
       "pu": {}}
for c in CS:
    m = models[c]
    out["pu"][str(c)] = {"pi": m.pi_, "mean_p_unlabelled": float(scores[c].mean()),
                         "below_tau_share_top1000": float(below[top_k(scores[c], 1000)].mean()),
                         **ranking_agreement(naive, scores[c], models[1.0].coef_, m.coef_, ks=(100, 1000))}
    print(c, out["pu"][str(c)], flush=True)


def top_coefs(m, n=15):
    order = np.argsort(-np.abs(m.coef_))[:n]
    return [[names[i], float(m.coef_[i])] for i in order]


out["top_coefficients"] = {"naive": top_coefs(models[1.0]), str(LIST_C): top_coefs(models[LIST_C])}
(ROOT / "reports" / "h3_real.json").write_text(json.dumps(out, indent=1, default=float))

nn = neurons.set_index("neuron_id")
u = sample[unl].reset_index(drop=True)
rank_naive = pd.Series(naive).rank(ascending=False, method="first").to_numpy()
idx = top_k(scores[LIST_C], 500)
lst = pd.DataFrame({"rank": np.arange(1, 501), "pre_id": u.loc[idx, "pre_id"].to_numpy(), "post_id": u.loc[idx, "post_id"].to_numpy(),
                    "pre_type": nn.loc[u.loc[idx, "pre_id"], "cell_type"].to_numpy(),
                    "post_type": nn.loc[u.loc[idx, "post_id"], "cell_type"].to_numpy(),
                    "recorded_synapses": u.loc[idx, "syn"].to_numpy(),
                    f"p_true_pu_c{LIST_C}": scores[LIST_C][idx], "p_true_naive": naive[idx], "naive_rank": rank_naive[idx].astype(int)})
lst.to_csv(ROOT / "reports" / "h3_top_unrecorded.csv", index=False)
print(json.dumps({k: v for k, v in out.items() if k != "pu"}, indent=1, default=float))
