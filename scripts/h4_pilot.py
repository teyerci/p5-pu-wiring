"""H4 pilot: are high-scoring unrecorded fafb pairs recorded at type level in the hemibrain
more often than matched control pairs? (proposal H4; corroborative only)

Scores: the real-data fit of scripts/h3_real.py (same sample, features and λ = 1e-4), PU at
c = 0.7; H3 showed the naive ranking is nearly the same. Eligible pairs: sampled unlabelled
fafb pairs with both neurons on the right side and both types in a match group that also
occurs on the hemibrain's right side (through fafb's `hemibrain_type`). Hemibrain v1.2
public export (no `cropped` flag, see README); cropped neurons can only lower hit rates.

Hit: the pair's type pair is recorded at τ in the hemibrain (src.h4, src.discrepancy), at
τ = 10 and at τ scaled by the type-pair total ratio hemibrain / fafb (src.discrepancy.scale).
Candidates: the top k eligible pairs by score, k ∈ {100, 500, 1000, 5000}. Controls: the other
eligible pairs, standardised to the candidates' strata (src.h4.matched_hit_rate):
  A  type-pair size decile (fafb right-side neurons in A × in B) × degree decile
     (out_deg_i + in_deg_j at τ): the proposal's two matching variables;
  B  A × whether fafb records the type pair at τ through other neuron pairs;
  C  B × same type.
Deciles are taken over ranks with ties split by row order (`rank(method="first")`), so equal
values (e.g. pairs of one type pair) can fall in neighbouring deciles; the split follows the
sampled pair order, which is fixed by the seed.
B and C are added here because the model can learn a type pair's connectivity from fafb
itself, and a type pair recorded in fafb is likely recorded in the hemibrain too.
Intervals: 95% percentile intervals from 500 bootstrap replicates that resample whole clusters,
the source neuron's type group (hits cluster by type pair; src.h4.cluster_bootstrap).
Writes reports/h4_pilot.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import fetch_fafb, fetch_hemibrain                                    # noqa: E402
from src.benchmark import GROUPS                                              # noqa: E402
from src.discrepancy import scale, type_pair_table                            # noqa: E402
from src.h4 import cluster_bootstrap, matched_hit_rate, other_type_table, pair_type_status, type_keys  # noqa: E402
from src.pair_features import build_pair_features, design, feature_columns    # noqa: E402
from src.pairs import cx_roster, region_pair_counts, sample_pairs             # noqa: E402
from src.pu import PULogistic                                                 # noqa: E402

TAU, LAM, SEED, C, SIDE, KS, N_BOOT = 10, 1e-4, 0, 0.7, "right", (100, 500, 1000, 5000), 500
neurons, edges, edgeroi = fetch_fafb.load_neurons(), fetch_fafb.load_edges(), fetch_fafb.load_edgeroi()
hn, her = fetch_hemibrain.load_neurons(), fetch_hemibrain.load_edgeroi()
ids = cx_roster(neurons)
pair_syn = region_pair_counts(edgeroi, ids)
sample = sample_pairs(ids, pair_syn, tau=TAU, per_positive=10, seed=SEED)
feats = build_pair_features(neurons, edges, edgeroi, sample, ids, tau=TAU)
X, _, _ = design(feats, None, feature_columns(feats, GROUPS))
s, a = sample["s"].to_numpy(float), sample["weight"].to_numpy()
score = PULogistic(c=C, lam=LAM).fit(X, s, a).predict_proba(X)

# type keys and tables (right side)
roster = neurons[neurons["neuron_id"].isin(ids)].reset_index(drop=True)
kf, kh = type_keys(roster, hn)
key_by_id = pd.Series(kf.to_numpy(), index=roster["neuron_id"].to_numpy())
side_by_id = pd.Series(roster["side"].to_numpy(), index=roster["neuron_id"].to_numpy())
groups_hb = set(kh[(hn["side"] == SIDE).to_numpy()].dropna())
groups_fafb = set(kf[(roster["side"] == SIDE).to_numpy()].dropna())
both = groups_hb & groups_fafb
t_hb = other_type_table(hn, her, kh, both, SIDE)
t_fafb = type_pair_table(roster, pair_syn, kf, SIDE)
sc = scale(t_fafb, t_hb, both)
tau_scaled = max(1, round(TAU * sc))

u = sample[(s == 0)].reset_index(drop=True)
su, fu = score[s == 0], feats[s == 0].reset_index(drop=True)
pre, post = u["pre_id"].to_numpy(), u["post_id"].to_numpy()
ka, kb = key_by_id.reindex(pre).to_numpy(), key_by_id.reindex(post).to_numpy()
elig = ((side_by_id.reindex(pre).to_numpy() == SIDE) & (side_by_id.reindex(post).to_numpy() == SIDE)
        & pd.Series(ka).isin(both).to_numpy() & pd.Series(kb).isin(both).to_numpy())
pre, post, su, fu, ka, kb = pre[elig], post[elig], su[elig], fu[elig].reset_index(drop=True), ka[elig], kb[elig]

hits = {f"tau{TAU}": (pair_type_status(pre, post, key_by_id, t_hb, TAU) == "recorded").to_numpy(),
        f"tau{tau_scaled}_scaled": (pair_type_status(pre, post, key_by_id, t_hb, tau_scaled) == "recorded").to_numpy()}
fafb_rec = (pair_type_status(pre, post, key_by_id, t_fafb, TAU) == "recorded").to_numpy()
n_right = pd.Series(kf[(roster["side"] == SIDE).to_numpy()]).value_counts()
size = n_right.reindex(ka).to_numpy() * n_right.reindex(kb).to_numpy()
size_dec = pd.qcut(pd.Series(size).rank(method="first"), 10, labels=False).to_numpy()
deg_dec = pd.qcut(pd.Series(fu["out_deg_i"] + fu["in_deg_j"]).rank(method="first"), 10, labels=False).to_numpy()
same = (ka == kb)
strata = {"A_size_x_degree": list(zip(size_dec, deg_dec)),
          "B_plus_fafb_type_recorded": list(zip(size_dec, deg_dec, fafb_rec)),
          "C_plus_same_type": list(zip(size_dec, deg_dec, fafb_rec, same))}
strata = {name: pd.factorize(pd.Series(st))[0] for name, st in strata.items()}       # integer labels, faster

order = np.argsort(-su, kind="stable")
out = {"n_boot": N_BOOT, "bootstrap_cluster": "source type group", "tau": TAU, "tau_scaled": tau_scaled, "scale_hemibrain_over_fafb": sc, "c": C, "lambda": LAM,
       "groups_both": len(both), "eligible_unlabelled": int(elig.sum()), "unlabelled": int(len(elig)),
       "base_rates": {**{h: float(v.mean()) for h, v in hits.items()}, "fafb_type_recorded": float(fafb_rec.mean()),
                      "same_type": float(same.mean())},
       "results": []}
for k in KS:
    cand = np.zeros(len(su), bool)
    cand[order[:k]] = True
    row = {"k": k, "fafb_type_recorded_share": float(fafb_rec[cand].mean()), "same_type_share": float(same[cand].mean()),
           "below_tau_share": float((u.loc[elig, "syn"].to_numpy()[cand] > 0).mean())}
    for h, hit in hits.items():
        row[h] = {name: {**matched_hit_rate(hit, cand, st), "bootstrap": cluster_bootstrap(hit, cand, st, ka, N_BOOT, seed=0)}
                  for name, st in strata.items()}
    out["results"].append(row)
    print(json.dumps(row, default=float), flush=True)
(ROOT / "reports" / "h4_pilot.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1, default=float))
