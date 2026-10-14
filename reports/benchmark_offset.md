# Where the held-out-type offset comes from

`reports/benchmark_relative.md` found that held-out source types are over-predicted even without thinning: the naive mean P(y=1) on held-out pairs is 1.25, 1.36 and 1.51 times the true rate (seeds 0–2, λ = 10⁻⁴). Two explanations were tested, both at λ = 10⁻⁴ on the same seeds and splits, with q = 1 and q = 0.7 synapse dropout.

## 1. The source cell-type one-hot: ruled out

`scripts/benchmark_type_onehot.py` → `reports/benchmark_type_onehot.json`. A held-out source type is never a source in training, so its `type_i` one-hot is all zeros at test time. Removing `type_i` from the features leaves the offset where it was:

| Features | Offset without thinning, seeds 0 / 1 / 2 | Mean | AUC (q = 1), mean |
|---|---|---|---|
| all | 1.246 / 1.364 / 1.506 | 1.372 | 0.945 |
| without `type_i` | 1.280 / 1.368 / 1.505 | 1.385 | 0.944 |

(A third arm, also without `sub_i`, was stopped after one run: 1.299 on seed 0. It is not in the JSON, which predates the `target_graph`, `fold` and `test_sources` fields.)

## 2. A train/test shift in the target-side features: the main cause

`scripts/benchmark_target_graph.py` → `reports/benchmark_target_graph.json`.

To keep held-out labels out of training, `split_features` builds the training features on a graph without any held-out source's edges into the roster. Every training target has therefore lost the inputs it gets from about a fifth of the roster. The held-out pairs' features, however, were built on the full graph. So their target-side features (`in_deg_j`, `frac_in_*`, and through them `overlap` and `spatial`) were systematically larger than anything seen in training, and the model, which has learnt that well-connected targets are likely partners, over-predicted.

The new option `target_graph="train"` (`src.pair_features.split_features`, `src.benchmark.run_condition`) takes the held-out pairs' target-side features from the training graph as well. Source-side features stay on the full graph, leave-pair-out, as before. No held-out label enters either side; it also removes a coupling the full-graph version had, where a held-out pair's target features included edges from other held-out sources (other test labels).

| Target-side graph | Seed | Offset (q = 1) | Naive, relative level (q = 0.7) | PU at true c, relative level | PU raw level / true rate | AUC (q = 1) | Top reliability bin, PU at q = 0.7 (true rate) |
|---|---|---|---|---|---|---|---|
| full (before) | 0 | 1.246 | 0.614 | 0.967 | 1.204 | 0.927 | 0.033 (0.030) |
| full (before) | 1 | 1.364 | 0.648 | 0.994 | 1.356 | 0.944 | 0.037 (0.028) |
| full (before) | 2 | 1.506 | 0.671 | 1.009 | 1.519 | 0.962 | 0.063 (0.043) |
| training | 0 | 0.972 | 0.628 | 0.972 | 0.945 | 0.909 | 0.026 (0.027) |
| training | 1 | 1.074 | 0.656 | 0.998 | 1.072 | 0.931 | 0.029 (0.027) |
| training | 2 | 1.102 | 0.663 | 1.004 | 1.106 | 0.956 | 0.045 (0.041) |

Realised c in training: 0.653, 0.658, 0.655. The "full" rows are the λ = 10⁻⁴ synapse runs of `reports/benchmark_relative.json`; the runs are deterministic, and the all-feature arm of the one-hot diagnostic reproduces them.

**What it shows:**
- **The offset falls from +37% to +5% on average** (per seed −3%, +7%, +10%). Most of it was this feature shift, a property of the evaluation split rather than of the model or of PU.
- **The raw mean level of PU at the true c is now close to the truth** (0.95, 1.07, 1.11 of the true rate at q = 0.7) without dividing by a same-split baseline. The relative levels are unchanged (naive 0.63–0.66 against c ≈ 0.66; PU 0.97–1.00).
- **The top reliability bin is now close to the true rate** for PU at q = 0.7 (0.026–0.045 against 0.027–0.041), where it overshot before (0.033–0.063 against 0.028–0.043).
- **The AUC falls by about 0.013** (0.007–0.019 per seed). Training-graph target features ignore the inputs a target gets from held-out sources, so they carry less information. This cost belongs to the evaluation design: a model fitted to all data and applied to the same data has no split and no shift.
- **A residual offset of up to +10% remains** (−3% to +10% per seed, 3 seeds). Its cause is not identified here; seed-to-seed variation in which types are held out is one candidate.

**Consequence for the design:** benchmark levels and calibration are read with `target_graph="train"` from now on; the homologue-group fold benchmark (running; reported separately) uses it. The full-graph AUCs (0.927–0.962) are higher, but they come with the shifted features, so they are not a clean reference either.
