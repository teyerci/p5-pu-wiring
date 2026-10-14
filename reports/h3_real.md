# H3 on real data: the PU model keeps the naive ranking

`scripts/h3_real.py` → `reports/h3_real.json`, `reports/h3_top_unrecorded.csv` (10 minutes).

**Setup.** This is the real-data fit, as at prediction time, so there is no split:
- **Pairs:** all 32,125 positives of the FAFB central-complex roster (τ = 10) plus a uniform sample of 321,982 unlabelled pairs (10 per positive).
- **Features:** leave-pair-out, on the full graph.
- **Fit:** λ = 10⁻⁴, the naive model and PU at c ∈ {0.9, 0.8, 0.7, 0.6, 0.5}. c is unknown on real data, so it is a sensitivity range. All six fits converged.

Every comparison is over the 321,982 sampled unlabelled pairs, the set a ranked list of missed connections is drawn from.

| c | π | Mean P(y=1), unlabelled | Spearman (naive, PU) | Top-100 overlap | Top-1000 overlap | Coefficients: Spearman, signed / by magnitude | Below-τ share, top 1000 |
|---|---|---|---|---|---|---|---|
| naive (1.0) | 0.0039 | 0.0035 | — | — | — | — | 0.376 |
| 0.9 | 0.0043 | 0.0039 | 0.99997 | 0.99 | 0.992 | 0.9996 / 0.9986 | 0.375 |
| 0.8 | 0.0048 | 0.0044 | 0.99988 | 0.98 | 0.983 | 0.9986 / 0.9943 | 0.373 |
| 0.7 | 0.0055 | 0.0051 | 0.99970 | 0.96 | 0.975 | 0.9962 / 0.9854 | 0.374 |
| 0.6 | 0.0065 | 0.0059 | 0.99942 | 0.92 | 0.967 | 0.9918 / 0.9695 | 0.373 |
| 0.5 | 0.0078 | 0.0071 | 0.99899 | 0.88 | 0.952 | 0.9829 / 0.9502 | 0.377 |

(π for the naive row is the training P(s=1) = 0.0039.)

**What it shows:**
- **H3 holds on this data.** PU at any c in the range ranks the unlabelled pairs almost exactly as the naive model does: Spearman 0.999 or higher (lowest 0.99899, at c = 0.5), and 95–99% of the top 1,000 are shared. The PU correction is mainly a rescaling: the mean P(y=1) on unlabelled pairs rises roughly as 1/c (0.0035 naive, 0.0051 at c = 0.7, 0.0071 at c = 0.5).
- **The differences grow as c falls** (top-100 overlap 0.99 → 0.88). They come from the change in functional form: the PU risk is not a monotone transform of the naive fit, so coefficients move a little (magnitude rank correlation 0.95 at c = 0.5). On real data SCAR violations cannot be separated from this; the benchmark runs now record the same statistics against known truth (`h3_naive_vs_pu_true_c` in `src.benchmark.run_condition`; from the next benchmark run on).
- **Consequence for the headline:** a ranked list does not distinguish the models. What PU changes is the level of the probabilities, so the project's main claim stays the calibration result against benchmark truth (H2), as planned.
- **The top of the ranking is enriched for weak recorded connections.** 37–38% of the top 1,000 unlabelled pairs have 1–9 recorded synapses, against 3.1% of all unlabelled pairs (about 12 times the base rate), although synapse counts are never a feature. This is the same internal check as the pilot's, now at λ = 10⁻⁴ with no split.

## Largest coefficients (naive; standardised numerics, one-hot types)

`spatial` +2.28, `overlap` +0.60, `type_i=PFNp` −0.45, `same_type` +0.42, `type_j=ExR1` −0.42, `in_deg_j` +0.38, `type_j=Delta7` −0.38, `out_deg_i` +0.36. The c = 0.7 model has the same leading features in nearly the same order (`reports/h3_real.json`).

## The ranked list (`reports/h3_top_unrecorded.csv`)

The top 500 unlabelled pairs of the c = 0.7 model, with types, recorded synapses (0 or 1–9), both models' P(y=1), and the naive rank. The naive ranks of these 500 run from 1 to 645.
- 43% have 1–9 recorded synapses.
- Same-type pairs dominate: EPG→EPG 61, ER3d→ER3d 44, Delta7→Delta7 38, ER2→ER2 24, PEN_b→PEN_b 13. That is consistent with two influential features, `spatial` (same-type neurons share arbors) and `same_type`, though this list alone does not show which features drive it. Whether these are missed connections or an artefact of the features is exactly what the hemibrain check (H4) is for. Some recurrent connections within these types are reported in the literature, but that has not been checked here.
- **Limits.** This is about a 1-in-26 sample of the unlabelled pairs, so the list is a sample of the full ranking, not the ranked list itself, which would score every roster pair. λ is fixed, and there is no bootstrap.
