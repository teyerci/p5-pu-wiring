# Benchmark levels against a same-split baseline, and λ sensitivity

`scripts/benchmark_relative.py` → `reports/benchmark_relative.json` (18 runs, about 20 minutes).
`scripts/benchmark_lambda.py` → `reports/benchmark_lambda.json` (synapse dropout) and `reports/benchmark_lambda_connection.json` (diagnostics, 3 seeds each). The synapse file was written before training P(s=1) and π were added to the output, so it lacks those two fields; nothing below uses them.

## Why a same-split baseline

The 5-seed benchmark (`reports/benchmark.md`) compared each model's mean P(y=1) on held-out pairs with the true rate. That comparison mixes two things:
- **the recording loss**, which PU is meant to correct;
- **a level offset of the held-out source types**, which is present even without thinning.

The split depends only on the seed, so each thinned run can be divided by the no-thinning run of the same seed and split. That divides the offset out, **assuming it is multiplicative and the same in the thinned fit**. The SCAR control (connection dropout, where the naive level should be exactly c) shows the assumption holds to about ±3% on average: naive level / c is 0.97 at λ = 10⁻² and 1.03 at λ = 10⁻⁴ (0.93–1.09 per seed over both λ). The quantity is the **relative level**: a model's mean P(y=1) / true rate, divided by the same ratio for the no-thinning naive model of that seed. Under SCAR it should be about c for the naive model and about 1 for PU at the true c.

## Result (q = 0.7, 3 seeds; mean, with the range over seeds)

Means are over seeds of the per-seed ratios. The offset is the no-thinning naive level / true rate, mean over seeds (range).

| λ | Condition | Realised c (training) | Offset without thinning | Naive, relative level | PU at true c, relative level |
|---|---|---|---|---|---|
| 10⁻² | connection dropout | 0.701 | +12% | 0.678 (0.646–0.694) | **0.993** (0.982–1.000) |
| 10⁻² | synapse dropout | 0.656 | +12% | 0.641 (0.630–0.649) | **1.002** (0.978–1.039) |
| 10⁻⁴ | connection dropout | 0.701 | +37% | 0.720 (0.693–0.759) | **1.027** (1.002–1.066) |
| 10⁻⁴ | synapse dropout | 0.656 | +37% | 0.645 (0.614–0.671) | **0.990** (0.967–1.009) |

**What it shows:**
- **The naive model's level falls by the factor c.** It averages 0.64–0.72 against c = 0.66–0.70, which is the attenuation the SCAR derivation predicts: P(s=1|x) = c·P(y=1|x).
- **PU at the true c recovers the level** to within about 3% on average (0.99–1.03), in both dropout modes and at both λ. This is the mean-level part of H2, and it now holds once the split offset is removed.
- **No effect of the SCAR violation is resolved at the level of the mean.** Under synapse dropout, where weak connections are lost first, the naive level is 0.641–0.645 against c = 0.656, about 2% short, and PU recovers 0.99–1.00. That 2% is within the spread the SCAR control itself shows (about ±3% on average, 0.93–1.09 of c per seed; synapse dropout itself spans 0.94–1.02 of c per seed). With 3 seeds the violation is not resolvable at this precision. Its cost in per-pair ranking and calibration is not measured here.
- **The largest calibration error is the held-out-type offset, not the recording loss.**
  - Without any thinning, held-out source types are over-predicted by 12% on average at λ = 10⁻² (per seed −3%, +16%, +23%) and 37% at λ = 10⁻⁴ (+25%, +36%, +51%).
  - Earlier, PU looked like it overshot by 8–26% (`reports/benchmark.md`) or by 37–43% (the λ diagnostic). That is this offset, not a failure of the correction.
  - It is a property of the split. **Update:** `reports/benchmark_offset.md` traces most of it to a train/test shift in the target-side features (training targets lose the held-out sources' inputs; held-out pairs' target features used the full graph), not to the all-zero type one-hot. With target features from the training graph the offset falls to +5% on average.

## λ sensitivity (diagnostic; λ is not chosen on these scores)

Synapse dropout, q = 0.7, 3 seeds, raw levels against the true rate:

| λ | AUC naive / PU | Mean level naive / PU | Top reliability bin naive / PU (true rate) | Recall of deleted, top 1% |
|---|---|---|---|---|
| 10⁻² | 0.922 / 0.923 | −29% / +11% | 0.008 / 0.016 (0.031) | 0.18 |
| 10⁻³ | 0.932 / 0.934 | −18% / +26% | 0.021 / 0.035 (0.033) | 0.23 |
| 10⁻⁴ | 0.942 / 0.943 | −10% / +37% | 0.028 / 0.044 (0.034) | 0.25 |

- **The naive model's under-confidence in `reports/benchmark.md` was mostly regularisation.** As λ falls from 10⁻² to 10⁻⁴, its top bin moves from 0.008 to 0.028 against a true 0.034, the AUC rises from 0.92 to 0.94, and recall of deleted connections rises from 0.18 to 0.25. PU's top bin moves from 0.016 to 0.044, overshooting the true 0.034. That is the same held-out-type offset, so this table alone cannot show that PU is calibrated.
- **The raw levels move with λ because the held-out-type offset does.** Read them through the relative levels above.
- **Elkan–Noto e1 rises with lower λ** (0.07–0.10 at λ = 10⁻⁴, against c = 0.66), but stays far below c.
- **λ for the final design** is chosen by inner cross-validation on training pairs. These held-out scores only show that the default of 10⁻² is too strong.
