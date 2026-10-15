# Homologue-group fold benchmark

`scripts/benchmark_folds.py`: 5 folds of homologue groups (worksheet D18: a cell type across sides is one group; untyped neurons are their own group; 238 groups for the 2,875-neuron roster). Each roster neuron is a held-out source exactly once per seed. Held-out target features are built on the training graph (`target_graph="train"`, `reports/benchmark_offset.md`). Conditions: no thinning and q = 0.7 synapse dropout. Models: naive and PU at the true c. Fold sizes for seed 0 are 461, 522, 589, 610 and 693 neurons; groups differ in size, so folds do too.

## 1. First run: λ by inner CV, no clipping, seed 0 (`reports/benchmark_folds_cv_noclip.json`)

| Fold | Held-out sources | λ (q = 1 / 0.7) | Offset without thinning | Naive, relative level (q = 0.7) | PU at true c, relative level | AUC (q = 1) |
|---|---|---|---|---|---|---|
| 0 | 461 | 10⁻⁵ / 10⁻⁶ | 0.922 | 0.556 | 0.841 | 0.911 |
| 1 | 522 | 10⁻⁴ / 10⁻⁵ | 0.969 | 0.624 | 0.946 | 0.936 |
| 2 | 589 | 10⁻⁴ / 10⁻⁵ | 1.164 | 0.792 | 1.146 | 0.923 |
| 3 | 610 | 10⁻⁵ / 10⁻⁵ | 1.882 | 0.884 | 1.065 | 0.954 |
| 4 | 693 | 10⁻⁵ / 10⁻⁶ | 1.077 | 0.627 | 0.961 | 0.943 |

Realised c in training: 0.649–0.660. Pooled over folds (Σ weight · mean P / Σ weight · true rate), the offset is 1.234; the pooled level is 0.922 for the naive model and 1.268 for PU at q = 0.7.

The offset varies from −8% to +88% across folds. The relative levels vary too (naive 0.56–0.88 against c ≈ 0.65), so dividing by a same-fold baseline no longer removes it. Fold 3 is the outlier.

## 2. Diagnosis: one type, one rare feature (`scripts/fold_offset_by_type.py`)

The fold-3 excess was split by source cell type, at a fixed λ = 10⁻⁴ to separate the fold from the λ choice (`reports/fold_offset_by_type.json`). One type, **ExR8**, accounts for about 90% of it: 6,021 predicted connections (weighted) against 0 true ones, an excess equal to 85% of all the fold's true connections (the fold's offset at this λ is 1.94).
- ExR8 has 4 neurons. They make most of their output synapses outside the central complex (IPS 1,972, LAL 645, WED 248) and about 60 onto roster neurons inside it.
- In training, almost no source has output in IPS_R, so the standardised `frac_out_IPS_R` has a tiny training SD. For ExR8 it reaches z ≈ 2,273, against a training maximum of 43. A coefficient of only 0.016 then adds about +35 to the logit. The model extrapolates linearly far outside its data, at λ = 10⁻² as well as 10⁻⁴; the mean P(y=1) of ExR8 pairs is 0.48 at both. (These feature-level values come from an interactive check of the fitted fold-3 model; they are not stored in the JSON files, which hold per-type aggregates only.)

**Fix:** `design(clip=True)` limits held-out numerics to the training range before standardising (`run_condition(clip=True)`). Training rows are unchanged, and so is any held-out value inside the range. With clipping (λ = 10⁻⁴, `reports/fold_offset_by_type_clip.json`), the fold offsets become 0.935, 0.961, 1.142, 0.990 and 1.160: fold 3 falls from 1.94 to 0.99. The remaining excess sits in a few large held-out types whose wiring differs from what training types predict: EPG in fold 2 (2,002 predicted against 895 true) and PFNp in fold 4 (1,082 against 161).

Clipping matters only when features are built for sources the model has not seen. A model fitted on all data and applied to the same pairs never extrapolates in this way.

## 3. Clipped benchmark

### 3.1 Fixed λ = 10⁻⁴, seeds 0–2 (`reports/benchmark_folds_0.0001.json`, summary `scripts/benchmark_folds_summary.py`)

Each seed draws its own fold assignment, so the 15 folds differ across seeds.

| Seed | Offset without thinning, pooled (fold range) | Naive, relative level (q = 0.7) | PU at true c, relative level | Brier naive / PU (q = 0.7) | Mean AUC naive / PU (q = 0.7) | Realised c (train) |
|---|---|---|---|---|---|---|
| 0 | 1.048 (0.935–1.160) | 0.665 | 1.009 | 0.0038 / 0.0040 | 0.926 / 0.928 | 0.654 |
| 1 | 1.030 (0.934–1.207) | 0.656 | 0.994 | 0.0038 / 0.0039 | 0.927 / 0.929 | 0.655 |
| 2 | 1.040 (0.832–1.265) | 0.678 | 1.027 | 0.0038 / 0.0039 | 0.935 / 0.937 | 0.657 |

Relative level = pooled level at q = 0.7 divided by the pooled no-thinning naive level of the same seed. Without thinning, AUC ranges 0.907–0.961 over the 15 folds.

- **Offset.** With clipping, single-fold offsets lie between 0.83 and 1.27 and the pooled offset is +3% to +5%. The offset is a property of which types are held out, not a constant: within a fold, dividing by that fold's own no-thinning level gives naive 0.60–0.71 and PU 0.91–1.09.
- **Level (H2).** Pooled per seed, the naive level falls to 0.66–0.68 of baseline (realised c ≈ 0.655) and PU at the true c restores it to 0.99–1.03. PU's fold-level ratio is closer to 1 than naive's in 13 of 15 folds.
- **Brier (H2).** PU at the true c has the *higher* Brier score in 13 of 15 folds (mean difference naive − PU = −1.1·10⁻⁴, SD 1.5·10⁻⁴; sign test p = 0.007). The top tenth of pairs by prediction (pooled reliability, seed 0) shows why: naive predicts 0.0215 against an observed 0.0298, PU 0.0335 against 0.0302. PU's error there is smaller in level but the correction is applied to every pair, and lower bins are already near the truth for both.
- **Ranking (H3).** The Spearman correlation between naive and PU scores on held-out unlabelled pairs is at least 0.9991 in every fold.

One candidate reason for the Brier result is that synapse dropout breaks SCAR. A pair with n true synapses stays recorded at τ = 10 with probability P(Binomial(n, 0.7) ≥ 10): 0.03 at n = 10, 0.25 at 12, 0.72 at 15, 0.97 at 19 and ≈ 1 from 25. Strong connections, which the model ranks highest, keep their labels; weak ones lose them. Dividing every prediction by one global c then over-corrects the top of the ranking and under-corrects the weak positives. The predictions agree with this. Per true stratum, pooled over the 15 folds, the naive model's mean prediction at q = 0.7 divided by its mean prediction without thinning (same fold) is 0.66 for true zeros, 0.68 for 1–9 synapses, 0.69 for 10–19, 0.73 for 20–49 and 0.79 for 50+ (single folds 0.63–0.96 for 50+). For the strongest pairs, then, the thinned model shrinks less than c ≈ 0.655, so dividing by c lifts them about 20% above the no-thinning model. The earlier random-type-split benchmark (λ = 10⁻², `reports/benchmark.md`) found PU with the lower Brier in 18 of 20 runs; the difference in λ and in the split may explain the change, but this has not been tested. Section 3.3 runs the SCAR-respecting control (connection dropout).

The 3-split result of `reports/benchmark_offset.md` (offset +5% with `target_graph="train"`) was computed without clipping.

### 3.2 λ by 3-fold inner CV with the 1-SE rule, seed 0 (`reports/benchmark_folds_cv3.json`)

`select_lambda_kfold`: grouped 3-fold CV on the training pairs (log-loss for s), then the largest λ whose mean loss is within one standard error of the best. Same folds as seed 0 in 3.1.

- It chooses 10⁻⁴ in 7 of 10 runs and 10⁻⁵ in 3 (fold 0 and fold 1 at q = 0.7, fold 2 at q = 1). The single-split CV of section 1 chose 10⁻⁵ or 10⁻⁶ in 8 of 10, so the 1-SE rule moves the choice toward stronger regularisation, as intended.
- Where it differs from the fixed 10⁻⁴, the naive AUC changes by +0.006, +0.002 and −0.003 (PU: +0.005, +0.002, −0.003), and the Brier of PU at the true c rises each time (by 2.6–13.5·10⁻⁵).
- Pooled: offset 1.070 (fold range 0.935–1.223), naive relative level 0.649, PU at true c 0.983; PU has the higher Brier in all 5 folds and the level closer to 1 in all 5.

Tuning λ by inner CV gives no gain over a fixed 10⁻⁴ here. One likely reason: the inner folds group pairs by source neuron, while the outer folds hold out whole homologue groups. Inner validation neurons share cell types with the inner training set, so the inner loss is optimistic for the type-level task and favours small λ. Grouping the inner folds by homologue group would match the two (tested in 3.6). The fixed value is kept for the benchmark; the final real-data fit can use `lam="cv3"`, which lands at or near it.

### 3.3 Connection dropout (SCAR holds by construction), λ = 10⁻⁴, seeds 0–2 (`reports/benchmark_folds_0.0001_connection.json`)

Here each true connection loses its label with probability 0.3, whatever its strength. The folds and the no-thinning rows are the same as in 3.1 (same seeds).

| Seed | Naive, relative level | PU at true c, relative level | Brier naive / PU | Realised c (train) |
|---|---|---|---|---|
| 0 | 0.716 | 1.013 | 0.0038 / 0.0039 | 0.704 |
| 1 | 0.709 | 1.008 | 0.0038 / 0.0038 | 0.700 |
| 2 | 0.709 | 1.013 | 0.0038 / 0.0038 | 0.697 |

- The naive model's shrinkage is now flat across true strata: 0.71, 0.71, 0.72, 0.72 and 0.70 (true zeros to 50+). Under synapse dropout it rose from 0.66 to 0.79. So the shrinkage pattern of 3.1 is due to the SCAR violation.
- PU at the true c still has the higher Brier in 13 of 15 folds (mean difference −6.6·10⁻⁵, SD 1.0·10⁻⁴; sign test p = 0.007), and its level is closer to 1 in 11 of 15. **SCAR violation does not explain PU's higher Brier.**

**Comparison with the model fitted on all labels.** Each fold also has a model fitted without thinning (q = 1). Its mean Brier over the 15 folds is 0.003988. In 15 paired comparisons with it:

| Dropout | Naive at q = 0.7: lower Brier than q = 1 | PU at true c: lower Brier than q = 1 | Mean Brier naive / PU |
|---|---|---|---|
| synapse | 10 of 15 (p = 0.30) | 1 of 15 (p = 0.001) | 0.003954 / 0.004069 |
| connection | 13 of 15 (p = 0.007) | 10 of 15 (p = 0.30) | 0.003920 / 0.003986 |

Under connection dropout, PU at the true c matches the fully labelled model in Brier (0.003986 against 0.003988). The naive model is *better* than both. Its predictions are scaled down by about c, and on held-out types that scaling lowers squared error. The fully labelled model does not predict held-out types well enough for its correct level to pay off in Brier: in one seed-0 fold its top bin predicts 0.049 against an observed 0.039. Shrinking toward zero reduces the squared error of such over-confident predictions more than the level bias it adds.

Under synapse dropout, PU is also worse than the fully labelled model (14 of 15 folds). That extra cost is consistent with the SCAR violation: dividing by the global c over-corrects strong pairs.

**Consequence for H2.** In these folds the Brier score rewards shrinkage, so "PU lowers Brier relative to naive" is not a reliable test of correct level. The level ratio is: PU restores it to 0.99–1.03 under both dropout schemes. The proposal's H2 should report level (and reliability) as the primary outcome and Brier as secondary. Brier should be compared with the fully labelled model, not only with naive. This rests on q = 0.7 (c ≈ 0.65–0.70), λ = 10⁻⁴ and 3 seeds. With heavier thinning the cost of the naive model's shrinkage grows roughly as (1 − c)², so the Brier ordering may reverse; the earlier random-split result at λ = 10⁻² (PU lower in 18 of 20) points the same way. Section 3.4 tests the mechanism directly.

### 3.4 Brier-optimal scale, seed 0 (`reports/benchmark_folds_0.0001_scale.json`, `reports/benchmark_folds_0.0001_connection_scale.json`)

The seed-0 runs of 3.1 and 3.3 were repeated with `run_condition` now also storing, per model, the common factor α* = Σ a·f·y / Σ a·f² that minimises the weighted Brier score of α·f. The rerun reproduces every stored Brier score exactly. α* < 1 means scaling the predictions down would lower the Brier score. α* times the model's level gives the level (predicted total over true total) at which Brier is lowest.

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Fully labelled model (q = 1): level | 0.935 | 0.961 | 1.142 | 0.990 | 1.160 |
| Fully labelled model: α* | 0.677 | 0.739 | 0.316 | 0.827 | 0.628 |
| Fully labelled model: Brier-optimal level, α* · level | 0.63 | 0.71 | 0.36 | 0.82 | 0.73 |
| Connection dropout, PU at true c: α* | 0.715 | 0.730 | 0.315 | 0.819 | 0.635 |
| Connection dropout, naive: α* | 1.016 | 1.064 | 0.393 | 1.100 | 0.908 |
| Synapse dropout, PU at true c: α* | 0.635 | 0.658 | 0.271 | 0.610 | 0.586 |
| Synapse dropout, naive: α* | 0.961 | 1.004 | 0.350 | 0.828 | 0.907 |

- Even the model fitted on all labels, whose level is 0.94–1.16, would have its Brier score lowered by scaling every prediction by 0.32–0.83. On held-out types the Brier score is minimised by predicting 36–82% of the true number of connections. Its ranking is good (AUC 0.91–0.95) but its probabilities spread too widely: too high at the top, where the squared error is largest. Fold 2, which holds out EPG, is the extreme.
- Under connection dropout, PU at the true c has almost the same α* as the fully labelled model in every fold (difference at most 0.04). It reproduces that model up to estimation noise, and inherits its spread. The naive model sits near α* ≈ 1 because its shrinkage by c ≈ 0.70 happens to land near the Brier-optimal scale.
- Under synapse dropout, PU's α* is lower than the fully labelled model's (0.27–0.66 against 0.32–0.83): the over-correction of strong pairs from 3.1.

So the Brier comparison of 3.1 and 3.3 measures whether a model's spread happens to be offset by shrinkage, not whether its level is right. A model with well-spread probabilities on held-out types would need recalibration (a fitted slope as well as a level), which is outside the PU correction. For H2 this supports level and reliability as primary outcomes, with the reliability slope reported as its own result.

### 3.5 Fixed λ = 10⁻², seed 0 (`reports/benchmark_folds_0.01.json`)

Same seed-0 folds, synapse dropout, λ = 10⁻² (the value of the earlier random-split benchmark).

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Level without thinning | 0.937 | 1.067 | 0.751 | 0.945 | 1.530 |
| AUC without thinning (λ = 10⁻⁴: see 3.1) | 0.883 | 0.926 | 0.889 | 0.945 | 0.915 |
| α* without thinning | 1.47 | 2.01 | 1.89 | 2.31 | 1.31 |
| q = 0.7: α* naive / PU at true c | 2.54 / 1.46 | 3.04 / 2.00 | 3.16 / 1.59 | 3.82 / 2.14 | 1.97 / 1.23 |

- Stronger regularisation reverses the spread: α* is now 1.3–2.3 without thinning, so the predictions are too narrow and Brier would fall if they were scaled *up*.
- Correspondingly, PU at the true c has the lower Brier than naive in 5 of 5 folds (mean difference naive − PU = +4.6·10⁻⁵; sign test p = 0.06 with 5 folds) and also beats the fully labelled model in 5 of 5. Its pooled relative level is 1.002 (naive 0.656). This matches the random-split result at λ = 10⁻² (PU lower in 18 of 20).
- The cost is ranking: mean AUC without thinning is 0.912 against 0.931 at λ = 10⁻⁴ (lower in every fold), and the fold offsets spread more widely (0.75–1.53, pooled 1.013).
- Under synapse dropout, the naive model's shrinkage by true stratum now *falls* with strength (0.67 for true zeros, 0.55 for 1–9 synapses, 0.50–0.51 for the positive strata), the opposite of λ = 10⁻⁴. The stratum pattern depends on λ, so it is not a clean read-out of the SCAR violation by itself.

### 3.6 λ by 3-fold inner CV grouped by homologue group, seed 0 (`reports/benchmark_folds_cv3h.json`)

`lam="cv3"` with `inner_groups="homologue"`: the inner folds hold out whole homologue groups, as the outer folds do (Fable's review of 3.2).

- It chooses λ = 10⁻³ in 8 of 10 runs and 10⁻² in fold 0 (both q). That is one to two decades stronger than with inner folds by source neuron (3.2: 10⁻⁴ or 10⁻⁵). So the source-grouped inner loss was optimistic for the type-level task, as expected.
- Mean AUC without thinning is 0.920, between λ = 10⁻⁴ (0.931) and 10⁻² (0.912). Pooled offset 1.018 (folds 0.92–1.32), naive relative level 0.659, PU at true c 1.012.
- α* without thinning is 1.47, 1.02, 0.58, 0.97 and 0.77: close to 1 in two folds. No Brier preference is detected (PU lower in 2 of 5, mean difference −2.4·10⁻⁵, sign test p = 1.0).

With a λ chosen on held-out types, α* lies on both sides of 1 and this comparison detects no Brier preference between naive and PU (5 folds, one seed). That is a preliminary result, not evidence of equivalence or of well-calibrated spread. AUC is lower than at λ = 10⁻⁴. If ranking matters most (H3, the H4 candidate list), λ = 10⁻⁴ ranks best; if probabilities matter (H2), the type-grouped choice is better matched to the task. The proposal's final design should state which one it uses, and why.

Taken together (3.1–3.6): changing λ reverses the Brier ordering of naive and PU, in step with whether the base model's predictions on held-out types are too wide (α* < 1) or too narrow (α* > 1). These comparisons (sections 3.4–3.6 on seed 0 only, all at q = 0.7) do not show that λ is the only factor; they show that a Brier difference between naive and PU cannot be read as evidence for or against the PU correction without checking the spread. PU at the true c restores the level in every setting tried (pooled relative level 0.98–1.03).
