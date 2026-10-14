# Dropout benchmark (5 seeds): naive vs PU when the truth is known

`src/benchmark.py`, `scripts/benchmark_run.py` → `reports/benchmark.json` (21 runs, about 20 minutes); `scripts/benchmark_summary.py` → `reports/benchmark_summary.csv` and figures `B1_…`–`B3_…` in `figures/eda/`; `scripts/benchmark_paired.py` → `reports/benchmark_paired.csv`. This supersedes the one-seed first run (`reports/benchmark_pilot.md`).

**Design.**
- The fafb central-complex roster's recorded synapses are the **truth**: y = 1 when a pair has ≥ 10 synapses over PB, FB, EB and NO.
- Models see a **thinned copy**:
  - synapse dropout keeps each synapse with probability q;
  - connection dropout keeps each pair whole with probability q (the SCAR control).
- s is relabelled at τ = 10. Pairs are sampled from s (all positives plus 10 unlabelled per positive), and features are rebuilt on the thinned graph.
- **Split first:** 20% of roster types are held-out sources, and training features never contain a held-out pair's synapses.
- **Evaluation:** on held-out sources, **every deleted connection** (a true positive left unlabelled by the thinning) is scored with weight 1. The other sampled held-out pairs keep their sampling weights.
- **Models:**
  - naive (c = 1);
  - PU at the true realised c of the training sources;
  - PU at the Elkan–Noto e1, from a naive model fitted on 80% of training source neurons and averaged over the other 20%'s positives.
- λ = 10⁻² throughout, not tuned.

## Results (mean over 5 seeds; one run without thinning)

| Condition | Realised c | e1 | Model | AUC (y) | Brier (y) | Mean P(y=1) | True rate | Recall of deleted, top 1% |
|---|---|---|---|---|---|---|---|---|
| none | 1.00 | 0.026 | naive = PU true c | 0.910 | 0.00372 | 0.00437 | 0.00378 | — |
| synapse, q = 0.9 | 0.888 | 0.020 | naive | 0.928 | 0.00397 | 0.00387 | 0.00407 | 0.18 |
| | | | PU, true c | 0.928 | 0.00395 | 0.00439 | | 0.19 |
| synapse, q = 0.7 | 0.656 | 0.012 | naive | 0.923 | 0.00401 | **0.00281** | 0.00408 | 0.18 |
| | | | PU, true c | 0.924 | 0.00396 | **0.00438** | | 0.18 |
| connection, q = 0.9 | 0.900 | 0.019 | naive | 0.928 | 0.00471 | 0.00467 | 0.00407 | 0.25 |
| | | | PU, true c | 0.928 | 0.00469 | 0.00514 | | 0.25 |
| connection, q = 0.7 | 0.700 | 0.012 | naive | 0.926 | 0.00437 | 0.00351 | 0.00407 | 0.24 |
| | | | PU, true c | 0.928 | 0.00454 | 0.00504 | | 0.25 |

PU at e1 is in the CSV. Its Brier score is 0.18–0.27 and its mean P(y=1) 0.20–0.29 in every condition, 53–71 times the true rate. The standard deviation of the Brier score over seeds is about 0.0007–0.0013 for the naive and true-c models. Recall uses the weighted 99th percentile of the unlabelled scores as its threshold, and pairs tied with it are included.

Mean P(y=1) given to **true connections** (q = 0.7), by their true synapse count:

| Condition | Model | 10–19 | 20–49 | 50+ |
|---|---|---|---|---|
| synapse | naive | 0.0086 | 0.0116 | 0.0135 |
| synapse | PU, true c | 0.0186 | 0.0271 | 0.0329 |
| connection | naive | 0.0095 | 0.0123 | 0.0136 |
| connection | PU, true c | 0.0184 | 0.0251 | 0.0281 |

## Paired comparison (H2): same seed, both models

`reports/benchmark_paired.csv`. Brier difference = naive − PU at the true c (> 0: PU better).

| Condition | Mean difference | sd | Seeds where PU is better | Paired t-test p | Sign test p |
|---|---|---|---|---|---|
| synapse, q = 0.9 | +0.000018 | 0.000016 | 5 of 5 | 0.065 | 0.063 |
| synapse, q = 0.7 | +0.000056 | 0.000038 | 5 of 5 | 0.030 | 0.063 |
| connection, q = 0.9 | +0.000013 | 0.000016 | 4 of 5 | 0.150 | 0.375 |
| connection, q = 0.7 | −0.000176 | 0.000517 | 4 of 5 | 0.488 | 0.375 |

## Reliability (weighted, ten bins of approximately equal weight; figure B3)

Mean prediction / true rate in the lowest and the top bin, averaged over 5 seeds (one seed without thinning). Pairs are indivisible, so bin weights differ slightly; here each bin is within about 0.03% of its ideal weight.

| Condition | Model | Lowest bin | Top bin |
|---|---|---|---|
| none | naive = PU | 0.0013 / 0.0000 | 0.0126 / 0.0275 |
| synapse, q = 0.7 | naive | 0.0010 / 0.0000 | 0.0079 / 0.0309 |
| synapse, q = 0.7 | PU, true c | 0.0011 / 0.0000 | 0.0154 / 0.0311 |
| connection, q = 0.7 | naive | 0.0010 / 0.0000 | 0.0133 / 0.0309 |
| connection, q = 0.7 | PU, true c | 0.0011 / 0.0000 | 0.0211 / 0.0310 |

> **Update:** `reports/benchmark_relative.md` separates the recording loss from an offset that held-out source types carry even without thinning. Against a same-split baseline, the naive level falls by the factor c and PU at the true c recovers it to within about 3% on average (0.97–1.07 per seed). The apparent PU overshoot below is that offset. The under-confidence below is mostly the untuned λ.

## What it shows (hypotheses where stated)

- **At q = 0.7 the naive model under-predicts connections on average.**
  - Its mean P(y=1) is 31% low under synapse dropout and 14% low under connection dropout. At q = 0.9 the picture is mixed: 5% low under synapse dropout, 15% high under connection dropout.
  - PU at the true c lifts the mean to 8–26% above the truth across the thinned conditions.
  - Both models are already 16% high without thinning (one seed only). A shift between source types is one possible explanation; misspecification and the untuned λ are others. The H2 comparison is therefore provisional.
- **Paired over seeds, PU at the true c has the lower Brier score in 18 of 20 runs.**
  - The evidence is strongest under synapse dropout: 5 of 5 seeds at both q, paired t-test p = 0.03 at q = 0.7.
  - Five seeds give little power: the exact sign test cannot go below 0.063.
  - The negative mean at connection q = 0.7 comes from one seed. More seeds are needed before H2 can be claimed.
- **The correction rescales more than it reorders.**
  - The condition means of the AUC move by at most 0.002 (by about 0.003 at most within a seed).
  - Recall of deleted connections in the top 1% is similar for both models: 0.18 under synapse dropout, 0.25 under connection dropout.
- **Both models are under-confident** (reliability). Their predictions are compressed toward the mean: too high in the low bins, too low in the top bin.
  - In the top bin the true rate is about 0.03. The naive model predicts 0.008–0.013 and PU 0.015–0.021.
  - PU moves the top bin a third to a half of the way towards the truth.
  - The compression is also present without thinning, so heavy regularisation (λ = 10⁻², untuned) is a likely cause. Tuning λ is the next step.
- **True connections get low scores.** Mean P(y=1) on true connections is 0.01–0.03, rising with synapse count (table above); PU lifts every stratum about 2-fold.
  - Low scores on positives at a 0.4% base rate are not, by themselves, miscalibration; the reliability table is the calibration evidence.
  - The ranking is good (AUC ≈ 0.92), but the scores stay small.
- **The SCAR violation does not show at this level.** Strong connections get higher scores than weak ones under both dropout modes, and the PU lift is similar in both.
- **Elkan–Noto e1 is unusable here (H1).**
  - It gives 0.012–0.026, with e1/c ≈ 0.02 in every condition, including without thinning. That is c·E[f | y = 1] for scores that stay low on positives.
  - A PU fit at e1 raises the mean probability to 53–71 times the true rate.
  - Its higher recall of deleted connections (0.25–0.54) comes from a very different fit, at a much larger π = P(s=1)/e1 (roughly 0.15–0.25 from the condition means; the exact training π is stored from the next run on). It is not evidence that e1 works.

**Next:** tune λ (the under-confidence); more seeds for the paired test; homologue-group folds instead of one random type split; the realistic c range for the real data.
