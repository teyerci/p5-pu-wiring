# λ chosen by inner cross-validation

`scripts/benchmark_cv.py` → `reports/benchmark_cv.json` (9 runs: seeds 0–2 × no thinning, q = 0.7 synapse and connection dropout; 88 minutes alongside other runs).

**Method** (`src.benchmark.select_lambda`): on the training pairs only, 20% of the source neurons are held out (one split, seeded), a naive model is fitted on the rest for each λ in `LAMBDA_GRID` = 10⁻², 10⁻³, …, 10⁻⁷, and λ minimises the weighted log-loss for the observed label s on the held-out sources. The true label is never used, so the same procedure runs on real data. The chosen λ is then used for every model of that run. These runs use full-graph target features for the held-out pairs (see `reports/benchmark_offset.md`), so their offsets are inflated by that shift; the relative levels divide it out. These runs predate the `target_graph`, `fold` and `test_sources` fields, which are absent from this JSON.

A first run with the grid ending at 10⁻⁵ chose its end point in 8 of 9 runs, so the grid was extended to 10⁻⁷.

## Chosen λ

| Seed | No thinning | Synapse q = 0.7 | Connection q = 0.7 |
|---|---|---|---|
| 0 | 10⁻⁵ | 10⁻⁵ | 10⁻⁵ |
| 1 | 10⁻⁶ | 10⁻⁴ | 10⁻⁶ |
| 2 | 10⁻⁶ | 10⁻⁷ | 10⁻⁶ |

- The minimum is inside the grid in 8 of 9 runs. In the ninth (seed 2, synapse) the loss is flat at the end (0.01042 at 10⁻⁶, 0.01041 at 10⁻⁷).
- Every choice is far below the earlier default of 10⁻², confirming that 10⁻² was too strong.
- **Below the optimum the loss either stays flat or rises steeply.** In 5 of 9 runs (seed 1 without thinning and with connection dropout, all three seed-2 runs) every λ from 10⁻⁵ to 10⁻⁷ is within 1.3% of the minimum, so the choice among them is noisy. In the other 4 (all seed-0 runs, seed 1 synapse) the loss at 10⁻⁷ is between 14% above and 8.9 times the minimum (seed 0 synapse: 0.0114 at 10⁻⁵, 0.101 at 10⁻⁷), a sign of over-fitting rare positives. Above the optimum it rises smoothly in every run (10⁻² is 22–32% worse).

## Results (relative to the no-thinning naive model of the same seed and split)

| Condition | Realised c (training) | Naive, relative level | PU at true c, relative level |
|---|---|---|---|
| synapse q = 0.7 | 0.653–0.658 | 0.644 (0.628–0.675) | **0.978** (0.936–1.036) |
| connection q = 0.7 | 0.696–0.704 | 0.718 (0.714–0.720) | **1.002** (0.992–1.011) |

Mean over seeds (range). All fits converged.

- **The relative levels repeat the fixed-λ result** (`reports/benchmark_relative.md`): the naive level falls by about c, and PU at the true c recovers it, here to within about 2% on average and about 6% per seed.
- **CV-chosen λ does not beat a fixed 10⁻⁴ on held-out pairs.** Against the λ = 10⁻⁴ runs of `reports/benchmark_relative.json` (same seeds and splits), PU's AUC is within 0.003 in 5 of 6 thinned runs, and lower by 0.010 in seed 2 under synapse dropout, where CV chose 10⁻⁷ (Brier 0.0049 against 0.0045). With one inner split and a flat loss, CV can pick a λ that is too small. The inner protocol also differs from the outer one: inner validation rows are standardised with statistics that include them, and their target features come from the full training graph (so they include other inner-validation sources' edges). Neither carries an outer held-out label, but the inner split does not reproduce the outer shift.
- **The held-out-type offset is +29% to +31%** without thinning at these λ. That is the target-feature shift explained in `reports/benchmark_offset.md`, not a CV effect.

**For the design:** keep inner CV, but make it more stable before relying on it: more than one inner split (e.g. 3 grouped folds), and the largest λ whose loss is within a small tolerance of the minimum, rather than the bare minimum. Until then, the fold benchmark uses CV as implemented here, and its results should be read with this caveat.
