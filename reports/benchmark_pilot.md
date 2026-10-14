# Dropout benchmark, first run (one seed, one split)

*Superseded by `reports/benchmark.md` (5 seeds, all deleted connections scored). Kept as the record of the first run.*

`src/benchmark.py`, `scripts/benchmark_pilot.py` → `reports/benchmark_pilot.json` (about 5 minutes, 9 GB).

**Setup.**
- The fafb central-complex roster's recorded synapses are the **truth**: y = 1 when a pair has ≥ 10 synapses over PB, FB, EB and NO.
- A **thinned copy** is what the models see:
  - synapse dropout keeps each synapse with probability q;
  - connection dropout keeps each pair whole with probability q (the SCAR control).
- s is relabelled at τ = 10 on the thinned copy.
- Pairs are sampled from s (all positives plus 10 unlabelled per positive), and features are rebuilt on the thinned graph.
- Split first, as in the pilot: 20% of roster types are held-out sources, and training features never contain a held-out pair's synapses.
- **Models:**
  - naive (c = 1);
  - PU at the **true realised c** (known here);
  - PU at the **Elkan–Noto e1** estimate, from a naive model's mean score over held-out training positives.
- λ = 10⁻². All three are scored against y on the held-out pairs.

| Condition | Realised c | e1 | Model | AUC (y) | Brier (y) | Mean P(y=1) | True rate |
|---|---|---|---|---|---|---|---|
| no thinning | 1.00 | 0.027 | naive | 0.910 | 0.00372 | 0.0044 | 0.0038 |
| synapse, q = 0.9 | 0.885 | 0.022 | naive | 0.913 | 0.00381 | 0.0038 | 0.0039 |
| | | | PU, true c | 0.913 | 0.00380 | 0.0043 | |
| | | | PU, e1 | 0.912 | 0.178 | 0.201 | |
| synapse, q = 0.7 | 0.654 | 0.015 | naive | 0.908 | 0.00388 | **0.0028** | 0.0039 |
| | | | PU, true c | 0.911 | 0.00387 | **0.0043** | |
| | | | PU, e1 | 0.909 | 0.208 | 0.228 | |
| connection, q = 0.9 | 0.901 | 0.022 | naive | 0.912 | 0.00363 | 0.0040 | 0.0037 |
| | | | PU, true c | 0.913 | 0.00363 | 0.0044 | |
| connection, q = 0.7 | 0.704 | 0.014 | naive | 0.917 | 0.00390 | **0.0030** | 0.0039 |
| | | | PU, true c | 0.918 | 0.00388 | **0.0043** | |
| | | | PU, e1 | 0.917 | 0.253 | 0.273 | |

**What it shows (one seed; the full benchmark repeats over 5 seeds with proper folds):**
- **The naive model underestimates the connection probability when recording is lossy.**
  - At q = 0.7 its mean P(y=1) is 0.0028–0.0030 against a true 0.0039.
  - PU at the true c restores the level (0.0043) and slightly improves the Brier score.
  - The ranking (AUC) barely changes: the correction mainly rescales.
  - This is the direction H2 predicts. Its size needs the 5-seed run.
- **Elkan–Noto e1 fails badly here (H1).**
  - It gives c ≈ 0.014–0.027 in every condition, including no thinning, where the true c is 1.
  - With 0.4% prevalence the naive model's scores on true positives are low, so their mean is far below c. That is the non-separable case, where e1 is only a lower bound, and here a uselessly loose one.
  - A PU fit at that c inflates the mean probability 47–63-fold.
  - So c must not be taken from e1 on this problem. The proposal's plan stands: performance at known c, estimator bias on the benchmark, and a sensitivity range of c on real data.
- **SCAR violation.** Synapse dropout (realised c 0.654 at q = 0.7) and connection dropout (0.704) give similar results at this level of detail. The cost of the violation should show in per-stratum calibration, which the full benchmark reports.
- **Recall of deleted connections** (true positives that the thinning made unlabelled, ranked in the top 1% of unlabelled held-out pairs) is reported in the JSON. It rests on only 15–55 sampled deleted pairs per condition, too few to compare models. The full benchmark scores all deleted connections, not only the sampled ones.
