# Pilot: naive vs PU models on the central-complex roster (fafb 783, τ = 10)

`scripts/pilot_pu.py` → `reports/pilot_pu.json` (about 2.5 minutes, 10 GB). A first end-to-end run, not the final design: one split, λ not tuned, no bootstrap, and c unknown, so the PU fits are a sensitivity grid.

**Setup.**
- **Sample:** all 32,125 positives plus 10 unlabelled pairs per positive (ρ = 0.039, weight 1/ρ; `src/pairs.py`).
- **Features:** leave-pair-out pair features (`src/pair_features.py`; 570 columns with every group).
- **Split:** 20% of the roster's cell types are held-out sources: 46 types, 95,839 test pairs, 8,176 test positives. **Split first:** training features are built on a graph from which every held-out candidate pair's synapses are removed, in all neuropils. Neither the training rows nor the standardisation can then carry a held-out label (`split_features`, with an invariance test). Held-out features use the full graph, leave-pair-out, as at prediction time. Target neurons are shared between train and test, a pilot simplification. The final design uses homologue-group folds and the neuron-clustered bootstrap.
- **Fit:** λ = 10⁻², inverse-inclusion weights.
- **Metrics** (on the test set, weighted):
  - AUC and Brier score for the observable label s, using the model's P(s=1|x) (naive g, PU c·f);
  - the **below-τ share among the top 1% of unlabelled pairs**. Synapse counts are never features, so this is an internal check: pairs recorded with 1–9 synapses are the most plausible missed connections. Their base share among all unlabelled pairs is 2.8%.

## Feature groups (naive model, added one at a time)

| Groups | Columns | AUC (s) | Brier (s) | Below-τ share, top 1% |
|---|---|---|---|---|
| identity | 18 | 0.707 | 0.00359 | 13.2% |
| + cell type | 432 | 0.721 | 0.00359 | 18.7% |
| + region profile | 567 | 0.778 | 0.00459 | 13.0% |
| + degree | 569 | 0.891 | 0.00370 | 22.4% |
| + spatial proxy | 570 | **0.924** | **0.00354** | **26.1%** |

## Naive vs PU (all groups)

| c | π | AUC (s) | Brier (s) | Mean P(y=1) on unlabelled pairs | Below-τ share, top 1% |
|---|---|---|---|---|---|
| 1 (naive) | 0.0040 | 0.924 | 0.00354 | 0.0049 | 26.1% |
| 0.9 | 0.0044 | 0.925 | 0.00353 | 0.0055 | 26.6% |
| 0.7 | 0.0057 | 0.927 | 0.00349 | 0.0072 | 27.7% |
| 0.5 | 0.0079 | 0.931 | 0.00343 | 0.0103 | 28.1% |

**What it shows (observed differences; causes are hypotheses until the final design tests them):**
- **The pipeline runs end to end.** Sampler, split-first leave-pair-out features, the NumPy PU fit (every fit converged in 8 Newton steps) and the metrics all work.
- **Adding groups raises the AUC for s on held-out source types.**
  - Identity alone gives 0.71, and cell type adds little (0.72).
  - Degree (0.89) and the spatial proxy (0.92) add the most.
  - A type-only model was not run, so this does not yet measure how much type alone predicts.
- **The model finds plausible missed connections without seeing synapse counts.** Among the top 1% of unlabelled pairs, 26–28% are recorded with 1–9 synapses, against a 2.8% base rate: about 9–10× enrichment. The share rises as c falls.
- **Lowering c raises the estimated probability of a true connection** among unlabelled pairs (0.0049 → 0.0103 at c = 0.5). On held-out types it slightly improves the AUC and the Brier score for s. This does not identify c: c needs the dropout benchmark, where the truth is known.
- **Every full-feature model over-predicts on held-out types:** mean P(s=1) is 0.0050–0.0054 against 0.0036 recorded (0.0044–0.0060 across the ablation steps).
  - Possible causes: a shift between source types; the lower training degrees that come from removing held-out edges; misspecification; or the untuned λ.
  - The final design checks calibration per fold.
- **Adding the region profile lowers the Brier score's quality** (0.00459) while raising the AUC, until degree is added. Overfitting of its 135 columns at the untuned λ is one hypothesis.

**Not yet done (by design):** λ tuning, homologue-group folds, the neuron-clustered bootstrap, the dropout benchmark (truth known), the hemibrain check of top-scored pairs.
