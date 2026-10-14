# H4 pilot: hemibrain type-level check of high-scoring unrecorded pairs

`scripts/h4_pilot.py` → `reports/h4_pilot.json` (3 minutes). The proposal gates H4 behind the core results (go/no-go at week 9), so this is an early pilot on one sample and one fit.

**Setup:**
- **Scores:** the real-data fit of `reports/h3_real.md` (same sample, features and λ = 10⁻⁴), PU at c = 0.7. Its ranking is nearly the naive one.
- **Eligible pairs:** sampled unlabelled FAFB pairs with both neurons on the right side and both types in a match group that also occurs on the hemibrain's right side (FAFB `hemibrain_type` → hemibrain `cell_type`, 177 groups). That leaves 29,561 of the 321,982 sampled unlabelled pairs.
- **Hit:** the pair's type pair is recorded at τ in the hemibrain: some neuron pair of those types has ≥ τ synapses in the central-complex neuropils. Both τ = 10 and τ = 20 are used; 20 is τ scaled by the hemibrain/FAFB type-pair total ratio of 2.0.
- **Candidates:** the top k eligible pairs by score.
- **Controls:** the other eligible pairs, standardised to the candidates' strata (`src.h4.matched_hit_rate`):
  - **A**, the proposal's matching: type-pair size decile × degree decile;
  - **B**: A × whether FAFB itself records the type pair at τ through other neuron pairs;
  - **C**: B × same type.

  A candidate in a stratum without controls is dropped from both rates; the n column counts those kept.

**Base rates among eligible pairs:**
- hemibrain hit: 5.2% at τ = 10, 2.7% at τ = 20;
- type pair recorded in FAFB: 4.2%;
- same type: 1.3%.

## Hit rates at τ = 10

| k | Type pair recorded in FAFB | Same type | 1–9 synapses in FAFB | A: candidates / controls | B: candidates / controls (n) | C: candidates / controls (n) |
|---|---|---|---|---|---|---|
| 100 | 42% | 36% | 34% | 0.470 / 0.086 | 0.470 / 0.403 (100) | 0.286 / 0.246 (70) |
| 500 | 25% | 17% | 27% | 0.320 / 0.060 | 0.320 / 0.242 (500) | 0.265 / 0.199 (452) |
| 1000 | 24% | 15% | 25% | 0.304 / 0.045 | 0.304 / 0.225 (1000) | 0.256 / 0.180 (900) |
| 5000 | 13% | 5% | 14% | 0.167 / 0.019 | 0.145 / 0.088 (4854) | 0.129 / 0.069 (4696) |

## Hit rates at τ = 20 (scaled)

| k | Type pair recorded in FAFB | Same type | 1–9 synapses in FAFB | A: candidates / controls | B: candidates / controls (n) | C: candidates / controls (n) |
|---|---|---|---|---|---|---|
| 100 | 42% | 36% | 34% | 0.400 / 0.059 | 0.400 / 0.358 (100) | 0.186 / 0.180 (70) |
| 500 | 25% | 17% | 27% | 0.206 / 0.036 | 0.206 / 0.173 (500) | 0.153 / 0.126 (452) |
| 1000 | 24% | 15% | 25% | 0.194 / 0.026 | 0.194 / 0.154 (1000) | 0.149 / 0.122 (900) |
| 5000 | 13% | 5% | 14% | 0.101 / 0.010 | 0.088 / 0.062 (4854) | 0.077 / 0.053 (4696) |

n: candidates kept in B and C; a candidate whose stratum has no control is dropped, which can change the candidate rate.

## Candidate / control ratios with cluster-bootstrap intervals

500 replicates resampling whole clusters, the source neuron's type group (177 clusters; hits cluster by type pair), with the candidate set fixed (the intervals condition on the original top-k selection) and the matched rates recomputed from the resampled pairs in each (`src.h4.cluster_bootstrap`); 95% percentile intervals. All 500 replicates were valid in every cell. The intervals are skewed to the right of the point estimate, as usual for a ratio.

| k | τ | A: ratio (95% interval) | B: ratio (95% interval) | C: ratio (95% interval) |
|---|---|---|---|---|
| 100 | 10 | 5.48 (2.63–12.77) | 1.17 (1.03–1.68) | 1.16 (0.88–1.66) |
| 100 | 20 | 6.78 (2.86–25.51) | 1.12 (1.00–1.62) | 1.03 (0.67–1.63) |
| 500 | 10 | 5.31 (3.44–8.13) | 1.32 (1.15–1.95) | 1.34 (1.14–1.93) |
| 500 | 20 | 5.66 (3.16–10.37) | 1.19 (0.95–2.13) | 1.21 (0.84–1.89) |
| 1000 | 10 | 6.75 (4.37–10.00) | 1.35 (1.18–2.00) | 1.42 (1.19–1.99) |
| 1000 | 20 | 7.48 (4.25–14.15) | 1.26 (0.97–2.09) | 1.22 (0.90–1.87) |
| 5000 | 10 | 8.57 (5.02–16.25) | 1.65 (1.36–3.64) | 1.87 (1.48–3.96) |
| 5000 | 20 | 10.21 (4.48–29.65) | 1.42 (1.01–2.93) | 1.47 (1.01–3.51) |

**What it shows:**
- **Against the proposal's matching (A), the enrichment is large.** At k = 1000, 30% of candidates are hits against 4.5% of controls (19% against 2.6% at τ = 20).
- **Most of it is type-level knowledge already in FAFB.** The model learns type-pair connectivity from the FAFB positives (`spatial`, `same_type`, the type one-hots), and a type pair that FAFB records through other neurons is very likely recorded in the hemibrain too. Matching on that as well (B) leaves 0.304 against 0.225 at k = 1000 and 0.145 against 0.088 at k = 5000: 1.2–1.7 times the controls at τ = 10 over k = 100–5000. Matching also on same type (C) gives 1.2–1.9 times at τ = 10 and 1.0–1.5 times at τ = 20.
- **So the proposal's two matching variables are not enough.** H4 as pre-registered would mostly measure that the model has learnt FAFB's type-level wiring, which the hemibrain shares. The informative comparison is B or C: whether the model's high scores point to hemibrain connections beyond what FAFB's own type-level wiring predicts.
- **That residual is modest but resolved at τ = 10.** For k ≥ 500 the B and C intervals exclude 1 (lower ends 1.14–1.48; ratios 1.3–1.9). At k = 100, C includes 1. At the scaled τ = 20, the intervals reach 1 or below for k ≤ 1000 (lower ends 0.67–1.00) and only just exclude it at k = 5000 (1.01). The result therefore depends on how strict a hemibrain hit is, and it is one sample, one fit and one c.

**Limits:**
- The bootstrap resamples type groups of the source neuron only, not target groups. The design's neuron-clustered bootstrap over the whole pipeline (refitting the model) is not done here; the intervals take the scores as fixed.
- The hemibrain export is v1.2, not v1.2.1, and has no `cropped` flag. Cropped neurons can only lower hit rates, in candidates and controls alike.
- Only right-side pairs with mapped types are eligible (9% of the sampled unlabelled pairs). The candidate set is drawn from a sample of the unlabelled pairs (about 1 in 26).
- A hemibrain hit is type-level corroboration. It does not confirm a single missed FAFB connection, and a miss can be biological variation.
