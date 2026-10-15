# P5 — Positive–unlabelled wiring (CENG8004 Machine Learning)

Predicting connections in the fly connectome when "no recorded connection" does not mean "no connection". This repository follows the course's Week 2 Data Conventions and the Week 3 EDA layout, so the shared code runs on it unchanged.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # exact environment: requirements-lock.txt
cp .env.example .env                           # fill in tokens; .env is gitignored
.venv/bin/python -m pytest                     # 174 own tests (most for the review gate) + 15 eda_shared tests
.venv/bin/python scripts/build_fafb_cache.py   # downloads raw files once, builds data/cache/
.venv/bin/python scripts/build_malecns_cache.py  # same for MaleCNS (8.6 GB download, ~17 GB RAM)
```

Python 3.10.12. Packages: caveclient 8.2.1, fafbseg 3.2.2, neuprint-python 0.6.4, navis 1.12.0, pandas 2.3.3, pyarrow 25.0.1, numpy 2.2.6, networkx 3.4.2, scikit-learn 1.7.2, matplotlib 3.10.9, pytest 9.1.1.

Tokens live only in environment variables or `.env`: `NEUPRINT_APPLICATION_CREDENTIALS` (neuPrint) and `CAVE_SECRET` (FlyWire CAVE). They are never written to a notebook or committed.

## Layout

```
config.py              course file, copied verbatim from the Data Conventions; never edited
eda_shared/            course package v0.1.1-eda, unchanged
src/                   paths, cache helper, validators, fetch functions, summary,
                       EDA workstream E, step 15 (pairs.py, discrepancy.py),
                       PU model (pu.py), dropout thinning (dropout.py),
                       leave-pair-out pair features (pair_features.py),
                       dropout benchmark (benchmark.py)
scripts/               cache build, week 2 report, notebook generator
tests/                 tests for src/
notebooks/00_integration.ipynb
figures/eda/  reports/
data/raw/  data/cache/ gitignored
```

## Pinned releases (from config.py)

| Dataset | Release | Source |
|---|---|---|
| FAFB | materialisation 783, flywire_annotations v3.2.0, Buhmann synapses | Zenodo 10676866 (v783.0) bulk files; annotations from the v3.2.0 tag |
| MaleCNS | male-cns:v1.0, Traced, synapse confidence ≥ 0.5 | Janelia's public flat-connectome files for v1.0 (male-cns.janelia.org/download, `minconf-0.5` files); no token needed. Addresses in `src/paths.py` |
| Hemibrain | v1.2.1 pinned by the EDA plan; **v1.2 used** | Janelia's public v1.2 export of non-cropped Traced neurons (`exported-traced-adjacencies-v1.2`); v1.2.1 is only in neuPrint (token). Used in E6. See project decisions |

## Schema

The neuron and edge schemas are defined once, in `config.py` (`NEURON_SCHEMA`, `EDGE_SCHEMA`, `EDGEROI_SCHEMA`), and every validator in `src/validate.py` reads from there. Cache files:

```
data/cache/fafb_neurons_783_ann3.2.0.parquet
data/cache/fafb_edges_783_buhmann_min1.parquet
data/cache/fafb_edgeroi_783_buhmann_min1.parquet
data/cache/malecns_neurons_v1.0_traced.parquet
data/cache/malecns_edges_v1.0_traced_min1_conf0.5.parquet
data/cache/malecns_edgeroi_v1.0_traced_min1_conf0.5.parquet
data/cache/hemibrain_neurons_v1.2_traced-uncropped.parquet
data/cache/hemibrain_edges_v1.2_traced-uncropped_min1.parquet
data/cache/hemibrain_edgeroi_v1.2_traced-uncropped_min1.parquet
```

### FAFB source mapping (checked against the pinned files)

| Target | FAFB source | Note |
|---|---|---|
| neuron_id | `root_id` | int64 end to end |
| cell_type, cell_class, cell_sub_class | `cell_type`, `cell_class`, `cell_sub_class` | verbatim |
| super_class | `super_class` | `sensory_ascending` → `sensory` (provisional, see open issues) |
| side | `side` | `na` → `unknown` |
| region | per-neuron neuropil counts, pre + post | the neuropil with most synapses, from all synapses; `None` (outside any neuropil) is ignored; ties → alphabetically first (provisional, until the week 3 tie rule) |
| nt_type / nt_confidence | `top_nt` / `top_nt_conf` | missing → `unknown`, with confidence left missing |
| n_pre, n_post | not provided | left missing, as the conventions say |
| hemibrain_type | `hemibrain_type` | |
| dimorphism, status | `dimorphism`, `status` | |
| hemilineage | `ito_lee_hemilineage` | |
| nt_known | `known_nt` | verbatim string |
| pre_id / post_id / syn_count / neuropil | `pre_pt_root_id` / `post_pt_root_id` / `syn_count` / `neuropil` | summed over neuropils first, then thresholded |

Printed source columns, value counts and missing-value counts are in `reports/week2_fafb.md`.

### MaleCNS source mapping (checked against the pinned files)

Files: `body-annotations` (annotations), `body-neurotransmitters`, `body-stats` (per-neuron synapse-site counts), `connectome-weights` (pair weights) and `syn-partners` (one row per synaptic connection, with its neuropil `primary_post`). All are v1.0 at confidence ≥ 0.5 except the neurotransmitter file, which has no threshold.

| Target | MaleCNS source | Note |
|---|---|---|
| neuron_id | `bodyId` | int64; rows with `status == "Traced"` only (165,122) |
| cell_type, cell_class, cell_sub_class | `type`, `class`, `subclass` | verbatim |
| super_class | `superclass` | **provisional** mapping onto the FAFB levels, see open issues |
| side | `somaSide`, else `rootSide` | `L/R/M` → `left/right/center` |
| region | `syn-partners`, pre + post | the neuropil with most synaptic connections, from all partners (the FAFB rule); labels containing `unspecified` and missing labels are ignored; ties → alphabetically first. Names keep neuPrint's spelling (`ME(R)`) |
| nt_type / nt_confidence | `predicted_nt` / `predicted_nt_confidence` | `unclear` or no prediction → `unknown`, confidence left missing |
| n_pre, n_post | body-stats `pre`, `post` | missing for the 333 neurons with no synapse at confidence ≥ 0.5. `pre` counts presynaptic sites (T-bars), each with several partners, so it is not an output-connection total (median outgoing `syn_count` / `n_pre` ≈ 3); `post` counts postsynaptic sites and is comparable to input `syn_count` |
| flywire_type, manc_type, hemibrain_type | `flywireType`, `mancType`, `hemibrainType` | |
| dimorphism, status | `dimorphism`, `status` | |
| hemilineage | `itoleeHl` | |
| nt_known | neurotransmitter file `ground_truth` | verbatim; set for 83,496 Traced neurons |
| pre_id / post_id / syn_count | `connectome-weights` `body_pre` / `body_post` / `weight` | already summed over neuropils; both ends Traced; self-connections dropped |
| edgeroi neuropil / syn_count | `syn-partners` grouped by (pre, post, `primary_post`) | every connection kept, so per-pair sums equal `syn_count` (checked on all 25,563,096 pairs); `unspecified` labels kept verbatim |

### Hemibrain (E6 only)

`src/fetch_hemibrain.py` builds the three tables from the public v1.2 export (46 MB; about 10 s): 21,739 non-cropped Traced neurons, 3,550,403 edges (14,329,229 synapses, no self-connections), 4,259,624 edge-ROI rows whose per-pair sums equal the edges. The export has only `bodyId`, `type` and `instance`, so: `cell_type` and `hemibrain_type` = `type`; `side` from the last `_R`/`_L` token of the instance (`EPG(PB08)_R4`, `PFNv(PB05)_L4_C7`; for central-complex columnar types the protocerebral-bridge half, assumed to be the soma side), else from the region's side (`LO(R)`), else unknown: right 18,117, left 2,992, unknown 630; `region` = the primary ROI with most synapses with partners in the export, that is non-cropped Traced neurons (`NotPrimary` ignored; 5 unknown; every name is accepted by `eda_shared`); `status` Traced; everything else unknown or missing. `NotPrimary` edge-ROI rows are kept as `UNASGD`.

## Counts and timings (FAFB)

| Quantity | Value | Course check |
|---|---|---|
| Annotated neurons | 139,248 | 139,248 ✓ |
| Edge ids with no annotation row, added with unknown annotations | 14 | 7 expected — to report |
| Neurons table rows | 139,262 | |
| Edges ≥ 5 synapses | 2,700,513 | 2,700,513 ✓ |
| Edges ≥ 1 / ≥ 10 | 15,091,983 / 1,066,822 | |
| Self-connections | 0 | 0 ✓ |
| Neurons with region `unknown` | 282 | |
| Neurons whose top two neuropils tie | 232 | |

The FAFB tables are built from bulk files downloaded once (about 1.1 GB, a few minutes). Building the cold cache takes about 30 s, and a warm run of `build_fafb_cache.py` takes about 20 s, mostly validation. Deleting one cache file rebuilds only that file.

## Counts and timings (MaleCNS)

| Quantity | Value | Course check |
|---|---|---|
| Traced neurons | 165,122 | 165,122 ✓ |
| Self-connections dropped | 101 pairs | 101 ✓ |
| Edges ≥ 1 | 25,563,096 | 25.56M ✓ |
| Edges ≥ 5 | 6,235,649 | Projects 1 and 4: 6.25M / 6.24M ✓ |
| Edges ≥ 10 | 2,749,394 | |
| Synapses between Traced neurons | 124,024,573 | |
| Edge-neuropil rows | 29,436,294 | |
| Side `unknown` | 630 | 630 ✓ |
| `dimorphism` labelled | 2,368 | 2,368 ✓ |
| `cell_class` missing | 140,598 (85%) | 85% ✓ |
| `hemilineage` missing | 127,377 (77%) | 79% in the conventions — to check |
| `nt_type` unknown | 14,867 | |
| Neurons with region `unknown` | 464 (333 without synapses, 131 only outside named neuropils) | |
| Neurons whose top two neuropils tie | 72 | |
| Synapses between Traced neurons outside named neuropils (`unspecified` labels) | 2.4% | |

The five raw files total 8.6 GB (the synapse-partner file is 6.8 GB). A cold build of all three tables takes about 80 s with a peak of 17 GB of memory, mostly from reading the synapse-partner columns; a warm run takes a few seconds plus validation.

## Open issues to raise with the group

1. **`sensory_ascending`** (612 FAFB neurons) is a FAFB super class that `config.SUPER_CLASS_VALUES` does not list. It is mapped to `sensory` for now. It needs a group decision: add the level to the vocabulary, or keep the mapping.
2. **Missing ids.** 14 edge ids have no annotation row, not the 7 the conventions expect. There are 139,255 proofread ids against 139,248 annotation rows, and the two lists differ in both directions.
3. **FAFB `n_post` could be filled.** Zenodo's `per_neuron_neuropil_count_post_783.feather` gives each neuron's input synapses across all partners. That would enable the plausibility check and item E4 on FAFB too.
4. **Region `None`.** FlyWire labels synapses outside every neuropil as `None`, and `eda_shared` rejects that value. Here it is excluded before choosing a neuron's region.
5. **MaleCNS `super_class` (provisional).** `config.SUPER_CLASS_VALUES` has no MaleCNS-only levels yet, so every MaleCNS superclass is mapped onto the FAFB levels in `src/fetch_malecns.py` (`SUPER_CLASS_MAP`): `ol_intrinsic` → optic, `cb_intrinsic` → central, every sensory variant (including `sensory_ascending` and the `_tbc` ones) → sensory, `ascending_neuron` / `descending_neuron` → ascending / descending, motor and efferent → motor (the 110 efferent neurons — `cb_efferent` 4, `vnc_efferent` 94, `efferent_ascending` 8, `efferent_descending` 4 — are non-motor efferents in the MaleCNS vocabulary, so this merge in particular needs a decision), endocrine → endocrine, `visual_projection(_tbc)` → visual_projection, `visual_centrifugal` → visual_centrifugal. `vnc_intrinsic` (13,151), `ENS` (47), `vnc_tbc` (1) and the 516 Traced neurons with no superclass become `unknown`, 13,715 in all. The `cell_class` and `cell_type` columns keep the detail.
6. **MaleCNS region names in `eda_shared`.** Every brain neuropil name is accepted. `add_provisional_flags` refuses 34 nerve-cord names, the regions of 22,037 neurons: `ANm`, `IntTct`, `LTct`, `LegNp(T1–T3)(L/R)`, `WTct(UTct-T2)`, `HTct(UTct-T3)`, `NTct(UTct-T1)`, `Ov`, `mVAC(T1–T3)`, `DMetaN`, `AbN2`, `AbN3`, `ProLN`, `MesoLN`, `MetaLN` (with sides) and `CRN`. They belong in the function's `nerve_cord` argument, not in `ALIASES`; the group should agree on one list so the five repositories pass the same set. The edge-neuropil table also carries the `unspecified` labels (`CentralBrain-unspecified`, `Optic-unspecified(L/R)`, `VNC-unspecified`, `CV-unspecified`, `<unspecified>`), which `canonical_neuropil` refuses as well; they could join `NO_NEUROPIL`. The edge-neuropil table has 143 labels against 114 neuron regions: besides these, it carries nerves that no neuron has as its region (`ADMN`, `AbN1`, `AbN4`, `AbNT`, `CvN`, `DProN`, `MesoAN`, `PDMN`, `PrN`, `ProAN`, `ProCN`, `VProN`, with sides, and `AbN3(R)`, `MesoLN(L)`), so the agreed `nerve_cord` list should cover them too.
7. **MaleCNS region is computed, not read.** neuPrint's per-neuron `roiInfo` is not in the bulk files, so the region is counted from the synapse-partner table with the FAFB rule (per synaptic connection, pre + post, all partners). This is a deviation from "neuPrint roiInfo" if another project reads that field; the two can differ where a T-bar has several partners.

## Step 15 (Project 5): pair sampler and discrepancy count

At the end of `notebooks/00_integration.ipynb`; summary in `reports/step15_discrepancy.md`, full table in `reports/step15_discrepancy.csv`.

- **Pair sampler** (`src/pairs.py`, `sample_pairs`): for the central-complex roster, the recorded positives (≥ τ synapses summed over PB, FB, EB, NO) plus a Bernoulli sample of the unlabelled ordered pairs at rate ρ with weight 1/ρ. The scheme (`uniform`), the rate or the ratio per positive, and the seed are explicit arguments. It reproduces the roster counts: 32,125 positives and 257,763 pairs with 1–9 synapses at τ = 10, and the weighted sample estimates them without bias.
- **Discrepancy count** (`src/discrepancy.py`): central-complex type pairs recorded at τ in one copy, and their status in another: recorded, below τ, or absent. Three comparisons: fafb left vs right, fafb vs hemibrain, and fafb vs malecns (reported only, as it includes sex differences). Between animals the other copy's threshold is scaled by the median type-pair total ratio (hemibrain/fafb 2.0, malecns/fafb 1.75), a proxy for how many more synapses it records per connection.
- **Result at τ = 10:** only 1.0–4.3% of recorded type pairs have no synapse at all in the other copy. 21–37% are present but below τ. The count is a lower bound on missed positives only if the copies share their true wiring; biological differences also count as discrepancies.

Cross-dataset type mapping (course week 3, `src/type_mapping.py`, also in the integration notebook): fafb `cell_type` ↔ malecns `flywire_type` only, compound values merged; 95.0% of fafb and 95.9% of malecns brain neurons are in a matched group. Summary in `reports/type_mapping_coverage.md`.

Open issues for the group, as one message: `reports/open_issues_for_group.md`.

## PU model and dropout benchmark (first parts)

- **NumPy PU module** (`src/pu.py`): logistic model for P(y=1|x) fitted with the convex unbiased PU risk (du Plessis et al. 2015), π = P(s=1)/c, inverse-inclusion weights, unpenalised intercept, damped Newton. Infeasible c (c ≤ P(s=1)) is refused. Also the Elkan–Noto estimators of c. The checks (`tests/test_pu.py`, numbers from `scripts/pu_checks.py`):
  - the gradient matches finite differences to 3·10⁻⁸;
  - at c = 1 it agrees with scikit-learn to 5·10⁻⁹;
  - it recovers known parameters on synthetic SCAR data, where the naive model halves both the probabilities and the coefficients;
  - on synthetic data with c = 0.6, the Elkan–Noto e1 gives 0.39 (a lower bound) and the maximum score e3 gives 0.98.

  Derivation and table in `reports/pu_module.md`.
- **Dropout thinning** (`src/dropout.py`): synapse dropout (binomial thinning) and connection dropout (the SCAR control), with the realised c by stratum of true synapse count, computed exactly and by simulation. On the fafb central-complex roster at q = 0.7, positives with 10–19 synapses keep c = 0.45 while those with 20 or more keep 1.00. That is 62% of positives in the weak stratum, so SCAR is strongly violated. See `reports/dropout_realised_c.md`. The full benchmark (features rebuilt on the thinned graph, both models, c estimators) follows the week 5–6 feature work.
- **Pair features** (`src/pair_features.py`): P5's leave-pair-out features. Each pair's own synapses are removed from i's output profile, j's input profile and both degrees before computing them. Groups: identity, cell type, region profile with overlap, degree, spatial proxy. Fold-local design matrices: one-hot levels and scaling are taken from the training set.
- **Pilot** (`scripts/pilot_pu.py`, `reports/pilot_pu.md`): naive and PU fits on the roster sample, with 20% of source types held out. The split comes first: training features never contain a held-out pair's synapses (`split_features`).
  - The AUC for s rises from 0.71 (identity) to 0.92 (all groups).
  - Among the top 1% of unlabelled pairs, 26–28% are recorded with 1–9 synapses, against 2.8% overall, although synapse counts are never features.
  - It is not the final design: one split, λ not tuned, c a sensitivity grid.
- **Dropout benchmark, first run** (`src/benchmark.py`, `scripts/benchmark_pilot.py`, `reports/benchmark_pilot.md`): truth = the recorded roster at τ = 10, models see a thinned copy, scored against the truth. One seed, one split.
  - At q = 0.7 the naive model's mean P(y=1) is 0.0028–0.0030 against a true 0.0039; PU at the true c restores 0.0043.
  - The Elkan–Noto e1 estimate is 0.014–0.027 even when the true c is 1, so c must not be taken from e1 here.
- **Dropout benchmark, 5 seeds** (`scripts/benchmark_run.py`, `scripts/benchmark_summary.py`, `reports/benchmark.md`, figures `B1_…` and `B2_…`): every deleted connection is scored.
  - Under synapse dropout at q = 0.7 the naive model is 31% low on the mean P(y=1); PU at the true c is 8% high.
  - The AUC hardly changes.
  - Paired over seeds, PU at the true c has the lower Brier score in 18 of 20 runs; p = 0.03 at synapse dropout q = 0.7 (5 seeds, provisional).
  - Both models are under-confident in a weighted reliability analysis, likely because λ is not tuned.
  - e1/c ≈ 0.02 in every condition.
- **Same-split baseline and λ** (`scripts/benchmark_relative.py`, `scripts/benchmark_lambda.py`, `reports/benchmark_relative.md`): each thinned run is divided by the no-thinning run of the same seed and split.
  - The naive level falls by the factor c (0.64–0.72 for c = 0.66–0.70).
  - PU at the true c recovers it to within about 3% on average (0.99–1.03; 0.97–1.07 per seed), in both dropout modes and at λ = 10⁻² and 10⁻⁴.
  - The larger error is an offset of held-out source types (+12% to +37% on average without thinning).
  - λ = 10⁻² looks too strong. That is a held-out diagnostic; λ is chosen by inner CV in the final design.
  - No cost of the SCAR violation is resolved at the level of the mean (within about ±3%, 3 seeds).
- **λ by inner CV** (`src.benchmark.select_lambda`, `scripts/benchmark_cv.py`, `reports/benchmark_cv.md`): grouped inner split on training pairs, log-loss for s, grid 10⁻² to 10⁻⁷.
  - It chooses 10⁻⁴ to 10⁻⁷ (inside the grid in 8 of 9 runs); PU at the true c recovers the level (0.98 synapse, 1.00 connection, relative).
  - It does not beat a fixed 10⁻⁴ on held-out pairs; with one inner split and a flat loss it can choose too small a λ.
  - `select_lambda_kfold` (`lam="cv3"`): grouped 3-fold CV with the 1-SE rule. On the seed-0 folds it chooses 10⁻⁴ in 7 of 10 runs and 10⁻⁵ in 3, with no gain over fixed 10⁻⁴ (`reports/benchmark_folds.md` §3.2).
- **The held-out-type offset explained** (`scripts/benchmark_type_onehot.py`, `scripts/benchmark_target_graph.py`, `reports/benchmark_offset.md`):
  - Not the all-zero source type one-hot: removing `type_i` leaves it unchanged.
  - Mostly a train/test shift in target-side features: training targets lose the held-out sources' inputs, and held-out pairs used the full graph. With target features from the training graph (`target_graph="train"`) the offset falls from +37% to +5% on average (λ = 10⁻⁴), PU's raw level at q = 0.7 is 0.95–1.11 of the truth, and the AUC falls by about 0.013.
- **Homologue-group folds** (`src.benchmark.homologue_groups`, `fold_of_groups`, `run_condition(fold=…)`): worksheet D18 groups, 5 folds, every roster neuron held out once per seed.
- **Fold benchmark** (`scripts/benchmark_folds.py`, `scripts/benchmark_folds_summary.py`, `reports/benchmark_folds.md`): 5 homologue-group folds, held-out target features from the training graph, λ = 10⁻⁴, 3 seeds, synapse and connection dropout at q = 0.7.
  - One held-out type (ExR8, output mostly outside the CX) got a standardised feature value of about 2,273 and drove one fold's offset to +88%. Held-out numerics are now clipped to the training range (`design(clip=True)`). With clipping, fold offsets are 0.83–1.27 and the pooled offset +3% to +5%.
  - PU at the true c restores the pooled level to 0.99–1.03 of the no-thinning level, in both dropout modes.
  - Its Brier score is still *higher* than naive in 13 of 15 folds in both modes (sign test p = 0.007), so the SCAR violation of synapse dropout does not explain it. Under connection dropout PU matches the model fitted on all labels (Brier 0.003986 vs 0.003988), and the naive model beats both: on held-out types, shrinking predictions toward zero lowers squared error. Under synapse dropout PU is also worse than that model (14 of 15 folds), a cost consistent with the SCAR violation. At this q and λ, level and reliability are the informative measures for H2; Brier should be compared with the fully labelled model as well as with naive.
  - Direct check (seed 0): even the fully labelled model's Brier score would fall if its predictions were scaled by 0.32–0.83, because on held-out types they spread too widely. PU at the true c has the same Brier-optimal scale as that model under connection dropout (`reports/benchmark_folds.md` §3.4).
  - At λ = 10⁻² the spread reverses (predictions too narrow, optimal scale 1.3–2.3) and PU has the lower Brier in 5 of 5 folds, at an AUC cost of about 0.02 (§3.5). Changing λ reverses the Brier ordering (seed 0, q = 0.7), so a Brier difference alone does not speak for or against the PU correction; PU restores the level in every setting tried.
  - Inner CV grouped by homologue group (`inner_groups="homologue"`, `lam="cv3h"` in the script) chooses 10⁻³ (8 of 10) or 10⁻²; no Brier preference is detected (PU lower in 2 of 5, preliminary), mean AUC 0.920 (§3.6).
- **H3 on real data** (`src/ranking.py`, `scripts/h3_real.py`, `reports/h3_real.md`, `reports/h3_top_unrecorded.csv`): no split, λ = 10⁻⁴, c from 0.9 to 0.5.
  - PU keeps the naive ranking of unlabelled pairs: Spearman about 0.999 or higher, top-1000 overlap 0.95–0.99. PU mainly rescales; a ranked list does not tell the models apart, so the headline stays calibration (H2).
  - 37–38% of the top 1,000 unlabelled pairs have 1–9 recorded synapses, against 3.1% overall. The top of the list is dominated by same-type pairs (EPG→EPG, ER3d→ER3d, Delta7→Delta7).
  - Benchmark runs now record the same statistics against known truth (`h3_naive_vs_pu_true_c`).
- **H4 pilot** (`src/h4.py`, `scripts/h4_pilot.py`, `reports/h4_pilot.md`; gated in the proposal, so a pilot; type-group cluster bootstrap):
  - Against the proposal's matched controls (type-pair size × degree decile), top-scored unrecorded pairs are recorded at type level in the hemibrain about 7 times as often (k = 1000: 0.30 vs 0.045).
  - Most of that is type-level wiring FAFB already records. Matching also on whether FAFB records the type pair leaves 1.2–1.7 times (k = 1000: 0.30 vs 0.23; interval 1.18–2.00). The proposal's two matching variables are not enough.
  - That residual excludes 1 at τ = 10 for k ≥ 500 but not at a scaled τ = 20 for k ≤ 1000.

## Project-specific decisions (deviations are stated, not configured)

- **Hemibrain release:** the EDA plan pins v1.2.1, which needs a neuPrint token; the public v1.2 export is used until one is available. Effects: 87% of fafb's distinct `hemibrain_type` values are found among v1.2 types (some misses, such as `PI1` or `PAM06`, look like v1.2.1 renames), and the cropped share can only be bounded (≤ 2.1% of typed neurons of the matched types; per matched type rather than per region as the plan asks), not read from neuPrint's `cropped` flag. Switching to v1.2.1 means a new fetch module reading neuPrint; E6 code does not change.
- **Roster:** the central complex roster is `cell_class == "CX"` (2,875 neurons, 229 types). This is annotation-based, not synapse-based, so the label does not define the population.
- **Threshold:** the course default is τ = 10. A pair recorded with 1–9 synapses is **unlabelled**, because under the label definition its true count may still be ≥ 10. That gives 257,763 such pairs in the roster, against 32,125 positives at τ = 10.

## Change log

- 2026-10-13: repository created; FAFB cache built and validated; MaleCNS pending.
- 2026-10-14: MaleCNS v1.0 cache built from Janelia's public bulk files and validated; course counts reproduced. `recode` and `_cast` shared between the two fetch modules.
- 2026-10-14: EDA workstream E run on malecns as well (brain part; E4 with all Traced partners); E2, E3 and E7a now compare both datasets in one figure. E6 waits on the hemibrain. Summary in `reports/eda_E.md`.
- 2026-10-14: hemibrain v1.2 cache (public export) and EDA item E6.
- 2026-10-14: step 15 (pair sampler, discrepancy count) in the integration notebook, which now loads and validates all three datasets; MaleCNS evidence in the decision worksheet; MaleCNS fetch tests; FlyWire–MaleCNS type mapping and coverage report; open issues collected for the group.
- 2026-10-14: NumPy PU module with its checks; dropout thinning and realised c by stratum.
- 2026-10-14: Newton-decrement stopping rule and a non-convergence warning in the PU fit; leave-pair-out pair features; first pilot of naive vs PU fits; review nits on the PU and dropout tests and reports.
- 2026-10-14: first run of the dropout benchmark (truth known); 5-seed benchmark with all deleted connections scored, a grouped inner split for e1 and per-stratum results.
- 2026-10-14: benchmark levels against a same-split baseline; λ sensitivity; π and training P(s=1) stored per run; review nits.
- 2026-10-15: λ by inner CV; the held-out-type offset traced to a target-side feature shift (`target_graph` option); homologue-group folds; review nits.
- 2026-10-15: H3 on real data (ranking agreement, sampled ranked list); H4 pilot against the hemibrain with a type-group cluster bootstrap; H3 statistics, `keep_scores` and `clip` options in benchmark runs.
- 2026-10-15: fold benchmark (homologue-group folds, clipping of held-out numerics, 3 seeds, both dropout modes); 3-fold 1-SE λ selection; PU's higher Brier is not explained by the SCAR violation and is consistent with shrinkage paying off on held-out types (q = 0.7, λ = 10⁻⁴).
- 2026-10-15: Brier-optimal scale per model in benchmark runs; inner λ search grouped by homologue group (`inner_groups`); folds at λ = 10⁻² and with type-grouped inner CV: changing λ reverses the Brier ordering of naive and PU, in step with the spread of the predictions.
