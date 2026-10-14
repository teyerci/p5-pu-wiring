# P5 — Positive–unlabelled wiring (CENG8004 Machine Learning)

Predicting connections in the fly connectome when "no recorded connection" does not mean "no connection". This repository follows the course's Week 2 Data Conventions and the Week 3 EDA layout, so the shared code runs on it unchanged.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # exact environment: requirements-lock.txt
cp .env.example .env                           # fill in tokens; .env is gitignored
.venv/bin/python -m pytest                     # 119 own tests (most for the review gate) + 15 eda_shared tests
.venv/bin/python scripts/build_fafb_cache.py   # downloads raw files once, builds data/cache/
.venv/bin/python scripts/build_malecns_cache.py  # same for MaleCNS (8.6 GB download, ~17 GB RAM)
```

Python 3.10.12. Packages: caveclient 8.2.1, fafbseg 3.2.2, neuprint-python 0.6.4, navis 1.12.0, pandas 2.3.3, pyarrow 25.0.1, numpy 2.2.6, networkx 3.4.2, scikit-learn 1.7.2, matplotlib 3.10.9, pytest 9.1.1.

Tokens live only in environment variables or `.env`: `NEUPRINT_APPLICATION_CREDENTIALS` (neuPrint) and `CAVE_SECRET` (FlyWire CAVE). They are never written to a notebook or committed.

## Layout

```
config.py              course file, copied verbatim from the Data Conventions; never edited
eda_shared/            course package v0.1.1-eda, unchanged
src/                   paths, cache helper, validators, fetch functions, summary
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
| Hemibrain | v1.2.1 | not fetched by default; needed for EDA item E6 and H4 |

## Schema

The neuron and edge schemas are defined once, in `config.py` (`NEURON_SCHEMA`, `EDGE_SCHEMA`, `EDGEROI_SCHEMA`), and every validator in `src/validate.py` reads from there. Cache files:

```
data/cache/fafb_neurons_783_ann3.2.0.parquet
data/cache/fafb_edges_783_buhmann_min1.parquet
data/cache/fafb_edgeroi_783_buhmann_min1.parquet
data/cache/malecns_neurons_v1.0_traced.parquet
data/cache/malecns_edges_v1.0_traced_min1_conf0.5.parquet
data/cache/malecns_edgeroi_v1.0_traced_min1_conf0.5.parquet
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

## Project-specific decisions (deviations are stated, not configured)

- **Roster:** the central complex roster is `cell_class == "CX"` (2,875 neurons, 229 types). This is annotation-based, not synapse-based, so the label does not define the population.
- **Threshold:** the course default is τ = 10. A pair recorded with 1–9 synapses is **unlabelled**, because under the label definition its true count may still be ≥ 10. That gives 257,763 such pairs in the roster, against 32,125 positives at τ = 10.

## Change log

- 2026-10-13: repository created; FAFB cache built and validated; MaleCNS pending.
- 2026-10-14: MaleCNS v1.0 cache built from Janelia's public bulk files and validated; course counts reproduced. `recode` and `_cast` shared between the two fetch modules.
