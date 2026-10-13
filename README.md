# P5 — Positive–unlabelled wiring (CENG8004 Machine Learning)

Predicting connections in the fly connectome when "no recorded connection" does not mean "no connection". This repository follows the course's Week 2 Data Conventions and the Week 3 EDA layout, so the shared code runs on it unchanged.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # exact environment: requirements-lock.txt
cp .env.example .env                           # fill in tokens; .env is gitignored
.venv/bin/python -m pytest                     # 11 own tests + 15 eda_shared tests
.venv/bin/python scripts/build_fafb_cache.py   # downloads raw files once, builds data/cache/
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
| MaleCNS | male-cns:v1.0, Traced, synapse confidence ≥ 0.5 | **not yet fetched**: needs a neuPrint token and the bulk-download addresses from Project 4's README |
| Hemibrain | v1.2.1 | not fetched by default; needed for EDA item E6 and H4 |

## Schema

The neuron and edge schemas are defined once, in `config.py` (`NEURON_SCHEMA`, `EDGE_SCHEMA`, `EDGEROI_SCHEMA`), and every validator in `src/validate.py` reads from there. Cache files:

```
data/cache/fafb_neurons_783_ann3.2.0.parquet
data/cache/fafb_edges_783_buhmann_min1.parquet
data/cache/fafb_edgeroi_783_buhmann_min1.parquet
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

## Open issues to raise with the group

1. **`sensory_ascending`** (612 FAFB neurons) is a FAFB super class that `config.SUPER_CLASS_VALUES` does not list. It is mapped to `sensory` for now. It needs a group decision: add the level to the vocabulary, or keep the mapping.
2. **Missing ids.** 14 edge ids have no annotation row, not the 7 the conventions expect. There are 139,255 proofread ids against 139,248 annotation rows, and the two lists differ in both directions.
3. **FAFB `n_post` could be filled.** Zenodo's `per_neuron_neuropil_count_post_783.feather` gives each neuron's input synapses across all partners. That would enable the plausibility check and item E4 on FAFB too.
4. **Region `None`.** FlyWire labels synapses outside every neuropil as `None`, and `eda_shared` rejects that value. Here it is excluded before choosing a neuron's region.

## Project-specific decisions (deviations are stated, not configured)

- **Roster:** the central complex roster is `cell_class == "CX"` (2,875 neurons, 229 types). This is annotation-based, not synapse-based, so the label does not define the population.
- **Threshold:** the course default is τ = 10. A pair recorded with 1–9 synapses is **unlabelled**, because under the label definition its true count may still be ≥ 10. That gives 257,763 such pairs in the roster, against 32,125 positives at τ = 10.

## Change log

- 2026-10-13: repository created; FAFB cache built and validated; MaleCNS pending.
