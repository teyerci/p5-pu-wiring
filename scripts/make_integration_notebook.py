"""Generate notebooks/00_integration.ipynb (kept as code so the notebook is reproducible)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 00 — Integration\n\nFrom a clean clone: config, cache, validators, counts for fafb and malecns "
       "(and the hemibrain, used in E6 and step 15), then the Project 5 step 15: the pair sampler and the "
       "first discrepancy count. Runs off `data/cache/` with no network I/O once the cache exists "
       "(a cold cache downloads the raw files once)."),
    code('import sys\nfrom pathlib import Path\n'
         'ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.py").exists())\n'
         'sys.path.insert(0, str(ROOT))'),
    code('import logging\nimport config\n'
         'from src.fetch_fafb import load_neurons, load_edges, load_edgeroi\n'
         'from src.validate import validate_dataset\nfrom src.summary import summary\n'
         'logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")\n'
         'print("FAFB", config.FAFB_MATERIALIZATION, config.FAFB_ANNOTATIONS_TAG, config.FAFB_SYNAPSE_TABLE)'),
    md("## Load (cache first) and validate"),
    code("neurons, edges, edgeroi = load_neurons(), load_edges(), load_edgeroi()\n"
         "validate_dataset(neurons, edges, edgeroi)\nsummary(neurons, edges)"),
    md("## Threshold table (course grid, central brain)"),
    code("from eda_shared.eda_helpers import add_provisional_flags, relative_weight\n"
         "import pandas as pd\n"
         "fl = add_provisional_flags(neurons)\n"
         "central = set(fl.loc[~fl.is_optic, 'neuron_id'])\n"
         "e = edges.assign(frac=relative_weight(edges))\n"
         "e = e[e.pre_id.isin(central) & e.post_id.isin(central)]\n"
         "rows = []\n"
         "for label, keep in [(f'>= {t}', e.syn_count >= t) for t in (1, 5, 10, 31)] + \\\n"
         "                   [(f'>= {f:.1%} input', e.frac >= f) for f in (0.009, 0.026)]:\n"
         "    k = e[keep]\n"
         "    connected = pd.unique(pd.concat([k.pre_id, k.post_id]))\n"
         "    rows.append({'criterion': label, 'edges': len(k), 'synapses': int(k.syn_count.sum()),\n"
         "                 'isolated': len(central) - len(connected),\n"
         "                 'median_out_degree': float(k.groupby('pre_id').size().median())})\n"
         "pd.DataFrame(rows)"),
    md("## malecns and hemibrain: load and validate"),
    code("from src import fetch_malecns, fetch_hemibrain\n"
         "mal = (fetch_malecns.load_neurons(), fetch_malecns.load_edges(), fetch_malecns.load_edgeroi())\n"
         "validate_dataset(*mal)\n"
         "hemi = (fetch_hemibrain.load_neurons(), fetch_hemibrain.load_edges(), fetch_hemibrain.load_edgeroi())\n"
         "validate_dataset(*hemi)\n"
         "from src.validate import assert_interchangeable\n"
         "assert_interchangeable(neurons, mal[0]); assert_interchangeable(neurons, hemi[0])\n"
         "pd.concat([summary(neurons, edges), summary(mal[0], mal[1]), summary(hemi[0], hemi[1])])"),
    md("## Project 5 region: central complex roster"),
    code("cx = neurons[neurons.cell_class == 'CX']\n"
         "ids = set(cx.neuron_id)\n"
         "ecx = edgeroi[edgeroi.neuropil.isin(['EB', 'FB', 'PB', 'NO']) & edgeroi.pre_id.isin(ids) & edgeroi.post_id.isin(ids)]\n"
         "pair = ecx.groupby(['pre_id', 'post_id']).syn_count.sum()\n"
         "n = len(ids)\n"
         "pd.Series({'roster': n, 'ordered pairs': n * (n - 1), 'pairs >= 10': int((pair >= 10).sum()),\n"
         "           'recorded below 10 (1-9 syn)': int(((pair >= 1) & (pair < 10)).sum())})"),
    md("## Cross-dataset type mapping and coverage (week 3)\n\n"
       "fafb `cell_type` ↔ malecns `flywire_type` only (never the type names), compound values merged into "
       "match groups; malecns is its brain part (`src/type_mapping.py`)."),
    code("from src.eda_e import brain_part\nfrom src.type_mapping import mapping_table, coverage\n"
         "mal_brain = brain_part(mal[0], mal[1])[0]\n"
         "tmap = mapping_table(neurons, mal_brain)\n"
         "overall, by_region = coverage(neurons, mal_brain, tmap)\n"
         "tmap.to_csv(ROOT / 'reports' / 'type_mapping_fafb_malecns.csv')\n"
         "by_region.to_csv(ROOT / 'reports' / 'type_mapping_coverage_by_region.csv', index=False)\n"
         "print(overall)\n"
         "by_region[by_region.level == 'region_group'].round(3)"),
    md("## Step 15 (Project 5): pair sampler\n\n"
       "Recorded positives (≥ τ synapses summed over PB, FB, EB, NO) plus a uniform Bernoulli sample of the "
       "unlabelled ordered pairs at rate ρ, weight 1/ρ (`src/pairs.py`). The weighted sample reproduces "
       "the population counts."),
    code("from src.pairs import cx_roster, region_pair_counts, sample_pairs\n"
         "ids = cx_roster(neurons)\n"
         "pair_syn = region_pair_counts(edgeroi, ids)\n"
         "sample = sample_pairs(ids, pair_syn, tau=10, scheme='uniform', per_positive=20, seed=0)\n"
         "print(sample.attrs)\n"
         "pd.DataFrame({'sampled rows': sample.stratum.value_counts(),\n"
         "              'weighted estimate': sample.groupby('stratum').weight.sum().round(0),\n"
         "              'population': pd.Series({'positive': int((pair_syn >= 10).sum()),\n"
         "                                      'below_tau': int(((pair_syn >= 1) & (pair_syn < 10)).sum()),\n"
         "                                      'unrecorded': len(ids) * (len(ids) - 1) - len(pair_syn)})})"),
    md("## Step 15 (Project 5): first discrepancy count\n\n"
       "Type pairs of the central complex recorded at τ in one copy, and their status in the other "
       "(`src/discrepancy.py`). Under shared true wiring the count is a lower bound on missed positives; "
       "biological differences also count as discrepancies. fafb vs malecns also contains sex differences "
       "and is only reported."),
    code("from src.discrepancy import fafb_left_right, fafb_vs_other, fig_discrepancy\n"
         "from IPython.display import Image, display\n"
         "lr, lr_info = fafb_left_right(neurons, edgeroi)\n"
         "hb, hb_info = fafb_vs_other(neurons, edgeroi, hemi[0], hemi[2], 'hemibrain_type', 'cell_type', 'hemibrain')\n"
         "mc, mc_info = fafb_vs_other(neurons, edgeroi, mal[0], mal[2], 'cell_type', 'flywire_type', 'malecns')\n"
         "print({'fafb left vs right': lr_info, 'fafb vs hemibrain': hb_info, 'fafb vs malecns': mc_info})\n"
         "table = pd.concat({'fafb L/R': lr, 'fafb/hemibrain': hb, 'fafb/malecns': mc}, names=['comparison']).reset_index(level=0)\n"
         "(ROOT / 'reports').mkdir(exist_ok=True)\n"
         "table.to_csv(ROOT / 'reports' / 'step15_discrepancy.csv', index=False)\n"
         "display(table[table.tau == 10].round(3))\n"
         "display(Image(filename=str(fig_discrepancy({'lr': lr, 'hb': hb, 'mc': mc}))))"),
    md("## What this shows\n\nCounts match the course checks for both datasets (fafb 2,700,513 and malecns "
       "6,235,649 edges at 5 synapses), and the cache works offline. The central complex gives 32,125 "
       "positives at τ = 10. Few central-complex connections recorded at τ vanish entirely in another copy "
       "(1–4% of type pairs), but many fall below the threshold (21–24% within fafb; 29–37% against the "
       "hemibrain even after scaling its threshold by the type-pair total ratio of 2.0). See `reports/step15_discrepancy.md`."),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/00_integration.ipynb")
