# Workstream E: regions, completeness and discrepancy (one-page summary)

Tarkan Eyerci (Project 5) · 15.10.2026 · FAFB 783 (annotations v3.2.0, Buhmann) and MaleCNS v1.0 (Traced, confidence ≥ 0.5, brain part: 142,745 neurons); E6 also uses the hemibrain v1.2 public export. Central = `is_optic` False. Figures: `figures/eda/E*`; notebook: `notebooks/eda_E.ipynb` (about 2 minutes from the cache, no network); full report with definitions and deviations: `reports/eda_E.md`.

| Item | Number | What it means |
|---|---|---|
| E1 One neuron, one region | 25% (FAFB) and 20% (MaleCNS) of central neurons have a top-vs-second gap below 0.10 (named neuropils); exact ties in the region assignment (all neurons, all partners): 232 and 90 | The "most synapses" rule is a close call for a fifth to a quarter of central neurons, so the tie rule matters (D15). Worked examples for D17: fractions sum to 1 while any input survives τ, to 0 when none does |
| E2 Optic-lobe share | Ambiguous between optic and central: 4.3% (FAFB) and 4.6% (MaleCNS); clean optic 65.0% and 67.6% | A boolean `is_optic` is the right shape (D13); super_class undercounts optic neurons, so the flag should come from synapses or region |
| E3 Region-to-region map | 59% (FAFB) and 60% (MaleCNS) of central synapses at τ = 10 stay within one region; the same main routes (AL→LH, PVLP→AVLP) | Same large-scale structure; MaleCNS has no AMMC, OCG or GA and twice the synapses, so comparisons need a common region list and shares, not raw counts |
| E4 Record completeness | Median 95% (FAFB, supplement) and 96% (MaleCNS) of a neuron's inputs arrive as recorded edges (neurons with inputs); lowest MaleCNS group: sensory neurons (0.84) | The record is nearly complete per neuron; the MaleCNS shortfall is mainly input from untraced fragments (plus 375 dropped self-synapses), not the confidence filter |
| E5 Left–right recurrence | 1-synapse type edges recur 32% (central, both datasets); 10–19 synapses 94% (FAFB) and 92% (MaleCNS) | Recurrence first exceeds 90% in the 10–19 bin, which supports τ = 10 (D10) |
| E6 Another animal | 10–19-synapse type edges recur 89% female–female (FAFB vs hemibrain) and 89.7% female–male on the same types | Between animals recurrence is 4–7 points lower than within one. On matched types female–male and female–female recurrence are similar (89.7% vs 89.5%); this does not isolate sex effects, but a sex claim must exceed that baseline |
| E7 Central complex | 2,617 (FAFB) and 2,657 (MaleCNS) neurons by region; in connected columnar type pairs a median 2.5% and 2.2% of neuron pairs are connected (τ = 5) | Cell type does not fix a connection, so neuron-level prediction (Project 5) is needed |

**What surprised me:** one-synapse edges recur only 32% within FlyWire, below the 42% Schlegel et al. found between FlyWire and the hemibrain; MaleCNS gives the same 32%. Population mix is a likely contributor: recurrence rises with type size (31% for one-neuron types to 56% for 20+) and singleton-type pairs are 80% of central type pairs, but this does not fully explain the gap to 42%.

**Choices to confirm with the group:** the MaleCNS nerve-cord list; the D15 rows for the `unspecified` labels in a new eda_shared release and the tie rule; a common region list (E3); E4 inputs from all Traced partners; E7 columnar types taken whole.
