# EDA workstream E — regions, completeness and discrepancy (Project 5)

FAFB 783, annotations v3.2.0, Buhmann synapses. Figures are in `figures/eda/`, and `notebooks/eda_E.ipynb` regenerates them from the cache. **Status:** done for FAFB. The malecns side of each item and E6 are pending, waiting on the neuPrint token and the hemibrain.

| Item | Number | What it means |
|---|---|---|
| E1 One neuron, one region | **25%** of central neurons (11,268 of 45,168) have a gap below 0.10 between their top and second neuropil; median top share 0.56 | The "most synapses" rule is a close call for a quarter of central neurons, so the tie rule (D15) matters. Worked example: a neuron that keeps any input at τ = 10 still has fractions summing to 1; two JO neurons lose all input and sum to 0 (D17: decide before or after thresholding, and 0 vs null). |
| E2 Optic-lobe share | **65.0%** clean optic, 30.2% clean central, **4.3% ambiguous** (5,947), 0.5% no synapses. By super_class 55.7%; by the region flag 67.3% | The boundary is clean enough for a boolean `is_optic` (D13). The super_class figure undercounts optic by about 10 points, so the flag should come from synapses or region, not super_class. |
| E3 Region-to-region map | **59%** of central synapses at τ = 10 stay within one region (38 base neuropils, 13.4M synapses) | Strong diagonal: AVLP (14%) and GNG (12%) carry the most within-region synapses. The largest cross-region routes: AL→LH (1.9%), PVLP→AVLP (1.5%), LH→MB medial lobe (1.1%). |
| E4 Record completeness (FAFB supplement) | Central neurons keep a median **95%** of their input synapses as edges to proofread partners (pooled 95%; never above 1) | FlyWire's all-synapse counts make E4 possible on FAFB as well. Inputs are nearly complete; outputs are not: only 44% of proofread neurons' output synapses land on proofread partners, matching FlyWire's 44.7%. |
| E5 Left–right recurrence | 1-synapse type edges recur **32%** (central) and 46% (optic); 5–9 synapses: 84%; **10+: 96%** (central) | The τ = 10 default is supported: recurrence passes 90% in the 10–19 bin (94%) and not before. Our 1-synapse figure is below Schlegel et al.'s 42%. |
| E6 Recurrence in another animal | pending | Needs malecns and the hemibrain. The FAFB–hemibrain curve is the baseline any sex claim must exceed. |
| E7 Central complex | 2,617 neurons whose region is PB/FB/EB/NO (FB 70%), 223 types; **39 types with 8–16 per side, 63 one per side**; type pairs connected at τ = 5 are sparse | **Type alone does not fix a connection.** In a connected pair of columnar types, a median **3%** of neuron pairs are connected (369 type pairs). Pair-level prediction is a real problem, not a type lookup. |

**Worksheet evidence:**
- **D10:** E5. τ = 10 is where recurrence crosses 90%.
- **D13, D14:** E2. A boolean flag works (4.3% ambiguous). Exclude the optic lobes by default for whole-brain claims.
- **D15:** E1. The tie rule affects a quarter of central neurons.
- **D17:** E1's worked example.

**What surprised me:**
- One-synapse type edges recur less often within FlyWire (32%) than Schlegel et al. found between FlyWire and the hemibrain (42%), even though a within-animal curve was expected to sit higher. One untested explanation is the size of the central type-pair space (1.66M pairs, against 88k in the optic lobes). This needs checking before the session.
- The two central-complex rosters disagree. By `cell_class == "CX"` there are 2,875 neurons; by region there are 2,617; 2,560 are in both. 315 CX-class neurons have most of their synapses outside the CX (for example in SMP and CRE), and 57 neurons whose region is CX belong to another class or none (TuBu 50, DAN 1, unclassified 6). Project 5 keeps the annotation-based roster and states why.
