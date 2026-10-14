# Step 15 (Project 5) — pair sampler and first discrepancy count

Central complex (PB, FB, EB, NO). fafb 783 roster: `cell_class == "CX"`, 2,875 neurons, 8,262,750 ordered pairs. Other copies: malecns v1.0 (Traced) and hemibrain v1.2 (public export of non-cropped Traced neurons; the plan pins v1.2.1). `notebooks/00_integration.ipynb` regenerates everything; the full table is `reports/step15_discrepancy.csv`.

## Pair sampler (`src/pairs.py`)

- **Label.** s = 1 when i makes at least τ synapses onto j, summed over the four central-complex neuropils. All other ordered pairs of distinct roster neurons are unlabelled (s = 0): 257,763 recorded with 1–9 synapses (stratum `below_tau`) and 7,972,862 with none (`unrecorded`).
- **Scheme** (an explicit argument; `uniform` is the only one so far). All 32,125 positives are kept with weight 1. Each unlabelled pair is included independently with probability ρ and gets weight 1/ρ, given as `rate` or as `per_positive` (ρ = k · P / U). The sample is deterministic given the seed.
- **Check.** With 20 unlabelled pairs per positive (ρ = 0.078), the weighted sample estimates 258,153 below-τ pairs (true 257,763) and 7,985,577 unrecorded pairs (true 7,972,862).

## Discrepancy count (`src/discrepancy.py`)

**Unit.** Neurons are matched across copies only by type, so the matched pair is a **type pair** on one side. In each copy, a type pair is:
- **recorded** if some neuron pair in it reaches τ (the P5 label);
- **below τ** if it has synapses but no neuron pair reaches τ;
- **absent** if it has no synapse in the central complex.

Types are matched by fafb `cell_type` (left vs right), fafb `hemibrain_type` ↔ hemibrain type, and fafb `cell_type` ↔ malecns `flywire_type`. Compound values are merged into match groups. For types with exactly one neuron per side in both copies, a type pair is a single neuron pair; these are reported as their own rows in the CSV.

**Synapse scale.** On type pairs with synapses in both copies, the median ratio of type-pair synapse totals is **1.0** for fafb right/left, **2.0** for hemibrain/fafb and **1.75** for malecns/fafb. This is a proxy for how many more synapses a copy records per connection: a type-pair total also grows with the number of connected neuron pairs. A fixed τ = 10 is therefore, as far as the proxy goes, a stricter label in fafb. Between-animal rows use a **scale-matched threshold**: τ anchored in fafb, round(τ · ratio) in the other copy, so 20 in the hemibrain and 18 in malecns.

**Results at τ = 10** (type pairs recorded in the first copy; the CSV has τ = 1, 5 and the same-τ rows as well):

| Direction | Recorded in first | Recorded in second | Below τ in second | Absent in second |
|---|---|---|---|---|
| fafb left → right | 1,310 | 73.5% | 24.0% | **2.4%** |
| fafb right → left | 1,236 | 77.9% | 21.1% | **1.0%** |
| fafb → hemibrain (τ 20) | 630 | 62.1% | 36.8% | **1.1%** |
| hemibrain (τ 20) → fafb | 588 | 66.5% | 29.3% | **4.3%** |
| fafb → malecns (τ 18) | 750 | 70.4% | 28.4% | 1.2% |
| malecns (τ 18) → fafb | 689 | 76.6% | 21.8% | 1.6% |

**What it means for P5:**
- **Few connections vanish; many cross the threshold.** Of the central-complex type pairs recorded at τ in one copy, only 1.0–4.3% have no synapse at all in the other. Another 21–37% are present but below τ, so the label is often decided by a few synapses near the threshold. This is the mechanism P5 models as label noise, and it supports keeping 1–9-synapse pairs unlabelled rather than negative.
- **How far the count bounds missed positives.** If the two copies shared their true wiring at type level, every discrepancy would be a connection missed in one copy, and the count would be a lower bound on missed positives, because connections missed in both copies stay invisible. Real copies differ (left–right asymmetry, individual variation; sex in fafb vs malecns), and each difference also counts as a discrepancy. So the count is a lower bound on missed positives **only under shared wiring**, and an overestimate of them where wiring truly differs. The within-animal recurrence share (74–78%) is likewise only a rough type-level indication of c: misses correlated across copies push it up, biological differences push it down. P5 therefore estimates c on the dropout benchmark, where the truth is known.
- **Neuron level.** Where a type pair is a single neuron pair (one neuron per type and side), the within-fafb discrepancy is higher: 40% left→right and 33% right→left at τ = 10, from 65 and 58 pairs. Pair-level labels are less stable than type-level ones, which a type-level description hides.
- **Between pipelines**, the absolute threshold does not transfer: without scale matching, 57% of hemibrain type pairs recorded at τ = 10 are below 10 or absent in fafb. Any cross-connectome validation of predicted-but-unrecorded pairs (the brief's external check) must use a scale-matched or relative threshold.
- **fafb vs malecns** includes sex differences on top of individual and pipeline differences. It is reported here and is never used to claim a missed connection.

**Limits.**
- The hemibrain is v1.2, not the pinned v1.2.1. Its side comes from the instance's last `_R`/`_L` token (for columnar types the protocerebral-bridge half), else the region's side.
- Type matching through annotation columns covers 177 groups with the hemibrain and 167 with malecns, of the 222 central-complex types present on both sides of fafb.
- A type pair counts as recorded when any of its neuron pairs reaches τ, so larger types have more chances to be recorded.
