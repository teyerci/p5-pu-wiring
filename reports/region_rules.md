# Region rule: counting measure and labels outside named neuropils (D15, workstream E)

`scripts/region_rules.py` → `reports/region_rules.json`. Context: `add_provisional_flags` rejects the MaleCNS labels `CentralBrain-unspecified` and `Optic-unspecified(L/R)`. Group members using different roiInfo fields counted 640 and 761 neurons with such a region. Two questions follow: which count defines "most synapses" (D15), and whether a label outside every named neuropil may be a neuron's region.

Populations: MaleCNS v1.0, the 165,122 Traced neurons, synapse-partner table at confidence 0.5, all partners. FAFB 783: the 139,262 neurons of the cache, FlyWire's per-neuron neuropil counts (the repo's region source). Ties go to the alphabetically first label throughout. The repo rule reproduces the cached `region` column for 100% of neurons in both datasets.

## 1. FAFB counts connections, not release sites

The course guide (section 1) says the datasets "record each sender–receiver contact as one synapse", so one release site adds a synapse to several edges. The data agrees:
- FlyWire's per-neuron output (pre) and input (post) count files have the same total over all root ids, 130,054,535 each. Each synapse has one sending and one receiving end, so equal totals mean both sides count the same units; with release sites on the output side, the pre total would be several times smaller.
- For the 138,005 neurons with proofread partners, the output counts total 121,905,410. The synapses from the same neurons to proofread partners alone total 54,492,922 (the guide's "about 54.5 million").
- For all 138,005 neurons present in both files, the output count is at least the proofread-partner total (ratio quantiles 1%: 1.14, median 1.92, 90%: 3.36). The partners missing from the second total are non-proofread fragments.
- If the output counts were release sites, the ratio would be far below 1. In MaleCNS, each T-bar contacts 6.9 partners on average (294,794,753 outgoing connections from Traced neurons; 42,658,913 T-bars).

So FAFB's region counts connections on both sides. In MaleCNS, the matching measure is upstream + downstream; pre + post counts each release site once.

## 2. MaleCNS neurons whose most-synapse label lies outside the named neuropils

"Literal" = any label can win; "repo" = labels containing `unspecified` never win (`src/fetch_malecns.py`). "Connections" = upstream + downstream (the repo's measure); "sites" = unique T-bars + post sites. A T-bar's neuropil is taken from its first partner row, which is an approximation.

| Label (literal rule) | Connections | Sites |
|---|---|---|
| CentralBrain-unspecified | 350 | 295 |
| Optic-unspecified(L) | 358 | 311 |
| Optic-unspecified(R) | 37 | 25 |
| Brain labels, total | 745 | 631 |
| VNC-unspecified | 230 | 207 |
| `<unspecified>` | 12 | 11 |

The group reported 761 (roiInfo upstream + downstream) and 640 (pre + post). The present totals lie 1–2% lower (745 against 761, 631 against 640); the synapse-partner table is not identical to roiInfo. The difference between 640 and 761 is a difference of measure.

FAFB, by contrast: 6 of 139,262 neurons have FlyWire's `None` label as their top label. Under the repo rule, 5 of them go to GNG or AMMC_L and 1 becomes unknown.

## 3. Where the repo rule puts the 745 brain-label neurons (connections)

| Literal label | Central | Optic | Nerve cord | Unknown | Median share of the named region |
|---|---|---|---|---|---|
| CentralBrain-unspecified (350) | 240 | 7 | 22 | 81 | 0.24 |
| Optic-unspecified(L) (358) | 10 | 339 | 0 | 9 | 0.32 |
| Optic-unspecified(R) (37) | 0 | 17 | 0 | 20 | 0.37 |

The most frequent named regions are ME(L) 226, LA(L) 54, LO(L) 44 and GNG 54, SMP 31 (L and R), SAD 17. The median share is the named region's synapses over all the neuron's labelled synapses: the assignment is weak, a quarter to a third of the synapses.

## 4. Size of the MaleCNS brain subset (in_brain: central or optic region)

| Treatment | Connections | Sites |
|---|---|---|
| Literal rule, with the proposed `neuropils_provisional.csv` rows (both unspecified brain labels count as brain) | 142,739 | 142,507 |
| Literal rule, rejected labels → unknown | 141,994 | 141,876 |
| Repo rule | 142,621 | 142,390 |

Under the repo rule, 132 of the 745 neurons leave the brain subset (110 unknown, 22 nerve cord). 14 neurons whose literal label is `VNC-unspecified` or `<unspecified>` enter it, so the net loss is 118. The other 613 keep a named region instead of region_group unknown.

## Reading

- The counting measure matters more than the label rule. It moves the count of such neurons from 631 to 745, and it applies to every neuron's region, not only these. FAFB fixes it to connections.
- The label rule is a trade-off. The repo rule gives 613 neurons a weakly supported but real region and loses 118 from the brain subset. The proposed CSV rows keep every neuron but give 745 the region_group unknown, which most per-region analyses then leave out. Whichever the group chooses, FAFB's `None` label should follow the same rule; it concerns only 6 FAFB neurons.
