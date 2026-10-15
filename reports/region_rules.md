# Region rule: counting measure and labels outside named neuropils (D15, workstream E)

`scripts/region_rules.py` → `reports/region_rules.json`. Context: `add_provisional_flags` with the shared table rejects the MaleCNS labels `CentralBrain-unspecified` and `Optic-unspecified(L/R)`. Group members using different roiInfo fields counted 640 and 761 neurons with such a region. Two questions follow: which count defines "most synapses" (D15), and whether a label outside every named neuropil may be a neuron's region.

Populations: MaleCNS v1.0, the 165,122 Traced neurons, synapse-partner table at confidence 0.5, all partners. FAFB 783: the 139,262 neurons of the cache, FlyWire's per-neuron neuropil counts (the repo's region source). Ties go to the alphabetically first label throughout. The repo's rule (literal, section 2) reproduces the cached `region` column for 100% of neurons in both datasets.

## 1. FAFB counts connections, not release sites

The course guide (section 1) says the datasets "record each sender–receiver contact as one synapse", so one release site adds a synapse to several edges. The data agrees:
- FlyWire's per-neuron output (pre) and input (post) count files have the same total over all root ids, 130,054,535 each. Each synapse has one sending and one receiving end, so equal totals mean both sides count the same units; with release sites on the output side, the pre total would be several times smaller.
- For the 138,005 neurons with proofread partners, the output counts total 121,905,410. The synapses from the same neurons to proofread partners alone total 54,492,922 (the guide's "about 54.5 million").
- For all 138,005 neurons present in both files, the output count is at least the proofread-partner total (ratio quantiles 1%: 1.14, median 1.92, 90%: 3.36). The partners missing from the second total are non-proofread fragments.
- If the output counts were release sites, the ratio would be far below 1. In MaleCNS, each T-bar contacts 6.9 partners on average (294,794,753 outgoing connections from Traced neurons; 42,658,913 T-bars).

So FAFB's region counts connections on both sides. In MaleCNS, the matching measure is upstream + downstream; pre + post counts each release site once.

## 2. MaleCNS neurons whose most-synapse label lies outside the named neuropils

"Literal" = any label can win, with `<unspecified>` (no ROI) spelled `UNASGD` as FlyWire does; this is the repo's rule since the group's D15 discussion of 2026-10-15 (`src/fetch_malecns.py`; the cached `region` equals it for every neuron). "Named-only" = labels containing `unspecified` and `UNASGD` never win; the repo's earlier rule. "Connections" = upstream + downstream (the repo's measure); "sites" = unique T-bars + post sites. A T-bar's neuropil is taken from its first partner row, which is an approximation.

| Label (literal rule) | Connections | Sites |
|---|---|---|
| CentralBrain-unspecified | 355 | 299 |
| Optic-unspecified(L) | 358 | 311 |
| Optic-unspecified(R) | 37 | 25 |
| Brain labels, total | 750 | 635 |
| VNC-unspecified | 230 | 207 |
| `UNASGD` (`<unspecified>`) | 6 | 6 |

The spelling sets the alphabetical tie order: kept as `<unspecified>`, the label won 6 more neurons (connections; 5 by sites) through ties: 5 of them now have `CentralBrain-unspecified` and 1 PRW. The brain totals were 745 and 631.

The group reported 761 (roiInfo upstream + downstream) and 640 (pre + post). The present totals lie about 1% lower (750 against 761, 635 against 640); the synapse-partner table is not identical to roiInfo. The difference between 640 and 761 is a difference of measure.

FAFB, by contrast: 6 of 139,262 neurons have FlyWire's `None` label (spelled `UNASGD` in the cache) as their top label. Under the named-only rule, 5 of them go to GNG or AMMC_L and 1 becomes unknown.

## 3. Where the named-only rule would put the 750 brain-label neurons (connections)

| Literal label | Central | Optic | Nerve cord | Unknown | Median share of the named region |
|---|---|---|---|---|---|
| CentralBrain-unspecified (355) | 240 | 7 | 22 | 86 | 0.24 |
| Optic-unspecified(L) (358) | 10 | 339 | 0 | 9 | 0.32 |
| Optic-unspecified(R) (37) | 0 | 17 | 0 | 20 | 0.37 |

The most frequent named regions are ME(L) 226, LA(L) 54, LO(L) 44 and GNG 54, SMP 31 (L and R), SAD 17. The median share is the named region's synapses over all the neuron's labelled synapses: the assignment is weak, a quarter to a third of the synapses.

## 4. Size of the MaleCNS brain subset (in_brain: central or optic region)

| Treatment | Connections | Sites |
|---|---|---|
| Literal rule, with the proposed `neuropils_provisional.csv` rows (both unspecified brain labels count as brain); the repo's rule | 142,745 | 142,512 |
| Literal rule, rejected labels → unknown | 141,995 | 141,877 |
| Named-only rule (earlier repo rule) | 142,621 | 142,390 |

Under the named-only rule, 137 of the 750 neurons leave the brain subset (115 unknown, 22 nerve cord). 13 neurons whose literal label is `VNC-unspecified` enter it, so the net loss is 124. The other 613 keep a named region instead of region_group unknown.

## Reading

- The counting measure matters more than the label rule. It moves the count of such neurons from 635 to 750, and it applies to every neuron's region, not only these. FAFB fixes it to connections. The group agreed on upstream + downstream on 2026-10-15.
- The label rule is a trade-off. The named-only rule gives 613 neurons a weakly supported but real region and loses 124 from the brain subset. The proposed CSV rows keep every neuron but give 750 the region_group unknown, which most per-region analyses then leave out. The group preferred the CSV rows, because a region holding a quarter to a third of a neuron's synapses is a weak basis for the assignment; the repo follows it, FAFB's `None` label included (6 neurons). No CX-class neuron (the Project 5 roster) changes region; one neuron whose region was FB now has CentralBrain-unspecified (E7 by region: 2,658 → 2,657).
