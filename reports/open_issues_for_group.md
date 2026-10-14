# Open issues for the group (Project 5, before the week 3 session)

Draft message to the other four projects. Each item says what I found, what I did for now, and what needs a group decision. Numbers are from this repository's caches: FAFB 783 (annotations v3.2.0, Buhmann synapses) and MaleCNS v1.0 (Traced, confidence ≥ 0.5).

## Needs a group decision

1. **MaleCNS `super_class` vocabulary.** `config.SUPER_CLASS_VALUES` has only the FAFB levels. I mapped MaleCNS onto them provisionally:
   - `ol_intrinsic` → optic; `cb_intrinsic` → central;
   - every sensory variant → sensory;
   - `ascending_neuron` and `descending_neuron` → ascending and descending;
   - motor and **efferent** → motor;
   - endocrine → endocrine;
   - the visual levels unchanged.

   `vnc_intrinsic` (13,151), `ENS` (47), `vnc_tbc` (1) and the 516 neurons with no superclass become unknown (13,715 in all).

   **Decide:**
   - Should MaleCNS-only levels be added (for example `vnc_intrinsic`)?
   - Should the 110 efferent neurons, which are non-motor efferents in the MaleCNS vocabulary, stay under motor?
2. **MaleCNS nerve-cord list for `add_provisional_flags(…, nerve_cord=…)`.** 34 MaleCNS region names are refused by `eda_shared`. They cover the regions of 22,037 neurons:
   - `ANm`, `IntTct`, `LTct`, `Ov`, `CRN`;
   - `LegNp(T1–T3)`, `mVAC(T1–T3)`;
   - `NTct(UTct-T1)`, `WTct(UTct-T2)`, `HTct(UTct-T3)`;
   - the nerves `DMetaN`, `AbN2`, `AbN3`, `ProLN`, `MesoLN`, `MetaLN`;
   - all of these with their sides.

   They are nerve-cord names, not missing aliases, so they belong in the function's `nerve_cord` argument. The edge-neuropil table adds more nerves that no neuron has as its region: `ADMN`, `AbN1`, `AbN4`, `AbNT`, `CvN`, `DProN`, `MesoAN`, `PDMN`, `PrN`, `ProAN`, `ProCN`, `VProN`.

   My provisional list is `MALECNS_NERVE_CORD` in `src/eda_e.py` of my repository.

   **Decide:** one shared list, so all five repositories use the same MaleCNS brain part. Mine gives 142,621 brain neurons.
3. **MaleCNS "unspecified" labels.** Synapses outside named neuropils are labelled `<unspecified>`, `CentralBrain-unspecified`, `Optic-unspecified(L/R)`, `VNC-unspecified` and `CV-unspecified`. Together they are 2.4% of synapses between Traced neurons.
   - `canonical_neuropil` refuses them.
   - I exclude them when choosing a neuron's region; 985 neurons would otherwise get one as their region.

   **Decide:** should they join `NO_NEUROPIL` in `eda_shared`?
4. **Region lists differ between datasets.** MaleCNS v1.0 has no AMMC, OCG or GA neuropil, and some neuropils differ a lot in size (IPS has 1,512 central neurons in FAFB vs 605 in MaleCNS; FLA 255 vs 77). **Decide:** a common region list for any region-level comparison between datasets.
5. **Synapse scale differs between datasets.** In central-complex type pairs, the median ratio of type-pair synapse totals is 2.0 (hemibrain/FAFB) and 1.75 (MaleCNS/FAFB). That is a proxy for synapses per connection, since a type-pair total also grows with its number of connected neuron pairs. MaleCNS has 28.3M central synapses at τ = 10 against FAFB's 13.4M, with a similar neuron count. An absolute τ = 10 is therefore a stricter label in FAFB. **Decide:** whether cross-dataset comparisons use a relative threshold (the 0.9% and 2.6% of input already in the grid) or a scale-matched absolute one.

## Reported, no decision needed unless someone disagrees

6. **FAFB `sensory_ascending`** (612 neurons) is not in the vocabulary. It is mapped to sensory for now.
7. **FAFB missing ids.** 14 edge ids have no annotation row, not the 7 the conventions expect. There are 139,255 proofread ids against 139,248 annotation rows, and the two lists differ in both directions.
8. **FAFB `n_post` can be filled.** Zenodo's per-neuron postsynapse counts give each neuron's inputs from all partners. That would enable E4 on FAFB too: the median input ratio is 95%.
9. **FlyWire region `None`** (synapses outside every neuropil) is refused by `eda_shared`. I exclude it before choosing a neuron's region.
10. **MaleCNS region source.** neuPrint's per-neuron `roiInfo` is not in the public bulk files. I computed the region from the synapse-partner table with the FAFB rule: the neuropil with most synaptic connections, inputs plus outputs, all partners. If you read `roiInfo`, our regions may differ for neurons near a tie (72 MaleCNS ties).
11. **Hemibrain release.** The plan pins v1.2.1 (neuPrint only). I used Janelia's public v1.2 export of non-cropped Traced neurons for E6. 87% of FAFB's `hemibrain_type` values are found among v1.2 types.
12. **MaleCNS counts for step 0:**
    - Traced neurons: 165,122.
    - Edges ≥ 5: **6,235,649** (Project 4: 6.24M).
    - Self-connections dropped: 101.
    - Edges ≥ 1: 25,563,096.

    No neuPrint token is needed: Janelia's public flat-connectome files reproduce these counts.
