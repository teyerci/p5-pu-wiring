# FlyWire ↔ male CNS type mapping and coverage (week 3)

`src/type_mapping.py`, run at the end of `notebooks/00_integration.ipynb`. The full mapping is in `reports/type_mapping_fafb_malecns.csv` (one row per match group) and coverage by region in `reports/type_mapping_coverage_by_region.csv`.

**Rule.** fafb `cell_type` is matched to malecns `flywire_type` only, never by comparing type names. Compound values ("A,B": one type matching the union of several) are merged into match groups. malecns is its brain part (142,621 neurons).

| | fafb 783 | malecns v1.0 (brain part) |
|---|---|---|
| Types | 8,846 | 8,272 |
| Types in a matched group | 7,610 (86%) | 7,329 (89%) |
| Neurons | 139,262 | 142,621 |
| Neurons in a matched group | **132,329 (95.0%)** | **136,766 (95.9%)** |
| Neurons without a `flywire_type` | — | 2,515 |

**Match groups:** 9,089 in all, of which 7,294 are matched, 1,235 are fafb-only and 560 are malecns-only.

**By region group** (share of neurons in a matched group, fafb / malecns):
- sensory periphery: 98.5% / 99.1%;
- higher order: 89.9% / 87.9%;
- motor periphery: 75.0% / 84.4%.

**By neuropil:**
- **Lowest in fafb:** OCG (8%; malecns has no OCG neuropil), CAN (50%, 2 neurons), FLA (64%), GNG (72%) and GA (73%; no GA in malecns).
- **Lowest in malecns:** EPA (23%), GOR (38%), FLA (56%) and SIP (58%).
- **Central complex** (by region; fafb / malecns): PB 91% / 95%, FB 87% / 88%, EB 84% / 76% and NO 100% / 100%. P5's cross-connectome check therefore covers most, not all, of its region. The unmatched part is reported with each comparison (step 15 matches 167 groups with malecns and 177 with the hemibrain, of 222 CX types).

**Reading it.** An unmatched type is not evidence that the type is sex-specific: annotation gaps and the volumes' coverage also leave types unmatched (EDA plan, workstream B). The lowest coverage sits in the motor and neuroendocrine periphery (GNG, FLA) and, for malecns, in EPA and GOR. Cross-dataset claims there need care.
