# Week 2 report — FAFB 783 (annotations v3.2.0, Buhmann synapses)

## Source columns

Annotation table (`Supplemental_file1_neuron_annotations.tsv`):

`cell_class`, `cell_sub_class`, `cell_type`, `dimorphism`, `fbbt_id`, `flow`, `fru_dsx`, `hartenstein_hemilineage`, `hemibrain_type`, `ito_lee_hemilineage`, `known_nt`, `known_nt_source`, `matching_notes`, `nerve`, `nucleus_id`, `pos_x`, `pos_y`, `pos_z`, `root_id`, `side`, `soma_x`, `soma_y`, `soma_z`, `status`, `super_class`, `supertype`, `supervoxel_id`, `synonyms`, `top_nt`, `top_nt_conf`, `vfb_id`

Connection table (`proofread_connections_783.feather`): `pre_pt_root_id`, `post_pt_root_id`, `neuropil`, `syn_count`, `gaba_avg`, `ach_avg`, `glut_avg`, `oct_avg`, `ser_avg`, `da_avg`

## Source value counts: `side`

| value | neurons |
|---|---|
| left | 69,956 |
| right | 69,088 |
| center | 174 |
| na | 30 |

## Source value counts: `super_class`

| value | neurons |
|---|---|
| optic | 77,541 |
| central | 32,383 |
| sensory | 16,907 |
| visual_projection | 8,038 |
| ascending | 1,750 |
| descending | 1,303 |
| sensory_ascending | 612 |
| visual_centrifugal | 524 |
| motor | 110 |
| endocrine | 80 |

## Source value counts: `top_nt`

| value | neurons |
|---|---|
| acetylcholine | 86,193 |
| glutamate | 24,875 |
| gaba | 19,171 |
| dopamine | 5,909 |
| serotonin | 2,282 |
| nan | 602 |
| octopamine | 216 |

## Missing values in the neurons table (after mapping)

| column | missing |
|---|---|
| neuron_id | 0 |
| dataset | 0 |
| release | 0 |
| cell_type | 1,542 |
| cell_class | 31,744 |
| cell_sub_class | 113,435 |
| super_class | 0 |
| side | 0 |
| region | 0 |
| nt_type | 0 |
| nt_confidence | 616 |
| n_pre | 139,262 |
| n_post | 139,262 |
| flywire_type | 139,262 |
| manc_type | 139,262 |
| hemibrain_type | 105,991 |
| dimorphism | 14 |
| status | 138,604 |
| hemilineage | 101,720 |
| nt_known | 51,425 |

## Counts

| quantity | value | expected |
|---|---|---|
| neurons in the annotation table | 139,248 | 139,248 |
| proofread root ids (Zenodo) | 139,255 | 139,255 published |
| edge ids with no annotation row, added as unknown | 14 | 7 expected |
| neurons table rows | 139,262 | |
| edges ≥ 1 synapse | 15,091,983 | |
| edges ≥ 5 synapses | 2,700,513 | 2,700,513 |
| edges ≥ 10 synapses | 1,066,822 | |
| synapses in edges | 54,492,922 | |
| distinct cell types | 8,846 | |
| self-connections dropped | 0 | 0 |
