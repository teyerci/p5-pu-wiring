# Step 0 on the corrected tables (eda_shared v0.2.0)

Built by `scripts/step0.py`. 23 of 23 reference values in Shared Data Corrections match.

| Quantity | Value |
|---|---|
| FAFB neurons | 139,255 = 139,241 kept + 14 added; 7 dropped |
| FAFB superclasses | optic 77,537; central 32,383; sensory 16,904; visual_projection 8,038; ascending 1,750; descending 1,303; sensory_ascending 612; visual_centrifugal 524; motor 110; endocrine 80; unknown 14 |
| FAFB edges | τ = 1: 15,091,983; ≥ 5: 2,700,513 |
| FAFB region | ties 232; unknown 281; per region in `step0.json` |
| MaleCNS brain part | kept 144,414; dropped 20,708; ascending kept 1,841; sensory_ascending kept 534; region ties 76 |
| MaleCNS brain edges | τ = 1: 21,706,010; ≥ 5: 5,143,528; synapses: brain 99,982,458, nerve cord 24,041,864, no location 251 |
| Self-connections dropped | FAFB 0; MaleCNS 101 |
| Central population | FAFB 45,203; MaleCNS 45,642 |
| Transmitter correction | fafb known_nt: 77,169 rows, 18,513 changed; fafb predicted: 62,086 rows, 0 changed; malecns kc_override: 4,064 rows, 4,064 changed; malecns predicted: 161,058 rows, 0 changed |
| Kenyon cells (FAFB) | 5,177 of 5,177 acetylcholine; dopamine 5,909 → 638 |
| CX roster (P5) | 2,875 neurons; transmitter changed for 171 |

Kenyon-cell types found (check that the list holds only Kenyon cells): fafb:KCa'b'-ap1 (1), fafb:KCab (1643), fafb:KCab-p (128), fafb:KCapbp-ap1 (280), fafb:KCapbp-ap2 (298), fafb:KCapbp-m (338), fafb:KCg-d (295), fafb:KCg-m (2189), fafb:KCg-s1 (2), fafb:KCg-s2 (2), fafb:KCg-s3 (1), malecns:KC (2), malecns:KCa'b'-ap1 (199), malecns:KCa'b'-ap2 (291), malecns:KCa'b'-m (205), malecns:KCab-c (488), malecns:KCab-m (536), malecns:KCab-p (129), malecns:KCab-s (657), malecns:KCg (1), malecns:KCg-d (206), malecns:KCg-m (1342), malecns:KCg-s1 (2), malecns:KCg-s2 (2), malecns:KCg-s3 (2), malecns:KCg-s4 (2).
