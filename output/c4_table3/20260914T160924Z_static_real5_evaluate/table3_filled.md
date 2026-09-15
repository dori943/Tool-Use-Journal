# Table 3: C4 Static Real-5 Evaluation Results

**Evaluate run**: 20260914T160924Z
**Inference run**: 20260914T064753Z_static_real5_inference
**manifest_sha256**: 89c476f3afd9f00223fff928e6d86c124905ad53d928d70b5b95e8e9f8726dbc
**GT**: evaluator_only_gt.yaml (Kandukuri et al., Table 5 / EV-RealPhys)
**Scoring**: object-level median → macro mean over 5 objects → mean ± std over 5 repeats

| Prior/SiPhy | Condition | Mass_MnRE | Mass_Acc | Suction_Acc | Suction_PF | Friction_MAE | Clearance_RelErr | Feas_Acc | DA | Crit |
|-------------|-----------|-----------|----------|-------------|------------|--------------|------------------|----------|----|------|
| Prior | Name-only | - | - | - | - | - | - | - | - | - |
| Prior | + Affordance labels | - | - | - | - | - | - | - | - | - |
| SiPhy | SiPhy (as adopted) | 0.9085 ± 0.0046 | - | - | - | - | - | - | - | - |
| SiPhy | + Geometric grounding (ours, §III-A) | 0.9085 ± 0.0046 | - | - | - | - | - | - | - | - |
|  | Ours (full: + friction, + spatial) | 0.9085 ± 0.0046 | - | - | - | 0.1550 ± 0.0048 | - | - | - | - |
|  | GT numerics (upper bound) | - | - | - | - | - | - | - | - | - |

## Notes

- **Mass MnRE** = mean over 5 objects of |object-level-median-pred − GT| / GT, averaged over 5 repeats
- **Friction MAE** = mean over 5 objects of |object-level-median-pred − GT|, averaged over 5 repeats
- This is the **Static Real-5 visual-prior variant**: 50 static RGB/RGB-D inputs (10 per object); the legacy RGB-D trajectory friction result is auxiliary and is not combined here.
- geometric_grounding and ours_full Mass MnRE are **identical to siphy_adopted** (shared_prediction_source=SiPhy)
- Mass Acc: no `mass_bins` in GT file; scoring.py `score_mass_acc` requires EE payload data not present → `-`
- Suction Acc, Suction PF: no suction predictions in inference run → `-`
- Clearance RelErr: no clearance predictions in inference run → `-`
- Feas. Acc, DA, Crit.: scoring functions require independent physical trial GT not in evaluator_only_gt.yaml → `-`
- 0 parsing failures; 0 excluded samples
- 95% CI uses t-distribution df=4, t=2.776
