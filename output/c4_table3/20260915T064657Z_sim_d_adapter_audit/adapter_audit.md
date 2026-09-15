# D_SIM condition adapter audit

이 문서는 adapter 인터페이스 연결 상태를 기록한다. `NEEDS_PROVIDER`는 코드 경로가 있으나 실제 model/provider 설정과 prediction 실행이 아직 없다는 뜻이다.

| condition | status | active modules | prediction source | supported metrics | GT channel |
|---|---|---|---|---|---|
| Name-only | INTERFACE_IMPLEMENTED_NEEDS_PROVIDER | name_prompt | name_only | Mass_Acc, Crit | forbidden_input_keys_rejected |
| + Affordance labels | INTERFACE_IMPLEMENTED_NEEDS_PROVIDER | affordance_prompt | affordance_labels | Mass_Acc, Crit | forbidden_input_keys_rejected |
| SiPhy (as adopted) | INTERFACE_IMPLEMENTED_NEEDS_PROVIDER | SiPhyBackend, shell_mass_integral | siphy_adopted | Mass_Acc, Crit | forbidden_input_keys_rejected |
| + Geometric grounding (ours) | INTERFACE_IMPLEMENTED_NEEDS_PROVIDER | SiPhyBackend, M1_geometry, clearance_normalization | siphy_adopted | Mass_Acc, Clearance_RelErr, Feasibility_Acc, DA, Crit | forbidden_input_keys_rejected |
| Ours (full) | INTERFACE_IMPLEMENTED_NEEDS_PROVIDER | SiPhyBackend, M1_geometry, visual_friction, M4_downstream | siphy_adopted | Mass_Acc, Suction_Acc, Suction_PF, Clearance_RelErr, Feasibility_Acc, DA, Crit | forbidden_input_keys_rejected |
| GT numerics (oracle) | ORACLE_ISOLATED | oracle_channel_only | oracle | Mass_Acc, Suction_Acc, Suction_PF, Clearance_RelErr, Feasibility_Acc, DA, Crit | forbidden_input_keys_rejected |

## 규칙

- `+ Geometric grounding`과 `Ours`의 Mass Acc는 SiPhy mass prediction cache를 공유하며 추가 model call을 하지 않는다.
- SiPhy mass cache가 없으면 shared adapter는 실패하고 새 호출을 조용히 만들지 않는다.
- GT 필드가 observation에 들어오면 `GT_LEAKAGE_DETECTED`로 중단한다.
- GT numerics는 `OracleAdapter(oracle=True)`로만 생성하며 일반 provider registry에 등록하지 않는다.
- 현재 상태는 end-to-end prediction 완료가 아니므로 condition matrix의 `NEEDS_IMPLEMENTATION`을 유지한다.
