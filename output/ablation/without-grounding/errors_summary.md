# without-grounding 에러 요약

갱신: 2026-10-24 15:14:53  총 8회 실행

## 태스크별 성공 / 실패 단계

| task | runs | success | m1 | m2 | independent_assignment | m4 | m5_planning | m5_execution | goal | unknown | 1차시도 exec 성공 | 최빈 원인 코드 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1_1 | 1 | 0/1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1 | - |
| c2_1 | 1 | 0/1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1 | - |
| c3_1 | 1 | 0/1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1 | - |
| c1_2 | 1 | 0/1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1 | - |
| c2_2 | 1 | 0/1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0/1 | EMPTY_FEASIBLE_EE (1) |
| c3_2 | 1 | 0/1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0/1 | IK_SEARCH_EXHAUSTED (1) |
| c4_1 | 1 | 0/1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1 | - |
| c4_2 | 1 | 0/1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1 | - |

## 원인 코드 빈도 (전체)

| code | count |
|---|---|
| IK_SEARCH_EXHAUSTED | 25 |
| EMPTY_FEASIBLE_EE | 10 |
| INFEASIBLE_NO_CANDIDATE | 2 |

## 실행별 상세

### FAIL c1_1 seed=0 rep=0  stage=m2  code=None  exit=1  1.4min

- [manifest.failure] ValueError at stage=m2
- [traceback] ValueError: visual_selection: invalid model response after 2 attempts: SG1: EE hypotheses must cover exactly ['obj_ladle_ladle']
- [run.log] raise ValueError(f"{stage}: invalid model response after 2 attempts: {error}")
- [run.log] ValueError: visual_selection: invalid model response after 2 attempts: SG1: EE hypotheses must cover exactly ['obj_ladle_ladle']
- traceback: `ValueError: visual_selection: invalid model response after 2 attempts: SG1: EE hypotheses must cover exactly ['obj_ladle_ladle']`

### FAIL c1_2 seed=0 rep=0  stage=m2  code=None  exit=1  0.5min

- [manifest.failure] ValueError at stage=m2
- [traceback] ValueError: visual_selection: invalid model response after 2 attempts: SG1: EE hypotheses must cover exactly ['obj_spatula_spatula']
- [run.log] raise ValueError(f"{stage}: invalid model response after 2 attempts: {error}")
- [run.log] ValueError: visual_selection: invalid model response after 2 attempts: SG1: EE hypotheses must cover exactly ['obj_spatula_spatula']
- traceback: `ValueError: visual_selection: invalid model response after 2 attempts: SG1: EE hypotheses must cover exactly ['obj_spatula_spatula']`

### FAIL c2_1 seed=0 rep=0  stage=m2  code=None  exit=1  0.4min

- [manifest.failure] ValueError at stage=m2
- [traceback] ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination
- [run.log] raise ValueError(f"{stage}: invalid model response after 2 attempts: {error}")
- [run.log] ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination
- traceback: `ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination`

### FAIL c2_2 seed=0 rep=0  stage=m4  code=EMPTY_FEASIBLE_EE  exit=1  0.5min

- [manifest.failure] RuntimeError at stage=m4
- [m4.rejection] EMPTY_FEASIBLE_EE: feasible_ee is empty; no EE can be assigned
- [traceback] RuntimeError: M4 found no plan: INFEASIBLE_NO_CANDIDATE; see m4.json rejections
- [run.log] raise RuntimeError(
- [run.log] RuntimeError: M4 found no plan: INFEASIBLE_NO_CANDIDATE; see m4.json rejections
- traceback: `RuntimeError: M4 found no plan: INFEASIBLE_NO_CANDIDATE; see m4.json rejections`

### FAIL c3_1 seed=0 rep=0  stage=m2  code=None  exit=1  0.3min

- [manifest.failure] ValueError at stage=m2
- [traceback] ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination
- [run.log] raise ValueError(f"{stage}: invalid model response after 2 attempts: {error}")
- [run.log] ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination
- traceback: `ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination`

### FAIL c3_2 seed=0 rep=0  stage=m5_planning  code=IK_SEARCH_EXHAUSTED  exit=1  7.3min

- [manifest.failure] SystemExit at stage=m5
- [run_result.detail] MotionPlanningPipelineError: no generated keyframe strategy produced a finalizable path: SG2_s6_d1:S1=IK_SEARCH_EXHAUSTED (SG2_s6_d1:S1:3:KF3: 0/0 IK branches passed state validity; solver: no full-pose IK solution converged from deterministic seeds; attempted eef position_m=[2.5
- [m5_summary.detail] MotionPlanningPipelineError: no generated keyframe strategy produced a finalizable path: SG2_s6_d1:S1=IK_SEARCH_EXHAUSTED (SG2_s6_d1:S1:3:KF3: 0/0 IK branches passed state validity; solver: no full-pose IK solution converged from deterministic seeds; attempted eef position_m=[2.5
- [m5_summary.failure_type] MotionPlanningPipelineError
- [m5_failure[0]] SG2_s6_d1:S1: IK_SEARCH_EXHAUSTED: SG2_s6_d1:S1:3:KF3: 0/0 IK branches passed state validity; solver: no full-pose IK solution converged from deterministic seeds; attempted eef position_m=[2.575985, -2.909425, 1.297421] frame_ref=object:fruit_b anchor=top
- [live.failure] MotionPlanningPipelineError: no generated keyframe strategy produced a finalizable path: SG2_s6_d1:S1=IK_SEARCH_EXHAUSTED (SG2_s6_d1:S1:3:KF3: 0/0 IK branches passed state validity; solver: no full-pose IK solution converged from deterministic seeds; attempted eef position_m=[2.575985
- [live.failed_step] SG2_s6_d1:
- [run.log] "status": "FAILED",

### FAIL c4_1 seed=0 rep=0  stage=m2  code=None  exit=1  0.5min

- [manifest.failure] ValueError at stage=m2
- [traceback] ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination
- [run.log] raise ValueError(f"{stage}: invalid model response after 2 attempts: {error}")
- [run.log] ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination
- traceback: `ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination`

### FAIL c4_2 seed=0 rep=0  stage=m2  code=None  exit=1  0.9min

- [manifest.failure] ValueError at stage=m2
- [traceback] ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination
- [run.log] raise ValueError(f"{stage}: invalid model response after 2 attempts: {error}")
- [run.log] ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination
- traceback: `ValueError: visual_selection: invalid model response after 2 attempts: SG1: select a scene tool distinct from targets/destination`
