# without-reasoner 에러 요약

갱신: 2026-10-24 18:31:07  총 8회 실행

## 태스크별 성공 / 실패 단계

| task | runs | success | m1 | m2 | independent_assignment | m4 | m5_planning | m5_execution | goal | unknown | 1차시도 exec 성공 | 최빈 원인 코드 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1_1 | 1 | 0/1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0/1 | EXECUTION_FAILED (1) |
| c2_1 | 1 | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1/1 | - |
| c3_1 | 1 | 0/1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0/1 | PAYLOAD_EXCEEDED (1) |
| c1_2 | 1 | 0/1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0/1 | PAYLOAD_EXCEEDED (1) |
| c2_2 | 1 | 0/1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0/1 | PAYLOAD_EXCEEDED (1) |
| c3_2 | 1 | 0/1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1 | BATCH_TIMEOUT (1) |
| c4_1 | 1 | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1/1 | - |
| c4_2 | 1 | 0/1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0/1 | - |

## 원인 코드 빈도 (전체)

| code | count |
|---|---|
| PAYLOAD_EXCEEDED | 9 |
| NO_CANDIDATES | 9 |
| EXECUTION_FAILED | 7 |
| GEOM_QUERY | 4 |

## 실패 메시지에 등장한 물체 (반복 drop/slip 확인용)

| object | count |
|---|---|
| scissors | 1 |

## 실행별 상세

### FAIL c1_1 seed=0 rep=0  stage=m5_execution  code=EXECUTION_FAILED  exit=1  3.9min

- [manifest.failure] SystemExit at stage=m5
- [run_result.detail] RuntimeError: object 'scissors' is 0.0847 m from the EE; attach limit is 0.0200 m; closest geom pair 'gripper0_right_right_fingertip_collision' <-> 'scissors_g9' measured by GEOM_QUERY
- [m5_summary.detail] RuntimeError: object 'scissors' is 0.0847 m from the EE; attach limit is 0.0200 m; closest geom pair 'gripper0_right_right_fingertip_collision' <-> 'scissors_g9' measured by GEOM_QUERY
- [m5_summary.failure_type] RuntimeError
- [live.failure] RuntimeError: object 'scissors' is 0.0847 m from the EE; attach limit is 0.0200 m; closest geom pair 'gripper0_right_right_fingertip_collision' <-> 'scissors_g9' measured by GEOM_QUERY
- [live.failed_step] SG1_s1_d1:
- [run.log] "status": "FAILED",
- [run.log] "detail": "RuntimeError: object 'scissors' is 0.0847 m from the EE; attach limit is 0.0200 m; closest geom pair 'gripper0_right_right_fingertip_collision' <-> 'scissors_g9' measured by GEOM_QUERY",

### FAIL c1_2 seed=0 rep=0  stage=m4  code=PAYLOAD_EXCEEDED  exit=1  0.8min

- [manifest.failure] SystemExit at stage=m4
- [m4.rejection] PAYLOAD_EXCEEDED: combined object/tool load exceeds EE payload
- [m4.rejection] NO_CANDIDATES: no feasible candidate survived thresholding and static checks
- [run.log] [중단] run_m4 단계가 exit=3 로 끝났습니다.

### OK  c2_1 seed=0 rep=0  stage=none  code=None  exit=0  38.2min


### FAIL c2_2 seed=0 rep=0  stage=m4  code=PAYLOAD_EXCEEDED  exit=1  1.2min

- [manifest.failure] SystemExit at stage=m4
- [m4.rejection] PAYLOAD_EXCEEDED: combined object/tool load exceeds EE payload
- [m4.rejection] NO_CANDIDATES: no feasible candidate survived thresholding and static checks
- [run.log] [중단] run_m4 단계가 exit=3 로 끝났습니다.

### FAIL c3_1 seed=0 rep=0  stage=m4  code=PAYLOAD_EXCEEDED  exit=1  0.7min

- [manifest.failure] SystemExit at stage=m4
- [m4.rejection] PAYLOAD_EXCEEDED: combined object/tool load exceeds EE payload
- [m4.rejection] NO_CANDIDATES: no feasible candidate survived thresholding and static checks
- [run.log] [중단] run_m4 단계가 exit=3 로 끝났습니다.

### FAIL c3_2 seed=0 rep=0  stage=m1  code=BATCH_TIMEOUT  exit=-9  90.0min

- [batch] TIMEOUT after 90 min

### OK  c4_1 seed=0 rep=0  stage=none  code=None  exit=0  35.0min


### FAIL c4_2 seed=0 rep=0  stage=m4  code=None  exit=1  0.7min

- [manifest.failure] SystemExit at stage=m4
- [run.log] [중단] run_m4 단계가 exit=2 로 끝났습니다.
