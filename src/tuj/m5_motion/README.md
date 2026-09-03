# Motion Planner

## 빠른 시작

Motion Planner(M5) 모듈 위치는 `src/tuj/m5_motion/`이며, 패키지 경로는
`tuj.m5_motion`이다. Task Planner(M4)의 패키지 경로는 `tuj.m4_taskplanner`다.
(파이프라인 재정의로 모듈 번호를 하나씩 밀어 단계명과 패키지 경로를 일치시켰다.) 별도 worktree라면 두 저장소의 `src/`가 모두
Python import 경로에 있어야 하며, 아래 M5 runner가 이를 자동 설정한다.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest src/tuj/m5_motion/tests -q
```

Windows PowerShell에서 가상환경을 활성화하려면 다음 명령을 먼저 실행한다.

```powershell
.\.venv\Scripts\Activate.ps1
```

테스트는 `tests/conftest.py`가 현재 `src/`와 같은 상위 폴더의 Task Planner worktree
`src/`를 import 경로에 넣으므로 별도 설치 없이 동작한다. 테스트 asset 경로는 저장소
기준으로 해석하므로 다른 작업 디렉터리에서도 실행할 수 있다.

OpenAI keyframe 기능을 쓸 때만 추가로 설치한다.

```bash
python -m pip install "openai>=2,<3"
```

현재 통합 단계 명칭은 **M4 Task Planner**, **M5 Motion Planner**이고 패키지 경로도
이에 맞춘다. 다만 기존 Git 브랜치 이름은 각각 `feature/dain-m3`, `feature/dain-m4`를
유지한다. 두 feature
branch를 병렬로 검토할 때는 별도 worktree를 사용한다.

```bash
git fetch origin \
  feature/dain-m3:refs/remotes/origin/feature/dain-m3 \
  feature/dain-m4:refs/remotes/origin/feature/dain-m4
git worktree add ../tuj-m3 origin/feature/dain-m3
git worktree add ../tuj-m4 origin/feature/dain-m4
python -m pip install -r ../tuj-m4/requirements.txt
```

### 범용 M4 → M5 실행

범용 M5 runner는 C1_1 경로나 환경을 기본값으로 추정하지 않는다. M4 결과와 초기
`WorldSnapshot`을 명시하거나, 지원되는 Tool-Use-Journal 환경 이름을 넘겨 현재 world를
캡처한다.

지원 환경은 M5 내부 상수로 관리하지 않는다. 새 환경 모듈을
`environments/__init__.py`에서 import해 robosuite에 등록하면 runner가 checkout-local
registry에서 환경 이름을 자동 발견한다. 등록된 환경은 reset 후 다음 공통 계약을
만족해야 한다.

- UR5e 로봇과 bare-flange 환경 생성
- `2F`, `3F`, `vac` 각각의 물리 gripper variant 생성
- 세 EE의 `ee_rack_info`
- M2/GK ID와 일치하는 `obj_body_id`
- `model.get_xml()` 및 동일 seed reset 상태 재현

목록 등록은 자동이지만 위 물리 계약은 생략되지 않는다. 계약이 없는 일반 robosuite
환경은 planning/simulation 전에 명확한 compatibility error로 중단한다.

```powershell
# 다른 Task의 입력 계약만 검증
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --initial-world C:\tasks\sorting\initial_world.json `
  --validate-input-only

# 환경에서 초기 world 캡처 후 실제 계획
$env:OPENAI_API_KEY = "<project-api-key>"
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --environment C2_1_ObjectSorting `
  --initial-ee none
```

`--constraints`와 `--options`는 공통 JSON 또는 subgoal id별 JSON mapping을 받는다.
생략하면 초기 world의 관절 이름을 기준으로 보수적인 기본 제약을 만든다. 출력 폴더를
생략하면 `artifacts/<Task 입력 폴더명>/`을 사용한다.

#### M4 입력 grounding 계약

- `target_ids`는 현재 `WorldSnapshot.objects` 또는 `rack`에 존재하는 구체 ID여야
  한다. `?tool` 같은 미해결 변수와 scene에 없는 ID는 계획 전에 거부한다.
- M4의 structured `grasp`는 선택 입력이다. 없으면 M5 keyframe provider가 선택된
  Tool/object와 현재 world에서 접촉 자세 후보를 만든다.
- `PICK_TOOL`은 Tool object를 물리적으로 attach하고 `attached_object_id`와
  `held_tool_id`를 함께 갱신한다. `RETURN_TOOL`은 같은 object를 detach하고 두 상태를
  해제한다. Tool을 든 동안 EE 교체는 허용하지 않는다.
- 실제 region geometry가 있는 sweep/push는 실행 후 모든 target footprint가 region
  안에 들어왔는지 판정한다. `tool_rest`처럼 M2의 개념적 위치 이름만 있고 geometry가
  없는 운반 단계는 M4→M5 predicted world에 보존된 Tool home pose를 사용한다.

계획부터 MuJoCo 실행까지 한 번에 확인하려면 다음 중 하나를 선택한다.

```powershell
# controller physics와 live viewer (기본 realtime factor 1.0)
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --environment C2_1_ObjectSorting `
  --initial-ee none `
  --simulate controller

# 빠른 deterministic qpos replay, viewer 없음
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --environment C2_1_ObjectSorting `
  --initial-ee none `
  --simulate kinematic `
  --headless

# controller physics를 offscreen MP4로 기록
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --environment C2_1_ObjectSorting `
  --initial-ee none `
  --video artifacts\sorting\run.mp4
```

`--video`는 `--simulate controller`를 암시한다. `--realtime-factor`,
`--hold-seconds`, `--camera`, `--width`, `--height`, `--video-fps`로 재생 및
렌더링을 조정할 수 있다. 전체 sequence의 run/report/goal 평가는
`<output-dir>/simulation/`에, 계획·실행 요약은 `m5_summary.json`에 저장된다.
simulation 실패 또는 목표 검증 실패 시 runner는 종료 코드 2를 반환하며 생성된
report와 manifest는 그대로 보존한다.

파일로 받은 `--initial-world`는 environment reset과 같은 관절·EE 상태여야 한다.
일치하지 않으면 잘못된 scene에서 궤적을 재생하지 않고 중단한다. held/attached
object가 있는 외부 snapshot도 fresh reset으로 재현할 수 없으므로 거부한다. 임의 Task를 바로
시뮬레이션할 때는 `--environment`를 사용해 계획과 실행이 같은 seed의 deterministic
reset을 공유하게 하는 것이 가장 안전하다. 범용 evaluator는 grounded Tool 상태,
pose/joint, 실제 region containment를 판정한다. 그 밖의 scenario-level 성공 조건은
`UNKNOWN`으로 처리된다. C1 sweep의 검증된 rim grasp, 마찰, closed-loop rollback과
cleanup 정책까지 필요하면 전용 runner를 사용한다.

### C1_1 전용 물리 실행

C1_1은 M4 결과에 없는 plate rim-grasp와 분할 sweep geometry를 전용 binder가
보완하므로 범용 runner와 분리한다. `tuj-m3`, `tuj-m4`가 같은 폴더에 있으면 C1_1
runner가 M4 결과를 자동으로 찾는다.

```bash
python ../tuj-m3/scripts/run_m4_task_planner.py
python ../tuj-m4/scripts/run_m5_c1_1_motion_planner.py --validate-input-only

export OPENAI_API_KEY="<project-api-key>"
python ../tuj-m4/scripts/run_m5_c1_1_motion_planner.py
```

PowerShell에서는 마지막 두 명령 앞에 다음과 같이 API key를 설정한다.

```powershell
$env:OPENAI_API_KEY = "<project-api-key>"
```

worktree 이름이 다르면 C1_1 runner에 `--task-planner path/to/task_planner.json`으로
M4 결과를 명시한다. `--controller-kp`, `--planning-time` 등 C1_1 Motion Planner
옵션은 runner가 그대로 전달한다. Tool과 EE는 두 runner 모두 M4 결과에서 읽는다.

이 입력에서는 GK bundle의 `roles.selected_tool`인 `light_plate`가 M4
`candidate_assignments`를 거쳐 M5까지 그대로 전달된다. 세 개로 분할된 sweep도 각각의
target 목록과 실행 순서를 유지한 채 차례대로 motion plan으로 변환된다.

처음에는 `--stop-after-pick`을 추가하면 전체 sweep 전에 설치와 물리 grasp까지만
짧게 검증할 수 있다.

### 최신 C1_1 물리 실행

최신 개선분은 다음을 포함한다.

- weld 없이 2F finger contact와 마찰로 `light_plate`를 파지하고, 양쪽 접촉·리프트·최종
  유지 시간을 실제 MuJoCo 관측값으로 검증
- live tool-in-EEF transform을 사용한 rim contact sweep과 table clearance 보정
- block별 짧은 closed-loop push, 접촉 손실·진행 부족 시 checkpoint rollback 및
  감소된 거리 재시도
- collection zone 포함 여부와 지지 안정성을 확인하는 cleanup pass
- 실행·goal 평가·recovery directive·artifact lineage 저장
- 로봇 제어, grasp, contact, push, recovery 값을 검증된 profile로 분리

검증된 C1_1 기본값은
`src/tuj/m5_motion/examples/c1_1_physical_grasp_profile.json`에 있다. 기존 raw PICK
keyframe을 재사용하면 API 호출 없이 물리 PICK까지만 재현할 수 있다.

```powershell
python scripts/run_m5_c1_1_motion_planner.py `
  --pick-keyframes path\to\pick_keyframes_raw.json `
  --motion-profile src\tuj\m5_motion\examples\c1_1_physical_grasp_profile.json `
  --stop-after-pick
```

영상까지 포함한 전체 실행은 `--video artifacts\c1_1\run.mp4`를 추가한다. PICK
keyframe을 전달하지 않거나 `--sweep-provider openai`를 선택한 경우에만
`OPENAI_API_KEY`가 필요하다. 실행에 사용하는 `configs/robot_spec.json`의 2F 항목에는
검증된 `grip_force_n`과 `fingerpad_friction`이 포함되어야 한다.

## 목표

Task Planner가 생성한 grounded task를 받아 **충돌 없는 시간축 궤적을 생성**하고,
동일한 궤적과 로봇 이벤트를 MuJoCo에서 재생·평가하는 모듈이다.

```text
Task Planner (Task Planner)
        │ MotionPlanRequest (one-way)
        ▼
Motion Planner
  strategy → relative pose resolution → multi-branch IK
           → connected branch search → path validation
           → time parameterization → final validation
        │ MotionPlan (internal artifact)
        ▼
Simulator / Controller ──► ExecutionReport 저장
  joint trajectory + synchronized events replay
        │ failure
        ▼
Recovery Orchestrator
  artifact lineage 역추적 → 원인 모듈만 재실행
```

Task Planner는 task/subgoal, EE, Tool, Grasp, 목표 pose를 결정한다. Motion Planner는
이를 변경하지 않고 현재 robot/world state에서 실행할 수 있는 joint trajectory로
구체화한다.

실행 결과를 Task Planner에 반환하지 않는다. 실패 시에도 Task Planner에 일반적인 callback을
보내는 대신 Recovery Orchestrator가 원인 artifact를 만든 모듈을 직접 재실행한다.
원인이 task 자체일 때만 Task Planner가 재실행 대상이 된다.

## v3 계약 스키마

기계 판독 가능한 JSON Schema는
[`schemas/motion_planner.schema.json`](schemas/motion_planner.schema.json)에 있다.

| 내부 동작 | 입력 | 생성 artifact |
|---|---|---|
| trajectory 생성 | `MotionPlanRequest` | `MotionPlan` |
| simulation 재생 | `SimulationRun` | `ExecutionReport` |
| failure recovery | `ExecutionReport` | `RecoveryDirective` |

### MotionPlanRequest

- `world`: scene, robot joint state, objects, obstacles, rack
- `task`: Task Planner가 grounded한 action/EE/Tool/Grasp/goal
- `constraints`: collision margin, pose tolerance, 속도·가속도 scale
- `constraints.joint_limits`: joint별 속도·가속도·선택적 jerk limit
- `constraints.min_jacobian_singular_value`, `max_jacobian_condition_number`:
  특이점 거부 기준
- `constraints.max_joint_path_step_rad`, `max_cartesian_speed_m_s`: swept-path
  검사 해상도와 Cartesian 속도 상한
- `options`: planning algorithm, timeout, attempts, interpolation step, seed

`MotionTask.action_type`과 `MotionGoal.goal_type`은 서로 다른 추상화 수준이다.

- `action_type`: Task/Manipulation Planner가 정한 `PICK`, `PLACE`, `DOCK`,
  `PUSH`, `SWEEP`, `INSERT` 등의 의미와 resource effect
- `goal_type`: Motion Planner가 실제로 해석할 target representation인 `POSE` 또는
  `JOINT`

따라서 PLACE/DOCK 같은 동작도 `target_pose`, `target_object_id`,
`target_region_id`, approach/retreat constraint를 가진 `POSE` goal로 전달한다. 새
manipulation action을 추가해도 `GoalType`을 늘릴 필요가 없다. 기존 v3.2 JSON의
action형 `goal_type` 값은 읽을 때 `POSE`로 마이그레이션하지만 새 artifact에는 쓰지
않는다. attach/detach/EE exchange 같은 상태 효과는 `action_type`과 명시적 task
metadata에서 판별한다.

접촉 조작은 선택적으로 `MotionTask.contact`에 geometric intent를 전달한다.
`ToolContactPatch`는 tool-local 접촉점·법선·접선·면적·collision geometry를 표현하며,
Task Planner의 action taxonomy와 독립적이다.

### SelectedPlan 자동 변환

Task Planner의 `SelectedPlan`은 선택 결과에 `action_type`, `mode`, source binding,
targets, goal region, EE, Tool, Grasp와 candidate action parameter를 보존한다. adapter는
`subgoal_order` 순서대로 하나의
`MotionPlanRequest`씩 생성하며, 관련 transition/subgoal step도 task metadata에 함께
보존한다.

```python
from tuj.m5_motion import selected_plan_to_motion_requests

requests = selected_plan_to_motion_requests(
    planning_result.selected_plan,
    worlds={
        "pick-part": pick_start_world,
        "place-part": place_start_world,
    },
    constraints=motion_constraints,  # 공통값 또는 subgoal별 mapping/callable
    options=planner_options,          # 공통값 또는 subgoal별 mapping/callable
    selected_plan_artifact_id="selected-plan:run-42",
)
```

여러 subgoal을 한 번에 변환할 때 `WorldSnapshot`은 subgoal별 mapping 또는 callable로
공급해야 한다. 앞 동작 이후 달라진 joint/object 상태를 다음 요청의 시작 상태로 잘못
재사용하는 것을 막기 위해 단일 snapshot 입력은 거부한다. 반면 constraints와 options는
공통 객체를 모든 요청에 재사용할 수 있다. 목표 pose나 PICK grasp가 누락된 plan은 추측해
채우지 않고 `SelectedPlanAdapterError`로 중단한다.

전체 순서를 자동 계획하고 결과를 저장할 때는 orchestrator를 사용한다. EE 교체와 Tool
반납/획득을 독립 motion work unit으로 먼저 만들고, 각 `MotionPlan.expected_final_state`를
다음 요청의 시작 `WorldSnapshot`으로 넘긴다.

```python
from tuj.m5_motion import (
    MotionPlanStore,
    SelectedPlanMotionOrchestrator,
)

orchestrator = SelectedPlanMotionOrchestrator(
    plan_one_request,  # MotionPlanRequest -> MotionPlan 또는 MotionPlanningResult
    store=MotionPlanStore("artifacts/run-42"),
)
sequence = orchestrator.plan(
    planning_result.selected_plan,
    initial_world=current_world,
    constraints=motion_constraints,
    options=planner_options,
)

motion_plans = sequence.plans
manifest = sequence.manifest_path
```

저장은 request/plan별 JSON과 full final `WorldSnapshot`을 포함한 ordered manifest를 임시
파일에 쓴 뒤 atomic replace한다. 중간 transition 이후에는 request identity와 scene
signature도 새 시작 상태에 맞게 다시 고정한다. 다른 process에서 simulation을 시작할 때는
`MotionPlanStore(root).load_manifest()`로 goal 정보까지 포함한 동일 sequence를 복원한다.

### MotionPlan

- `joint_names`: 모든 waypoint가 따르는 joint ordering
- `segments`: APPROACH, GRASP, LIFT, TRANSFER, PLACE, RETREAT 등
- `waypoints`: 시간, joint position/velocity/acceleration, 선택적 EEF pose
- `events`: gripper, suction, object attach/detach, Tool lock/unlock
- `expected_final_state`: 정상 실행 시 도달해야 하는 robot state

### ExecutionReport

- 실제 종료 robot state
- joint/EEF tracking error
- collision count와 goal error
- event 실행 결과
- 실패 위치(`segment_id`, `waypoint_index`, `event_id`)와 관측 증거
- 대용량 trace의 외부 참조(`trace_ref`)

이 report는 Task Planner로 반환하는 응답이 아니라 내부 로그·원인 분석 artifact다.

### ArtifactProvenance / RecoveryDirective

- 모든 artifact에 생성 모듈, invocation, 입력 artifact, 재시도 횟수를 기록한다.
- 실패 관측에서 parent artifact를 거꾸로 따라 root-cause module을 찾는다.
- `RecoveryDirective`는 해당 모듈, 다시 시작할 입력 artifact, 무효화할 downstream
  artifact와 parameter override만 지정한다.

## 설계 규칙

- 좌표·시간·관절 단위는 SI로 고정한다.
- trajectory 시간은 strict monotonic이며 segment끼리 겹치지 않는다.
- 모든 waypoint의 DOF는 `joint_names`와 일치해야 한다.
- gripper·suction·attach 동작은 trajectory clock에 동기화된 event로 표현한다.
- request/plan/result id와 `scene_signature`로 계획과 실행을 추적한다.
- planning과 simulation seed를 명시해 결과를 재현할 수 있게 한다.
- 실행 결과는 Task Planner로 반환하지 않고 내부 `ExecutionReport`로 저장한다.
- `RecoveryDirective.target_module`은 분석된 `RootCause.module`과 같아야 한다.

## Keyframe 전략과 IK branch

VLM이 world 좌표와 quaternion을 직접 쓰지 않도록 내부 후보 artifact는
`KeyframePlanCandidate`를 사용한다. 후보 하나는 독립 pose 집합이 아니라
`PRE_GRASP → GRASP → LIFT → ...`처럼 의미적으로 일관된 전략이다.

각 `RelativeKeyframeSpec`은 다음 정보만 가진다.

- `frame_ref`: `object:<id>` 또는 `rack:<ee>`
- named `anchor`
- frame-local unit approach axis
- 접근 축 방향 offset과 roll
- segment planner 종류와 event
- segment에서 사용할 collision context

`RelativePoseResolver`가 immutable `WorldSnapshot`을 이용해 실제 world pose와
정규화된 quaternion을 계산한다.

`UR5eKinematics.solve_all_ik()`는 keyframe별 full-pose IK 해집합을 보존한다.
`FirstFeasibleBranchSelector`는 `(keyframe, ik_branch)`를 노드로 하는 계층 그래프에서
모든 구간이 연결되는 첫 시퀀스를 찾는다. 현재 정책은 비용 최적화가 아니라
`FIRST_FEASIBLE_CONNECTED_SEQUENCE`이며, 동일 branch와 짧은 관절 이동은 결정적인
탐색 순서로만 사용한다.

Keyframe의 `planner`는 실제 알고리즘으로 dispatch된다.

- `CARTESIAN`: 시작 FK pose와 목표 pose 사이를 SE(3) 직선 보간하고 각 sample의
  연속 IK branch를 선택한다.
- `SAMPLING_BASED`: joint limit 안에서 deterministic seeded bidirectional
  RRT-Connect를 실행한다.
- `JOINT`: 목표 관절 상태까지 bounded-step joint interpolation을 사용한다.

모든 planner가 생성한 edge는 `max_joint_path_step_rad` 간격으로 다시 검사된다.
Jacobian 최소 singular value와 condition number를 함께 검사하여 충돌이 없더라도
특이점 기준을 위반한 상태는 제거한다.

현재 IK 구현은 exact MuJoCo model에 대한 deterministic multi-start DLS이므로 발견한
branch는 모두 보존하지만 analytic completeness를 주장하지 않는다
(`enumeration_complete=false`). 이후 calibrated analytic UR solver를 연결하면 이
flag를 `true`로 올리고 최대 branch를 완전 열거할 수 있다.

## 일반화된 contact manipulation

접촉 작업의 공통 부분은 다음 계층으로 분리되어 있다.

- `ToolAffordanceProvider`: tool geometry/registry에서 `ToolContactPatch` 후보를 제공
- `PushToRegionStrategyProvider`: 관측한 target·region footprint와 선택된 patch로
  approach → engage → push → retract keyframe을 생성
- `ClosedLoopContactExecutor`: checkpoint → 실행 → contact/progress 관측 → rollback →
  감소된 step 재시도를 bounded loop로 수행
- `CompositeGoalEvaluator`: region containment, support stability, grasp retention,
  tool clearance predicate를 fail-closed 방식으로 합성
- `profiles.py`: robot control, grasp, contact, push planning, recovery, tool affordance
  설정을 역할별 validated profile로 분리

generic provider는 구체적인 tool 이름이나 C1_1 block 수를 알지 않는다. 단순 tool-frame
계획에는 `DirectToolContactPoseResolver`를 사용하고, 실제로 파지한 tool은 측정된
tool-in-EEF transform을 적용하는 resolver를 주입한다. C1_1의 기존 물리 실행 경로는
live grasp transform과 MuJoCo contact를 사용하므로 후자에 해당하며, 공통 target ordering,
step-distance reduction, contact schema를 재사용한다.

## EE 교체 템플릿

EE 교체는 VLM 전략을 사용하지 않는다. `EEExchangeTemplateGenerator`가 rack slot의
`dock_pose`, `approach_axis_xyz`, staging/pre-dock 거리를 이용해 다음 macro를 만든다.

```text
old staging → old pre-undock → old dock → unlock/verify
→ bare-flange retreat → new staging → new pre-dock
→ new dock → lock/verify → retreat
```

unlock/lock event 전후의 collision context가 분리되므로, 기존 EE가 장착된 형상,
bare flange, 새 EE 장착 형상이 같은 segment 상태로 섞이지 않는다.

초기 `current_ee`가 `null`이면 Task Planner는 `DETACH_EE` 없이 첫 `ATTACH_EE`만
생성한다. 운영 Motion Planner는 이 요청을 일반 keyframe/IK/RRT pipeline 전에
가로채고 `configs/precomputed_ee_paths/<environment>/bare_to_<EE>.json`의 검증 완료
관절 궤적을 사용한다. 저장 궤적은 시작 관절, UR5e base, rack/dock, collision model
version, lock/verify event, 동역학 limit, 현재 scene의 보간 전 구간 collision을 모두
통과해야 새 요청의 `MotionPlan`으로 바인딩된다.

```text
EE_ATTACH → registry lookup → static/start validation
→ dense current-scene collision validation → request-bound MotionPlan
```

일반 `EE_EXCHANGE`도 pipeline 전에 가로챈다. EE별 반납 파일 3개와 기존 장착
파일 3개를 사용하므로 pairwise EE→EE 파일 6개는 필요 없다.

```text
EE_EXCHANGE_ENTRY(from)
→ load and validate from→bare.exchange-entry joint target
→ direct joint path under ee-attached:<from>, else bounded RRT-Connect
→ dense + final timed collision validation
→ expected final joints = stored exchange-entry

EE_EXCHANGE(from, to)
→ from→bare registry lookup + bare→to registry lookup
→ return start/workcell/dense collision validation
→ exact bare-home joint seam validation
→ attach validation
→ timeline/event offset
→ one request-bound MotionPlan
```

합성 계획의 event 순서는 `TOOL_UNLOCK`, `VERIFY_TOOL_RELEASE`, `TOOL_LOCK`,
`VERIFY_TOOL_LOCK`이며 `dynamic_planner_invoked=false`다. 반납 시작 자세는 해당
`bare→EE` 장착 궤적의 마지막 exchange-entry 자세로 고정된다. Selected Plan
orchestrator는 교환 전 `EE_EXCHANGE_ENTRY`를 자동 삽입하며, 현재 자세에서 저장된
EE별 entry까지 장착 상태 collision model로 동적 계획한다. 이 선행 계획이 성공한
경우에만 반납·재장착 사전 궤적을 실행한다. 선행 요청 없이 `EE_EXCHANGE`를 직접
호출하거나 controller가 entry 허용 오차에 도달하지 못하면 `START_STATE_MISMATCH`로
fail-closed 한다.

기본 `--ee-attach-policy precomputed-required`에서는 miss나 검증 실패를 명시적
failure code로 종료하고 일반 planner로 조용히 fallback하지 않는다. 개발 commissioning
시에만 `precomputed-or-plan`을 사용할 수 있다. 저장 경로 생성과 fresh bare runtime
반복 재생은 `scripts/generate_precomputed_ee_attach.py`가 담당한다. 이때 사용하는
4-keyframe macro는 저장 경로를 최초 생성하기 위한 도구이며 운영 attach 경로가 아니다.
반납 후보 생성, return-only 재생, 두 pairwise 합성 교환 검증은
`scripts/generate_precomputed_ee_return.py`가 담당한다.

lock 직후에는 장착 EE가 rack support에서 빠져나올 때까지
`ee-attached-dock-contact:<EE>`를 사용하고, retreat 완료 후 자유공간
`ee-attached:<EE>`로 전환한다. 이 경계가 없으면 정상적인 dock 이탈 접촉을 collision으로
오판한다.

unlock 직후에도 QC master가 rack의 기존 EE에서 빠져나올 때까지
`bare-flange-dock-contact:<old-EE>`를 사용하고, retreat 뒤 `bare-flange`로 전환한다.

## Trajectory processing

`QuinticTimeParameterizer`는 초기 구현용 보수적인 rest-to-rest parameterizer다.
joint별 velocity/acceleration/jerk limit로 segment 시간을 계산하고 모든 waypoint의
position/velocity/acceleration을 채운다. `MotionPlanBuilder`는 시간 파라미터화 후
호출자가 제공한 final segment validator를 반드시 통과시킨 뒤에만 MotionPlan을 만든다.
FK가 연결된 pipeline에서는 모든 waypoint에 EEF pose를 기록하고
`max_cartesian_speed_m_s`를 넘지 않도록 segment 시간을 추가 scale한다.
추후 TOTG/TOPP-RA와 Ruckig adapter로 교체할 수 있도록 이 경계를 별도 모듈로 유지한다.

## MuJoCo collision validation

`MuJoCoCollisionValidator`는 격리된 robot XML이 아니라 robosuite가 compile한 전체
workcell `MjModel`을 사용한다. 현재 rack 데모의 table, curved rack, support, 분리 EE,
UR5e, QC master collision geom이 모두 검사 대상이다.

- arm joint state를 planner 전용 `MjData`에 적용하고 `mj_forward`로 contact를 계산한다.
- moving geom의 MuJoCo contact margin을 요청의 `collision_margin_m`까지 확장하여
  penetration뿐 아니라 안전거리 미달도 검출한다.
- robot-world, non-adjacent self-collision, active EE, attached object를 같은 결과 형식으로
  검사한다.
- `allowed_collision_pairs`와 `touch_links`는 geom/body/logical entity 이름으로 적용한다.
- `MuJoCoInterpolatingEdgePlanner`는 양 끝점 사이 모든 joint sample을 검사하므로,
  endpoint가 각각 안전하지만 중간에 rack을 통과하는 경로도 거부한다.
- time parameterization 후에는 같은 backend를 `MotionPlanBuilder.final_segment_validator`
  로 다시 사용한다.

EE lock/unlock은 ACM만 바꾸지 않고 `MuJoCoCollisionModelRegistry`가
`collision_model_version`별 hard-attached 모델을 선택한다. 운반 물체는
`AttachedObjectTransform`의 free joint와 body/site 기준 grasp transform을 각 검사
상태에 실제 적용한다. OPEN/CLOSED/HOLDING gripper 형상도 collision context의
`kinematic_joint_positions`로 고정한다. attachment transform, free joint 또는 기준
body/site가 없으면 fail-closed 한다.

`EEWorkcellCollisionModelCompiler`는 reset된 `EERackLayoutEnv`의 MJCF를 source of
truth로 사용해 bare-flange와 3F/Vacuum/2F hard-attached 모델을 자동 compile한다.
QC coupling 위치는 `quick_changer_master.xml`의 `eef` body에서 읽고, 선택 EE root를
rack의 world body에서 제거해 QC master 하위에 reparent한다. 장착 모델도 arm DOF는
6개로 유지되며, 선택되지 않은 EE는 rack에 world-fixed 상태로 남는다.

`renders/`는 Tool 방향과 접촉 자세의 시각 검증에만 사용한다. 길이, coupling offset,
body hierarchy, collision geom은 MJCF 및 compile된 `MjModel`에서만 읽는다.

`EEExchangeTemplateGenerator.build_collision_contexts()`는 자유 이동 context와 실제
dock 접촉이 허용되는 context를 분리한다. 따라서 EE-support / QC-target EE 접촉 허용은
dock segment에만 적용되고, TOOL_LOCK/TOOL_UNLOCK 뒤에는 collision model 자체가 바뀐다.

## Tool-Use-Journal 환경 연결

`ToolUseJournalEnvironmentAdapter`와 `ToolUseJournalCollisionModelCompiler`는
`feature/yebin-task-environments`의 다음 환경을 직접 지원한다.

- `C1_1_LegoSweep`
- `C2_1_ObjectSorting`
- EE ID `2F`, `3F`, `vac`

검증 기준 revision은 `113f84686d94203dbd90f1836187e351aa0b246d`다. 장면 pose,
object free-joint state, UR5e base, EE root와 rack slot은 reset된 환경의 compile된 MJCF와
`MjData`에서 읽는다. 렌더 이미지는 수치 입력으로 사용하지 않는다.

`robot_spec.json`의 `current_ee`는 `null`이며 초기에는 장착 EE가 없다. Motion Planner는
이 값을 임의로 `2F`로 바꾸지 않고 M4가 선택해 전달한 EE를 `active_ee`로 명시해 물리
환경을 만든다. `gripper_types`를 생략하면 `NullGripper`가 되므로 반드시 아래 helper를
사용하고, planning 전에 `require_physical_ee()`로 실제 모델을 확인한다.

```python
from tuj.m5_motion import (
    ToolUseJournalCollisionModelCompiler,
    ToolUseJournalEnvironmentAdapter,
    make_tool_use_journal_env,
)

repository = r"C:\path\to\Tool-Use-Journal"
env = make_tool_use_journal_env(
    repository,
    "C2_1_ObjectSorting",
    active_ee="2F",
    seed=0,
)
env.reset()

adapter = ToolUseJournalEnvironmentAdapter(env)
adapter.require_physical_ee("2F")
world = adapter.world_snapshot()
kinematics = adapter.make_kinematics()

compiler = ToolUseJournalCollisionModelCompiler.from_repository(
    env,
    repository,
    seed=0,
)
contexts = compiler.build_ee_exchange_contexts(from_ee="2F", to_ee="vac")
collision_registry = compiler.build_collision_registry(
    contexts,
    collision_margin_m=0.005,
)
```

실제 전체 계획에서는 registry를 수동으로 하나 고정하지 않고 환경에 묶인
`plan_one_request`를 만든다. 이 callable은 keyframe 생성 직후 요청별 context를 만들고
동일한 registry를 IK branch, edge sample, 최종 timed waypoint 검증에 전달한다.

```python
from tuj.m5_motion import (
    MotionPlanStore,
    SelectedPlanMotionOrchestrator,
    ToolUseJournalEnvironmentAdapter,
    ToolUseJournalMotionRequestPlanner,
)

# env는 make_tool_use_journal_env(..., active_ee="2F") 후 reset된 실제 workcell env
adapter = ToolUseJournalEnvironmentAdapter(env)
plan_one_request = ToolUseJournalMotionRequestPlanner.from_environment(
    env,
    repository,
    seed=0,
)

orchestrator = SelectedPlanMotionOrchestrator(
    plan_one_request,
    store=MotionPlanStore("artifacts/run-42"),
)
sequence = orchestrator.plan(
    selected_plan,
    initial_world=adapter.world_snapshot(),
    constraints=motion_constraints,
    options=planner_options,
)
```

factory가 적용하는 collision state는 GPT 출력이 아니라 deterministic 후처리다.

- 일반 이동: 모든 keyframe에 현재 mounted-EE context를 명시한다.
- PICK: GRASP 구간만 대상 접촉을 허용하고 `ATTACH_OBJECT` 뒤에는 후보별
  object-to-hand transform으로 물체를 EE와 함께 이동시킨다.
- PLACE: PLACE 구간에만 `allowed_touch_objects`/`target_region_id` 접촉을 허용하고
  `DETACH_OBJECT` 뒤에는 물체 free joint를 목표 world pose에 고정해 RETREAT를 검사한다.
- EE_EXCHANGE: attached → dock-contact → bare → dock-contact → 새 attached 모델을
  keyframe event 경계에서 전환한다.

PLACE 목표 표면과의 의도된 접촉이 필요하면 `MotionTask.allowed_touch_objects` 또는
`target_region_id`에 MuJoCo entity/body/geom selector를 넣어야 한다. 허용 목록은 PLACE
contact context에만 적용되며 전체 경로에 전역 허용되지 않는다.
PICK 직후 물체가 지지면에서 떨어지는 첫 LIFT 구간에만 지지면 접촉을 허용해야 하면
`MotionTask.metadata["support_collision_selectors"]`에 selector 목록을 넣는다. 첫 LIFT가
끝나면 strict attached context로 자동 복귀한다. 물체를 도구처럼 들고 의도적으로 다른
물체를 접촉하는 일반 동작은 `allowed_touch_objects`가 그 attached-object context에
적용된다.

runtime에서 이미 물체를 들고 있는 snapshot을 만들 때는 실제 attachment state도 함께
전달한다.

```python
from tuj.m5_motion import attached_object_transform_from_state

transform = (
    attached_object_transform_from_state(runtime.attachment)
    if runtime.attachment is not None
    else None
)
world = ToolUseJournalEnvironmentAdapter(runtime.env).world_snapshot(
    attached_object_transform=transform,
)
```

여러 환경을 한 process에서 계획하면 환경별 planner를 만든 뒤
`WorkcellMotionRequestRouter`에 `environment_name`으로 등록한다. 등록되지 않은 환경이나
요청 EE와 물리 active EE가 다른 경우에는 다른 모델로 추측하지 않고 실패한다.

대상 rack의 base, support와 보관 EE는 원본에서 모두 `contype=0`,
`conaffinity=0`인 시각화 전용 형상이다. compiler는 live simulator나 원본 파일을
수정하지 않고 planner-owned MJCF 복사본에서 rack collision geom만 활성화한다.
장착 상태에서는 해당 EE의 rack display 복제본을 제거하고, robosuite가 실제로
UR5e에 장착한 gripper subtree를 사용한다. bare/2F/3F/vac 모델의 arm 및 object
joint state는 하나의 live snapshot으로 동기화된다.

### EE 교체 runtime과 MotionPlan replay

`ToolUseJournalEERuntime`은 MuJoCo topology를 한 모델 안에서 억지로 바꾸지 않는다.
`TOOL_UNLOCK`/`TOOL_LOCK`에서 각각 bare 또는 대상 EE가 실제로 장착된 robosuite
환경을 새로 만들고, 다음 상태를 이름 기준으로 원자적으로 이관한 뒤 live env를 교체한다.

- arm 및 object joint qpos/qvel
- 공통 actuator control
- simulation time
- 현재 EE metadata
- rack EE 표시 상태(장착 EE의 rack 복제본은 숨김)

`VERIFY_TOOL_RELEASE`는 bare model과 반환된 rack EE 표시를 확인하고,
`VERIFY_TOOL_LOCK`은 실제 mounted gripper class와 rack 복제본 제거를 확인한다.
중간 생성이나 상태 검증이 실패하면 기존 env를 유지하고 event를 실패 처리한다.

두 player가 같은 event runtime을 사용한다.

- `ToolUseJournalKinematicTrajectoryPlayer`: timestamp의 qpos를 직접 적용하는 빠른
  deterministic 검증용 player
- `ToolUseJournalControllerTrajectoryPlayer`: absolute `JointPositionController` 목표를
  주고 robosuite torque/controller/actuator physics loop를 실제로 진행하는 player

둘 다 실행 결과를 `ExecutedEvent`, 최종 robot state, tracking/collision metric을
포함하는 `ExecutionReport`로 저장한다. planner collision registry를
`collision_probe`로 넘기면 replay 중 segment context별 충돌도 다시 검사한다.

```python
from tuj.m5_motion import (
    ToolUseJournalEERuntime,
    ToolUseJournalKinematicTrajectoryPlayer,
)

runtime = ToolUseJournalEERuntime.from_repository(
    repository,
    "C2_1_ObjectSorting",
    active_ee="2F",
    seed=0,
)
player = ToolUseJournalKinematicTrajectoryPlayer(
    runtime,
    collision_probe=collision_registry,
)
report = player.execute(simulation_run)
```

Controller-backed 실행은 controller 전용 factory로 모든 bare/2F/3F/vac variant를
같은 설정으로 생성해야 한다.

```python
from tuj.m5_motion import (
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
)

runtime = ToolUseJournalEERuntime.from_repository_for_controller(
    repository,
    "C2_1_ObjectSorting",
    active_ee="vac",
    seed=0,
    control_timestep_s=simulation_run.config.control_timestep_s,
)
report = ToolUseJournalControllerTrajectoryPlayer(runtime).execute(
    simulation_run
)
```

### 전체 MotionPlan 시퀀스 실행과 목표 검증

`SelectedPlanMotionOrchestrator`가 만든 전체 plan sequence는
`SelectedPlanSimulationOrchestrator`로 순서대로 실행한다. 각 실행 사이에는 직전
`ExecutionReport.final_robot_state`와 다음 plan의 시작 waypoint를 비교한다. 오차가 실행
tolerance를 넘거나 attached object/held Tool 상태가 다르면 남은 plan을 실행하지 않고
`STATE_DIVERGED`로 중단한다.

개별 `ExecutionReport.SUCCESS`는 controller/physics replay가 정상 종료됐다는 뜻이다.
전체 sequence의 `SUCCESS`는 여기에 grounded goal 검증까지 통과해야 한다. 따라서
collision-free replay가 끝났더라도 PICK 대상이 붙어 있지 않거나, PLACE 물체가 목표 pose에
없거나, scenario-level evaluator가 task goal 실패를 반환하면 전체 결과는 `GOAL_FAILED`다.
평가된 position/orientation error와 근거는 report metadata와 simulation manifest에 함께
저장된다. 관측값이 없으면 PASS로 추정하지 않고 `UNKNOWN`으로 처리한다.

Tool-Use-Journal에서는 실행 adapter가 plan에 고정된 event-scoped collision context로
registry를 다시 만들고, controller player와 실제 world snapshot을 공급한다.

```python
from tuj.m5_motion import (
    SimulationArtifactStore,
    ToolUseJournalExecutionAdapter,
)

execution_adapter = ToolUseJournalExecutionAdapter.from_repository(
    runtime,
    repository,
    seed=0,
    controller=True,
)
execution = execution_adapter.execute(
    sequence,
    store=SimulationArtifactStore("artifacts/run-42/simulation"),
)
if not execution.successful:
    raise RuntimeError(f"{execution.status.value}: {execution.detail}")
```

전체 task의 성공 조건이 개별 `MotionGoal`보다 넓으면 `sequence_goal_evaluator`를 전달한다.
예를 들어 C1 sweep은 "모든 block이 collection zone 안에 있음"을 별도 evaluator로 검사해야
한다. `src/tuj/m5_motion/examples/c1_1_openai_motion_run.py`도 controller replay와 이 task goal이 모두
성공한 경우에만 최종 `status=SUCCESS`와 exit code 0을 반환한다.

실패 report나 goal-failed sequence는 `RecoveryOrchestrator`가 artifact lineage를 따라
원인 모듈의 입력 artifact에서만 재시작하도록 `RecoveryDirective`를 만든다. 알 수 없는
failure code는 임의 모듈에 귀속하지 않으며, 실제 재실행은 명시적으로 등록된 handler에만
전달된다.

```python
from tuj.m5_motion import RecoveryOrchestrator
from tuj.m5_motion.schema import ModuleName

recovery = RecoveryOrchestrator(
    {
        ModuleName.MOTION_PLANNER: retry_motion_planner,
        ModuleName.SIMULATOR: retry_simulation,
        ModuleName.TASK_PLANNER: retry_task_planner,
    }
)
directive = recovery.directive_for_sequence(execution)
outcome = recovery.execute(directive)
```

지원 event는 `GRIPPER_OPEN/CLOSE`, `SUCTION_ON/OFF`,
`ATTACH_OBJECT/DETACH_OBJECT`, EE lock/unlock/verify, `WAIT`이다. grasp event의
정상 순서는 다음과 같다.

```python
events = [
    TrajectoryEvent(
        event_id="vac-on", time_from_start_s=1.0,
        event_type=EventType.SUCTION_ON,
    ),
    TrajectoryEvent(
        event_id="attach", time_from_start_s=1.0,
        event_type=EventType.ATTACH_OBJECT, target_id="apple",
        parameters={
            "attachment_mode": "BREAKABLE_WELD",
            "max_attach_distance_m": 0.02,
            "max_attach_penetration_m": 0.005,
            "max_weld_force_n": 40.0,
            "max_weld_torque_nm": 12.0,
            "max_contact_force_n": 250.0,
            "require_contact": True,
        },
    ),
    TrajectoryEvent(
        event_id="detach", time_from_start_s=3.0,
        event_type=EventType.DETACH_OBJECT, target_id="apple",
    ),
    TrajectoryEvent(
        event_id="vac-off", time_from_start_s=3.0,
        event_type=EventType.SUCTION_OFF,
    ),
]
```

attach는 target object에 정확히 하나의 world-relative free joint가 있는지, 선택 EE가
실제로 장착되어 있는지, close/suction-on 명령이 선행했는지, 충돌 활성화된
EE-object MJCF geometry 거리가 `max_attach_distance_m` 이내인지 검사한다.
동시에 penetration이 `max_attach_penetration_m`보다 크면 초기 contact impulse로
물체가 튀는 잘못된 grasp pose로 보고 attach 전에 거부한다.

`attachment_mode`는 두 가지다.

- `KINEMATIC`: grasp site 상대 transform을 매 substep에 직접 투영한다. 빠르고
  deterministic하지만 grasp slip/failure는 발생하지 않는다.
- `BREAKABLE_WELD`: object와 grasp frame 사이에 6-DoF spring-damper wrench와
  equal/opposite robot reaction wrench를 적용한다. 실제 EE-object contact 수와
  `mj_contactForce`를 측정하고 weld force/torque, contact force, pose error, contact loss
  중 하나가 debounce 기간 동안 한계를 넘으면 자동 detach한다.

C1/C2 물체 질량이 2.4 g부터 0.8 kg까지 크게 다르므로 spring/damping gain은 object
mass/inertia와 `natural_frequency_hz`를 사용해 안정 범위로 자동 축소된다. 조절 가능한
주요 값은 `max_weld_force_n`, `max_weld_torque_nm`, `max_contact_force_n`,
`max_position_error_m`, `max_orientation_error_rad`, `startup_grace_steps`,
`contact_loss_grace_steps`, `break_debounce_steps`다. 실패하면 player는 즉시
`GRASP_LOST`를 반환하고 required wrench, pose error, contact count/force를
`FailureObservation.observed`와 report metadata에 저장한다.

detach는 현재 world pose와 velocity를 보존한 채 weld force를 제거한다. 물체를 든
상태의 EE 교체나 detach 전 gripper open/suction-off는 fail-closed 한다.

실제 target checkout에서 event runtime만 독립 확인하려면 stationary smoke plan을 쓴다.

```bash
python src/tuj/m5_motion/examples/tool_use_journal_ee_runtime_smoke.py \
  . \
  --env C2_1_ObjectSorting --from-ee 2F --to-ee vac --controller
```

`C1_1_LegoSweep`와 `C2_1_ObjectSorting`은 `reward()`를 구현하지 않아 public
`env.step()` 마지막 단계에서 예외가 난다. Controller player는 동일한 robosuite
`_pre_action` controller와 MuJoCo substep loop를 사용하되 reward/episode packaging만
호출하지 않는다. event의 `actual_time_s`는 control tick 기준이므로 scheduled time보다
최대 한 control timestep 늦을 수 있다.

## OpenAI GPT Keyframe provider

`OpenAIKeyframeProvider`는 OpenAI Responses API의 Pydantic Structured Output을 사용해
일반 task용 `KeyframePlanArtifact` 후보를 여러 개 생성한다. API key는 파일이나 artifact에
저장하지 않고 프로세스의 `OPENAI_API_KEY`에서만 읽는다.

```powershell
python -m pip install "openai>=2,<3"
$env:OPENAI_API_KEY = "<new project API key>"
$env:OPENAI_KEYFRAME_MODEL = "gpt-5.4-mini"       # 선택 사항
$env:MOTION_PLANNER_KEYFRAME_CACHE = ".keyframes" # 선택 사항
```

```python
from tuj.m5_motion import (
    MotionPlanningPipeline,
    MuJoCoInterpolatingEdgePlanner,
    OpenAIKeyframeProvider,
    UR5eKinematics,
)

provider = OpenAIKeyframeProvider()
pipeline = MotionPlanningPipeline(provider, UR5eKinematics())
result = pipeline.plan(
    request,
    state_validator=collision_registry,
    edge_planner=MuJoCoInterpolatingEdgePlanner(collision_registry),
    collision_contexts=collision_contexts,
    initial_collision_context_id=initial_context_id,
    final_segment_validator=collision_registry.final_segment_validator,
)
motion_plan = result.plan
```

GPT의 권한 경계는 다음과 같다.

1. task가 확정된 뒤 scene-relative keyframe 전략 후보만 제안한다.
2. 출력은 world pose/quaternion이나 joint path가 아니라 frame, anchor, approach axis,
   offset, roll로 제한한다.
3. Structured Output을 다시 내부 Pydantic contract로 검증하고 모든 frame/anchor를 실제
   `WorldSnapshot`에 resolve한 뒤 artifact로 고정한다.
4. API 응답이 여러 후보를 내더라도 IK, joint limit, collision, edge 연결에 실패한 후보는
   deterministic compiler가 제거하고 첫 fully-connected 후보만 `MotionPlan`으로 확정한다.
5. 응답 provenance에 model, prompt hash, provider request id를 보존하며 동일
   `(scene_signature, subgoal_id, prompt_hash)`는 선택적 persistent cache에서 재사용한다.

요청 JSON에서 keyframe artifact만 확인하는 smoke CLI:

```powershell
python src\tuj\m5_motion\examples\openai_keyframe_generation.py request.json `
  --candidates 4 --output keyframes.json
```

실제 API network test는 비용과 외부 의존성을 피하기 위해 opt-in이다.

```powershell
$env:RUN_OPENAI_LIVE_TEST = "1"
python -m pytest src\tuj\m5_motion\tests\test_vlm_provider_live.py -q
```

스키마 타입을 변경한 뒤 정적 JSON Schema를 갱신한다.

```bash
PYTHONPATH=src python -m tuj.m5_motion.schema
```

## 구현 현황

현재 구현된 범위:

- Task Planner용 deterministic reachability oracle
- full-pose multi-branch IK와 endpoint branch 보존
- object/rack-relative keyframe pose resolver
- first-feasible branch backtracking
- OpenAI Responses API Structured Output 기반 multi-candidate keyframe provider
- Task Planner `SelectedPlan` → subgoal별 `MotionPlanRequest` adapter
- Task Planner resource transition → motion work unit 변환과 예상 상태 순차 전달
- plan JSON + ordered manifest atomic 저장
- prompt/scene 기반 frozen keyframe artifact cache와 provenance
- keyframe provider → IK/edge validation → MotionPlanBuilder orchestration pipeline
- SE(3) 직선 Cartesian planner와 deterministic bidirectional RRT-Connect
- Jacobian singularity, bounded swept-path, 최종 timed-waypoint 재검증
- Cartesian 속도 제한을 반영한 trajectory rescaling
- deterministic EE dock/undock template
- segment별 collision context와 attach/EE model 전환 계약
- deterministic shortcut과 quintic time parameterization
- 연결된 branch 결과를 MotionPlan으로 만드는 builder
- 전체 robosuite workcell 기반 MuJoCo endpoint / sampled-path collision validator
- collision model version registry와 dock 전용 ACM context
- bare / 3F / Vacuum / 2F attached workcell collision model compiler
- Tool-Use-Journal C1/C2 world adapter와 planner-only rack collision model compiler
- Tool-Use-Journal bare/2F/3F/vac runtime model 교체와 named-state 이관
- EE event를 실행하고 ExecutionReport를 생성하는 kinematic MotionPlan player
- robosuite absolute joint-position torque controller 기반 MotionPlan player
- finger/vacuum command와 geometry-gated object attach/detach runtime
- free-joint 운반 물체 collision transform과 gripper joint collision snapshot
- mass/inertia-scaled breakable 6-DoF weld와 contact-force 기반 `GRASP_LOST`
- Vacuum cup뿐 아니라 stalk까지 포함하는 Tool collision geometry
- 전체 MotionPlan sequence의 controller/kinematic 순차 실행과 actual-state 검증
- `POSE`/`JOINT` representation evaluator와 action-semantic PICK/PLACE/EE/Tool 상태
  evaluator의 분리, fail-closed UNKNOWN 처리
- tool-local contact patch/affordance provider와 geometry-grounded push-to-region 전략
- checkpoint/rollback 기반 closed-loop contact executor와 합성 task evaluator
- robot/grasp/contact/push/recovery/affordance 역할별 validated profile
- run/report/goal evaluation/sequence manifest atomic 저장
- artifact lineage 기반 RecoveryDirective 생성과 등록 handler 재실행

1~7 이후 고도화 범위:

- bounded swept sampling을 대체하는 geometry-level exact continuous collision detection
- TOTG/TOPP-RA/Ruckig adapter

미구현 검사는 PASS로 간주하지 않으며 기존 oracle에서는 계속 `UNKNOWN`을 반환한다.

## 테스트

```bash
python -m pytest src/tuj/m5_motion/tests -q
python -m pytest ../tuj-m3/src/tuj/m4_taskplanner/tests -q
```

테스트는 IK oracle뿐 아니라 schema version, joint dimension, timeline, event,
plan/result 관계, 정적 JSON Schema 동기화, Tool-Use-Journal 상태/충돌 호환성,
goal-aware sequence 실행, recovery attribution, 실제 robosuite controller physics loop를
검증한다. workspace 전체 `pytest`는 별도 `upstream_planner_a`의 import 설정까지
요구하므로 위처럼 두 모듈의 test directory를 명시한다.
