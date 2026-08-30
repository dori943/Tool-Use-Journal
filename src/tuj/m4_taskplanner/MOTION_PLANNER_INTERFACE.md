# Task Planner → Motion Planner v3 인터페이스

## 책임 분리

Task Planner는 Task Planner다. 사용자 작업을 서브골로 분해하고 실행 순서, EE,
Grasp와 목표 pose를 결정하며, 앞단에서 확정된 Tool의 전환 시점을 계획한다.
Motion Planner는 확정값을 바꾸지 않고 현재 장면에서
실행할 수 있는 시간축 joint trajectory를 생성한다. 이 연결은 one-way command이며,
Motion Planner와 Simulator의 결과를 Task Planner에 직접 반환하지 않는다.

| Task Planner | Motion Planner |
|---|---|
| task/subgoal 생성 | IK 해 계산 |
| action 순서 결정 | collision-free path 생성 |
| EE·Grasp 선택, 고정 Tool 전환 배치 | path smoothing 및 time parameterization |
| 목표 pose 제공 | trajectory segment/event 생성 |
| 원인 모듈일 때만 Orchestrator가 재실행 | trajectory와 실행 report artifact 생성 |

## 호출 계약

```text
MotionPlanRequest -> MotionPlan -> SimulationRun -> ExecutionReport
                                                   │ failure
                                                   ▼
                                           RecoveryDirective
                                                   │
                                                   └─ 원인 모듈만 재실행
```

### MotionPlanRequest

Task Planner는 다음 정보를 한 번에 전달한다.

- `WorldSnapshot`: scene identity, robot state, objects, obstacles, rack
- `MotionTask`: task/subgoal, action, EE, Tool, targets, Grasp, grounded goal
- `MotionConstraints`: 충돌 여유, 목표 tolerance, 속도·가속도 제한
- `PlannerOptions`: 알고리즘, 시간 제한, 시도 횟수, interpolation step, seed

### MotionPlan / SimulationRun

MotionPlan은 semantic segment, joint waypoint, trajectory clock에 동기화된 discrete
event, expected final state로 구성된 내부 artifact다. SimulationRun은 이 plan과
simulation config를 묶어 Simulator에 직접 전달한다.

### ExecutionReport / RecoveryDirective

Simulator는 Motion Planner가 생성한 plan을 수정하지 않고 그대로 재생한다.
ExecutionReport에는 tracking error, collision count, goal error, event 실행 결과,
최종 robot state와 trace reference가 포함된다. 이 report는 Task Planner 응답이 아니라
내부 저장 및 원인 분석 입력이다.

실패 시 Recovery Orchestrator가 artifact lineage를 역추적해 RootCause를 정하고,
RecoveryDirective로 원인 모듈과 재시작 artifact만 지정한다. downstream artifact는
무효화하고 관련 없는 upstream 모듈은 다시 실행하지 않는다.

## 핵심 불변조건

1. 모든 waypoint는 plan의 `joint_names` ordering과 같은 DOF를 사용한다.
2. waypoint 시간은 strict monotonic이고 segment는 서로 겹치지 않는다.
3. gripper/suction/attach 동작은 trajectory와 같은 clock을 사용한다.
4. metric 단위는 metre, second, radian 기반 SI로 고정한다.
5. 계획과 simulation은 명시된 random seed로 재현 가능해야 한다.
6. 현재 `scene_signature`와 다른 장면에서는 기존 plan을 재사용하지 않는다.
7. 모든 artifact는 `produced_by`, `invocation_id`, `input_artifact_ids`를 보존한다.
8. `RecoveryDirective.target_module`은 `RootCause.module`과 반드시 같아야 한다.

정식 JSON Schema는
[`../Motion_Planner/schemas/motion_planner.schema.json`](../Motion_Planner/schemas/motion_planner.schema.json)에 있다.

## 마이그레이션 상태

Task Planner의 결과에는 motion grounding에 필요한 action, targets, EE, 앞단에서 고정된
Tool, Grasp와 candidate parameter가 보존된다. Motion Planner의
`selected_plan_to_motion_requests()`가
이를 선택된 subgoal 순서의 `MotionPlanRequest`로 변환한다. 다중 subgoal 변환은 각 동작
시작 시점의 `WorldSnapshot`을 mapping/callable로 받아 stale state를 거부한다.

Task Planner 탐색 중 빠른 pruning을 위한 legacy `CandidateQuery` feasibility oracle은 아직
남아 있다. 이것은 최종 trajectory 생성 경로가 아니며, 선택 확정 후에는 v3
`MotionPlanRequest → MotionPlan` 경로를 사용한다. `SelectedPlanMotionOrchestrator`는
EE/Tool transition을 독립 work unit으로 확장하고, 확정된 `MotionPlan.expected_final_state`
로 다음 subgoal의 predicted snapshot을 갱신하며, plan JSON과 ordered manifest를 저장한다.
실제 Tool-Use-Journal workcell에서는 `ToolUseJournalMotionRequestPlanner`가 환경별 MuJoCo
compiler를 `plan_one_request`로 감싼다. 요청별 collision registry와 keyframe event-scoped
context를 생성하며, PICK attachment transform과 PLACE 이후 free-object pose도 predicted
snapshot으로 이관한다.
Simulation 단계에서는 predicted snapshot을 실제 `ExecutionReport.final_robot_state`와
대조해 다음 실행 snapshot을 확정한다.
