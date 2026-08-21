# Planner A → Task Planner 계약

## 책임 분리

| 항목 | 담당 |
|---|---|
| 서브골과 hard DAG | Planner A |
| 서브골별 Tool identity | Planner A에서 고정 |
| feasible EE 집합 | Planner A 또는 scenario |
| 서브골 실행 순서 | Task Planner |
| 최종 EE 배정 | Task Planner |
| Tool 집기·반납 및 EE 전환 | Task Planner |
| 외부 작업 좌표, IK, 충돌, 궤적 | Motion Planner |

Task Planner는 Planner A의 `tool_id`를 다른 Tool로 바꾸지 않는다. 가능한 DAG
순서 중에서 같은 고정 Tool을 사용하는 서브골을 연속 실행해 전환 비용을 줄일 수
있지만, 이는 Tool 선택이 아니라 순서 및 전환 최적화다.

## Adapter

현재 Planner A의 `detailed_subgoals`와 `edges`를 각각 `subgoals`와
`order_constraints`로 변환한다. scenario에서 초기 EE, rack, 전체 feasible EE,
symbolic EE/Tool/Object catalog를 복원한다.

`mutex`, `open_conditions`, `disjunctive_threats`, `deferred_conditions`,
observation request, redecompose signal은 typed contract로 보존한다. 미래의 미지
필드는 Pydantic extra field로 유지한다.

외부 작업 좌표는 이 adapter가 만들지 않는다. Motion Planner 또는 별도 perception
모듈이 좌표를 공급하고 `CandidateQuery.target_pose` 또는 자체 world model을 통해
사용한다.
