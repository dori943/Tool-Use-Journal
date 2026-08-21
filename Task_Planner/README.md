# Task Planner

Planner A의 정규화된 출력 또는 `GK + M1`을 입력받아 다음을 하나의 Dijkstra
검색으로 계획한다.

- 실행 가능한 서브골 순서
- 서브골 또는 group별 End-Effector
- 앞단에서 확정한 Tool의 집기·반납 시점
- EE 교체와 terminal cleanup

Planner A와 GK/M1 경로 모두 앞단에서 전달한 `tool_id`를 그대로 고정한다.
Task Planner는 Tool 후보를 비교하거나 다른 Tool로 대체하지 않는다. GK/M1 입력은
`tool_id`를 표준 필드로 사용하며 `selected_tool_id`, `selected_tool`도 호환 입력으로
받는다. `tool_candidate_ids`만 전달되면 입력 오류를 반환한다. 접촉 자세 후보와 작업
좌표, IK, 충돌, 접근·후퇴 경로는 Motion Planner가 담당한다.

## 최적화 비용

비용은 가중합이 아니라 다음 튜플의 사전식 순서로 비교한다.

```text
(ee_switches, tool_switches, motion_cost, execution_cost)
```

우선순위는 EE 교체 최소화, Tool 교체 최소화, 이동 비용, 실행 비용 순서다.
Terminal Tool 반납과 초기 EE 복원 비용도 포함한다.

## 검색 상태

```python
SearchState(
    completed_subgoals: frozenset[str],
    current_ee: str,
    held_tool: str | None,
    group_ee_bindings: tuple[tuple[str, str], ...],
    symbolic_facts: frozenset[FluentKey],
    scene_signature: str,
    rack_signature: tuple[tuple[str, str], ...] | None,
)
```

물체 보유 상태는 별도 접촉 binding 대신 `holding(object)` symbolic fact로
표현한다. 물체를 든 동안에는 EE 교체와 Tool 변경을 허용하지 않는다.

## 주요 계약

- Planner A의 hard DAG는 추가하거나 삭제하지 않는다.
- 같은 `group_id`의 모든 서브골은 같은 EE를 사용한다.
- `tool_id`는 앞단이 고정하며 Task Planner가 선택하거나 대체하지 않는다.
- `None→A`, `A→None`, `A→B`는 Tool identity 변경 1회다.
- Tool을 든 상태에서 EE를 바꿀 때는 `RETURN_TOOL`이 `DETACH_EE`보다 먼저다.
- 입력에 없는 EE, Tool, 작업 pose를 만들어내지 않는다.
- 작업 pose는 Motion Planner에 opaque 입력으로 전달할 수 있으며 Task Planner는
  이를 생성·해석·선택하지 않는다.
- 물리 suitability는 task-level payload와 wrench만 평가한다.
- IK, 충돌, docking은 lazy geometry checker 또는 Motion Planner가 검증한다.

## 실행

GK 입력 연결:

```bash
python -m task_planner.cli plan \
  --gk output/c1_1/gk.json \
  --m1 output/c1_1/m1.json \
  --m0 output/c1_1/m0.json \
  --robot-spec robot_spec.json \
  --output results/c1_1_task_planner.json
```

GK에는 action detail과 실행 partial-order가 없으므로 `--m1`은 필수다. `--m0`와
`--robot-spec`은 scene geometry와 EE payload/capability를 보강하며, 동일 정보가
`--resources`에 있으면 생략할 수 있다. `obj_<class>_<instance>` 형식 ID는 기본적으로
`<instance>`로 정규화하고, 예외는 `--id-aliases aliases.json`으로 지정한다.

Planner A 출력과 scenario 연결:

```bash
python -m task_planner.cli plan \
  --planner-a ../planner_a/outputs_vlm/t2_2_stack_tower.json \
  --scenario ../planner_a/scenarios/t2_2_stack_tower.json \
  --output results/t2_2_stack_tower_task_planner.json
```

외부 resource와 candidate를 사용하는 예제:

```bash
python -m task_planner.cli plan \
  --planner-a examples/heavy_crate_planner_a.json \
  --resources examples/heavy_crate_resources.json \
  --candidates examples/heavy_crate_candidates.json \
  --output examples/heavy_crate_result.json
```

재계획:

```bash
python -m task_planner.cli replan \
  --request request.json \
  --execution-state execution_state.json \
  --failure failure.json \
  --output replanned_result.json
```

## Motion Planner 경계

Task Planner가 전달하는 `CandidateQuery`에는 서브골, EE, 선택된 Tool, target id,
action type, 선택적인 `target_pose`가 포함된다. Motion Planner는 다음을 판단한다.

- 주어진 작업 pose의 IK 도달 가능성
- 접근·후퇴 및 전체 경로의 충돌
- EE rack docking/undocking
- Tool 반납과 terminal cleanup의 기하 실현 가능성

`target_pose`가 candidate metadata에 있으면 그대로 전달할 뿐 Task Planner가 값을
변경하지 않는다.

## 재계획 no-good 범위

| 실패 유형 | 기본 범위 |
|---|---|
| `CANDIDATE_INVALID`, `RESOURCE_UNAVAILABLE` | candidate 전역 금지 |
| `TRANSITION_INVALID`, `ATTACHMENT_FAILED` | transition signature 금지 |
| `SCENE_CHANGED` | 금지 없이 새 상태에서 재탐색 |

필요하면 `FailureFeedback.scope`로 scene/global/transition 범위를 명시할 수 있다.

## 테스트

```bash
python -m pytest -q
```

테스트는 입력 검증, Planner A 제약, GK/M1 변환, ID 정규화, group EE binding,
고정 Tool 검증, 후보 필터링, Tool/EE 전환, lazy geometry cache, terminal policy, replanning,
brute-force 대비 Dijkstra 최적성, CLI end-to-end를 포함한다.

관련 연구와 논문 근거는 [`RELATED_WORK.md`](RELATED_WORK.md)와
[`references.bib`](references.bib)에 정리되어 있다.
