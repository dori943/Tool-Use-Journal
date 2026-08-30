# Task Planner

Task Planner(M4) 모듈 위치는 `src/tuj/m3_taskplanner/`이며, 기존 Git 단계명을
유지한 패키지 경로는 `tuj.m3_taskplanner`다.

## 빠른 시작

### M4 한 명령 실행

저장소에 포함된 C1_1 입력으로 Task Planner를 바로 실행하려면 저장소 루트에서
다음 명령을 사용한다. 실행 위치와 관계없이 저장소 기준 경로를 사용하며, 결과 폴더도
자동으로 만든다.

```bash
python scripts/run_m4_task_planner.py
```

기본 입력과 출력은 다음과 같다.

| 구분 | 경로 |
|---|---|
| GK bundle | `output/c1_1/gk_bundle.json` |
| M1 | `output/c1_1/m1.json` |
| Robot spec | `configs/robot_spec.json` |
| M5 전달 결과 | `output/c1_1/task_planner.json` |

다른 입력을 사용할 때만 경로를 덮어쓴다.

```bash
python scripts/run_m4_task_planner.py \
  --gk path/to/gk_bundle.json \
  --m1 path/to/m1.json \
  --robot-spec path/to/robot_spec.json \
  --output path/to/task_planner.json
```

이 runner는 Tool 이름을 인자로 받지 않는다. Tool은 항상 GK bundle의
`roles.selected_tool`에서 읽고 결과의 `candidate_assignments`로 전달한다.

저장소 루트에서 Python 3.11 이상 가상환경을 만든 뒤 의존성을 설치한다.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest src/tuj/m3_taskplanner/tests -q
```

Windows PowerShell에서 가상환경을 활성화하려면 다음 명령을 먼저 실행한다.

```powershell
.\.venv\Scripts\Activate.ps1
```

테스트는 `tests/conftest.py`가 `src/`를 import 경로에 넣으므로 별도 설치 없이
저장소 루트에서 바로 돌아간다. CLI를 모듈로 실행할 때는 `src/`를 `PYTHONPATH`에
넣는다 — `scripts/` 실행기들과 같은 규약이다.

```bash
PYTHONPATH=src python -m tuj.m3_taskplanner.cli --help
```

PowerShell에서는 다음과 같다.

```powershell
$env:PYTHONPATH = "src"
python -m tuj.m3_taskplanner.cli --help
```

`GK + M1`을 필수 입력으로 받아 다음을 하나의 Dijkstra 검색으로 계획한다.

- 실행 가능한 서브골 순서
- 서브골 또는 group별 End-Effector
- 앞단에서 확정한 Tool의 집기·반납 시점
- EE 교체와 terminal cleanup

현재 GK bundle 경로에서는 각 `gk_by_subgoal[].roles.selected_tool`을 그대로
고정한다. Task Planner는 Tool 후보를 비교하거나 다른 Tool로 대체하지 않는다.
기존 M1 action/DAG 형식의 `tool_id`, `selected_tool_id`, `selected_tool`도 호환
입력으로 받는다. 후보 목록만 있고 선택값이 없으면 입력 오류를 반환한다. 접촉 자세
후보와 작업 좌표, IK, 충돌, 접근·후퇴 경로는 Motion Planner가 담당한다.

## 최적화 비용

비용은 가중합이 아니라 다음 튜플의 사전식 순서로 비교한다.

```text
(ee_switches, tool_switches, motion_cost, execution_cost)
```

우선순위는 EE 교체 최소화, Tool 교체 최소화, 이동 비용, 실행 비용 순서다.
초기 `current_ee`가 `null`이면 첫 EE 장착은 교체로 세지 않고, 첫 EE까지 검색에서
선택해 전체 작업의 이후 EE 교체 횟수를 최소화한다. Terminal Tool 반납과, 구체적인
초기 EE가 있을 때의 EE 복원 비용도 포함한다.

## 검색 상태

```python
SearchState(
    completed_subgoals: frozenset[str],
    current_ee: str | None,
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

- M1의 hard DAG는 추가하거나 삭제하지 않는다.
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

GK + M1 입력 연결:

```bash
PYTHONPATH=src python -m tuj.m3_taskplanner.cli plan \
  --gk output/c1_1/gk_bundle.json \
  --m1 output/c1_1/m1.json \
  --robot-spec configs/robot_spec.json \
  --output output/c1_1/task_planner.json
```

현재 입력 계약에서 `gk_bundle.json`은 action detail, partial-order, Tool 후보와
앞단에서 확정한 `roles.selected_tool`을 제공하고, `m1.json`은 scene의 `nodes/edges`를
제공한다. 이전 `m1_subgoals` 형식도 계속 지원하며, scene graph가 별도 파일이면
`--m0`로 명시할 수 있다. `--robot-spec`은 EE payload/capability를 보강한다.
`obj_<class>_<instance>` 형식 ID는 기본적으로 `<instance>`로 정규화하고, 예외는
`--id-aliases aliases.json`으로 지정한다. 외부 resource catalog와 candidate
proposal은 각각 `--resources`, `--candidates`로 보강할 수 있다.

### 초기 로봇 상태

`configs/robot_spec.json`에는 최소한 `current_ee`와 `hand_empty`를 명시해야 한다.
EE가 아직 장착되지 않았다면 `"current_ee": null`로 지정한다. 이 경우 Task Planner는
가능한 모든 첫 EE를 비교하며, 첫 장착에는 `ATTACH_EE`만 생성하고 EE 교체 횟수는
증가시키지 않는다. 첫 작업 이후 실제 EE가 바뀔 때만 교체로 계산한다.

현재 Tool 또는 rack 점유 상태가 있으면 `held_tool`, `rack_occupancy`도 함께 전달한다.
`current_ee`가 `null`이면 `held_tool`도 `null`이어야 한다. Task Planner는
`current_ee` 필드가 누락됐을 때 EE 목록의 첫 항목을 임의로 추정하지 않는다.

로봇 사양과 실행 상태를 별도로 관리할 때는 정규화된 상태 파일을 사용한다.

```bash
PYTHONPATH=src python -m tuj.m3_taskplanner.cli plan \
  --gk output/c1_1/gk_bundle.json \
  --m1 output/c1_1/m1.json \
  --robot-spec configs/robot_spec.json \
  --initial-state src/tuj/m3_taskplanner/examples/initial_state.json \
  --output output/c1_1/task_planner.json
```

`--initial-state`는 `InitialState` 스키마를 그대로 사용하며 robot spec의 초기 상태보다
우선한다. 예시는 `src/tuj/m3_taskplanner/examples/initial_state.json`에 있다.

재계획:

```bash
PYTHONPATH=src python -m tuj.m3_taskplanner.cli replan \
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
python -m pytest src/tuj/m3_taskplanner/tests -q
```

테스트는 입력 검증, M1 DAG 제약, GK/M1 변환, ID 정규화, group EE binding,
고정 Tool 검증, 후보 필터링, Tool/EE 전환, lazy geometry cache, terminal policy, replanning,
brute-force 대비 Dijkstra 최적성, CLI end-to-end를 포함한다.
