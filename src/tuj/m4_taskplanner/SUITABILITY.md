# Candidate suitability

Task Planner의 기본 suitability는 앞단에서 고정된 Tool과 선택된 EE 조합의 실행
가능성을 판단하는 두 항목만 평가한다. 실패해도 다른 Tool로 대체하지 않는다.

## Payload

EE가 운반해야 하는 object mass와 Tool mass의 합을 EE payload와 비교한다.
`object_remains_supported=true`인 flatten/sweep 동작은 object 전체 질량을 운반
하중에 더하지 않는다.
PICK_TOOL/RETURN_TOOL처럼 같은 물리 자원이 object와 Tool 양쪽에 나타나면 질량을
한 번만 센다. 두 catalog view의 값이 다르면 알려진 값 중 큰 값을 사용한다.

## Wrench

서브골 또는 candidate metadata의 `required_wrench`를 Tool의
`deliverable_wrench`와 비교한다. 요구가 없으면 `NOT_APPLICABLE`, 필요한 수치가
없으면 `UNKNOWN`이다.
flatten/sweep은 `requires_wrench=true`로 표시하므로 접촉력 요구 수치가 누락된
경우에도 `NOT_APPLICABLE`로 잘못 기록하지 않고 `UNKNOWN`으로 남긴다.

## 집계와 정책

적용 가능한 component의 최소 margin score를 물리 suitability로 사용한다.
component가 `FAIL`이면 후보를 제거하지만, 모두 `PASS`인 후보는 margin score가
낮다는 이유로 제거하지 않는다. 이 점수는 후보 정렬과 plan tie-break에만 쓴다.
`UNKNOWN`은 완전한 candidate보다 뒤에 정렬되며
`unknown_suitability_policy`의 `reject/allow/defer` 설정을 따른다.

`vlm` 또는 `knowledge_graph`가 제공한 외부 후보는 별도의 제안 품질 점수를
반드시 포함해야 하며, `candidate_score_threshold`는 이 제공 점수에만 적용한다.
M4가 생성한 `deterministic_rule` 후보와 명시적인 `manual` 후보에는 이 품질
threshold를 적용하지 않는다. 제공 점수는
`provided_suitability_score` metadata로 보존한다. 기본 물리 scorer가
활성화된 이후의 `suitability_score`는 물리 margin만 나타낸다.

품질 threshold, 물리 feasibility, top-k를 통과한 후보는 먼저 EE/Tool 교체,
motion, execution의 운영 비용으로 비교한다. 이 운영 비용이 모두 같을 때는 plan
전체의 candidate suitability 합이 높은 쪽을 선택하고, 그마저 같을 때만 candidate
ID를 결정론적 최종 tie-breaker로 사용한다.

접촉 위치, 손목 방향, 개구 폭, 접촉력, suction seal처럼 작업 pose에 종속되는
평가는 Task Planner에서 수행하지 않는다. 외부 좌표를 사용하는 Motion Planner와
Controller가 해당 검증을 담당한다.

결과는 다음 위치에 기록된다.

```text
selected_plan.candidate_assignments[].suitability
```
