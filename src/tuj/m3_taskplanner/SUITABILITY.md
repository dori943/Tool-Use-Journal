# Candidate suitability

Task Planner의 기본 suitability는 앞단에서 고정된 Tool과 선택된 EE 조합의 실행
가능성을 판단하는 두 항목만 평가한다. 실패해도 다른 Tool로 대체하지 않는다.

## Payload

EE가 운반해야 하는 object mass와 Tool mass의 합을 EE payload와 비교한다.
`object_remains_supported=true`인 push/pull 동작은 object 전체 질량을 운반 하중에
더하지 않는다.

## Wrench

서브골 또는 candidate metadata의 `required_wrench`를 Tool의
`deliverable_wrench`와 비교한다. 요구가 없으면 `NOT_APPLICABLE`, 필요한 수치가
없으면 `UNKNOWN`이다.

## 집계와 정책

적용 가능한 component의 최소 margin score를 사용한다. `UNKNOWN`은 완전한
candidate보다 뒤에 정렬되며 `unknown_suitability_policy`의
`reject/allow/defer` 설정을 따른다.

접촉 위치, 손목 방향, 개구 폭, 접촉력, suction seal처럼 작업 pose에 종속되는
평가는 Task Planner에서 수행하지 않는다. 외부 좌표를 사용하는 Motion Planner와
Controller가 해당 검증을 담당한다.

결과는 다음 위치에 기록된다.

```text
selected_plan.candidate_assignments[].suitability
```
