# C4_2 바게트 해제·지지 접촉 검증 — 2026-10-31

> 공유 결과는 [전체 진행 요약](ee_gear_checkpoint_20261102.md)과 [조합별 결과](ee_gear_coverage_results_20261102.json)에 정리했다. 이 문서의 `output/` 경로는 로컬 보관 자료이며 GitHub 업로드에 포함하지 않는다.

## 검증 질문

이전 milk-only d에서는 바게트만 안착 속도 기준을 실패했다. 안착 관측창 2.3~2.7초에 각속도가 0.327→4.358rad/s로 증가했고, 2.8초에 상자 벽 접촉이 추가되며 0.087rad/s로 급감했다. 처음 손을 여는 과정의 충격인지, 지지물 위에서 접촉 위치가 바뀌며 회전하는지, 해제 뒤 인위적 부착 외력이 남는지 구분한다. 이전 시간은 해제 시점이 아니라 첫 안착 검사 기준이다.

## 실제 검증 a 조건

`output/c4_2-release-contact-20261031-a`. frozen M1/M2/G_k/M4, 우유만 최소 바닥 면적 진단, seed 0, gpt-4o-mini, controller, 재시도 0을 그대로 유지한다. 물성/힘/성공 허용치/5초 안착 관측창 변경 없음. 새 M5 응답을 생성한다.

해제 직전·직후와 이후 각 0.02초 controller tick 뒤에 다음을 읽기 전용으로 기록한다: 바게트 자세/속도, 실제 부착 모드, 접촉 상대·접촉 위치·world 힘/토크·물체 subtree COM에 대한 접촉 모멘트, 사용자 적용 일반화 힘/물체 외력. 원래 이벤트와 controller advance를 정확히 한 번 호출하고 결과를 그대로 반환하며 추가 물리 step은 없다. 공통 소스 변경 없음. 계측은 진단 entry에만 있다.

실제 runtime에서 snapshot 사전 검사 통과: qpos/qvel/ctrl/act/qfrc_applied/xfrc_applied/time 모두 불변, finite JSON 직렬화 확인. 바게트 실제 subtree 질량은 0.3718563294kg. 이는 계측 smoke check이지 전체 성공 검증이 아니다. 기존 관련 71개 검사 통과 이후 공통 코드 변경 없음.

원자료: release_contact_trace.py, trace_preflight.json, comparison_scope.json, policy_difference.json, experiment_metadata.json, release_contact_trace.jsonl, settle_trace.jsonl, API/영상/판정 보고서. 실행 중 소스 및 frozen 입력 해시 보존 여부를 종료 후 확인한다.

## 진행 상태

실제 검증 a 완료. 실행 중인 실험 없음. 공통 소스 수정 없음.


## 완료 결과와 해석

- 실제 실행 523.217초, 새 M5 응답 10개 / 401,539토큰. source_unchanged, frozen_prefix_preserved, 진단 entry/trace 해시 모두 true. M5는 매번 새로 생성했으므로 고정 응답 paired replay가 아니다.
- 이전 milk-only d와 0003/0006/0009/0012 및 최종 world까지 정확히 같음. 전체 목표는 실패 유지: 0015 PLACE `CONTAINER_NOT_SETTLED`, 뚜껑 미실행. 물체 5개 모두 수납, 바게트만 속도 검사 실패.
- 해제 직전 KINEMATIC → 해제 직후 부착 없음. 해제 시 위치·방향 변화 0, 선속도·각속도 모두 0. 이후 기록의 적용 일반화 힘/물체 외력 모두 0, 재부착 없음.
- 해제 직후에는 손 접촉 5개가 아직 있으나 0.02초 간격의 이후 276개 표본에는 손 접촉 없음. 표본 사이 순간 접촉까지 없었다고 단정하지 않는다.
- **새 핵심 관측:** 해제 0.02~0.16초의 8개 표본은 접촉 없음, 각속도 최대 5.81e-16rad/s. 0.16초에 약 124.739mm 낙하·하향 속도 1.559m/s. 0.18초 표본에서는 거품기 접촉과 함께 각속도 21.062rad/s, 총 낙하 147.493mm를 관측. 정확한 최초 충돌 시각은 20ms 표본으로 확정하지 않는다.
- 이후 3.18초에 거품기·시리얼 접촉 중 각속도 5.307rad/s까지 다시 증가. 3.20초 상자 접촉이 생겼다가 잠시 끊기고 3.26초부터 다시 접촉하며 0.131rad/s로 감소한다.
- 최종 해제 후 5.52초(기존 안착 대기 5.02초 포함)에도 바게트 선속도 2.895mm/s, 각속도 0.167451rad/s. 기존 기준 1mm/s·0.01rad/s를 모두 초과한다. 다른 네 물체는 통과. 마지막 1초 이동 2.686mm·회전 9.076°도 d와 동일하다.

## 해제 높이 코드 감사

`src/tuj/m5_motion/scripted_grasps/container_orientation.py:packing_destination_center`는 PLACE에서 물체 최하단을 max(상자 입구, 내용물 상단)보다 기존 clearance만큼 위에 둔다. 실제 해제 최하단 Z=1.165001m, 상자 입구 Z=1.160000m: 차이 5.001mm로 이 정책과 일치한다. 손 열림 envelope의 추가 상승 `container_release_lift_m`은 이 실행에서 0이다. 즉 이번 낙하는 손 envelope가 추가로 올린 결과가 아니라 기존 상자 입구 위 방출 정책에서 비롯된다.

현재 증거는 잔류 부착 외력이나 해제 순간의 지속적인 회전 주입보다, 입구 위에서 놓고 거품기에 착지하는 과정 및 이후 지지 접촉을 후속 대상으로 가리킨다. **새 구현 버그 확정이나 해결 완료로 분류하지 않는다.** Full 배치 정책/물리 동역학 문제이며 frozen w/o Reasoner M2 거절과 분리한다. 관측 접촉력은 1ms 충격량 적분 또는 반사실 인과 실험이 아니다.

## 다음 검증

현재 측정 파지 변환·방향·XY를 유지한 채 더 낮은 해제 위치를 검토한다. 먼저 손이 열리는 전체 envelope, 충돌 여유, IK 및 내려가는 경로를 기존 기준 그대로 검사하고, 통과하는 후보에 한해 별도 실제 실행으로 비교해야 한다. 무조건 Z를 낮추거나 대기/성공 허용치를 늘리지 않는다. 낮은 해제가 가능하거나 성공한다는 것은 아직 검증하지 않았다.

## 원자료와 조회 위치

- 실행 감사 (로컬: `output/c4_2-release-contact-20261031-a/diagnostic_result.json`), 이전 d와 상태 비교 (로컬: `output/c4_2-release-contact-20261031-a/measured_state_comparison.json`).
- 해제·접촉 분석 (로컬: `output/c4_2-release-contact-20261031-a/release_analysis.json`), 낙하·높이 감사 (로컬: `output/c4_2-release-contact-20261031-a/release_drop_audit.json`), 원시 278개 표본 (로컬: `output/c4_2-release-contact-20261031-a/release_contact_trace.jsonl`).
- 접촉 동역학 그래프 (로컬: `output/c4_2-release-contact-20261031-a/release_contact_dynamics.png`), 물체별 안착 그래프 (로컬: `output/c4_2-release-contact-20261031-a/final_settle_speeds.png`), 실제 영상 (로컬: `output/c4_2-release-contact-20261031-a/full/c4_2/seed_0/m5/c4_2.mp4`).
- 공통 소스는 이전 관련 71개 검사 통과 이후 변경 없음. 이번에는 계측 사전 불변 검사, 실제 재실험, 최종 상태 대조, 판정 원자료 감사와 그래프 검토를 수행했고 전체 회귀 검사를 새로 실행한 것은 아니다.
