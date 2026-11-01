# EE-GEAR 검증 체크포인트 (2026-11-02)

- 8개 태스크 중 C2_1부터 C4_2까지 6개의 독립 EE 파지 행렬을 완료했다. 대상 38개에 대해 2F, 3F, vac 총 114조합을 실행했다.
- 범위는 파지, 들어 올리기, 유지 검증이다. 전체 태스크 성공이나 논문 최종 ablation 점수는 아니다.
- 구현 오류, 실행 경로 누락, 현재 후보의 물리 실패, 추정과 계획 단계의 거절을 구분했다. 실패를 Reasoner 제거의 효과로 바로 귀속하지 않는다.
- 진행 중인 실험은 없다. C1_1/C1_2의 실제 실험 근거 확인과 최종 공통 코드의 Full / w/o Reasoner 8개 비교가 남아 있다.

## 주 행렬 결과

| 태스크 | 조합 수 | PASS | 검토 후 FAIL | 자유 유지 성공 | 획득 부착 보조 성공 | 기록 |
|---|---:|---:|---:|---:|---:|---|
| C4_2 | 18 | 14 | 4 | 7 | 7 | [상세](c4_2_coverage_validation_20261031.md) |
| C4_1 | 15 | 7 | 8 | 3 | 4 | [상세](c4_1_coverage_validation_20261101.md) |
| C3_2 | 36 | 19 | 17 | 3 | 16 | [상세](c3_2_coverage_validation_20261102.md) |
| C3_1 | 15 | 10 | 5 | 7 | 3 | [상세](c3_1_coverage_validation_20261102.md) |
| C2_2 | 15 | 5 | 10 | 0 | 5 | [상세](c2_2_coverage_validation_20261102.md) |
| C2_1 | 15 | 9 | 6 | 6 | 3 | [상세](c2_1_coverage_validation_20261102.md) |
| 합계 | 114 | 64 | 50 | 26 | 38 | 집계용 건수 |

- 원본 주 행렬 합계는 64 PASS, 49 FAIL, 1 ERROR다. C3_2 fork_a/vac의 분류만 검토 후 FAIL로 정정했다. 원본 ERROR를 덮어쓰지 않았다. 시간 초과는 없다.
- 태스크별로 기존 레시피의 힘, 접촉, 부착, 상승 거리와 허용값이 다르다. 합계를 균일 프로토콜의 성공률이나 EE의 인과 효과로 해석하지 않는다.
- 자유 유지 성공은 획득과 유지 중 부착 보조 없이 통과하고 이후 운반용 부착을 한 결과다. 획득 부착 보조 성공은 `ACQUISITION` 정책을 사용한 결과다. 흡착 모델은 접촉 조건 기반 운동학적 부착이며 실제 밀봉이나 독립 진공 유지력을 별도로 증명하지 않는다.
- C3_2 자유 유지 3개는 fruit_b 2F와 bread_a/b 3F다. spoon_a/b 3F는 획득 부착 보조를 사용했다.

## 수정과 원판정 보존

- C4_2: 독립 프로세스 실행, 사례별 종료 보고, 원래 예외 유형과 실패 단계 보존, 제어 주기 영상 기록, 초기 상태와 소스 해시 기록을 보강했다. 중단된 예비 실행을 주 행렬에 합치지 않았다.
- C4_1: 실제 장면 대상은 5개로 확인했다. 과거 감사 목록의 4개와 분모가 다르다. 3F 손끝 마찰 적용 코드가 2F 형상 이름만 찾던 오류를 수정했다. 선언된 마찰과 힘은 유지했고 수정 후 15조합 전체를 다시 실행했다. 초기 실행의 완료 결과, 오류, 중단 상태는 별도 보존했다.
- C3_2: 부착 거리와 침투 차단을 전용 예외로 구분했다. fork_a/vac는 `ERROR`에서 `FAIL/MOTION_OR_GRASP`로 재분류했다. 같은 조건의 음성 대조와 fork_b/vac 양성 대조에서 초기, 파지 직전, 최종 상태 배열과 지표가 원본과 일치했다. 물리 기준과 성공 여부는 바뀌지 않았다.
- C3_1: apple/mug 3F가 바인딩된 C3_1 설정 대신 기본 C2_1 설정을 다시 만들던 오류를 수정했다. 수정 전 8 PASS, 5 FAIL, 2 ERROR에서 전체 재실행 후 10 PASS, 5 FAIL이 됐다. 나머지 13조합의 저장 상태와 지표는 동일했다.
- C2_2와 C2_1: 기존 등록 설정을 보존하고 누락 조합에 검증 전용 후보를 추가했다. 두 태스크에서 새로운 공통 실행 오류는 발견하지 않았다.
- 전체 포장 실행에서는 배치 탐색, 실제 지지 접촉, 안착 속도 측정과 판정 근거를 보완했다. 수정 이후 남은 물리 실패를 성공으로 바꾸지 않았다. [Full 대조](c4_2_full_control_20261031.md), [물체별 안착 진단](c4_2_settle_diagnostics_20261031.md).

## 공유 직전 검토

- 공유 브랜치는 `fix/subin-ee-gear-validation-20261102`, 실험 기준 커밋은 `6156e34`다.
- `--registered-only`와 `--experimental-only`가 C3_2의 타입 별칭 12개를 잘못 분류하는 검증 옵션 오류를 수정했다. 실제 등록 우선순위인 정확한 물체 ID, 타입 별칭 순서를 따르며, 정확한 ID가 있으면 다른 EE의 타입 경로로 넘어가지 않는다.
- 이 수정은 검증 대상 선택에 한정한다. 과거 주 행렬의 원자료와 물리 동작을 바꾸지 않았으며 새 시뮬레이션은 실행하지 않았다. 문서의 소스 불변 확인은 각 실험 실행 당시의 기록이다.
- 별칭 12개, 정확한 ID 우선순위, 두 선택 옵션의 동작을 추가 검사했다. 해당 검사 파일은 29개 모두 통과했다.
- 업로드 직전 관련 확대 회귀: **648 PASS, 2 SKIP, 4 FAIL**, NumPy 경고 36개. 실패 4개는 `test_attachment_sync.py::test_projection_uses_current_hand_pose_after_qpos_changes`의 기존 속도 동기화 검사다. 기준 커밋 `6156e34`의 runtime 소스를 메모리에 로드한 대조에서도 동일한 4 FAIL과 1 PASS가 재현됐다.

## 후속 검증과 실행 반영

- C4_2: 실패 4조합에 대해 사전에 정한 자세 후보 12개를 추가 실행했으나 모두 실패했다. 새 손가락 성공 후보 4종을 seed 1/2에서 반복한 8건은 모두 통과했다.
- C4_2의 밀대, 바게트, 시리얼 3F/vac와 우유 2F/vac, 총 8개 대안을 검증한 레시피 그대로 일반 실행에 등록했다. 흡착 4개는 주 행렬 seed 0 근거이며 별도 seed 반복은 하지 않았다. 추정 feasible EE 집합은 확장하지 않았다.
- C4_1: 도마 3F seed 1/2 반복 2건 통과. 새 성공 후보 5개는 검증 전용이다.
- C3_2: 부착 분류 대조 2건을 완료했고 새 fruit_b 2F seed 1/2 반복 2건도 통과했다. 새 성공 후보 6개는 검증 전용이다.
- C3_1: 오류를 수정한 apple/mug 3F의 seed 1/2 반복 4건 통과. 새 성공 후보 2개는 검증 전용이다.
- C2_2: 빵 3F의 높이만 변경한 추가 후보 4건 모두 CLOSE 실패였다. 원래 자세 대조 1건도 동일 실패를 재현했다. 기존 흡착 5개만 성공했고 새 후보의 일반 실행 등록은 없다.
- C2_1: 새 bread/mug 2F seed 1/2 반복 4건 통과. 두 손 후보와 apple/mug vac를 포함한 새 성공 후보 4개는 검증 전용이다. 새 apple 2F는 5N/5도 기준에서 회전 15.119도로 실패했다. C3_1 기존 apple 2F의 8N/25도 기준 통과와 직접 비교하지 않는다.
- 위 후속 실행은 114조합 분모에 합치지 않는다. 고정 장면의 재현성 확인이며 임의 배치 강건성 검증은 아니다. 후보 실패는 해당 자세의 실패이며 가능한 모든 파지의 불가능성을 증명하지 않는다.

## Full 포장과 w/o Reasoner의 별도 상태

- C4_2 Full 포장은 아직 전체 목표 실패다. 원래 정책에서는 바게트 수납 문제가 남았고, 제한된 후속 진단에서는 5개 내용물 수납 후에도 안착 기준을 통과하지 못했다.
- 최신 3F 전체 진단은 파지, 운반, 5개 내용물 수납을 통과했다. 그러나 바게트 각속도 0.042461rad/s가 기준 0.01rad/s를 넘어 실패했다. 뚜껑 획득과 닫기는 실행하지 않았다. [3F 전체 진단](c4_2_third_ee_20261101.md).
- frozen w/o Reasoner 입력은 M2에서 `MISSING_REQUIRED_TOOL:SG1`로 중단됐다. 이는 Full의 물리 포장 실패와 다른 단계의 결과다. 독립 파지 성공으로 기존 추정과 계획 거절을 덮어쓰지 않는다.
- 현재 기록은 Full 성공 대비 w/o Reasoner 실패라는 최종 비교를 성립시키지 않는다. 동일 최종 코드와 비교 조건의 전체 실행이 필요하다.

## 검증 근거와 보관

- [기계 판독 결과](ee_gear_coverage_results_20261102.json)에 114개 원판정, 실패 단계, 부착 여부, 유지 지표와 원본 요약별 SHA-256을 정리했다. C3_2 분류 정정과 대조 근거는 별도 필드에 있다.
- 주 행렬은 seed 0, 사례별 새 프로세스와 환경, 초기 안정화, 사례당 600초 제한을 사용했다. EE 검증과 해당 후속은 API 호출 없이 수행했다. 이전 전체 포장 진단의 호출 이력과 혼동하지 않는다.
- 초기 상태 차이, 실행 중 소스 보존, 저장 상태와 판정, 영상 표본을 각 실험 감사에서 확인했다. 처음, 중간, 마지막 영상 표본을 확인한 것을 전체 프레임의 의미 판독으로 표현하지 않는다.
- EE별 초기 안착 상태와 레시피 프로토콜에 차이가 있다. C2_2에서는 주변 국자의 자세 차이도 확인했으므로 장면 전체가 완전히 동일한 인과 비교라고 주장하지 않는다.
- `output/` 원자료, 영상, 상태 배열, 소스 보관본은 로컬에 보관하며 이번 GitHub 업로드에 포함하지 않는다. JSON의 원자료 경로는 저장소 기준 상대 식별자이며 다운로드 링크가 아니다. 전체 원자료가 필요한 재실행 감사에는 해당 로컬 보관본이 필요하다.
- 동일 내용은 기존 경로를 유지한 채 저장 공간을 공유했다. 고유 실패와 결과를 삭제하지 않았다. C2_2/C2_1의 신규 출력에서 제거한 중복분은 727,638,016바이트다. 실험 전 대비 디스크 순감소량은 아니다.

## 재개 순서

1. C1_1/C1_2의 실제 3EE 실행 결과와 담당 범위를 확인한다. 기존 규칙 및 등록표 감사만으로 실제 파지 검증 완료를 표시하지 않는다.
2. 검증 전용 성공 후보의 일반 실행 반영 범위와 운반 및 배치 적합성을 확정한다. 실패 후보와 원래 추정 판정은 보존한다.
3. 공통 코드를 고정하고 Full / w/o Reasoner 8개 태스크를 같은 조건으로 실행한다. 구현 오류, 접근 실패, 파지 실패, 계획 거절, 전체 목표 실패를 분리한다.
4. 최종 논문 표의 대상 목록, 분모, 부착 정책과 성공 정의를 맞추고 수치를 확정한다. 현재 후보 검증 건수를 기존 Table II/III에 바로 대입하지 않는다.

## 관련 회귀 검사 재실행

Python과 MuJoCo 의존성이 설치된 환경에서 다음 파일 묶음을 실행했다. API 키는 비우고 라이브 API 검사는 비활성화했다. 로컬 실행은 `MUJOCO_GL=cgl`을 사용했다.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:scripts MUJOCO_GL=cgl \
RUN_OPENAI_LIVE_TEST=0 OPENAI_API_KEY= GEMINI_API_KEY= GOOGLE_API_KEY= \
python -m pytest -q -p no:cacheprovider \
  tests/test_c2_1_coverage.py \
  tests/test_c2_2_coverage.py \
  tests/test_c3_1_coverage.py \
  tests/test_c3_2_coverage.py \
  tests/test_c4_1_coverage.py \
  tests/test_c4_1_finger_friction.py \
  tests/test_c4_2_pose_diagnostics.py \
  tests/test_c4_2_probe_recipes.py \
  tests/test_c4_2_production_constraints.py \
  tests/test_catalog_recipe_binding.py \
  tests/test_table2_coverage.py \
  tests/test_failure_taxonomy.py \
  tests/test_planning_preservation.py \
  tests/test_task_goal_evaluation.py \
  src/tuj/m5_motion/tests/test_container_orientation.py \
  src/tuj/m5_motion/tests/test_container_settle.py \
  src/tuj/m5_motion/tests/test_container_stable_face.py \
  src/tuj/m5_motion/tests/test_container_support.py \
  src/tuj/m5_motion/tests/test_placement_contacts.py \
  src/tuj/m5_motion/tests/test_scripted_grasp_validate.py \
  src/tuj/m5_motion/tests/test_scripted_task_constraints.py \
  src/tuj/m5_motion/tests/test_tool_use_journal.py \
  src/tuj/m5_motion/tests/test_vlm_provider.py \
  src/tuj/m5_motion/tests/test_c4_2_packing.py \
  src/tuj/m5_motion/tests/test_container_release.py \
  src/tuj/m5_motion/tests/test_container_surface.py \
  src/tuj/m5_motion/tests/test_container_surface_support.py \
  src/tuj/m5_motion/tests/test_scripted_transport.py \
  src/tuj/m5_motion/tests/test_packing_occupancy.py \
  src/tuj/m5_motion/tests/test_packing_position.py \
  src/tuj/m5_motion/tests/test_packing_validity.py \
  src/tuj/m5_motion/tests/test_attachment_geometry.py \
  src/tuj/m5_motion/tests/test_attachment_contact_timing.py \
  src/tuj/m5_motion/tests/test_attachment_sync.py \
  src/tuj/m5_motion/tests/test_live_execution.py \
  src/tuj/m5_motion/tests/test_vlm_provider_config.py
```
