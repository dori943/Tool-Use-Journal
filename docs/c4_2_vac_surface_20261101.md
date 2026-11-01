# C4_2 흡착 표면 후보 검증 — 2026-11-01

> 공유 결과는 [전체 진행 요약](ee_gear_checkpoint_20261102.md)과 [조합별 결과](ee_gear_coverage_results_20261102.json)에 정리했다. 이 문서의 `output/` 경로는 로컬 보관 자료이며 GitHub 업로드에 포함하지 않는다.

## 결론

기하로 정한 길이 방향 +30mm/−45mm 후보를 채택하지 않는다. 원래 controller_validate_case의 영상 포함 경로에서도 둘 다 CLOSE `GRASP_CONTACT_NOT_STABLE`로 실패했다. 원래 중심 파지는 같은 경로에서 성공했고, 과거 독립 성공의 저장 초기/GRASP 종료/최종 상태와 파지 지표가 정확히 일치했다. 특정 공통 코드 버그나 수정안은 이번 검증에서 확정하지 않았다.

## 기하 후보 선정

기존 선택기는 컵 아래 collision vertex가 있으면 그곳의 최고 높이를 쓰고, 접촉 유지 여유를 후보끼리 비교하지 않는다. 실제 컵은 반경30mm, 바게트 폭59.554mm. 단, 이 치수만으로 흡착 불가능을 주장하지 않는다.

기존80mm 근방 안에서 X={−5,0,+5}mm/Y=−75..75mm(5mm 간격), 총93개 후보를 고정했다. 컵 아래 최고점0.2mm 밴드의 점들로 만든 투영 볼록껍질 면적을 중심거리로 보정해 순위를 정했다. 중심15mm 주변을 제외하고 서로25mm 이상 떨어진 상위2개를 선택했다. +30mm 후보의 상부 면적64.627mm², 기준3.381mm². 이 값은 기하 제안 점수이며 접촉/밀봉/물리 성공의 증명이 아니다.

선정 범위·전체 순위 (로컬: `output/c4_2-vac-surface-screen-20261101-a/screen_result.json`), 실제 충돌 형상 (로컬: `output/c4_2-vac-surface-screen-20261101-a/geometry.json`).

!충돌 형상과 컵 위치 (로컬: `output/c4_2-vac-surface-screen-20261101-a/surface_candidates.png`)

## 실제 파지 검증

레시피의 offset_m X/Y만 바꿨다. 힘·물성·접촉수·정렬·부착·침투·상승·유지·slip 조건은 그대로다. 등록 파일이나 공통 소스는 수정하지 않았다. 모든 실패와 로그 보존, API0.

| 실행 조건 | 원래 중심 | +30mm | −45mm |
|---|---|---|---|
| a: 독립 초기 qpos/qvel/ctrl/time 복원 | CLOSE 실패 | CLOSE 실패 | CLOSE 실패 |
| a: 실제 순차 실행의 acquire 전 world pose/팔 상태 재구성 | 파지 성공 | CLOSE 실패 | CLOSE 실패 |
| b: 새 장면2초 안정화 + 진단용 초기화 | CLOSE 실패 | CLOSE 실패 | CLOSE 실패 |
| c: 기존 controller_validate_case 및 영상 경로 그대로 | 파지 성공 | CLOSE 실패 | CLOSE 실패 |

총12건, 변경 후보 검증8건은 모두 실패. 이는 조건별 유한 진단이며 무작위 성공률이 아니다. 전체 감사 (로컬: `output/c4_2-vac-surface-screen-20261101-a/grasp_audit.json`).

### 초기화 차이의 해석

a의 복원/재구성은 solver warm-start, 제어기 이력, 선행 물체의 접촉 모델 변경 등을 복원하지 못한다. 측정 world 재구성은 free-object velocity/actuator commands도0으로 초기화했다. b도 원래 검증기와 달리 추가 mj_forward 및 controller.new_update 초기화가 있고 영상 경로가 없다. 따라서 a/b를 원래 실행의 정확한 replay로 표현하지 않는다. 원래 파지의 결과가 뒤집힌 것을 특정 오차·영상·제어기 하나의 원인으로 단정할 수 없다. 실제 순차 실패를 a의 성공으로 교체하지 않는다.

c에서는 진단 초기화 대신 기존 controller_validate_case를 그대로 사용했다. 원래 파지는 과거 독립 검증과 initial_state.npz/grasp_state.npz/final_state.npz의 기록 배열이 전부 동일하고, 최소 유지 상승0.160904m·5초 유지·부착 유지율1.0 등의 모든 metrics도 동일했다. 따라서 이 파지에 대해 과거 독립 성공을 재현했고, 해당 경로가 새 공통 코드에서 퇴행했다고 볼 근거는 없다. 모든 EE 조합의 재검증을 뜻하지 않는다.

기준 재현 감사 (로컬: `output/c4_2-vac-patch-grasp-20261101-c/baseline_reproduction_audit.json`), 원래 파지 영상 (로컬: `output/c4_2-vac-patch-grasp-20261101-c/baseline/trial.mp4`), +30mm 실패 영상 (로컬: `output/c4_2-vac-patch-grasp-20261101-c/crown_1/trial.mp4`), −45mm 실패 영상 (로컬: `output/c4_2-vac-patch-grasp-20261101-c/crown_2/trial.mp4`).

## 결과에 따른 후속 선택

기하 면적만 넓힌 두 후보는 더 검증할 성공 근거가 없으므로 승격/전체 실행하지 않는다. 앞서 실제 전체 실행에서 관측한 2F PLACE 안착 실패와 vac ACQUIRE 접촉 실패를 각각 기존 M4 scene_candidates에만 반영했다. 추정/feasible EE/전역 등록은 그대로다. M4는 남은3F를 선택했고 바게트 그룹만 변경, 순서 동일, EE교체1→2/motion24→27. 새3F 실제 전체 진단도 완료했다. 파지·운반·5개 수납은 통과했으나 바게트 안착에서 실패했다. 이는 새 초기 상태에서 시작하는 별도 개발 진단이며 실제 실패 끝 상태의 recovery나 최종 ablation 점수가 아니다.

3F 전체 결과는 [별도 기록](c4_2_third_ee_20261101.md)에 남긴다.
