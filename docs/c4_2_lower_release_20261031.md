# C4_2 낮은 해제 후보 검증 — 2026-10-31

> 공유 결과는 [전체 진행 요약](ee_gear_checkpoint_20261102.md)과 [조합별 결과](ee_gear_coverage_results_20261102.json)에 정리했다. 이 문서의 `output/` 경로는 로컬 보관 자료이며 GitHub 업로드에 포함하지 않는다.

## 진행 방식

사용자가 후속 판단과 실행을 자율적으로 이어가고 결과를 보고하도록 요청했다. 단계마다 재개 여부를 묻지 않고, 관측→검증 가능한 가설→기존 조건의 검사→실제 대조 실행→채택/기각 순서로 진행한다. 전체 ablation 목적은 실패 원인 분리이며 전부 성공시키기 위한 임의 조정이 아니다.

## 정적 후보 검사 완료

원본: `output/c4_2-release-contact-20261031-a`의 실제 0015 PLACE 요청·계획과 0013 파지 종료 qpos. 새 폴더 `output/c4_2-lower-release-probe-20261031-a`.

- 실제 물체와 팔 자세를 PLACE 요청에서 복원하고 파지 종료 손 상태를 사용했다. 정확한 동적 체크포인트 재생은 아니다. 이 한계는 manifest에도 명시했다.
- 같은 파지 변환·방향·XY에서 기존 해제 높이 대비 0~155mm를 5mm씩 낮춘 32개 높이 검사. 공식 PLACE 충돌 context/5mm margin 유지, 손 개방 7개 표본, 연결되는 관절 경로(최대 0.02rad 간격) 검사. 기존 손 envelope 기준은 collision margin+position tolerance=10mm.
- 0~45mm까지 통과. 50mm부터 손 개방 envelope가 입구를 넘어서 최소 3.442mm 재상승 필요. 45mm는 표본 범위에서 가장 낮은 연속 통과 후보다. 연속 공간의 전역 최적 높이라는 주장은 아니다.
- nominal 후보 통과, 원본 입력과 runtime qpos/qvel/ctrl/act 불변. API 0회, 25.362초. 이것만으로 실제 하강·동적 해제·전체 목표 성공을 인정하지 않는다.

## 실제 대조 a

`output/c4_2-lower-release-20261031-a`. 45mm 낮춘 바게트 PLACE 목표만 별도 진단 entry에서 변경. 기존 clear_container_rim, M5 생성, IK/경로/충돌, controller, 5초 안착 및 전체 목표 기준을 그대로 실행한다. 이전 milk-only footprint 진단과 frozen 실제 Full M1/M2/G_k/M4를 유지. seed0, gpt-4o-mini, 실제 SiPhy/controller, 재시도0. 공통 소스 수정·정책 승격 없음. 실제 대조 완료. 결과는 아래와 같다.

원자료: probe.py/height_candidates.json/result.json, launch.py/observed_entry.py/policy_difference.json/height_policy_trace.jsonl, API/영상/실제 판정/접촉 기록.


## 실제 대조 a 완료 및 결정

532.087초, 새 M5 응답 10개 / 401,190토큰. 공통 소스/frozen prefix/진단 entry·trace 보존. 실제 해제 높이는 기준보다 45.001mm 낮고 XY 차이는 0.00182mm. 첫 네 물체 배치 후 world는 모두 기준과 정확히 같았다.

| 항목 | 원래 높이 | 45mm 낮춤 |
|---|---:|---:|
| 첫 관측 접촉 시각(해제 기준) | 0.18초 | 0.14초 |
| 첫 관측 접촉까지 낙하 | 147.493mm | 92.597mm |
| 관측 최대 각속도 | 21.062rad/s | 19.930rad/s |
| 최종 바게트 선속도 | 2.895mm/s | 0.644mm/s |
| 최종 바게트 각속도 | 0.167451rad/s | 0.031398rad/s |
| 최종 불안착 물체 | 바게트 | 바게트, 밀대 |
| 전체 목표 | 실패 | 실패 |

낮춘 실행도 거품기 접촉에서 급회전했다. 회전 peak는 5.38% 감소에 그쳤다. 5개 수납은 유지했으나 최종 바게트 0.031398rad/s·밀대 0.010792rad/s가 기존 0.01rad/s 기준을 초과한다. 뚜껑 미실행. 바게트 마지막 1초 순이동은 0.149mm, 순회전은 0.428°로 감소했지만 원래 실패 판정을 바꾸지 않는다.

주의: 안착 대기는 모두 5.02초지만 하강/후퇴 길이가 달라 최종 관측은 해제 후 5.52초와 6.34초다. 끝점 숫자 감소를 높이 하나의 순수 인과 효과로 해석하지 않는다. 같은 5.52초에서 낮춘 바게트의 각속도는 0.002051rad/s이나, 전체 물체의 원래 판정은 실패했다. M5 응답도 새 생성했으며 완전 고정 응답 replay가 아니다.

**결정: 45mm 낮춤을 해결책으로 채택하지 않고 공통 코드에 반영하지 않는다.** 다른 높이들을 통과할 때까지 반복하거나 안착 기준을 느슨하게 하지 않는다. 다음은 기존 허용 후보 안에서 다른 파지 선택으로 지지/착지 구성이 달라지는지 확인한다.

대조 수치 (로컬: `output/c4_2-lower-release-20261031-a/release_comparison.json`), 판정 감사 (로컬: `output/c4_2-lower-release-20261031-a/diagnostic_result.json`), 물체별 안착 (로컬: `output/c4_2-lower-release-20261031-a/settle_analysis.json`), 영상 (로컬: `output/c4_2-lower-release-20261031-a/full/c4_2/seed_0/m5/c4_2.mp4`).

해제 시각 기준 비교 그래프 (로컬: `output/c4_2-lower-release-20261031-a/height_comparison.png`). 두 끝점의 관측 시각 차이도 표시했다.
