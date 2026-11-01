# C2_2 EE 파지 검증 — 2026-11-02

> 공유 결과는 [전체 진행 요약](ee_gear_checkpoint_20261102.md)과 [조합별 결과](ee_gear_coverage_results_20261102.json)에 정리했다. 이 문서의 `output/` 경로는 로컬 보관 자료이며 GitHub 업로드에 포함하지 않는다.

현재 샌드위치 재료 5개 × 3EE의 첫 후보 검증은 **5 PASS / 10 FAIL / 0 ERROR / 0 TIMEOUT**이다. 성공은 모두 기존 흡착 경로의 접촉 조건을 통과한 부착 보조 성공이다. 빵 3F의 높이만 낮춘 추가 후보 4건도 CLOSE 실패였고, 원래 자세 대조 1건은 기존 실패를 그대로 재현했다. 물체/EE 조합의 모든 파지 전략이 불가능하다는 결과나 전체 샌드위치 태스크 성적은 아니다.

## 대상과 조건

실제 장면 entity 14개를 확인했다. Table II 재료 분모는 bread_a, bread_b, turkey_1, cheese_1, tomato_slice 5개다. knife/spatula/spoon/ladle과 받침·목적지 접시들은 원래 장면에 남겨 두고 분모에서 구분했다. 재료 로컬 크기는 사전 점검에 보존했다. 햄 두께 약 2.516mm, 치즈 4mm, 토마토 4.736mm, 빵 폭 약 113mm/두께 22.920mm다.

기존 ingredient_vac 5개 설정은 등록 파라미터 전체가 변경 전과 동일하다. 미등록 2F/3F 10조합만 validator 전용으로 추가했다. 새 후보는 CatalogRecipe 기본 힘·물성·접촉/충돌/부착 게이트·lift 0.18m/minimum 0.10m·hold 5s·slip 5mm/5°를 유지하고, 위치/자세/초기 개구만 정했다. 일반 M4 지원으로 승격하지 않았다.

- 빵 2F: 폭이 85mm 개구를 넘으므로 측면에서 두께를 집는 후보.
- 슬라이스 2F: 짧은 로컬 Y방향으로 닫는 상부 접근.
- 3F: 상부 중심 enclosure, 완전 개방으로 접근.
- 기존 vac는 별도 교정된 접촉 수·흡착 명령/이득·접촉 물성 등을 그대로 사용한다. 새 손 후보와 균일 프로토콜의 인과적 EE 비교가 아니다.

seed 0, 매 조합 독립 프로세스/새 환경, 2초 초기 안정화, 최대 동시 2개, 600초 제한, agentview 960×540/20fps, API 0. 사전 회귀 검사 **115 PASS**. 첫 배치 **801.177초**, 후속 5건 **76.653초**. 총 20건이며 원래 15조합 분모에 후속 5건을 더하지 않는다. 두 배치 모두 소스 불변 확인. 후속은 프로세스 내부에서 recipe 높이만 바꿨으며 그 worker/driver 해시도 보존했다.

## 첫 15조합 결과

| 물체 | 2F | 3F | vac |
|---|---|---|---|
| bread_a | FAIL, PRE_CLEARANCE IK | FAIL, CLOSE | PASS, 부착 보조 |
| bread_b | FAIL, PRE_CLEARANCE IK | FAIL, CLOSE | PASS, 부착 보조 |
| turkey_1 | FAIL, 받침 충돌 | FAIL, CLOSE | PASS, 부착 보조 |
| cheese_1 | FAIL, 받침 충돌 | FAIL, CLOSE | PASS, 부착 보조 |
| tomato_slice | FAIL, 받침 충돌 | FAIL, CLOSE | PASS, 부착 보조 |

**2F 0/5, 3F 0/5, vac 5/5.** 역사적 Table II는 0/5, 2/5, 5/5다. 후보 범위·실행 프로토콜을 확인해야 하므로 새 수치를 바로 논문 표에 대입하지 않는다.

실패 단계는 PRE_CLEARANCE IK 2건, PLAN_GRASP 충돌 3건, CLOSE 안정 접촉 미확보 5건이다. 슬라이스 2F에서 손끝과 받침의 관통은 각각 햄 2.576mm, 치즈 3.314mm, 토마토 2.739mm로 차단됐다. 허용치를 늘리거나 받침을 제거하지 않았다.

흡착 5건은 모두 ACQUISITION attachment를 사용한다. minimum hold lift는 빵 a/b 179.957/179.954mm, 햄 175.961mm, 치즈 176.127mm, 토마토 176.542mm다. 전부 5초 유지했다. CONTACT_GATED_KINEMATIC_ATTACHMENT의 성공이며 부착 보조 없는 유지나 실제 흡착 밀봉을 따로 입증한 것은 아니다.

## 빵 3F 높이 진단

원래 z fraction 0.5에서는 두 빵 모두 CLOSE 300표본 중 282표본에 세 손가락 접촉이 있었다. 그러나 접촉 중심은 윗면 약 +11.4mm이고 opposing-normal 최댓값이 0(요구 0.5)이었다. 위에서 누르는 접촉이며 단순 힘 부족이나 코드 오류로 바꾸어 해석하지 않았다.

사전 고정한 두 높이 z fraction 0, -0.25를 각각 bread_a/b에 적용한 **4후보 모두 CLOSE 실패**. 힘·판정·초기 개구·물성은 바꾸지 않았다. 원래 자세 대조 1건도 동일 실패이며 초기/파지/최종 상태 파일의 모든 배열과 판정·지표가 기존 배치와 정확히 같다. 네 새 후보도 각각 원래 배치와 초기 상태 배열이 정확히 같다. 추가 성공이나 일반 실행 승격은 없다. 후속 결과를 본 뒤 더 많은 높이를 계속 탐색하지 않는다.

## 증거와 비교 한계

두 배치의 모든 저장 JSON/NPZ 유한값을 확인했다. 20개 영상의 처음/중간/끝 프레임은 모두 읽을 수 있고 검은 화면이 아니다. 모든 프레임을 시각적으로 해석한 검사는 아니다. API 호출 0, 원본 소스·공유 보관 참조의 해시 검증 완료.

첫 배치의 초기 pose payload는 EE별로 3종이다. 파지 대상 재료의 최대 위치 차이는 0.001465mm(치즈), 최대 각도 차이는 0.002931°(햄)다. 주변 국자는 최대 **7.526mm / 18.777°** 차이가 있으므로 장면 전체가 완전히 동일하다고 가정하지 않는다. 그 차이가 개별 실패에 영향을 줬다는 인과적 결론도 내리지 않는다. 후속 3F 후보는 모두 같은 초기 상태에서 비교했다.

## 보관·원자료·다음 작업

- 실제 장면·기존 설정 (로컬: `output/c2_2-ee-coverage-20261102-a/scene_preflight.json`), 첫 배치 manifest (로컬: `output/c2_2-ee-coverage-20261102-a/run_manifest.json`), 15조합 감사 (로컬: `output/c2_2-ee-coverage-20261102-a/evidence_audit.json`), 레시피별 프로토콜 (로컬: `output/c2_2-ee-coverage-20261102-a/recipe_protocols.json`).
- 초기 상태 차이 (로컬: `output/c2_2-ee-coverage-20261102-a/initial_pose_audit.json`), 후속 후보 선택 근거 (로컬: `output/c2_2-ee-coverage-20261102-a/bread_contact_diagnosis.json`), 사전 검사 (로컬: `output/c2_2-ee-coverage-20261102-a/tests.log`).
- 후속 manifest (로컬: `output/c2_2-bread-height-20261102-a/run_manifest.json`), 후속 판정 감사 (로컬: `output/c2_2-bread-height-20261102-a/evidence_audit.json`), 원래 자세 대조 (로컬: `output/c2_2-bread-height-20261102-a/control_comparison.json`).

완료 파일 중 byte hash와 권한이 같은 **69개**를 hardlink로 공유했다. 할당 블록 기준 **170,315,776 bytes(약 170MB)**의 중복 저장분을 제거했고, 두 배치 실측 합계는 공유 inode를 한 번만 세면 약 104MiB다(공유 소스 pool 별도). 이는 새 출력 중복 제거량이며 실험 전 대비 순감소량은 아니다. 모든 원래 경로와 고유 실패를 보존했고, 보관 파일은 제자리 수정하지 않는다. 각 storage_audit.json에 원본·경로·해시가 있다.

이번에 새 공통 실행 오류는 발견하지 않았다. 기존 흡착 설정과 물리 기준은 보존했다. GitHub `fix/dohee-ablation_study`는 2026-11-02 재확인 시 `6156e34ef4b8800459e15a7338bfb4e5e5622632` 그대로다. 실험 완료 시점에는 커밋과 업로드 전이었다. 다음 담당 검증은 C2_1이며 최종 Full/w/o Reasoner 8개 공통 코드 재실험은 별도 남아 있다.
