# C2_1 EE 파지 검증 — 2026-11-02

> 공유 결과는 [전체 진행 요약](ee_gear_checkpoint_20261102.md)과 [조합별 결과](ee_gear_coverage_results_20261102.json)에 정리했다. 이 문서의 `output/` 경로는 로컬 보관 자료이며 GitHub 업로드에 포함하지 않는다.

현재 물체 5개 × 3EE 검증은 **9 PASS / 6 FAIL / 0 ERROR / 0 TIMEOUT**이다. 새 빵·머그 2F 후보는 각각 seed 1·2 반복까지 **4/4 PASS**다. 성공 9개 중 6개는 부착 없이 유지한 뒤 부착하고, 3개 vac는 파지 중 부착 보조를 사용한다. 독립 파지·들어 올리기·유지 결과이며 전체 분류 태스크나 w/o Reasoner 최종 점수는 아니다.

## 대상·설정·검사

실제 장면은 apple, bread, mug, plate, spoon과 목적지 blue_tray다. 기존 6경로(apple/bread/mug 3F, plate vac, spoon 2F/3F)의 모든 등록 파라미터는 변경 전후 동일하다. 누락 9조합을 validator 전용으로 추가했다. 일반 M4 실행 지원에는 추가하지 않았다.

새 후보는 CatalogRecipe 기본 힘·물성·접촉/충돌/부착 게이트·lift 0.18m/minimum 0.10m·hold 5s·slip 5mm/5°를 유지했다. apple/bread/mug 2F는 C3_1에서 사용한 위치·자세·개구만 참고한다. 기존 레시피의 force, contact region, post-grasp gain, lift 거리나 회전 허용값은 복사하지 않는다. 특히 새 apple 2F는 **5N/5°**이며 C3_1 기존 경로의 **8N/25°**와 다르다. plate 2F/3F는 측면 rim 후보, 새 vac는 상부 접촉 후보를 사용했다. 수치 비교는 레시피별 프로토콜 차이를 고려해야 한다.

사전 검사 **131 PASS**. EE_MISMATCH 단위 검사는 validator 후보가 늘어도 production-only 미지원 분류를 검증하도록 실험 후보를 fixture에서 제외했다. 실제 판정 코드는 바꾸지 않았다.

첫 배치는 seed 0, 각 조합 새 프로세스·환경, 2초 안정화, 최대 동시 2개, 제한 600초/조합, agentview 960×540/20fps, API0. **192.518초**. 성공한 새 손 후보 반복은 seed 1·2, 같은 코드/조건으로 **56.745초**. 총19건이다. 두 배치 소스 불변과 보관 참조 해시를 확인했다. 소스 전체 복사 없이 압축 tracked diff와 해시 기반 공유 untracked 원본을 사용하며, 반복은 첫 배치의 소스를 참조한다.

## 15조합 결과

| 물체 | 2F | 3F | vac |
|---|---|---|---|
| apple | FAIL, HOLD 회전 15.119° | PASS, 자유 유지 | PASS, 부착 보조·새 후보 |
| bread | PASS, 자유 유지·새 후보 | PASS, 자유 유지 | FAIL, CLOSE 접촉 불안정 |
| mug | PASS, 자유 유지·새 후보 | PASS, 자유 유지 | PASS, 부착 보조·새 후보 |
| plate | FAIL, PRE_CLEARANCE IK | FAIL, PRE_CLEARANCE IK | PASS, 부착 보조 |
| spoon | PASS, 자유 유지 | FAIL, HOLD | FAIL, CLOSE 접촉 불안정 |

**2F 3/5, 3F 3/5, vac 3/5.** 역사적 Table II 값은 4/5, 4/5, 1/5다. 현재 후보/프로토콜 검증 결과를 그대로 논문 표로 대체하지 않는다. 새 9후보 중 4 PASS/5 FAIL, 기존 6경로 중 5 PASS/1 FAIL이다.

- apple 2F: 176.155mm lift, 5초, 접촉 유지율1.0, slip 0.688mm/15.119°. 위치 유지는 되지만 회전이 5°를 넘으므로 HOLD FAIL. C3_1의 더 넓은 기준으로 통과한 사과와 직접 성능 비교하지 않는다.
- bread 2F: 166.087mm lift, 5초, 접촉 유지율1.0, slip 0.934mm/0.762°. 기본 force 5N, post-grasp gain 변경 없이 자유 유지 PASS.
- mug 2F: 179.598mm lift, 5초, 접촉 유지율1.0, slip 0.059mm/0.104°. 기본 force 5N으로 자유 유지 PASS.
- spoon 3F: 117.028mm lift 후 slip 34.414mm/60.086°, 접촉 유지율0으로 HOLD FAIL. 나머지는 판정표의 CLOSE/IK 실패다. 해당 후보의 실패를 모든 자세의 불가능성으로 확장하지 않는다.

성공의 부착 구분: **자유 유지 6개 → POST_VALIDATED_GRASP**, **vac 3개 → ACQUISITION/CONTACT_GATED_KINEMATIC_ATTACHMENT**. 후자는 부착 보조 없는 유지나 실제 밀봉을 별도로 증명하지 않는다.

## 반복·공통 장면·증거 감사

새 bread/mug 2F를 seed 1·2로 반복한 4건 모두 부착 없이 5초 유지 PASS. 각 대상의 초기·파지·최종 상태 파일 모든 배열과 유지 지표가 seed 0과 정확히 같다. 물체 배치가 고정되어 있으므로 고정 장면의 재현성 근거이며 배치 무작위화 강건성 실험이 아니다.

기존 6경로는 이전 C3_1 수정 후 배치와도 판정·유지 지표·실제 저장된 상태 파일18개의 모든 배열이 정확히 같다. 같은 정렬 자산/장면과 같은 교정값의 재현 결과다. 새 9후보는 이 동일 프로토콜 비교에 포함하지 않는다.

두 배치 저장 JSON/NPZ는 모두 유한값. 19개 영상의 처음/중간/끝 프레임 모두 읽을 수 있고 검은 화면이 아님을 확인했다. 모든 영상 프레임의 의미를 판독한 검사는 아니다. 첫 배치 EE별 초기 pose payload 3종, 최대 위치 차이0.076466mm/각도0.133550°(bread). 완전히 동일한 초기 장면의 인과적 EE 비교로 표현하지 않는다.

## 보관·원자료

- 장면과 기존 설정 (로컬: `output/c2_1-ee-coverage-20261102-a/scene_preflight.json`), 첫 배치 manifest (로컬: `output/c2_1-ee-coverage-20261102-a/run_manifest.json`), 15조합 감사 (로컬: `output/c2_1-ee-coverage-20261102-a/evidence_audit.json`), 레시피별 프로토콜 (로컬: `output/c2_1-ee-coverage-20261102-a/recipe_protocols.json`).
- 초기 상태 차이 (로컬: `output/c2_1-ee-coverage-20261102-a/initial_pose_audit.json`), C3_1 공통 경로 비교 (로컬: `output/c2_1-ee-coverage-20261102-a/shared_scene_comparison.json`), 검사 로그 (로컬: `output/c2_1-ee-coverage-20261102-a/tests.log`).
- 반복 manifest (로컬: `output/c2_1-finger-repeat-20261102-a/run_manifest.json`), 반복 감사 (로컬: `output/c2_1-finger-repeat-20261102-a/evidence_audit.json`), 반복 상태 비교 (로컬: `output/c2_1-finger-repeat-20261102-a/repeat_state_comparison.json`).

C2_1 출력 내부와 기존 C3_1 중 동일 bytes/권한 파일만 hardlink로 공유했다. 209회 교체 작업은 고유 경로188개에 해당하며, 할당 블록 기준 **557,322,240 bytes(약557MB)**의 중복 저장분을 줄였다. 같은 경로를 단계별로 다시 공유한 경우가 있어 작업 횟수와 경로 수를 구분한다. 이전 C3_1·C2_1 원본 경로와 고유 결과는 모두 유지했다. 저장 파일은 제자리 수정하지 않는다.

이번 턴 C2_2까지 합하면 39건/API0, 중복 절감 **727,638,016 bytes(약0.73GB)**, 고유 공유 경로257개다. 이는 새 결과물의 중복 제거량이지 실험 전 디스크 대비 순감소량은 아니다. 통합 경로·해시 재검증 (로컬: `output/c2_1-finger-repeat-20261102-a/combined_storage_audit.json`).

새 공통 실행 오류는 발견하지 않았다. 실험 완료 시점에는 커밋과 업로드 전이었다. C2_1~C4_2 여섯 태스크의 첫 EE 검증과 해당 후속 진단이 완료됐다. C1_1/C1_2는 상대 작업자 쪽 검증 범위/원자료 확인과 맞춰 이어가야 하며, 현재 공유된 audit는 규칙·등록표 조사라 실제 3EE 실험 완료 증거로 취급하지 않는다. 검증 후보의 일반 실행 반영 범위, 전체 Full/w/o Reasoner 8개 동일 코드 재실험, 최종 논문 수치 정리가 남아 있다.
