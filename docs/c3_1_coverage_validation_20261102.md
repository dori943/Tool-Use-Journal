# C3_1 EE 파지 검증 — 2026-11-02

> 공유 결과는 [전체 진행 요약](ee_gear_checkpoint_20261102.md)과 [조합별 결과](ee_gear_coverage_results_20261102.json)에 정리했다. 이 문서의 `output/` 경로는 로컬 보관 자료이며 GitHub 업로드에 포함하지 않는다.

현재 장면의 5개 물체 × 3 EE를 수정 후 같은 코드로 검증한 결과는 **10 PASS / 5 FAIL / 0 ERROR / 0 TIMEOUT**이다. 사과·머그 3F에서 발견한 환경별 설정 전달 오류를 수정했으며, 나머지 13조합은 수정 전후 저장된 상태와 지표가 정확히 일치했다. 이 검증의 범위는 독립 파지·들어 올리기·유지다. 전체 분류 태스크 성공이나 w/o Reasoner 논문 최종 점수가 아니다.

## 실험 조건과 기존 설정 보존

- 실제 환경 `C3_1_ObjectSorting`: apple, bread, mug, plate, spoon. 고정 목적지 blue_tray는 파지 분모에서 제외. 사전 장면에서 실제 ID와 물체 로컬 크기를 확인했다.
- 기존 9개 레시피의 등록 파라미터를 변경 전후 전부 비교하여 동일함을 확인했다. 2F/3F apple·bread·mug·spoon, vac plate.
- 빠진 6개만 validator 전용 후보로 추가했다: apple/bread/mug/spoon vac, plate 2F/3F. 일반 M4 실행 지원에는 추가하지 않았다.
- 새 후보는 CatalogRecipe 기본 힘·접촉·충돌·부착·물리 설정, lift 0.18m/minimum 0.10m, hold 5s, slip 5mm/5°를 유지한다. 흡착은 윗면(숟가락 bowl +Y, 머그 handle 방향 upper rim), 접시는 지름이 손 개구보다 커서 두께를 잡는 측면 rim 자세를 제안했다. 실패는 이 후보에 관한 결과다.
- 기존 레시피의 교정값은 그대로다. 특히 apple 2F는 8N/25°이며 실제 회전 12.900°로 PASS했다. 이를 5° 기준 통과로 해석하지 않는다. 레시피별 전체 설정은 각 배치 `recipe_protocols.json`에 보관했다.
- seed 0, 각 조합 독립 프로세스·새 환경, 2초 초기 안정화, 최대 동시 2개, 제한 600초/조합, agentview 960×540/20fps, API 호출 0.
- 작업 branch `fix/subin-ee-gear-validation-20261102`, base commit `6156e34ef4b8800459e15a7338bfb4e5e5622632`. 각 배치의 전체 Python 해시, 압축 tracked patch, hash 기반 untracked 원본 참조를 저장했다. 기존 출력 폴더 전체를 백업하지 않았다. 배치 실행 중 소스 불변 확인.

## 구현 오류와 수정 전후 대조

수정 전 a는 **8 PASS / 5 FAIL / 2 ERROR**, 177.022초였다. apple/mug 3F의 `bind_context`는 C3_1 레시피를 준비했지만, 물체 함수가 C2_1 기본 레시피를 다시 만들어 `Context must be initialized with the same recipe`를 발생시켰다. 실제 파지 전에 발생하는 구현 오류이며 물리적 실패로 세지 않는다. 두 ERROR에는 initial scene/state만 있고 파지 입력·trace·유효 영상 프레임은 없다.

`objects/apple.py`, `objects/mug.py`가 명시적 recipe → 바인딩된 context.recipe → standalone 기본 recipe 순서로 사용하도록 수정했다. 힘, 물성, 성공 기준, 레시피 수치는 바꾸지 않았다. 환경 설정이 같은 경우에도 바인딩된 교정값을 보존하는 회귀 검사는 수정 전 4 FAIL/2 PASS, 수정 후 관련 검사 전체 **188 PASS**다. 기존 NumPy 경고 36개는 로그에 남아 있다.

수정 후 b는 **10 PASS / 5 FAIL**, 189.828초. 나머지 13조합은 status, failure reason, metrics가 동일하며 저장 상태 파일 37개(초기·파지·최종 중 각 단계에서 실제 존재하는 파일)의 모든 배열이 정확히 일치했다. 파일 존재 여부도 일치한다. 수정한 두 사례는 초기 상태 배열이 정확히 같다. 오류 전에는 이후 상태가 없으므로 이후 궤적의 수정 전후 동일성을 주장하지 않는다.

채팅에서 첫 배치 완료 직후 10 성공/3 실패라고 먼저 말한 것은 마지막 숟가락 두 건을 확인하기 전의 잘못된 집계다. 이후 8 성공/5 실패/2 오류로 정정했고, a 원본 판정은 보존했다.

## 수정 후 15조합 결과

| 물체 | 2F | 3F | vac |
|---|---|---|---|
| apple | PASS, 자유 유지·25° 기존 기준 | PASS, 자유 유지·오류 수정 | PASS, 부착 보조·새 후보 |
| bread | PASS, 자유 유지 | PASS, 자유 유지 | FAIL, CLOSE 접촉 불안정 |
| mug | PASS, 자유 유지 | PASS, 자유 유지·오류 수정 | PASS, 부착 보조·새 후보 |
| plate | FAIL, PRE_CLEARANCE IK | FAIL, PRE_CLEARANCE IK | PASS, 부착 보조 |
| spoon | PASS, 자유 유지 | FAIL, HOLD | FAIL, CLOSE 접촉 불안정 |

EE별 결과: **2F 4/5, 3F 3/5, vac 3/5**. 역사적 Table II 값 4/5, 4/5, 1/5와는 실험 목적·파지 프로토콜을 맞춰야 하므로 표 숫자를 즉시 교체하지 않는다.

성공 10개 중 **7개는 부착 없이 유지한 뒤 POST_VALIDATED_GRASP**, **3개 vac는 접촉 조건을 통과한 ACQUISITION 부착 보조**다. 흡착 성공은 물리적 밀봉을 별도로 실증한 결과가 아니라 기존 접촉 게이트 기반 kinematic attachment 모델의 성공이다. 새로운 6후보 중 2 PASS/4 FAIL이며, 성공 후보도 일반 실행으로 승격하지 않았다.

- 수정된 apple 3F: minimum hold lift 174.922mm, slip 0.556mm/0.136°, 5초, 모든 손가락 접촉 유지율 1.0, acquisition attachment 없음.
- 수정된 mug 3F: minimum hold lift 138.916mm, slip 0.632mm/1.327°, 5초, 접촉 유지율 1.0, acquisition attachment 없음.
- spoon 3F: minimum hold lift 117.028mm지만 slip 34.414mm/60.086°이고 모든 손가락 접촉 유지율 0.0으로 HOLD 실패. 들어 올렸다는 사실만으로 성공으로 바꾸지 않는다.
- bread/spoon vac: CLOSE에서 안정 접촉 미확보. plate 2F/3F: PRE_CLEARANCE에서 전체 자세 IK 해 없음. 각각 현재 후보의 실패이며 다른 모든 자세의 불가능성은 증명하지 않는다.

## 증거 감사와 반복 검증

수정 전후 배치의 저장 JSON 및 NPZ 전부 유한값. 수정 후 15개 영상의 처음/중간/끝 프레임 모두 읽을 수 있고 검은 화면이 아님을 확인했다. 전체 영상의 모든 프레임을 해석한 검사는 아니다. 동일 seed에서 EE별 초기 상태는 미세하게 다르며, 모든 조합 쌍의 최대 물체 차이는 bread의 0.076466mm / 0.133550°다. 완전히 동일한 초기 상태를 가정한 통제 EE 비교로 표현하지 않는다.

수정된 apple/mug 3F를 seed 1·2에서 각각 반복한 **4건 모두 PASS**, 49.539초/API 0. 네 사례 모두 부착 보조 없이 5초 유지했다. 각각 seed 0과 초기·파지·최종 상태 파일의 모든 배열 및 유지 지표가 정확히 같다. 대상과 이웃 물체의 초기 위치 차이는 0, quaternion 비교의 최대 각도 차이는 약 1.52e-15°로 수치 반올림 수준이다. 따라서 서로 다른 배치의 강건성 검증이 아니라 고정된 장면의 재현성 확인이다. 반복 영상 샘플과 저장 JSON/NPZ 유한값, 소스 불변도 확인했다.

반복 상태 비교 (로컬: `output/c3_1-binding-repeat-20261102-a/repeat_state_comparison.json`)에 원본 seed 0과의 비교가 있다.

## 원자료와 다음 작업

- 수정 전: 배치 a (로컬: `output/c3_1-ee-coverage-20261102-a/run_manifest.json`), 판정 감사 (로컬: `output/c3_1-ee-coverage-20261102-a/evidence_audit.json`), 설정 전달 재현 (로컬: `output/c3_1-ee-coverage-20261102-a/recipe_dispatch_reproduction.log`).
- 수정 후: 배치 b (로컬: `output/c3_1-ee-coverage-20261102-b/run_manifest.json`), 15조합 감사 (로컬: `output/c3_1-ee-coverage-20261102-b/evidence_audit.json`), 13조합 대조 (로컬: `output/c3_1-ee-coverage-20261102-b/baseline_comparison.json`), 테스트 (로컬: `output/c3_1-ee-coverage-20261102-b/tests.log`).
- 반복: manifest (로컬: `output/c3_1-binding-repeat-20261102-a/run_manifest.json`), 감사 (로컬: `output/c3_1-binding-repeat-20261102-a/evidence_audit.json`).
- 각 원래 case 경로에 report, scene/state, trace와 영상이 남아 있다. 두 pre-grasp ERROR의 산출물 한계는 위에 명시했다. 완료된 파일 중 byte hash·mode가 같은 것만 hardlink로 공유한다. 이 보관 파일들은 제자리 수정하지 않는다.

다음 담당 coverage는 **C2_2**: 실제 장면 분모·기존 등록 경로부터 확인하고 같은 방식으로 구현 미지원과 실제 파지 실패를 나눈다. C3_1의 다섯 후보 실패는 현 근거로 보존한다. 전체 Full/w/o Reasoner 8개를 최종 동일 코드 조건으로 재검증하고 논문 수치와 연결하는 작업은 별도 남아 있다. 실험 완료 시점에는 커밋과 업로드 전이었다.

## 보관 정리 완료

세 배치에서 동일 내용·권한 파일 **256개**를 hardlink로 공유해 할당 블록 기준 **1,058,648,064 bytes (약 1.06 GB)**의 중복 저장분을 줄였다. 기존 경로·고유 실패·소스 참조는 보존했다. 이는 이번 새 출력의 중복 제거량이며 실험 전 디스크보다 그만큼 순감소했다는 뜻이 아니다. 공유 inode를 한 번만 세면 세 배치 합계 약 241 MiB다(공용 소스 pool 별도). 각 배치 storage_audit.json에 경로·원본·hash·절감량을 기록했다. 실행 중인 C3_1 실험은 없다.
