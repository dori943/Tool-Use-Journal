# C4_2 실제 Full 대조 실행과 배치 탐색 오류

> 공유 결과는 [전체 진행 요약](ee_gear_checkpoint_20261102.md)과 [조합별 결과](ee_gear_coverage_results_20261102.json)에 정리했다. 이 문서의 `output/` 경로는 로컬 보관 자료이며 GitHub 업로드에 포함하지 않는다.

후속 배치·순서 진단은 [별도 기록](c4_2_baguette_layout_20261031.md)을 참고한다.
새 정책 순서 진단은 a~g의 동일 입력 Full 대조와 구분한다.

## 최신 상태 — g 완료

- Full: 계획 통과. 시리얼·우유·밀대·거품기 수납 유지. 바게트는 5.02초 안착 관측 뒤
  움직임은 충분히 줄었으나 상자에서 30.566mm 돌출하여 **CONTAINER_NOT_SETTLED/goal**.
  손 접촉 0, 부착 없음, 팔 위치 오차 0.00672mm. 뚜껑은 미실행이며 **전체 목표 실패**.
- frozen v3 w/o Reasoner: 보존한 추정값으로 **MISSING_REQUIRED_TOOL:SG1**, M5 미실행.
  이후 변경은 M5/진단/테스트이며 이 M2 거절에 영향이 없다. Full의 포장 실패와 따로 분류한다.
- 최신 관련 회귀 검사 **391 passed, 2 skipped**. 별도 통합 184 passed/1 기존 실패는
  아래에 기준 커밋 재현 근거를 기록했다. 전체 테스트 스위트가 모두 통과했다는 뜻은 아니다.
- g 실제 시간 509.457초, M5 원응답 10개, 소스 및 frozen prefix 보존 확인. 지지 증거와
  대응 world snapshot 12개의 해시 일치 확인. 실행 중인 실험은 없다.
- 다음 대상은 **바게트의 선언된 배치 방향·XY 후보와 기존 적층 사이의 접촉/최종 돌출**이다.
  현재 입력/순서의 실패를 보존하고 후보별 진단으로 분리한다. 추정값·힘·성공 허용치는 바꾸지 않는다.
- C4_2 개발 진단 결과이며 논문 최종 성공률이 아니다. 다른 7개 및 공통 최종 코드 재검증은 남아 있다.

감사 자료: `output/c4_2-full-control-20261031-g/full_control_run_audit.json`,
`remaining_issue.json`, `regression_tests.xml`. 아래 a~g 내용은 각 실행 당시의 역사 기록이다.

## 실험 조건

공통 작업 브랜치는 `fix/ee-gear-validation-20261102`, 기준 커밋은 `6156e34`다.
새로 검증한 파지 8개가 연결된 공통 코드에서 gpt-4o-mini, seed 0,
Full planner, controller 실행, task-goals-v1, 물리 재시도 0회로 비교했다.
실제 Full의 M1은 SiPhy 백엔드와 새 빈 메모리에서 생성했다. 기존 ablation의
mock `m1_full.json`은 대조군으로 사용하지 않았다.

Full 새 관측과 기존 frozen v3 ablation의 장면 이미지 및 EE 경로 금지영역은
파일 해시가 일치했다. 물체 identity/class/center_mm/bbox_mm와 edge topology도
같았다. 원본 ablation 추정값은 수정하지 않고 복사하여 새 M2를 생성했다.
관측·수치 비교는 `output/c4_2-full-control-20261031-a/input_comparison.json`에 있다.

## 첫 Full 및 같은 코드의 w/o Reasoner

출력 루트: `output/c4_2-full-control-20261031-a`.

| 조건 | 계획 | 실제 실행 | 전체 목표 | 중단 원인 |
|---|---|---|---|---|
| 실제 Full | M4 SUCCESS | 시리얼 파지·운반·배치와 우유 2F 파지 통과 | 실패 | 우유 운반 전 배치 탐색의 PLACEMENT_SPACE_EXHAUSTED |
| w/o Reasoner, frozen v3 입력 | M2 REJECTED | 미실행 | 미평가 | MISSING_REQUIRED_TOOL:SG1 |

Full의 SiPhy와 계획에 새 API 응답 11개, frozen ablation의 새 M2에 6개가
기록됐다. Full에서 시리얼은 2F/3F/흡착, 뚜껑은 흡착이 가능으로 판정됐다.
기존 ablation 추정에서는 두 물체가 모든 EE 불가다. 두 실행 모두 실패했으므로
이 결과를 Full 성공 대비 ablation 실패라는 최종 비교로 보고하면 안 된다.

NumPy 수치 경고가 있었지만 저장된 M1 수치와 점군은 모두 유한했다.
일반 행렬 곱과 다른 계산식으로 재계산한 PCA 치수도 모든 물체에서 일치했다.
이 검사는 모든 내부 수치 연산의 완전한 보증을 뜻하지 않는다.

## 확인된 공통 코드 오류와 수정

안정면 방향을 순회하는 `configure_stable_face`가 특정 방향의 빈자리 탐색에서
`PLACEMENT_SPACE_EXHAUSTED`를 받으면 나머지 방향을 검사하지 않고 전체 탐색을
중단했다. 실제 저장된 우유 운반 요청에서도 같은 호출 경로로 재현됐다.

수정은 이 오류만 해당 방향의 기각으로 기록하고 다음 방향을 검사하도록 한다.
다른 오류는 다시 발생시키고, 모든 방향이 막히면 여전히 실패한다. 이미 선택해
유지하는 방향을 임의로 바꾸는 경로에는 적용하지 않는다. 치수, 점유 검사,
충돌 여유 5mm, 힘, 접촉 조건 및 목표 성공 판정은 바꾸지 않았다.

수정 후 동일한 저장 장면에서 점유 검사에 통과하는 우유 배치 후보를 찾았다.
이 오프라인 재현은 실제 팔 IK/경로/물리 실행 성공을 대신하지 않는다.
증거: `output/c4_2-full-control-20261031-a/stable_face_fix_replay.json`.
관련 회귀 테스트는 100 passed, 2 skipped다.

## 수정 후 실행

`output/c4_2-full-control-20261031-b`에서 첫 Full의 실제 M1/M2/G_k/M4를 그대로
복사하고 M5를 초기 장면부터 다시 실행한다. 새 M1/M2 호출 없이 동일 계획을
사용해 수정 전후를 분리한다. 이전 실패 결과는 보존한다.

b 실행 완료: 우유 배치까지 성공했고 밀대 운반 목표 생성에서 중단됐다. 아래에 결과와 후속 수정을 기록한다.

### 두 번째 실행 결과와 3D 포장 경로 수정

b 실행은 시리얼과 우유의 파지·운반·배치를 통과했다. 밀대 파지도 성공한 뒤,
밀대 운반 목표 생성에서 PLACEMENT_SPACE_EXHAUSTED로 중단됐다. 따라서 첫 안정면
탐색 수정은 실제 우유 배치까지 확인됐으나 전체 목표는 여전히 실패다.

밀대의 방향은 충돌 형상에 맞춰 기울여 상자 입구 위에서 놓도록 명시되어 있고,
상자는 기존 NEAREST_CLEAR_VOLUME 정책을 사용한다. 이 경로에 앞서 일반 배치의
2D 바닥 빈자리 검사를 요구하여 기존 3D 공간 선호·IK·해제 경로가 호출되기 전에
중단되는 회귀를 재현했다. 원래 포장 경로는 하부 물체 위의 상부 공간을 사용하고,
최종 수납은 물리 해제와 task-goals-v1로 확인한다.

수정은 collision-mesh packing_bounds와 CONTAINER/NEAREST_CLEAR_VOLUME가 모두
명시된 경우에만 입구 XY 후보를 별도로 생성한다. 실제 collision vertex 경계와
5mm 여유로 다시 제한하며, 기존 3D 공간 선호·열린 손 해제 통로·IK·경로 충돌·
물리 해제·최종 포함 검사는 유지한다. 같은 조건의 일반 바닥 배치는 여전히
겹침을 거부한다. 이 분리 자체로 물리적 성공이 보장되지는 않는다.

관련 테스트: 122 passed, 2 skipped. 같은 실패 요청의 오프라인 목표 생성 결과는
b 폴더의 `volume_entry_fix_replay.json`에 저장했다.

c 실행 (`output/c4_2-full-control-20261031-c`)은 두 수정을 포함하여 초기 장면부터
같은 실제 Full M1/M2/G_k/M4로 다시 실행 중이다. 동일 최신 코드의 frozen v3
w/o Reasoner도 새 M2로 실행해 비교 조건을 맞춘다.

동일 c 코드의 frozen v3 w/o Reasoner 실행은 완료됐다. 새 M2 응답 6개를 기록했고,
`MISSING_REQUIRED_TOOL:SG1`로 다시 중단됐다. 원본 및 복사한 frozen 입력 해시는
모두 유지됐다. 실제 실행과 목표 판정은 진행하지 않았다.


### 세 번째 실행 결과와 생성 후보 누락

c 실행은 시리얼·우유의 전체 배치와 밀대 파지·운반까지 통과했다.
밀대 PLACE의 첫 PRE_PLACE에서 네 후보 모두 IK_SEARCH_EXHAUSTED로 중단됐다.
실행 시간은 330.147초이며 소스와 frozen prefix 보존을 확인했다.

저장된 생성 결과에서 네 후보 모두 측정된 시작점에 0.1m를 더했다.
기존 프롬프트가 요구하는 측정 물체 시작점 → 배치점 → 측정 손 반환점의
zero-offset 후보는 하나도 없었다. 오프라인 변환 재현에서 추가 거리는 정확히
0.1m이며, 해당 compact 후보의 시작·반환 EEF 위치는 측정 위치와 오차 0m다.
증거는 c 폴더의 `compact_place_fix_replay.json`이다. 이 재현은 IK/물리 성공을
대신하지 않는다. 이 실패는 앞의 배치 탐색 예외와 구분하여 **M5 생성 후보의
계약 누락 및 도달 불가 제안**으로 기록한다. Reasoner 제거 실패로 세지 않는다.

공통 생성기는 기존 transport direct 후보와 같은 방식으로 측정 상태가 모두
주어진 배치에서 compact 후보가 누락되면 한 후보를 보존한다. 후보 수는 그대로,
출처는 TASK_GEOMETRY/GROUNDED_PLACE_COMPACT_V1로 표시하고 기존 해제 이벤트를
유지한다. 객체와 빈 손의 좌표를 구분하며, 일반 IK·충돌·해제·수납 판정을 거친다.
특정 C4_2 물체나 EE에 대한 성공 강제 분기는 없다. 관련 테스트 166 passed,
2 skipped. d 실행은 동일 M1/M2/G_k/M4로 다시 물리 실행 중이다.

**API 계측 범위 정정:** a~c의 `api_responses`는 Chat Completions 관측 수다.
M5는 Responses API를 사용하므로 0이라는 값은 M5 API 호출이 없었다는 뜻이
아니다. 저장된 generated artifact에서 확인되는 서로 다른 M5 응답 ID는
각각 a=2, b=4, c=6개다(미보존된 거절 응답까지 포함하는 총 호출 수는 아님).
67963 토큰도 계측된 M1/M2 범위이며 M5를 포함한 전체 비용으로 쓰면 안 된다.
d부터 Chat과 Responses 원요청·원응답을 모두 저장한다. M1/M2/M4는 고정했지만
M5 생성은 매 실행 새로 수행하므로 완전히 동일한 동작 후보의 paired replay는
아니다. 결정적 오류 재현은 별도의 저장 요청 audit으로 확인한다.

c→d 소스 차이는 `vlm_provider.py`와 그 테스트뿐이다. w/o Reasoner의 중단은
M2에서 발생하고 M5를 호출하지 않으므로, c의 frozen v3 중단 결과는 해당 수정의
영향을 받지 않는다. d에서 동일 M2 호출을 추가 반복하지 않았다.


### 네 번째 실행: 실제 수납 성공과 지지 검사 불일치

d는 317.956초에 종료했고 실행 중 소스 및 frozen prefix가 보존됐다. M5 Responses
원응답 6개가 기록됐다. 밀대 compact 후보가 선택되어 IK·경로·해제·빈 손 반환을
통과했다. 실제 segment의 container_settle도 succeeded=true, attached=false,
hand_contact_count=0이었다. 최종 task-goals-v1에서 밀대·시리얼·우유의 3D 수납은
모두 true였다. 밀대 overhang은 0.000000540m다.

그러나 일반 PlacementPreservation은 컨테이너에도 `_resting(target, box)`라는
바닥 직접 지지 검사를 적용해 밀대 supported=false로 중단했다. 이는 수납 실패와
구별한다. d의 접촉 진단은 target↔region만 저장하여 내용물 사이 지지 접촉을
증명하지 못한다. 기존 기하학적 support_chain도 기울어진 mesh의 중간 접촉을
표현하지 못하므로 그 결과만으로 적층 지지 성공을 강제하지 않았다.

새 연결은 명시적 CONTAINER에 대해 매 실행 요청 뒤 현재 MuJoCo 접촉력에서
상향 지지 경로를 측정한다. 경로의 peer는 3D 내부에 있고 해제된 PACKABLE_OBJECT만
허용한다. 실제 접촉거리 -5mm~0, 양의 상향 접촉력, 상자로 이어지는 경로를 요구하고,
관측을 현재 world 해시와 묶는다. 공중·옆면만 접촉·로봇 지지·순환·오래된 관측은
거부한다. 일반 트레이의 기존 검사와 task-goals-v1 전체 목표 함수는 변경하지 않았다.
컨테이너 누적 포함 검사는 오히려 기존 2D에서 3D 전체 포함으로 확인한다.

실제 MuJoCo 두 물체 적층과 음성 사례를 포함한 관련 테스트 84 passed,
목표 평가·기존 포장 관련 별도 검사 146 passed, 2 skipped.
e 실행은 동일 Full M1/M2/G_k/M4로 진행 중이며, 실제 상향 지지 경로와 이후
거품기·바게트·뚜껑 단계를 확인한다. d의 전체 목표는 후속 물체 미실행으로 실패다.


### 다섯 번째 실행: 5개 배치 후 EE 진입 중 보존 실패

e는 475.833초, M5 Responses 10개(400512 토큰), 실행 중 소스·frozen prefix 보존.
밀대 지지는 실제 contact force로 `rolling_pin → milk → cereal → packing_box`
경로가 관측됐고, 다음 단계에서는 `rolling_pin → milk → packing_box` 경로도
관측됐다. 거품기와 바게트도 배치·해제·3D 수납·접촉 지지 검사를 통과했다.

그러나 이어진 EE_EXCHANGE_ENTRY(2F 반환 경로 진입) 3.44초 뒤 바게트의 3D
최대 돌출이 19.999mm가 되어 PLACEMENT_PRESERVATION_FAILED로 중단됐다.
시리얼·우유·밀대·거품기는 최종 3D 수납을 유지했고 뚜껑은 미실행이다.
바게트는 해당 구간에서 약 33.7mm 이동, 29.72도 회전했다. 동작 보고서의
충돌 수는 0이지만 이것만으로 팔의 물리 영향을 완전히 배제하지 않는다.
최종 직전/중간/종료 영상 프레임은 e 폴더의 `final_minus_*_frames.png`에 있다.

바게트 PLACE 직후 관측 속도는 0.0165778m/s, 각속도 0.169685rad/s였는데,
container_settle은 인접 표본 변위 0.378mm/회전 0.00331rad를 안정으로 판단했고
RETREAT 추가 대기는 0.08초에 끝났다. 짧은 표본 간격에서 움직임이 작다는 조건이
실제 정지를 보장하지 않는 구현상 한계다. 팔 영향과 자발적 적층 변화의 인과적
분리는 정지 대조 실험 없이 확정하지 않는다.

f에는 기존 pose tolerance와 free_object_settle_seconds=5를 이용한 속도 검사를
추가했다. 현재 속도를 5초로 환산한 이동·회전이 동일 허용치 안이어야 안착으로
확인한다. 새 관측 규칙이며 task-goals-v1 전체 목표/물리 파라미터/5초 대기 상한은
변경하지 않는다. 속도 미관측·비유한 값도 거절한다. 기록된 바게트 속도 재현과
기존 검사 63 passed. f는 동일 M1/M2/G_k/M4를 사용하여 실행 중이다.

a~e의 초기 물체 7개 위치 차이는 모두 0m, 회전 차이는 수치 오차 수준이었다.
비교 자료는 e의 `initial_scene_comparison.json`이다.


### 여섯 번째 실행과 안착 대기 조기 중단 수정

f는 472.315초, M5 응답 10개, 소스·frozen prefix 보존. 시리얼·우유·밀대·거품기는
수납·속도 안착·실제 지지를 통과했다. 바게트는 EE 교환 이전 PLACE/RETREAT 안착
관측에서 밖으로 나왔다(최종 3D 돌출 32.237mm, hand_contact_count=0).
이는 이전 e의 EE 진입 뒤 실패와 구별되는 관측이며, 현재 배치 후보가 자동으로
안정 포장을 보장하지 않음을 보여준다. 전체 목표는 실패이고 뚜껑은 미실행이다.

이때 팔 위치 오차는 0.0133mm로 이미 허용치 안이었지만 기존 arm-error 정체
검사가 1.28초에 대기를 종료했다. 설정은 max_wait_s=5였으므로 5초 관측을 완료한
결과라고 보고하면 안 된다. g에서는 컨테이너 안착을 기다리는 동안 팔의 위치·각도·
관절 조건이 충족되면 팔 오차의 정체만으로 종료하지 않는다. 실제 팔이 막혔을 때의
조기 중단은 유지하고 기존 5초 상한도 유지한다. 물체 안착 시간 초과는
CONTAINER_NOT_SETTLED/goal로 분류하여 TRACKING_NOT_SETTLED와 구분한다.

제어 루프 테스트는 1.6초 뒤 물체 안착, 2초 시간 초과, 실제 팔 정체를 각각 검증했다.
관련 검사 80 passed. g는 같은 Full M1/M2/G_k/M4로 실행 중이다. g부터 접촉 증거와
동일 해시의 world snapshot도 `container_support_world.json`에 저장한다.

추가 통합 검사는 184 passed, 1 failed였다. 실패는
`test_generic_cli_video_runs_controller_simulation_and_writes_summary`의
`KeyError: completed`이며, 기준 HEAD 6156e34의 src/scripts를 별도 임시 디렉터리에
추출하여 같은 단일 테스트 실패를 재현했다. 이번 scripted C4_2 변경의 회귀로
분류하지 않았으며 기존 비스크립트 runner 문제로 남겨 둔다.


### 일곱 번째 실행 완료

g의 물리 안착 대기는 5.02초(제어 주기 경계 포함)까지 유지됐고 settle_stalled=false였다.
최종 container_settle은 stable=true/velocity_settle=true였지만 outside_target_ids에
baguette가 남아 실패했다. 즉 이번 관측에서는 대기 종료 시점에도 최종 3D 수납이
충족되지 않았다. 이를 추가 시간이나 허용치 변경으로 성공 처리하지 않았다.
M5 생성 후보는 실행마다 달라지므로 과거 e와의 차이를 오직 코드 효과라고 단정하지 않는다.

## Full h 완료: 전체 접촉력 계측 수정 후 원래 정책 대조

2026-10-31 실제 제어기 h는 508.487초 종료, 새 M5 원응답 10개 / 400,227토큰. frozen M1/M2/G_k/M4와 실행 중 소스 보존 확인. 안정면 정책은 원래 MINIMUM_HEIGHT이며 footprint 진단 override 없음.

시리얼·우유·밀대·거품기 4개 수납 유지, 바게트가 Z 30.565696mm 돌출하여 CONTAINER_NOT_SETTLED/goal 실패. 뚜껑 미실행. g와 시리얼·우유·밀대·거품기 PLACE 직후 및 최종 저장 world가 정확히 같다. 새 M5를 생성했지만 이 실행들에서는 실제 저장 상태가 일치했다. 전체 모션 후보 응답까지 동일하게 재생했다고 주장하지 않는다.

공통 마찰력 계측 수정은 실제 오류를 해결했으나 원래 정책의 바게트 수납 실패를 바꾸지 않았다. 잔여 실패는 계산에서 빠진 지지력을 인정하는 문제와 구분해야 한다. 산출물은 `output/c4_2-full-control-20261031-h`의 diagnostic_result.json, measured_state_comparison.json, completed_comparison.json, 실제 영상과 원응답. 별도 우유만 최소 면적 진단 c는 진행 중이며 최종 ablation 성공률에 합치지 않는다.

## 우유만 최소 면적 c 완료 — 5개 수납, 안착 실패

501.060초 실제 제어기 실행. 새 M5 원응답 10개 / 401,249토큰. frozen prefix 및 실행 중 소스 보존 확인. 바게트 PLACE(0015)까지 실행하여 **다섯 내용물 모두 기존 허용치 안에서 3D 수납**됐다. 바게트는 높이 방향 8.702mm 여유가 있었고, 가장 큰 옆벽 초과는 거품기 약 0.305mm로 기존 1.5mm 접촉 허용치 이내였다.

그러나 5.02초 안착 관측 종료 시 최대 선속도 0.00289548m/s, 최대 각속도 0.167451rad/s가 남았다. 기존 5초 horizon으로 투영한 이동 14.477mm / 회전 0.837255rad가 각각 5mm / 0.05rad 허용치를 넘어서 CONTAINER_NOT_SETTLED/goal 실패다. 붙어 있는 물체 없음, 손 접촉 0. 뚜껑은 미실행. 최종 전체 성공으로 표시하지 않는다.

이 속도는 안착 대상들에 대한 최댓값이다. 현재 최종 report는 물체별 속도/최댓값의 물체 ID를 저장하지 않으므로 바게트 하나의 속도라고 단정할 수 없다. 다음 검증은 물체별 속도·접촉을 동일 관측창에서 기록하여 회전/미끄러짐이 남는 물체를 식별하는 것이다. 관측 시간을 늘린 후속 진단을 하더라도 이번 기존 5초 실패는 보존하고 정식 성공으로 바꾸지 않는다.

c의 마지막 프레임과 영상, 최종 형상/접촉 world 해시/API 감사는 c/diagnostic_result.json, final_frame.png, video_audit.json, full/c4_2/seed_0/m5/c4_2.mp4에 있다. 영상 검사는 마지막 프레임만 수행했으며 전체 프레임 무결성 검사로 표현하지 않는다.

### 이번 단계의 분류와 종료 상태

- 확인된 공통 구현 오류: 접촉 좌표계의 접선 힘을 버린 지지력 계측. 전체 world 힘 변환으로 수정, 관련 70 tests passed, 실제 b에서 같은 저장 장면의 벽 마찰 지지 2.648941959N 확인.
- 원래 정책 h: g와 최종 world까지 동일, 바게트 30.566mm 돌출. 계측 수정만으로 해결되지 않는 실제 Full 포장 실패.
- 두 carton 최소 면적 b: 거품기 39.753mm 돌출, 3개 수납.
- 우유만 최소 면적 c: 5개 수납, 안착 미충족, 뚜껑 미실행. 진단 정책은 공통 코드에 채택하지 않음.
- frozen v3 w/o Reasoner: 기존 M2 MISSING_REQUIRED_TOOL:SG1 결과 유지. 이번 M5 계측/배치 진단으로 해당 거절을 Reasoner의 물리 실행 실패라고 바꾸지 않음.
- 실행 중인 이번 실험 없음. 전체 8개 태스크의 최종 공통 조건 재실험과 C4_2 전체 목표 성공은 아직 미완료. 실험 완료 시점에는 커밋과 업로드 전이었다.
