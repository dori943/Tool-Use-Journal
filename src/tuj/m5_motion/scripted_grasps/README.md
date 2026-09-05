# M5 객체별 파지 함수 연결

이 모드는 **물체를 집는 구간만 고정된 객체 함수로 실행**한다. 성공한 뒤의 이동, 사용, 놓기와 EE 교체는 기존 M5 계획기를 사용한다. `--simulate controller` 또는 `--video` 실행에서 기본으로 켜진다. `--scripted-grasps`로 명시할 수도 있고, 이전의 일괄 LLM 계획·재생을 비교하려면 `--no-scripted-grasps`로 끈다. 계획만 생성하거나 kinematic으로 재생하는 모드는 기존 동작을 유지한다.

## 현재 활성화 범위

| 상태 | 객체와 EE |
|---|---|
| 검증됨: 파지·유지와 후속 이동·release 통과 | bottle(3F), spatula(3F), spoon(2F), apple(3F), bread(3F), mug(3F), knife(2F), rolling_pin(2F), baguette(2F), whisk(2F), cereal(2F), milk(3F), lid(vac attach) |
| 실험 연결: 함수는 호출하지만 성공 미검증 | plate(2F): 현재 runtime에서 lift 전 접촉 안정성 실패 |
| 이식·등록 제외 | tongs, ladle |

자동 분기는 plate를 포함한 14개 객체에 연결되어 있다. 검증 완료는 13개이고 plate는 `EXPERIMENTAL`로 manifest에 기록된다. 2026-09-04의 8개 후속 통과 기록에 09-05 보정으로 bottle, spatula, bread, mug, lid가 추가됐다. 후속 검사는 일반 M5 계획기를 통한 2cm 이동·2초 유지 및 실제 release이며, 전체 task PLACE나 모든 배치의 성공률을 뜻하지 않는다. plate 파지가 실패하면 해당 단계에서 중단하며 일반 LLM 파지로 바꾸지 않는다.

파지·5초 유지·2cm 후속 이동·2초 유지·release 기준으로 13개 객체를 검증했다. 사과는 실제 전체 task에서 파지·유지·운반, 뒤집개는 파지·5초 유지를 통과했다. 전체 task 4개는 이후 계획 단계에서 중단되었고 완료 판정은 아직 없다. plate 연결 변경을 포함한 회귀검사는 273 passed, 2 skipped이다. `SOURCE.json`의 원본 lab 소스·보정값 31개 SHA-256은 작업 전과 일치한다.

## GitHub 기준

- 기준 브랜치: `origin/main`, `13317e405c1e0bdef96a36aef7d022a019246075`.
- 확인 당시 열린 PR은 없으며 #20, #21, #23, #25~#29가 병합되어 있었다.
- 기존 로컬 `main`과 원격 이력이 달라 새 worktree와 `feature/dain-m5-scripted-grasps` 브랜치에서 구현했다.
- 원래 작업 폴더의 미커밋 M5 변경 전체를 옮기지 않았다. 필요한 접촉 파지 상태 전달만 현재 main에 추가했다.
- `SOURCE.json`은 가져온 실험 코드와 보정값의 원본 SHA-256을 기록한다. 원본 lab, vendor 코드와 객체 자산은 수정하지 않는다.

## 실행 흐름

1. M4의 SelectedPlan을 기존 어댑터로 MotionPlanRequest에 연결한다.
2. 요청 직전에 현재 runtime에서 관절 상태, 물체 위치·방향, EE 상태를 읽는다.
3. acquire/PICK 요청이면 `registry.resolve(request)`로 **환경 + 객체 + EE**를 확인한다.
4. 지원 객체이면 `grasp_<object>(context)`를 호출한다. 해당 함수의 접근, 접촉 검사, 들어 올리기, 유지 검증을 실행한다. 파지 중에는 LLM을 호출하지 않는다.
5. 파지 성공 후 실제 `T_gripper_object`, 관절값, 손가락 제어 상태를 보존한다.
6. 그다음 요청은 기존 M5 계획기로 전달한다. LLM이 생성한 키프레임을 기존 IK, 경로 계획, 충돌 검사, 컨트롤러로 실행한다.
7. 실패하면 해당 단계에서 멈추고 실제 실패 상태와 원인을 저장한다. plate는 연결 상태와 검증 상태를 구분하기 위해 `EXPERIMENTAL`로 기록한다. 실패한 파지를 성공으로 처리하거나 다른 파지 방식으로 자동 재시도하지 않는다.

지원하지 않는 객체와 일반 동작은 기존 M5로 전달한다. 알려진 보정 보류 항목은 registry의 명시적 상태를 따른다. EE가 다른 경우 임의로 바꾸지 않고 입력 오류를 반환한다.

전체 동작을 먼저 계획한 뒤 재생하는 기존 모드와 달리, 이 모드는 **계획 → 실행 → 실제 상태 읽기**를 요청마다 반복한다. 따라서 파지 결과와 다른 예측 상태에서 다음 경로를 시작하지 않는다.

## 호출 방법

Gemini M1~M3 결과가 있는 폴더에서 M4의 정확한 물체 ID와 레시피별 EE를 맞춘다. 이 준비 명령은 기존 `m4.json`을 보존하고 `m4-scripted.json`을 새로 만든다. 원래 task의 지시문과 대상 물체를 유지하며, 초기 EE는 실제 실행 환경과 일치시켜야 한다.

```powershell
python scripts/prepare_scripted_task.py c4_2 --input-dir <M1-M4-output-folder> --initial-ee 2F
python scripts/run_m5.py c4_2 --task-planner <M1-M4-output-folder>/m4-scripted.json --initial-ee 2F --provider gemini --model gemini-3.8-flash --scripted-grasps --simulate controller --headless --output-dir <new-output-folder>
```

API 키는 `GEMINI_API_KEY` 환경변수로 제공한다. 후속 계획은 실제 Gemini provider를 사용한다. 전체 task 안의 미등록 물체나 미준비 EE 교체는 기존 M5 검증에서 중단될 수 있으며, 객체 단위 통과와 별도로 확인해야 한다.

이번 전체 실행의 응답 길이 문제에는 `GEMINI_KEYFRAME_MAX_OUTPUT_TOKENS=32000`, `GEMINI_KEYFRAME_REASONING_EFFORT=low`를 사용했다. JSON schema 오류는 최대 한 번만 모델에 재생성을 요청하며, 잘못된 응답은 실행 계획이나 캐시에 저장하지 않는다. 재생성 후에도 기존 M5 검증을 통과해야 한다.

기존 generic M5 실행기에 옵션을 추가한다.

```powershell
python scripts/run_m5.py --task-planner <M4-result.json> --environment <environment> --initial-ee <EE> --simulate controller --headless --scripted-grasps --output-dir <output>
```

EE 교체의 기존 `--ee-attach-*`, `--ee-return-trajectory` 옵션을 그대로 사용한다. 현재 저장된 EE 경로가 다른 환경이나 다른 손 모델과 맞지 않으면 기존 검증에서 중단된다. Kitchen용 교체 경로가 준비되지 않은 경우 먼저 필요한 EE를 장착한 상태로 시작한다.

이미 M5 runtime이 있는 Python 코드에서는 다음 구조로 호출한다.

```python
from tuj.m5_motion.scripted_grasps.live import ScriptedGraspSession

session = ScriptedGraspSession(runtime, repository, output_dir, seed=0)
session.execute_request(pick_request)
session.execute_request(next_motion_request)
# 또는 session.execute_selected_plan(selected, constraints=constraints)
```

runtime은 `ToolUseJournalEERuntime.from_repository_for_controller(..., scripted_grasps=True)`로 만든다. 필요한 손 모델 보정은 **환경 초기화와 EE 교체 시점**에 적용된다. 파지 함수는 runtime을 다시 만들거나 reset/close하지 않는다. runtime의 소유자가 마지막에 `close()`한다.

M5 계획기에 이미 사용하는 provider가 있으면 `provider=...`로 전달할 수 있다. 파지 요청만 실행할 때는 OpenAI 키가 필요 없다. LLM을 사용하는 후속 요청에서는 기존 provider의 인증이 필요하다.

## 좌표와 입력

- 단위: 미터, 라디안. 쿼터니언: `xyzw`.
- `T_WB`: 현재 물체 body의 위치와 방향.
- `T_BC`: body 원점에서 실제 bbox 중심까지의 보정.
- `T_CG`: 객체별 레시피에 저장한 중심 기준 그리퍼 자세.
- 최종 목표: `T_WG = T_WB @ T_BC @ T_CG`.
- 경로는 현재 관절값에서 다시 계산한다. 이전 배치의 관절 경로를 재생하지 않는다.
- 현재 구현은 M5의 simulator geometry record에서 물체 로컬 bbox와 중심 보정을 읽는다. M1을 다시 실행하거나 물성을 추론하지 않는다.
- 회전된 world AABB 크기를 물체 로컬 크기로 취급하지 않는다. 위치·방향만 바뀐 동일 자산 재사용과, 자산 크기·형태가 달라진 경우를 구분한다.
- 접시는 고정된 부분 관측 bbox 보정값으로 rim 목표를 유지하고, 실제 전체 자산 크기를 별도로 검사한다.

이름은 정확히 일치해야 한다. `obj_plate_plate`와 같은 명시적 M1 별칭만 정규화한다. 부분 문자열로 다른 물체를 추측하지 않는다.

## 물체를 들고 있는 동안

- 2F/3F 물체에는 attachment나 weld를 만들지 않는다.
- `held_tool_id`와 `world.metadata.contact_friction_held_objects`에 **측정된** 상대 자세를 기록한다.
- 충돌 검사에서만 이 상대 자세로 물체가 그리퍼를 따라가는 모델을 사용한다. 실행 시 실제 물체 qpos를 따라 쓰지 않는다.
- 일반 M5 컨트롤러에서도 파지에 사용한 손가락 힘 피드백을 유지한다. 접촉 손실과 미끄러짐이 허용치를 넘으면 중단한다.
- 물체를 놓으면 손가락 제어 유지와 held 상태를 해제한다. 물체를 든 채 EE 교체를 시도하면 거부한다.
- bread/lid는 물체를 들고 있는 동안에만 팔 제어 강성을 높이고 release 때 기존 kp/kd를 복원한다. 현재 관절값에 가까운 IK 동치각을 각 요청마다 다시 선택한다.
- M4의 암묵적 `transport`는 물체의 현재 자세를 목적지로 재사용하지 않는다. 현재 파지 offset을 유지하며 목적지 bbox 위에 EEF 목표 anchor를 만들고, Gemini에 목적지와 자세 유지 조건을 전달한다. 명시적인 M4 목표 자세는 덮어쓰지 않는다.
- 실행 보고서의 EE 자세도 IK와 동일한 `grip_site`를 사용한다.

**lid의 vac 예외:** 컵 접촉 검사를 통과한 뒤 기존 M5의 `KINEMATIC` attachment를 사용한다. 자유 물체에 대한 흡착 물리 시뮬레이션 성공으로 해석하지 않는다. release 시 attachment를 해제하고 실제 낙하를 확인한다.

## 파일과 산출물

| 파일 | 역할 |
|---|---|
| `registry.py` | 객체·환경·EE 분기와 활성화 상태 |
| `objects/*.py` | 객체별 고정 파지 함수와 중심 기준 자세 |
| `context.py` | 기존 runtime을 객체 함수에 연결 |
| `motion.py` | 기존 M5 IK/RRT/Cartesian 검증을 사용하는 공통 이동 단계 |
| `retention.py` | 다음 M5 동작 중 손가락 제어, 접촉·미끄러짐 검사 |
| `live.py` | 요청별 실행과 실제 상태 전달 |
| `profiles.py` | 초기화 시 적용하는 손 모델·수치 설정 |
| `reference_adapter.py`, `contact_gate.py`, `robot_response.py` | 접시의 기존 성공 제어 방식 |
| `calibrations/` | 접시의 자세 및 로봇 응답 보정값 |

`live-execution-manifest.json`은 각 요청이 `SCRIPTED_GRASP`와 `M5_MOTION_PLAN` 중 어느 경로로 실행됐는지 기록한다. 스크립트 파지를 가짜 MotionPlan으로 저장하지 않는다. 파지는 `grasp/result.json`, `trace.json`, 단계별 `motion_plan.json`을 저장하고, 일반 동작은 기존 M5 plan/run/report/goal artifact를 저장한다.

객체별 회귀 실행:

```powershell
python -m tuj.m5_motion.examples.scripted_grasp_smoke apple --output <new-output-dir> --followup
```

`--followup`은 고정된 테스트 키프레임을 기존 M5 계획기에 넣어 2cm 이동과 release를 검사한다. LLM 품질이나 전체 task 성공률을 측정하는 테스트가 아니다. 검증된 배치 범위 밖의 위치·방향은 새 경로와 접촉 검증을 통과해야 한다.
