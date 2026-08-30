# Tool-Use-Journal — M5 Motion Planner

이 worktree는 최신 M5 Motion Planner 실행 브랜치다. Python 패키지 경로는 기존
이름을 유지해 `tuj.m4_motion`이다.

## 바로 확인하기

저장소 루트에서 의존성을 설치하고 테스트한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest src\tuj\m4_motion\tests -q
```

범용 M5 runner는 특정 Task 경로를 기본값으로 사용하지 않는다. M4 결과와 초기
`WorldSnapshot`을 명시하면 C1_1이 아닌 Task도 OpenAI API 호출 없이 입력 계약을
검증할 수 있다.

```powershell
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --initial-world C:\tasks\sorting\initial_world.json `
  --validate-input-only
```

지원되는 Tool-Use-Journal 환경에서 초기 world를 직접 캡처할 수도 있다.

```powershell
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --environment C2_1_ObjectSorting `
  --initial-ee none `
  --validate-input-only
```

환경 이름은 M5 코드의 별도 고정 목록에 추가하지 않는다. 새 robosuite 환경 클래스를
저장소의 `environments/__init__.py`에서 import해 등록하면 runner가 자동 발견한다.
실행 시 해당 환경이 UR5e, `2F`/`3F`/`vac` rack, `obj_body_id`, `ee_rack_info` 계약을
만족하는지는 fail-closed 방식으로 검증한다.

실제 범용 계획에서는 일반 Task keyframe 생성을 위해 OpenAI API key를 설정한다.

```powershell
$env:OPENAI_API_KEY = "<project-api-key>"
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --environment C2_1_ObjectSorting
```

계획 직후 MuJoCo controller simulation을 화면으로 확인하려면 `--simulate`를
추가한다. 라이브 viewer는 기본적으로 실제 시간 속도로 재생된다.

```powershell
$env:OPENAI_API_KEY = "<project-api-key>"
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --environment C2_1_ObjectSorting `
  --initial-ee none `
  --simulate controller
```

MP4가 필요하면 `--video`만 추가해도 controller simulation이 자동으로 실행된다.
영상 실행은 offscreen이므로 viewer 창을 열지 않는다.

```powershell
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --environment C2_1_ObjectSorting `
  --initial-ee none `
  --video artifacts\sorting\run.mp4
```

빠른 궤적 검증은 `--simulate kinematic --headless`를 사용한다. 실행 결과는 계획
artifact와 별도로 `<output-dir>\simulation\simulation-manifest.json` 및
`m5_summary.json`에 기록된다. 외부 `--initial-world` 파일로 simulation할 때는 해당
snapshot이 지정한 environment와 seed의 reset 상태여야 한다. 현재 상태를 확실히
맞추려면 `--environment`로 같은 실행 안에서 world를 캡처하는 방법을 권장한다.

M4의 `PICK_TOOL`/`RETURN_TOOL`은 M5에서 실제 Tool object의 attach/detach event로
실행되고 `held_tool_id`로 다음 subgoal에 전달된다. M4에 grasp pose가 없어도 정상이다.
M5가 현재 world와 선택된 Tool을 기준으로 접촉 keyframe 후보를 만든다. 대신 모든
`target_ids`는 `WorldSnapshot.objects` 또는 `rack`에 존재하는 구체 ID여야 하며
`?tool` 같은 미해결 변수는 입력 검증에서 거부한다.

### C1_1 물리 실행

C1_1의 plate rim-grasp, 마찰 검증, rollback이 포함된 분할 sweep은 전용
geometry/profile을 사용하므로 별도 runner로 실행한다. 범용 runner도 C1_1 입력
계약과 region containment를 처리하지만, 검증된 C1_1 물리 정책이 필요하면 전용
runner를 사용한다. 같은 상위 폴더의 `dain-m3` M4 결과를 기본으로 탐색한다.

```powershell
python scripts\run_m5_c1_1_motion_planner.py --validate-input-only
```

```powershell
python scripts\run_m5_c1_1_motion_planner.py `
  --pick-keyframes C:\path\to\pick_keyframes_raw.json `
  --stop-after-pick
```

두 runner 모두 EE와 Tool을 M4 `candidate_assignments`에서 읽는다. runner가 `2F`나
`heavy_plate`를 선택값으로 주입하지 않는다. M4가 빈 장착 상태에서 만든 첫
`ATTACH_EE`는 M5에서 `bare-flange → selected EE` 초기 장착 모션으로 처리한다.

상세 구조, collision context, recovery와 전체 C1_1 옵션은
[`src/tuj/m4_motion/README.md`](src/tuj/m4_motion/README.md)를 참고한다.
