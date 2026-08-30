<<<<<<< HEAD
# Tool-Use-Journal — M4 Task Planner

이 worktree는 M4 Task Planner 실행 브랜치다. Python 패키지 경로는 기존 이름을
유지해 `tuj.m3_taskplanner`이며, 저장소 루트에서 아래 순서로 실행한다.
=======
# Tool-Use-Journal — M5 Motion Planner

이 worktree는 최신 M5 Motion Planner 실행 브랜치다. Python 패키지 경로는 기존
이름을 유지해 `tuj.m4_motion`이다.

## 바로 확인하기

저장소 루트에서 의존성을 설치하고 테스트한다.
>>>>>>> 9d4b9d3 (feat(motion-planner): add closed-loop planning and initial EE attach)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
<<<<<<< HEAD
python scripts\run_m4_task_planner.py
```

기본 입력은 `output/c1_1/gk_bundle.json`, `output/c1_1/m1.json`,
`configs/robot_spec.json`이고 결과는 `output/c1_1/task_planner.json`에 생성된다.
다른 `--gk` 경로를 주고 `--output`을 생략하면 해당 GK 파일과 같은 폴더에
`task_planner.json`을 생성한다.
Tool은 GK의 `roles.selected_tool`에서 읽으며 runner나 Planner가 `heavy_plate`로
고정하지 않는다. `current_ee: null`이면 장착된 EE 없이 시작해 전체 계획에서 이후
교체가 가장 적어지는 첫 EE를 검색하고, 첫 장착 자체는 교체 횟수에 포함하지 않는다.
M1의 acquire/place detail은 각각 한 번의 `PICK_TOOL`/`RETURN_TOOL` subgoal이 되며,
동일 동작을 transition으로 다시 만들지 않는다.

검증:

```powershell
python -m pytest src\tuj\m3_taskplanner\tests -q
```

입력 형식, 비용 기준, 다른 경로를 넘기는 방법은
[`src/tuj/m3_taskplanner/README.md`](src/tuj/m3_taskplanner/README.md)를 참고한다.
=======
python -m pytest src\tuj\m4_motion\tests -q
```

M4 결과와 M5의 입력 계약만 확인할 때는 OpenAI API key나 MuJoCo 실행이 필요 없다.
기본값은 같은 상위 폴더의 `dain-m3/output/c1_1/task_planner.json`도 자동 탐색한다.

```powershell
python scripts\run_m5_motion_planner.py --validate-input-only
```

다른 위치의 M4 결과를 사용할 때는 명시한다.

```powershell
python scripts\run_m5_motion_planner.py `
  --task-planner C:\path\to\task_planner.json `
  --validate-input-only
```

실제 계획·시뮬레이션은 PICK keyframe을 OpenAI로 생성하거나 기존 artifact로 넘긴다.

```powershell
$env:OPENAI_API_KEY = "<project-api-key>"
python scripts\run_m5_motion_planner.py --stop-after-pick
```

```powershell
python scripts\run_m5_motion_planner.py `
  --pick-keyframes C:\path\to\pick_keyframes_raw.json `
  --stop-after-pick
```

EE와 Tool은 M4 `candidate_assignments`에서 읽는다. runner가 `2F`나
`heavy_plate`를 선택값으로 주입하지 않는다. M4가 빈 장착 상태에서 만든 첫
`ATTACH_EE`는 M5에서 `bare-flange → selected EE` 초기 장착 모션으로 처리한다.

상세 구조, collision context, recovery와 전체 C1_1 옵션은
[`src/tuj/m4_motion/README.md`](src/tuj/m4_motion/README.md)를 참고한다.
>>>>>>> 9d4b9d3 (feat(motion-planner): add closed-loop planning and initial EE attach)
