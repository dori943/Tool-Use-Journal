# Tool-Use-Journal — M4 Task Planner

이 worktree는 M4 Task Planner 실행 브랜치다. Python 패키지 경로는 기존 이름을
유지해 `tuj.m3_taskplanner`이며, 저장소 루트에서 아래 순서로 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
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
