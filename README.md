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
