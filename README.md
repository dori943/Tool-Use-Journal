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

실제 범용 계획에서는 일반 Task keyframe 생성을 위해 OpenAI API key를 설정한다.

```powershell
$env:OPENAI_API_KEY = "<project-api-key>"
python scripts\run_m5_motion_planner.py `
  --task-planner C:\tasks\sorting\task_planner.json `
  --environment C2_1_ObjectSorting
```

### C1_1 물리 실행

C1_1의 plate rim-grasp와 분할 sweep은 전용 geometry/profile을 사용하므로 별도
runner로 실행한다. 같은 상위 폴더의 `dain-m3` M4 결과를 기본으로 탐색한다.

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
