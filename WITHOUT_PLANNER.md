# without-planner 실험 모드

`scripts/run.py`의 기본 `--planner-mode full` 경로는 그대로 둔다. 별도
`--planner-mode without-planner` 경로는 M4 EE Swap-Aware Planner의 공동 탐색을
호출하지 않는 **per-subgoal independent EE/tool assignment without global planning**
baseline이다. `without_grounding`과 동시에 사용할 수 없다.

## 선택 규칙과 실행 경계

M0/M1과 M2의 일반 산출물을 사용한다. 이 저장소의 통합 파이프라인에서 M3는 독립
실행 단계가 아니라 M1의 물리·기하 정보와 G_k 후보 생성, feasibility/suitability
판정에 녹아 있다. `without-planner`는 그 입력 계약과 M2 subgoal 순서를 유지한다.
각 subgoal의 제안 후보(없으면 catalog 후보)를 공통 정적 feasibility 및 suitability
규칙으로 평가한 다음, 후보 자체의 순위가 가장 높은 하나를 **독립적으로** 고른다.
현재 장착 EE, 앞뒤 subgoal, 전환 비용을 후보 순위에 반영하지 않는다. 선택이
실행 불가능하거나 M2 순서가 hard constraint를 위반하면 실패로 기록하며 다른 EE
탐색이나 순서 변경으로 복구하지 않는다.

공통 transition 및 직렬화 모델을 사용해 도구 pick/place와 EE 교체 primitive를
M5가 이해하는 `selected_plan` 형태로 만든다. 이는 결정된 순서를 실행 계약으로
변환하는 adapter이며, M4 `plan()`/`run_search()`를 호출하지 않는다. M5 입력 검증,
motion validation 및 실행 코드는 기본 모드와 동일하다. 스크립트 grasp 지원 EE
제약도 기본 모드와 같은 위치에서 적용한다.

## 실행

```powershell
python scripts/run.py c1_1 --seed 0 --planner-mode without-planner --provider openai --model gpt-4o --stop-after m4
```

M1/M2 산출물이 **같은 task, scene, seed**로 준비된 경우, 빠른 planner 경계 확인:

```powershell
python scripts/run.py c1_1 --seed 0 --planner-mode without-planner --provider openai --model gpt-4o --start-from m4 --stop-after m4 --output-dir <empty-run-directory>
```

`--start-from m4`일 때 그 폴더에 `m1.json`, `m2.json`이 필요하다. M5 입력 검증
또는 motion 실행에는 앞서 생성한 **동일한 출력 폴더**를 사용한다.

```powershell
python scripts/run.py c1_1 --seed 0 --planner-mode without-planner --provider openai --model gpt-4o --start-from m5 --m5-validate-only --output-dir <same-run-directory>
python scripts/run.py c1_1 --seed 0 --planner-mode without-planner --provider openai --model gpt-4o --start-from m5 --m5-simulate controller --output-dir <same-run-directory>
```

실제 keyframe 생성에는 선택한 provider의 API 키를 환경변수로 제공한다. 키는
명령문이나 설정 파일에 저장하지 않는다. `--m5-args`로 M5의 세부 옵션을 전달할
수 있으며 대조군에도 동일 옵션을 적용한다. `--stop-after-subgoal` 및
`--stop-after-pick` 실행은 `evaluation_scope=partial`, `success=null`로 기록한다.

## 산출물과 비교

기본 출력은 `output/without-planner/<task>/seed_<seed>/`이다. `m4_request.json`은
동일한 계획 입력, `m4.json`은 M5와 호환되는 계획 및 거절 사유를 담는다. 이
파일에서 `method=without-planner`, `m4_invoked=false`, `search_stats.states_expanded=0`
을 확인할 수 있다. 로그도 `M4 ... SKIPPED`, `M4 search calls=0`을 표시한다.
`ablation_manifest.json`은 task/seed/단계/계획 해시를 기록하여 다른 실험의
M5 산출물을 잘못 재사용하지 않게 한다.

`without_planner_result.json`은 method, task, seed, plan_status, m5_status,
success와 N_EE(EE 교체), N_tool(tool pick), motion_cost, execution_cost,
planning_time_ms(공동 탐색 생략으로 0), assignment_time_ms(독립 선택 소요)를 담는다.
N_EE는 공통 M4 비용 정의상 처음 bare 상태에서 EE를 장착하는 횟수는 제외한다.
별도 결과 집계기가 없는 현재 저장소에서는 이 method 필드로 full 결과와 구분한다.
M5 입력 검증만 수행한 결과와 M4까지만 수행한 결과는 task success가 아니다.

한 task의 성공/실패만으로 planner ablation의 성공률 차이나 인과효과를 추정할 수
없다. 같은 입력 장면, seed, M1/M2 산출물, M5 설정으로 paired 평가를 반복하고,
API/계약 실패와 물리·기하 실패를 별도로 분류해야 한다. N_EE/N_tool은 계획상의
수치이며 실제 motion 실행 완료 횟수와 동일하다고 가정하지 않는다.
