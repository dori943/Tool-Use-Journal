# D_SIM 시뮬레이터 패널

실물 독립 조작 데이터가 없는 상태에서 Table III의 D 열을 임의로 채우지 않기 위해, MuJoCo 전용 패널을 `D_SIM`으로 분리했다. 기존 `panel_d_manifest.yaml`은 실물 D 패널의 계약과 `BLOCKED` 상태를 그대로 유지한다. `D_SIM` 점수는 Real-5 점수와 합치거나 실물 조작 성능으로 표현하지 않는다.

## 현재 준비 상태

- MuJoCo 컴파일 fixture 5개를 생성했다.
- 각 fixture에서 subtree mass, opening geometry, support state를 읽어 evaluator-only GT를 만들었다.
- 각 sample에 두 suction pose와 pose별 5회 scripted trial, EE별 5회 feasibility trial, clearance margin, 독립 허용 EE 집합, vac 0.50±0.05 kg Crit subset을 기록했다.
- 입력 manifest에는 mass, trial 결과, 허용 EE, signed margin을 넣지 않았다.
- VLM/API prediction은 생성하지 않았다. 따라서 D_SIM condition cells는 adapter 구현 전까지 `NEEDS_IMPLEMENTATION`이다.

## 재현 명령

```powershell
Set-Location C:\Users\SAMSUNG\Downloads\EE\Tool-Use-Journal
$run = "output/c4_table3/$(Get-Date -AsUTC -Format yyyyMMddTHHmmssZ)_sim_d"
.venv\Scripts\python.exe experiments\c4_table3\sim_d\generate_sim_gt.py --output $run
.venv\Scripts\python.exe experiments\c4_table3\sim_d\validate_sim_gt.py $run
.venv\Scripts\python.exe -m pytest experiments\c4_table3\sim_d\test_sim_d.py -q --basetemp .pytest-sim-d
```

## 해석 제한

`sim_gt.yaml`의 trial은 고정된 simulator hidden state와 scripted contract에 따른 simulation label이다. 실제 vacuum pressure/누설 계측, 실물 저울, 실물 clearance 측정이 아니다. `configs/c4_sim_d.yaml`의 prediction disabled 상태를 해제하려면 조건별 adapter, simulator 관측 입력과 prediction schema, raw response/checkpoint, 그리고 `score_sim_d.py`에 전달할 독립 prediction 파일을 먼저 구현해야 한다.

조건별 현재 상태는 `condition_matrix_sim_d.csv`에 고정했다. 기존 `condition_matrix.csv`의 R/실물 D 상태를 덮어쓰지 않는다.
