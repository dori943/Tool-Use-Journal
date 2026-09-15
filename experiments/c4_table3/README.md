# C4 Table 3 평가 실행기

`run.py`는 [`condition_matrix.csv`](../table3_protocol/condition_matrix.csv)에서 `READY`인 셀만 실행 대상으로 잡는다. 현재 검증된 Static Real-5 R 결과는 별도 `run_locked_inference.py`와 평가 bundle에서 보존하며, `run.py`는 기존 25-sequence trajectory 경로용 legacy 실행기다. 수치 GT는 추론 입력으로 만들지 않으며 evaluator가 결과 디렉터리에 snapshot을 만들 때만 읽는다. GT numerics oracle 입력은 일반 prediction과 별도 로그로 격리한다.

2026-09-14의 Static Real-5 실행은 잠긴 50개 정지 입력으로 5회 inference와 평가를 완료했다. SiPhy Mass MnRE, shared + Geometric Mass MnRE, shared Ours Mass MnRE는 `0.9085 ± 0.0046`, Ours Static visual Friction MAE는 `0.1550 ± 0.0048`이다. 이 결과는 `output/c4_table3/20260914T160924Z_static_real5_evaluate/`에서 재현한다. D 패널과 Name-only/Affordance는 독립 GT 또는 adapter 부족으로 계속 차단된다. 기존 trajectory friction 경로는 이 static 결과에 섞지 않는다.
표 원본은 `output/c4_table3/20260914T160924Z_static_real5_evaluate/table3_filled.md`와 `.csv`이며, 값은 저장된 per-repeat macro 점수에서 독립 재집계된다.

실물 D 패널은 독립 조작 GT가 없어 계속 `BLOCKED`다. 시뮬레이터만으로 후속 작업할 때는 [D_SIM protocol](../table3_protocol/D_SIM_PROTOCOL_KO.md)과 `experiments/c4_table3/sim_d/`를 사용한다. D_SIM은 MuJoCo hidden state로 evaluator-only GT와 입력 manifest를 만들며, Real-5 또는 실물 D 점수와 섞지 않는다. 현재 준비 run에는 prediction이 없고 condition cells는 adapter 구현 전 `NEEDS_IMPLEMENTATION`이다.

## YCB mesh를 robosuite에 연결하는 경우

공식 robosuite 배포물에는 YCB 객체 mesh가 포함되어 있지 않다. EV-RealPhys의 5개 ID를 시뮬레이터에서 재현하려면 YCB 공식 모델을 `data/external/ycb/`에 별도로 내려받고, `ycb_robosuite_audit.py`로 파일 hash·단위·매핑 상태를 먼저 기록한다.

```powershell
.venv\Scripts\python.exe experiments\c4_table3\ycb_robosuite_audit.py `
  --robosuite external\robosuite `
  --ycb-root data\external\ycb `
  --output output\c4_table3\<UTC>_ycb_robosuite_audit
```

`--emit-mjcf`는 source unit을 독립적으로 검증한 뒤에만 `--unit-scale`과 함께 사용한다. wrapper에는 질량·마찰을 넣지 않는다. 모래 충전 real-world mass와 Table 5 friction은 시뮬레이터 입력으로 주입할 수 없고, calibration split에서 정한 별도 파라미터로만 관리한다. YCB 파일명이 같아도 EV-RealPhys 이미지와 동일한 실물 인스턴스라는 증거가 아니므로, 해당 결과는 `SIM-PROXY`로 별도 보고한다. Tuna Fish Can(007)은 항상 제외한다.

YCB-V/BOP의 공개 `models_info.json`과 mesh bounding box를 비교할 때는 `compare_ycb_scale.py`를 사용한다. 이 비교는 축 순서를 정렬한 geometry 점검이며, 자동 rescale이나 GT 주입을 수행하지 않는다.

정적 RGB-D 입력을 simulator 좌표계로 옮길 때는 `estimate_static_sim_pose.py`를 사용한다. 이 도구는 잠긴 QC bounding box 안의 depth centroid와 `scene_camera.json`의 intrinsics만 계산하고, 물체 회전·테이블 평면·metric scale은 `UNRESOLVED`/`NEEDS_CALIBRATION`으로 남긴다.

```powershell
.venv\Scripts\python.exe experiments\c4_table3\estimate_static_sim_pose.py `
  --static-manifest output\c4_table3\<prep>\selected_static_inputs.csv `
  --qc-jsonl output\c4_table3\<prep>\candidate_qc.jsonl `
  --output output\c4_table3\<UTC>_static_sim_pose_bridge
```

테이블 평면은 같은 입력과 QC bbox를 사용해 `fit_static_table_plane.py`로 보조 추정할 수 있다. 전역 5 depth-unit RANSAC 기준을 적용하며, 성공하지 못한 프레임은 `UNRESOLVED`로 남긴다. 이 결과만으로 world-frame pose나 접촉 GT를 만들지 않는다.

```powershell
.venv\Scripts\python.exe experiments\c4_table3\fit_static_table_plane.py `
  --static-manifest output\c4_table3\<prep>\selected_static_inputs.csv `
  --qc-jsonl output\c4_table3\<prep>\candidate_qc.jsonl `
  --output output\c4_table3\<UTC>_static_table_plane
```

매핑·pose·평면 결과를 합친 scene manifest는 `build_static_sim_manifest.py`로 만든다. 평면이 추정된 행은 `READY_FOR_CALIBRATION`, 나머지는 `BLOCKED_FOR_TRIAL`로 기록되며, 회전·단위가 확정되기 전에는 simulator trial을 실행하지 않는다.

```powershell
.venv\Scripts\python.exe experiments\c4_table3\build_static_sim_manifest.py `
  --mapping output\c4_table3\<mapping>\ycb_static_mapping.csv `
  --pose output\c4_table3\<pose>\static_sim_pose_bridge.csv `
  --plane output\c4_table3\<plane>\table_plane_estimates.csv `
  --output output\c4_table3\<UTC>_static_sim_scene_manifest
```

EV 정적 crop과 YCB shape를 직접 비교하려면 `render_ycb_ev_contact_sheet.py`를 사용한다. 렌더는 고정 preview pose이므로 이미지와 mesh의 instance/회전 일치를 자동 판정하지 않는다.

```powershell
Set-Location C:\Users\SAMSUNG\Downloads\EE\Tool-Use-Journal
.venv\Scripts\python.exe experiments\c4_table3\run.py
.venv\Scripts\python.exe experiments\c4_table3\run.py --resume output\c4_table3\20260914T053234Z
.venv\Scripts\python.exe -m unittest discover -s experiments\c4_table3 -p test_scoring.py -v
.venv\Scripts\python.exe experiments\c4_table3\independent_arithmetic.py
```

새 실행은 `output/c4_table3/<run_id>/`에 `config.json`, `manifest.json`, 원본 matrix/registry/QC 정책, `gt_snapshot/`, `raw_predictions.jsonl`, `oracle_inputs.jsonl`, `failures.json`, 객체별·반복별·전체 summary, `checkpoint.json`, `table3.csv`, `table3.md`, `run.log`를 새로 만든다. 재개 시 config/manifest 및 prediction 파일 해시와 고유 `(condition, metric, input_id, repeat_id)` 키를 검사하고 완결된 실행은 그대로 반환한다. model version·API에 실제 전달된 seed·prompt hash·raw response 등은 prediction마다 필수 필드이며 호출이 없는 현재 run에서는 null/빈 파일로 남는다.

`scoring.py`는 Real-5에서 **시퀀스 예측 중앙값→객체 GT 오차→5개 객체 동일 가중 macro**를 주 점수로 구현한다. 객체별 평균 예측은 `sensitivity_object_mean_prediction`이라는 별도 이름으로만 산출한다. 연속량의 조건별 예측 누락은 오차를 대입하거나 complete-case 분모를 줄이지 않고 `INCOMPLETE`와 coverage로 기록한다. accuracy는 고정 단위의 누락 예측을 오답으로 센다. Suction PF는 같은 객체의 두 pose가 모두 맞아야 1이고, DA는 독립 허용 EE 집합의 복수 정답을 인정한다. Clearance의 5 mm 분모 바닥값과 Crit의 vac 0.50±0.05 kg 범위는 기존 규약을 그대로 사용한다. run 간 표본 std(ddof=1), 객체 간 표본 std(ddof=1), 시퀀스 예측 std(ddof=1)를 별도 필드로 둔다.

## YCB mesh SIM_PROXY 실행

uniform-scaled YCB mesh를 사용하는 simulator-only proxy는 `sim_d/generate_ycb_proxy.py`로 준비한다. 이 생성기는 Real-5 GT를 읽지 않고 고정된 proxy density/occupancy 정책으로 evaluator-only `sim_gt.yaml`을 만든다. 기존 D_SIM adapter/runner를 그대로 재사용할 수 있으며, 결과는 `SIM_PROXY`로만 보고한다.

```powershell
.venv\Scripts\python.exe experiments\c4_table3\sim_d\generate_ycb_proxy.py `
  --ycb-root output\c4_table3\<uniform_scaled_run> `
  --output output\c4_table3\<proxy_prep_run>
.venv\Scripts\python.exe experiments\c4_table3\sim_d\run_inference.py `
  --prep output\c4_table3\<proxy_prep_run> `
  --output output\c4_table3\<proxy_inference_run> --repeats 3
.venv\Scripts\python.exe experiments\c4_table3\sim_d\score_repeats.py `
  --gt output\c4_table3\<proxy_prep_run>\sim_gt.yaml `
  --predictions output\c4_table3\<proxy_inference_run>\parsed_predictions.jsonl `
  --output output\c4_table3\<proxy_eval_run>\per_repeat_summary.json
```

Suction Acc/PF를 채점하려면 adapter가 반환한 `suction_pose_predictions.pose_A/pose_B`를
두 pose unit으로 확장한다. `--conditions ours_full`과
`--shared-predictions <SiPhy parsed_predictions.jsonl>`를 사용하면 Ours의 Mass_Acc가
동일 repeat의 SiPhy 결과를 provenance와 함께 공유하며 추가 SiPhy 호출을 만들지 않는다.
SIM_PROXY에서 추가 구현된 조건은 Name-only/Affordance의 suction·feasibility·DA,
SiPhy의 feasibility·DA, Geometric grounding의 suction·PF·feasibility·DA이다.
이 확장은 D_SIM proxy 조건에만 적용되며, 공식 Real-5 조건의 `—` 셀을 변경하지 않는다.
기존 raw/parsed JSONL을 `--reuse-predictions`로 넘기면 manifest·model·parse schema가
맞는 성공 행만 재사용하고, 새로 필요한 metric만 호출한다. 오래된 sample-level suction
행은 pose schema 불일치로 자동 재사용하지 않는다.

향후 입력/GT가 준비되더라도 **matrix만 READY로 수정해 실행할 수는 없다**. 정확한 조건별 어댑터와 공통 M3/M4 downstream 연결, `SiPhyBackend`의 전체 raw VLM response 보존, frame-15 mass와 활주 마찰의 분리 실행, oracle 숫자 채널 격리를 구현·검증해야 한다. 기존 `experiments/c4_real5/run_inference.py`는 마찰 추적 실패 시 질량도 건너뛰고 raw response를 기록하지 않으므로 그대로 Table 3 READY 실행기로 사용할 수 없다. 어댑터가 아직 없는데 READY가 나타나면 `run.py`는 `READY_ADAPTER_NOT_IMPLEMENTED`로 중단한다. 이는 불완전한 점수를 내지 않기 위한 명시적 보호다.

검증: focused test 14개 통과. config를 바꾼 재개는 `RESUME_CONFIG_OR_MANIFEST_HASH_MISMATCH`로 거부됐고, 동일 config의 재개는 `UNCHANGED`·예측 0개를 반환했다. [`independent_arithmetic.py`](independent_arithmetic.py)는 보존된 과거 24개 prediction의 숫자만 독립 재집계해 Mass MnRE `0.9227010043238545`, Friction MAE `0.0241`을 확인한다. 과거 `result.json`은 다른 run 디렉터리에 있고 prediction hash가 없어, 이 일치는 실행 계보나 어느 Table 3 조건의 결과도 증명하지 않는다. 새 run은 이를 재사용하거나 재채점하지 않았다.
