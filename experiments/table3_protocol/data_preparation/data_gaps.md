# 지표별 표본과 우선순위별 부족 데이터

기계 판독 개수는 [`metric_sample_counts.json`](metric_sample_counts.json), 행별 선택·차단 이유는 [`real5_input_manifest.json`](real5_input_manifest.json)에 있다. `READY`는 물리 GT나 원시 입력만 있는 상태를 뜻하지 않으며, 조건 실행·독립 QC·공통 scorer까지 준비됐을 때만 준다. Static Real-5 R의 Mass MnRE와 Ours visual friction은 현재 점수가 저장되어 있고, D 패널과 legacy trajectory friction은 별도 차단/auxiliary 상태다.

| 지표 | 패널 | 원본/계획 분모 | 입력 후보 | 평가 가능 | 상태와 선행 조건 |
| --- | --- | --- | --- | --- | --- |
| Mass MnRE | R | 5 객체, 25 시퀀스에서 정지 이미지 50장 | 질량 crop 50 | 25 sequence-equivalent | READY: Static Real-5 평가 run `20260914T160924Z_static_real5_evaluate` |
| Friction MAE-Real5 | R | 5 객체, 정지 context 50장 (visual variant) | context 50 | 25 sequence-equivalent | READY: visual prior 평가; 자유 활주 trajectory 경로는 TARGET_SEMANTICS_MISMATCH auxiliary |
| Mass Acc | D | 객체별 고정 sample×3 EE | 0 | 0 | BLOCKED: 독립 저울 질량·실장 payload 적용 증명·D 영상 |
| Suction Acc | D | 객체마다 2 pose | 0 | 0 | BLOCKED: 동일 vac/압력/접촉·pose별 5 trial, 가능/불가능 쌍 |
| Suction PF | D | 객체마다 1 pose 쌍 | 0 | 0 | BLOCKED: 두 pose GT 모두 완비 필요 |
| Clearance RelErr | D | 객체별 고정 물체–개구 관계 | 0 | 0 | BLOCKED: mm signed margin, 좌표계·독립 교정 측정·불확도 |
| Feasibility Acc | D | sample×3 EE | 0 | 0 | BLOCKED: EE별 5회 독립 task 실행, proxy/실측 분리 |
| DA | D | 객체·task별 선택 이벤트 | 0 | 0 | BLOCKED: 독립 허용 EE 집합, 제약·목적·동률 규칙 |
| Crit | D | vac 임계 근처 사전 고정 sample | 0 | 0 | BLOCKED: 독립 질량이 [0.45,0.55] kg인 실물과 양쪽 경계 사례 |

현재 R 객체별 원시 수는 모두 5다. 24개의 마찰 창 통과는 목표가 같다는 뜻이 아니고, 질량의 25개 분할 성공도 흡착/조작 GT로 변환되지 않는다. GT numerics에서 정답 수치를 주입하면 연속량 **추정 성능 셀은 구조적 `-`**이며, D의 downstream 셀은 독립 검증 때까지 `TBD/BLOCKED`다. Real-5 Table 5 GT를 다른 객체 사진의 Mass Acc·Crit·Feasibility GT에 옮기지 않는다.

1. **P0, R 의미/입력:** Static Real-5의 50개 입력은 `output/c4_table3/20260914T062054Z_static_real5_prep/`에서 crop/context와 warning 상태를 잠갔고, `000015`는 WARNING 유지, `000017` 두 번째 frame은 35로 교체했다. 이 정지 visual-prior 결과는 평가 run에 저장되어 있다. 기존 자유 활주에는 독립 테이블 level/중력 기준, push-tool release와 접촉/충돌 annotation, pixel RGB-depth 재투영·시간 검증, 회전 및 mask/tracking 변형을 고려한 물리 추정이 여전히 필요하며, 그 legacy trajectory MAE는 static 점수에 섞지 않는다.
2. **P0, D 독립 수집:** [`panel_d_manifest.yaml`](panel_d_manifest.yaml) schema와 [`measurement_protocol.md`](measurement_protocol.md)에 따라 동일 물리 인스턴스의 2F·3F·vac 실장 상태, 저울·vac 압력, 두 pose×각 5회의 seal/lift/hold, 각 EE×5회의 terminal 성공, 교정된 개구·물체 mm 측정, task 제약/허용 선택 집합을 채운다. 추론과 동일 규칙의 proxy label을 실행 성공 GT로 바꾸지 않는다. vac payload 0.50 kg±0.05 kg 사례를 예측 오류가 아닌 저울값으로 미리 모집한다. 최소 수는 D 객체 수를 `N_D`, 객체별 sample 1개라 하면 pose `2N_D`, suction trial `10N_D`, EE 성공 trial `15N_D`, EE 질량 판정 `3N_D`, 선택 이벤트 `N_D`다. `N_D`는 실제 확보 후 확정한다.
3. **P0, 실행 코드:** 이전 규약의 여섯 조건 게이트와 공통 downstream, metric scorer, SHA 연결·결측 정책은 별도 구현이 필요하다. Name-only prior/Affordance label은 train-only 출처를 잠그고 test GT 유출을 검사한다. 별도 gripper-contact μ GT가 없으면 R의 table μ를 grip slip 개선 주장의 입력에 쓰지 않는다. Real-5의 mass/friction GT를 D에 이식하지 않는다.
4. **P1, calibration:** prompt, 영상 선택이나 임계값을 GT와 대조해 바꿀 필요가 생기면 공개 `val_sliding` 또는 물리적으로 분리된 D calibration split에서만 조정한다. 현재 test 결과를 보고 후보 시퀀스를 바꾸거나 0/0을 점수화하지 않는다. 모든 변경은 새 정책 버전·해시, 사전 고정 입력 manifest와 실행 ID를 요구한다.

현재 상태를 읽기 전용으로 재검증하는 명령:

```powershell
Set-Location C:\Users\SAMSUNG\Downloads\EE\Tool-Use-Journal
.venv\Scripts\python.exe experiments\table3_protocol\data_preparation\validate_prepared.py
```

수정한 segmentation/관측의 **새** QA run을 남기는 명령(기존 출력은 덮어쓰지 않음):

```powershell
.venv\Scripts\python.exe experiments\table3_protocol\data_preparation\audit_inputs.py --fixed-mass-frame 15
```

독립 QC·마찰 target 의미·D 수집이 끝난 뒤 새 버전 정책과 output 디렉터리를 지정해 manifest를 다시 잠근다. 기존 legacy trajectory manifest와 D 패널은 여전히 score-ready가 아니지만, Static Real-5 R의 잠긴 manifest는 별도 evaluator에서 Mass MnRE와 visual Friction MAE를 채점할 수 있다. 따라서 아래 명령은 legacy 준비 상태를 검사하는 용도로만 사용하며, Static 평가 결과와 혼동하지 않는다:

```powershell
.venv\Scripts\python.exe experiments\table3_protocol\data_preparation\validate_prepared.py --require-score-ready
```

`experiments/c4_real5/run_inference.py`는 원본의 GT-pose 프레임 선택과 마찰 실패 시 질량 스킵 경로를 사용한다. 새 분리 manifest용 조건 adapter/scorer를 구현·검증하기 전에는 이 명령으로 Table 3을 실행하지 않는다.
