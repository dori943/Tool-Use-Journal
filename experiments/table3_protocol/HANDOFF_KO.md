# SiPhy Table III 인수인계 메모

## 1. 현재 결론

새 API 호출 없이 기존 Static Real-5 산출물을 검증·재사용해 Table III의 4개 셀을 READY로 반영했다.

| 조건 | 지표 | 결과 |
|---|---|---:|
| SiPhy (as adopted) | Mass MnRE | **0.9085 ± 0.0046** |
| + Geometric grounding | Mass MnRE | **0.9085 ± 0.0046** |
| Ours (full) | Mass MnRE | **0.9085 ± 0.0046** |
| Ours (full) | Friction MAE-Real5 | **0.1550 ± 0.0048** |

질량은 객체별 정지 입력 중앙값 → 5개 객체 macro 평균 → 5회 반복 평균±표본 표준편차(ddof=1)이다. Geometric/Ours 질량은 SiPhy 결과를 공유하며 독립 model call이 아니다.

현재 condition matrix 상태는 `READY=4`, `NEEDS_DATA=38`, `NOT_APPLICABLE=12`이다.

## 2. 기준 평가와 입력

- Benchmark: Kandukuri et al. EV-RealPhys `test_sliding`의 5개 객체
- 입력: 객체당 10장, 총 50장 Static RGB/RGB-D 입력
- 질량: object crop / RGB-D 기반 SiPhy 입력
- 마찰: 객체와 지지 테이블이 함께 보이는 정지 context image를 이용한 visual-prior 추정
- 기존 자유 활주 trajectory 마찰은 별도 auxiliary 결과이며 Static 점수에 섞지 않는다.
- 000015: 배경 잔여는 WARNING으로 유지
- 000017: 두 번째 입력을 frame 35로 교체
- 최종 제외: 0개

잠긴 manifest SHA-256:

`89c476f3afd9f00223fff928e6d86c124905ad53d928d70b5b95e8e9f8726dbc`

## 3. 핵심 결과 파일

- [Table III 결과](../../output/c4_table3/20260914T160924Z_static_real5_evaluate/table3_filled.md)
- [수치 집계](../../output/c4_table3/20260914T160924Z_static_real5_evaluate/aggregated_metrics.csv)
- [반복별 집계](../../output/c4_table3/20260914T160924Z_static_real5_evaluate/per_repeat_metrics.csv)
- [평가 실행 로그](../../output/c4_table3/20260914T160924Z_static_real5_evaluate/run.log)
- [Inference 검증 결과](../../output/c4_table3/20260914T064753Z_static_real5_inference/inference_validation.json)
- [입력 manifest](../../output/c4_table3/20260914T062054Z_static_real5_prep/selected_static_inputs.csv)
- [Inference 설정](../../output/c4_table3/20260914T062054Z_static_real5_prep/inference_config.yaml)
- [Adapter 감사](../../output/c4_table3/20260914T062054Z_static_real5_prep/adapter_audit.md)
- [Evaluator-only GT](../../output/c4_table3/20260914T062054Z_static_real5_prep/evaluator_only_gt.yaml)

## 4. 프로토콜·상태 파일

- [Evaluation protocol](evaluation_protocol.md)
- [Metric registry](metric_registry.yaml)
- [Condition matrix](condition_matrix.csv)
- [표본 수와 차단 상태](data_preparation/metric_sample_counts.json)
- [Data gaps](data_preparation/data_gaps.md)
- [C4 실행기 README](../c4_table3/README.md)

## 5. 아직 채우면 안 되는 셀

- Name-only / + Affordance labels Mass MnRE: 실제 adapter와 prediction 없음
- Mass Acc: `mass_bins`, EE payload 사양과 적용 GT 없음
- Suction Acc/PF: pose별 독립 흡착 GT와 prediction 없음
- Clearance RelErr: 독립 좌표계·간격 GT와 prediction 없음
- Feasibility Acc: EE별 실제 실행 성공 GT 없음
- DA: 독립 task constraint와 허용 EE 집합 GT 없음
- Crit: 사전 고정 경계 사례와 독립 GT 없음
- GT numerics 연속량: 추정 성능이 아니므로 `-`

기존 trajectory 결과 `Mass MnRE=0.922701`, `Friction MAE=0.0241`은 역사적 잠정 결과다. 현재 Static Table III 셀에 넣거나 Static 결과와 평균내면 안 된다. trajectory 경로는 target semantics mismatch 상태다.

## 6. 이어서 작업할 때 지킬 것

1. `experiments/c4_table3/run.py`는 legacy trajectory 실행기다. Static Real-5 재실행에는 사용하지 않는다.
2. 이미 생성된 raw prediction을 우선 재사용한다. 새 inference 전에 정확한 조건·지표·GT가 모두 준비됐는지 확인한다.
3. GT를 inference 코드나 prompt에 넣지 않는다. GT는 evaluator에서만 읽는다.
4. Static manifest, crop, prompt를 prediction을 보고 수정하지 않는다.
5. Geometric/Ours 질량을 독립 실험처럼 보고하지 않는다. `shared_prediction_source=SiPhy`를 유지한다.
6. 새 결과가 필요하면 기존 output을 덮어쓰지 말고 새 UTC run directory를 사용한다.

## 7. 재현·검증 명령

저비용 검증:

```powershell
Set-Location C:\Users\SAMSUNG\Downloads\EE\Tool-Use-Journal
.venv\Scripts\python.exe -m pytest experiments/c4_table3/test_scoring.py -q
.venv\Scripts\python.exe -m pytest experiments/c4_static_real5 -q
```

기대 결과는 각각 `14 passed`, `6 passed`다. 현재 결과를 재집계할 때는 `aggregated_metrics.csv`와 `per_repeat_metrics.csv`를 사용한다. 새 모델 호출은 필요하지 않다.

## 8. 산출물 작성 위치

논문 반영본:

[EE-GEAR_ICRA_편집본.md](../../../EE-GEAR_ICRA_편집본.md)

현재 Table III의 부분 결과는 위 문서에 반영되어 있다. 다른 팀원이 수정할 때는 평가 산출물과 수치가 먼저 존재하는지 확인한 뒤 manuscript를 수정한다.
