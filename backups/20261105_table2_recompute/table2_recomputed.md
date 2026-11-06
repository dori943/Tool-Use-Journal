# Table II 재계산 — 추정 feasible vs 실제 파지 (seed 0, 파지·상승·유지)

기호: ! 구현/실행 오류(ERROR·TIMEOUT, 물리 실패 아님 → 재실행) · ✓ 추정 feasible & PASS · ✗ 추정 feasible & FAIL · ? 추정 feasible 미실행 · + 추정 불가인데 PASS · ⊗ vac: 시뮬 부착은 됐으나 GT vac 규칙 불통과(성공 아님) · ∅ 미실행(레시피 없음) · · 해당 없음

## 요약 (값 = ✓ / 추정 feasible 수, 괄호는 역사적 Table II)

| task | 2F | 3F | vac | 추정 없음 물체 | 추정 소스 |
|---|---|---|---|---|---|
| c1_1 | 0/4 (4/5) ≠ | 1/4 (4/5) ≠ | 1/2 (1/5) ≠ | - | c1_1/m1.json |
| c1_2 | 1/4 (3/4) ≠ | 3/4 (3/4) | 0/1(?) (0/4) ≠ | - | c1_2/m1.json |
| c2_1 | 3/4 (4/5) ≠ | 3/4 (4/5) ≠ | 1/2 (1/5) ≠ | - | c2_1/m1.json |
| c2_2 | 0/2 (0/5) ≠ | 0/3 (2/5) ≠ | 2/2 (5/5) ≠ | turkey_1,cheese_1 | c2_2/m1.json |
| c3_1 | 4/4 (4/5) ≠ | 3/4 (4/5) ≠ | 1/2 (1/5) ≠ | - | c3_1/m1.json |
| c3_2 | 1/8 (8/12) ≠ | 10/10 (10/12) ≠ | 2/4 (4/12) ≠ | - | c3_2/m1.json |
| c4_1 | 2/5 (4/4) ≠ | 1/5 (3/4) ≠ | 4/4 (2/4) ≠ | - | c4_1/m1.json |
| c4_2 | 5/5 (4/6) ≠ | 4/5 (1/6) ≠ | 1/4 (1/6) ≠ | - | c4_2/m1.json |

## vac: 시뮬에서는 붙었지만 GT vac 규칙 불통과 (성공으로 세지 않음)

- c1_1: -
- c1_2: -
- c2_1: mug
- c2_2: -
- c3_1: mug
- c3_2: mug_a, mug_b, fork_b, bread_b
- c4_1: -
- c4_2: rolling_pin, cereal, milk, lid

## 전체 물체 기준 커버리지 (PASS / 전체 물체, 추정과 무관)

| task | 2F | 3F | vac | 추정 밖 PASS | ⚠ 미완료 (재실행 필요) |
|---|---|---|---|---|---|
| c1_1 | 0/5 | 1/5 | 2/5 | vac:scissors | - |
| c1_2 | 1/4 | 3/4 | 0/4* | - | vac 오류:bottle,spatula |
| c2_1 | 3/5 | 3/5 | 2/5 | vac:apple | - |
| c2_2 | 0/5 | 0/5 | 5/5 | vac:bread_a,turkey_1,cheese_1 | - |
| c3_1 | 4/5 | 3/5 | 2/5 | vac:apple | - |
| c3_2 | 1/12 | 10/12 | 4/12 | vac:spoon_a,spoon_b | - |
| c4_1 | 2/5 | 1/5 | 4/5 | - | - |
| c4_2 | 5/6 | 4/6 | 1/6 | - | - |

## c1_1

| object | 추정 feasible | 2F | 3F | vac |
|---|---|---|---|---|
| plate | vac | · | · | ✓ PASS |
| ladle | 2F,3F,vac | ✗ FAIL | ✗ FAIL | ✗ FAIL |
| fork | 2F,3F | ✗ FAIL | ✗ FAIL | · |
| scissors | 2F,3F | ✗ FAIL | ✗ FAIL | + PASS |
| bottle_distractor | 2F,3F | ✗ FAIL | ✓ PASS | · |

## c1_2

| object | 추정 feasible | 2F | 3F | vac |
|---|---|---|---|---|
| bottle | 2F,3F | ✗ FAIL | ✓ PASS | ! ERROR |
| spatula | 2F,3F,vac | ✗ FAIL | ✓ PASS | ! ERROR |
| spoon | 2F,3F | ✓ PASS | ✓ PASS | · |
| tongs | 2F,3F | ✗ FAIL | ✗ FAIL | · |

## c2_1

| object | 추정 feasible | 2F | 3F | vac |
|---|---|---|---|---|
| apple | 2F,3F | ✗ FAIL/free-hold | ✓ PASS/free-hold | + PASS/acq-assist |
| bread | 2F,3F,vac | ✓ PASS/free-hold | ✓ PASS/free-hold | ✗ FAIL/free-hold |
| mug | 2F,3F | ✓ PASS/free-hold | ✓ PASS/free-hold | ⊗ PASS/acq-assist [상면 RMS 2.748 vs 허용 1.5; 흡착 패치 RMS 2.034 vs 허용 1.5mm] |
| plate | vac | · | · | ✓ PASS/acq-assist |
| spoon | 2F,3F | ✓ PASS/free-hold | ✗ FAIL/free-hold | · |

## c2_2

| object | 추정 feasible | 2F | 3F | vac |
|---|---|---|---|---|
| bread_a | 2F,3F | ✗ FAIL/free-hold | ✗ FAIL/free-hold | + PASS/acq-assist |
| bread_b | 3F,vac | · | ✗ FAIL/free-hold | ✓ PASS/acq-assist |
| turkey_1 | (추정 없음) | · | · | + PASS/acq-assist |
| cheese_1 | (추정 없음) | · | · | + PASS/acq-assist |
| tomato_slice | 2F,3F,vac | ✗ FAIL/free-hold | ✗ FAIL/free-hold | ✓ PASS/acq-assist |

## c3_1

| object | 추정 feasible | 2F | 3F | vac |
|---|---|---|---|---|
| apple | 2F,3F | ✓ PASS/free-hold | ✓ PASS/free-hold | + PASS/acq-assist |
| bread | 2F,3F,vac | ✓ PASS/free-hold | ✓ PASS/free-hold | ✗ FAIL/free-hold |
| mug | 2F,3F | ✓ PASS/free-hold | ✓ PASS/free-hold | ⊗ PASS/acq-assist [상면 RMS 2.748 vs 허용 1.5; 흡착 패치 RMS 2.034 vs 허용 1.5mm] |
| plate | vac | · | · | ✓ PASS/acq-assist |
| spoon | 2F,3F | ✓ PASS/free-hold | ✗ FAIL/free-hold | · |

## c3_2

| object | 추정 feasible | 2F | 3F | vac |
|---|---|---|---|---|
| plate_a | vac | · | · | ✓ PASS/acq-assist |
| plate_b | vac | · | · | ✓ PASS/acq-assist |
| mug_a | 3F | · | ✓ PASS/acq-assist | ⊗ PASS/acq-assist [상면 RMS 3.435 vs 허용 1.5; 흡착 패치 RMS 2.298 vs 허용 1.5mm] |
| mug_b | 3F | · | ✓ PASS/acq-assist | ⊗ PASS/acq-assist [상면 RMS 3.435 vs 허용 1.5; 흡착 패치 RMS 2.298 vs 허용 1.5mm] |
| fork_a | 2F,3F | ✗ FAIL/free-hold | ✓ PASS/acq-assist | · |
| fork_b | 2F,3F | ✗ FAIL/free-hold | ✓ PASS/acq-assist | ⊗ PASS/acq-assist [접촉면 폭 25 vs seal 직경 30mm; 흡착 패치 RMS 99.9 vs 허용 1.5mm] |
| spoon_a | 2F,3F | ✗ FAIL/free-hold | ✓ PASS/acq-assist | + PASS/acq-assist |
| spoon_b | 2F,3F | ✗ FAIL/free-hold | ✓ PASS/acq-assist | + PASS/acq-assist |
| bread_a | 2F,3F,vac | ✗ FAIL/free-hold | ✓ PASS/free-hold | ✗ FAIL/free-hold |
| bread_b | 2F,3F,vac | ✗ FAIL/free-hold | ✓ PASS/free-hold | ⊗ PASS/acq-assist [상면 RMS 1.661 vs 허용 1.5; 흡착 패치 RMS 1.894 vs 허용 1.5mm] |
| fruit_a | 2F,3F | ✗ FAIL/free-hold | ✓ PASS/acq-assist | · |
| fruit_b | 2F,3F | ✓ PASS/free-hold | ✓ PASS/acq-assist | · |

## c4_1

| object | 추정 feasible | 2F | 3F | vac |
|---|---|---|---|---|
| tool_2_spatula_a | 2F,3F,vac | ✓ PASS/free-hold | ✗ FAIL/free-hold | ✓ PASS/acq-assist |
| tool_3_spatula_b | 2F,3F,vac | ✓ PASS/free-hold | ✗ FAIL/free-hold | ✓ PASS/acq-assist |
| tool_4_spatula_c | 2F,3F,vac | ✗ FAIL/free-hold | ✗ FAIL/free-hold | ✓ PASS/acq-assist |
| tool_5_cutting_board | 2F,3F,vac | ✗ FAIL/free-hold | ✓ PASS/free-hold | ✓ PASS/acq-assist |
| card | 2F,3F | ✗ FAIL/free-hold | ✗ FAIL/free-hold | · |

## c4_2

| object | 추정 feasible | 2F | 3F | vac |
|---|---|---|---|---|
| rolling_pin | 2F,3F,vac | ✓ PASS/free-hold | ✓ PASS/free-hold | ⊗ PASS/acq-assist [상면 RMS 1.519 vs 허용 1.5] |
| baguette | 2F,3F,vac | ✓ PASS/acq-assist | ✓ PASS/free-hold | ✓ PASS/acq-assist |
| whisk | 2F,3F | ✓ PASS/acq-assist | ✗ FAIL/free-hold | · |
| cereal | 2F,3F,vac | ✓ PASS/free-hold | ✓ PASS/free-hold | ⊗ PASS/acq-assist [상면 RMS 5.849 vs 허용 1.5] |
| milk | 2F,3F | ✓ PASS/free-hold | ✓ PASS/free-hold | ⊗ PASS/acq-assist [상면 RMS 4.011 vs 허용 1.5; 흡착 패치 RMS 2.954 vs 허용 1.5mm] |
| lid | vac | · | · | ⊗ PASS/acq-assist [질량 0.6804 vs 페이로드 0.5kg] |

* = 해당 EE 열에 오류·미실행 조합이 있어 커버리지가 확정되지 않음.

주의: 레시피별 힘·허용값·부착 정책이 달라 PASS 건수를 균일 프로토콜 성공률로 읽지 않는다. 'free-hold' 는 부착 보조 없이 유지, 'acq-assist' 는 획득 단계 부착 보조 사용.
