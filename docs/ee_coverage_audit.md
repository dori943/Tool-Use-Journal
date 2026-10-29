# EE coverage audit (GT rule vs Reasoner vs scripted registry vs Table II)

## c1_1  (C1_1_LegoSweep, seed=0)

| object | mass kg | size mm | GT-rule feasible | M1(Full) | registry prod (exp) | gap: GT ok but no recipe | recipe but GT infeasible |
|---|---|---|---|---|---|---|---|
| plate | 0.2 | 186.4×192.5×17.9 | vac |  ≠GT | 2F,vac |  | 2F |
| ladle | 0.026 | 231.6×78.1×148.8 | 2F,3F,vac |  ≠GT | ∅ | 2F,3F,vac |  |
| fork | 0.001 | 223.4×36.0×46.2 | 2F,3F,vac |  ≠GT | ∅ | 2F,3F,vac |  |
| scissors | 0.0176 | 181.0×91.3×12.3 | 3F,vac |  ≠GT | ∅ | 3F,vac |  |
| bottle_distractor | 0.2786 | 59.8×60.8×145.4 | 2F,3F,vac |  ≠GT | ∅ | 2F,3F,vac |  |

| EE | GT-rule coverage | registry coverage | executable (GT ∩ registry) | Table II | match |
|---|---|---|---|---|---|
| 2F | 3/5 | 1/5 | 0/5 | 4/5 | MISMATCH |
| 3F | 4/5 | 0/5 | 0/5 | 4/5 | OK |
| vac | 5/5 | 1/5 | 1/5 | 1/5 | MISMATCH |

**SCRIPTED_GRASP_EE_INFEASIBLE 위험 조합** (규칙상 가능하지만 레시피 없음): ladle→2F,3F,vac; fork→2F,3F,vac; scissors→3F,vac; bottle_distractor→2F,3F,vac

<details><summary>GT 규칙 탈락 사유</summary>

- plate / 2F: 치수 186 vs 개구 85mm
- plate / 3F: 치수 186 vs 개구 140mm
- scissors / 2F: 치수 91 vs 개구 85mm

</details>

## c1_2  (C1_2_DoughFlatten, seed=0)

| object | mass kg | size mm | GT-rule feasible | M1(Full) | registry prod (exp) | gap: GT ok but no recipe | recipe but GT infeasible |
|---|---|---|---|---|---|---|---|
| bottle | 0.2786 | 59.8×60.8×145.4 | 2F,3F | 2F,3F | 3F | 2F |  |
| spatula | 0.0253 | 250.0×46.8×5.1 | 2F,3F,vac | 2F,3F,vac | 3F | 2F,vac |  |
| spoon | 0.0244 | 178.1×69.2×35.8 | 2F,3F | 2F,3F | 2F,3F |  |  |
| tongs | 0.0 | 259.1×33.9×31.5 | 2F,3F,vac | 2F,3F ≠GT | ∅ | 2F,3F,vac |  |

| EE | GT-rule coverage | registry coverage | executable (GT ∩ registry) | Table II | match |
|---|---|---|---|---|---|
| 2F | 4/4 | 1/4 | 1/4 | 3/4 | MISMATCH |
| 3F | 4/4 | 3/4 | 3/4 | 3/4 | MISMATCH |
| vac | 2/4 | 0/4 | 0/4 | 0/4 | MISMATCH |

**SCRIPTED_GRASP_EE_INFEASIBLE 위험 조합** (규칙상 가능하지만 레시피 없음): bottle→2F; spatula→2F,vac; tongs→2F,3F,vac

<details><summary>GT 규칙 탈락 사유</summary>

- bottle / vac: 상면 RMS 1.93 vs 허용 1.5
- spoon / vac: 흡착 패치 RMS 99.9 vs 허용 1.5mm

</details>

## c2_1  (C2_1_ObjectSorting, seed=0)

| object | mass kg | size mm | GT-rule feasible | M1(Full) | registry prod (exp) | gap: GT ok but no recipe | recipe but GT infeasible |
|---|---|---|---|---|---|---|---|
| apple | 0.2066 | 103.0×109.2×84.0 | 3F | 2F,3F ≠GT | 3F |  |  |
| bread | 0.4875 | 101.4×148.6×93.6 | 3F,vac | 2F,3F,vac ≠GT | 3F | vac |  |
| mug | 0.0132 | 76.6×131.0×118.0 | 2F,3F | 2F,3F | 3F | 2F |  |
| plate | 0.0051 | 186.4×192.5×17.9 | vac | vac | vac |  |  |
| spoon | 0.0244 | 178.1×69.2×35.8 | 2F,3F | 2F,3F | 2F,3F |  |  |

| EE | GT-rule coverage | registry coverage | executable (GT ∩ registry) | Table II | match |
|---|---|---|---|---|---|
| 2F | 2/5 | 1/5 | 1/5 | 4/5 | MISMATCH |
| 3F | 4/5 | 4/5 | 4/5 | 4/5 | OK |
| vac | 2/5 | 1/5 | 1/5 | 1/5 | MISMATCH |

**SCRIPTED_GRASP_EE_INFEASIBLE 위험 조합** (규칙상 가능하지만 레시피 없음): bread→vac; mug→2F

<details><summary>GT 규칙 탈락 사유</summary>

- apple / 2F: 치수 103 vs 개구 85mm
- apple / vac: 상면 RMS nan vs 허용 1.5; 흡착 패치 RMS 99.9 vs 허용 1.5mm
- bread / 2F: 치수 101 vs 개구 85mm
- mug / vac: 상면 RMS 2.84 vs 허용 1.5
- plate / 2F: 치수 186 vs 개구 85mm
- plate / 3F: 치수 186 vs 개구 140mm
- spoon / vac: 흡착 패치 RMS 99.9 vs 허용 1.5mm

</details>
