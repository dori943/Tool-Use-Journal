# vac GT admissibility (seed 0) — Table II vac 성공 = 시뮬 PASS ∧ 아래 admissible

| task | object | admissible | surface RMS | seal patch RMS | contact width | mass (src) | material | porous? | reason |
|---|---|---|---|---|---|---|---|---|---|
| c1_1 | plate | YES | 0.287 | 0.454 | 184.1 | 0.279 (ycb) | rigid_plastic |  |  |
| c1_1 | ladle | no | 2.922 | 1.634 | 65.3 | 0.026 (sim) | metal |  | 상면 RMS 2.922 vs 허용 1.5; 흡착 패치 RMS 1.634 vs 허용 1.5mm |
| c1_1 | fork | no | 0.565 | 99.9 | 36.1 | 0.034 (ycb) | metal |  | 흡착 패치 RMS 99.9 vs 허용 1.5mm |
| c1_1 | scissors | YES | 0.359 | 0.43 | 80.9 | 0.082 (ycb) | metal_plastic |  |  |
| c1_1 | bottle_distractor | no | 5.652 | 2.802 | 52.2 | 0.2786 (sim) | plastic |  | 상면 RMS 5.652 vs 허용 1.5; 흡착 패치 RMS 2.802 vs 허용 1.5mm |
| c1_2 | bottle | no | 5.652 | 2.802 | 52.2 | 0.2786 (sim) | plastic |  | 상면 RMS 5.652 vs 허용 1.5; 흡착 패치 RMS 2.802 vs 허용 1.5mm |
| c1_2 | spatula | YES | 0.148 | 0.0 | 46.8 | 0.0515 (ycb) | metal_plastic |  |  |
| c1_2 | spoon | YES | 0.978 | 0.346 | 56.2 | 0.03 (ycb) | metal |  |  |
| c1_2 | tongs | YES | 0.793 | 1.291 | 31.0 | 0.0 (sim) | metal_plastic |  |  |
| c2_1 | apple | YES | 1.285 | 0.618 | 75.4 | 0.068 (ycb) | organic_tissue |  |  |
| c2_1 | bread | no | 2.159 | 1.762 | 85.6 | 0.4875 (sim) | bread | ⚠ | 상면 RMS 2.159 vs 허용 1.5; 흡착 패치 RMS 1.762 vs 허용 1.5mm |
| c2_1 | mug | no | 2.748 | 2.034 | 110.1 | 0.118 (ycb) | ceramic |  | 상면 RMS 2.748 vs 허용 1.5; 흡착 패치 RMS 2.034 vs 허용 1.5mm |
| c2_1 | plate | YES | 0.287 | 0.454 | 184.1 | 0.279 (ycb) | rigid_plastic |  |  |
| c2_1 | spoon | YES | 0.978 | 0.346 | 56.2 | 0.03 (ycb) | metal |  |  |
| c2_2 | bread_a | YES | 0.838 | 0.906 | 132.1 | 0.2711 (sim) | bread | ⚠ |  |
| c2_2 | bread_b | YES | 0.838 | 0.906 | 132.1 | 0.2711 (sim) | bread | ⚠ |  |
| c2_2 | turkey_1 | YES | 0.068 | 0.04 | 51.0 | 0.0059 (sim) | processed_meat |  |  |
| c2_2 | cheese_1 | YES | 0.152 | 0.223 | 60.6 | 0.0008 (sim) | cheese |  |  |
| c2_2 | tomato_slice | YES | 0.064 | 0.0 | 61.2 | 0.0137 (sim) | tomato |  |  |
| c3_1 | apple | YES | 1.285 | 0.618 | 75.4 | 0.068 (ycb) | organic_tissue |  |  |
| c3_1 | bread | no | 2.159 | 1.762 | 85.6 | 0.4875 (sim) | bread | ⚠ | 상면 RMS 2.159 vs 허용 1.5; 흡착 패치 RMS 1.762 vs 허용 1.5mm |
| c3_1 | mug | no | 2.748 | 2.034 | 110.1 | 0.118 (ycb) | ceramic |  | 상면 RMS 2.748 vs 허용 1.5; 흡착 패치 RMS 2.034 vs 허용 1.5mm |
| c3_1 | plate | YES | 0.287 | 0.454 | 184.1 | 0.279 (ycb) | rigid_plastic |  |  |
| c3_1 | spoon | YES | 0.978 | 0.346 | 56.2 | 0.03 (ycb) | metal |  |  |
| c3_2 | plate_a | YES | 0.264 | 0.424 | 169.2 | 0.279 (ycb) | rigid_plastic |  |  |
| c3_2 | plate_b | YES | 0.264 | 0.424 | 169.2 | 0.279 (ycb) | rigid_plastic |  |  |
| c3_2 | mug_a | no | 3.435 | 2.298 | 137.6 | 0.118 (ycb) | ceramic |  | 상면 RMS 3.435 vs 허용 1.5; 흡착 패치 RMS 2.298 vs 허용 1.5mm |
| c3_2 | mug_b | no | 3.435 | 2.298 | 137.6 | 0.118 (ycb) | ceramic |  | 상면 RMS 3.435 vs 허용 1.5; 흡착 패치 RMS 2.298 vs 허용 1.5mm |
| c3_2 | fork_a | no | 0.395 | 99.9 | 25.2 | 0.034 (ycb) | metal |  | 접촉면 폭 25 vs seal 직경 30mm; 흡착 패치 RMS 99.9 vs 허용 1.5mm |
| c3_2 | fork_b | no | 0.395 | 99.9 | 25.2 | 0.034 (ycb) | metal |  | 접촉면 폭 25 vs seal 직경 30mm; 흡착 패치 RMS 99.9 vs 허용 1.5mm |
| c3_2 | spoon_a | YES | 0.783 | 0.283 | 45.0 | 0.03 (ycb) | metal |  |  |
| c3_2 | spoon_b | YES | 0.783 | 0.283 | 45.0 | 0.03 (ycb) | metal |  |  |
| c3_2 | bread_a | no | 1.661 | 1.894 | 65.9 | 0.2219 (sim) | bread | ⚠ | 상면 RMS 1.661 vs 허용 1.5; 흡착 패치 RMS 1.894 vs 허용 1.5mm |
| c3_2 | bread_b | no | 1.661 | 1.894 | 65.9 | 0.2219 (sim) | bread | ⚠ | 상면 RMS 1.661 vs 허용 1.5; 흡착 패치 RMS 1.894 vs 허용 1.5mm |
| c3_2 | fruit_a | YES | 1.371 | 0.585 | 80.5 | 0.2508 (sim) | - |  |  |
| c3_2 | fruit_b | YES | 1.371 | 0.585 | 80.5 | 0.2508 (sim) | - |  |  |
| c4_1 | tool_2_spatula_a | YES | 0.118 | 0.013 | 36.0 | 0.0515 (ycb) | metal_plastic |  |  |
| c4_1 | tool_3_spatula_b | YES | 0.217 | 0.0 | 40.0 | 0.0515 (ycb) | metal_plastic |  |  |
| c4_1 | tool_4_spatula_c | YES | 0.35 | 0.0 | 56.0 | 0.0573 (sim) | - |  |  |
| c4_1 | tool_5_cutting_board | YES | 0.275 | 0.0 | 72.6 | 0.0253 (sim) | - |  |  |
| c4_1 | card | YES | 0.01 | 0.0 | 32.1 | 0.0021 (sim) | - |  |  |
| c4_2 | rolling_pin | no | 1.519 | 0.532 | 43.2 | 0.27 (sim) | wood |  | 상면 RMS 1.519 vs 허용 1.5 |
| c4_2 | baguette | YES | 1.226 | 0.445 | 59.8 | 0.3719 (sim) | bread | ⚠ |  |
| c4_2 | whisk | no | 1.388 | 1.563 | 67.1 | 0.0002 (sim) | metal_plastic |  | 흡착 패치 RMS 1.563 vs 허용 1.5mm |
| c4_2 | cereal | no | 5.849 | 0.0 | 126.1 | 0.072 (sim) | cardboard | ⚠ | 상면 RMS 5.849 vs 허용 1.5 |
| c4_2 | milk | no | 4.011 | 2.954 | 50.3 | 0.2724 (sim) | paper_carton | ⚠ | 상면 RMS 4.011 vs 허용 1.5; 흡착 패치 RMS 2.954 vs 허용 1.5mm |
| c4_2 | lid | no | 0.123 | 0.0 | 210.2 | 0.6804 (sim) | - |  | 질량 0.6804 vs 페이로드 0.5kg |

규칙은 Full M1 과 동일(evaluate_ee vac). 재질/다공성은 규칙에 포함되지 않으며 ⚠ 는 산업 자료 기준 참고 표시다.
