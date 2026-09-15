# Table III — Static Real-5 + YCB mesh SIM_PROXY

- R: Static EV-RealPhys Real-5 (5 objects, 5 repeats).
- S: Uniform-scaled YCB mesh MuJoCo simulator proxy (5 objects, 3 repeats).
- S 수치는 실제 EV 측정 결과가 아니며 YCB proxy GT와 독립적인 simulator trial 규칙으로 계산했다.

| Prior/SiPhy | Condition | Mass MnRE↓ | Mass Acc↑ | Suction Acc↑ | Suction PF↑ | Friction MAE↓ | Clearance RelErr↓ | Feas. Acc↑ | DA↑ | Crit.↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior | Name-only | TBD | 0.8889 ± 0.0385 (S; n=3) | 0.5667 ± 0.0577 (S; n=3) | 0.1333 ± 0.1155 (S; n=3) | — | — | 0.5111 ± 0.0385 (S; n=3) | 0.8000 ± 0.0000 (S; n=3) | 1.0000 ± 0.0000 (S; n=3) |
| Prior | + Affordance labels | TBD | 0.9333 ± 0.0000 (S; n=3) | 0.6000 ± 0.0000 (S; n=3) | 0.2000 ± 0.0000 (S; n=3) | — | — | 0.4667 ± 0.0000 (S; n=3) | 0.5333 ± 0.1155 (S; n=3) | 1.0000 ± 0.0000 (S; n=3) |
| SiPhy | SiPhy (as adopted) | 0.9085 ± 0.0046 (R) | 0.8889 ± 0.0385 (S; n=3) | — | — | — | — | 0.6000 ± 0.0000 (S; n=3) | 0.2000 ± 0.2000 (S; n=3) | 0.0000 ± 0.0000 (S; n=3) |
| SiPhy | + Geometric grounding (ours, §III-A) | 0.9085 ± 0.0046 (R; shared) | 0.8889 ± 0.0385 (S; n=3) | 0.2667 ± 0.0577 (S; n=3) | 0.0667 ± 0.1155 (S; n=3) | — | 0.1485 ± 0.1414 (S; n=3) | 0.5333 ± 0.0000 (S; n=3) | 0.2667 ± 0.1155 (S; n=3) | 0.6667 ± 0.5774 (S; n=3) |
|  | Ours (full: + friction, + spatial) | 0.9085 ± 0.0046 (R; shared) | 0.8889 ± 0.0385 (S; n=3) | 0.4333 ± 0.0577 (S; n=3) | 0.0000 ± 0.0000 (S; n=3) | 0.1550 ± 0.0048 (R-static) | 0.1990 ± 0.0557 (S; n=3) | 0.5333 ± 0.0667 (S; n=3) | 0.6667 ± 0.1155 (S; n=3) | 0.3333 ± 0.5774 (S; n=3) |
|  | GT numerics (upper bound) | — | TBD | TBD | TBD | — | TBD | TBD | TBD | TBD |

## 집계 및 주의

- S 점수는 객체 5개, 3 repeat, repeat 간 sample std(ddof=1)이다.
- Suction Acc/PF는 각 입력의 pose_A/pose_B를 별도 unit으로 확장해 같은 객체의 두 pose 쌍으로 채점했다.
- Name-only/Affordance/Geo의 추가 D_SIM 수치는 확장된 adapter 결과이며 공식 Real-5 조건의 `—` 셀을 변경하지 않는다.
- SiPhy/+Geo/Ours의 R Mass MnRE는 기존 Static Real-5 결과이며 D_SIM Mass Acc와 다른 지표·benchmark이다.
- Ours의 R-static Friction MAE는 기존 정적 visual combined-friction 결과이며 trajectory friction 결과와 결합하지 않았다.
- GT numerics는 estimator 성능이 아닌 evaluator-only oracle이며 이번 inference에서는 호출하지 않았다.
