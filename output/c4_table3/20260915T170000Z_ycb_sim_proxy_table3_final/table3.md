# Table III — Static Real-5 + YCB mesh SIM_PROXY (최종 집계)

- R: Static EV-RealPhys Real-5 (기존 5회 결과).
- S: Uniform-scaled YCB mesh MuJoCo simulator proxy (5 objects, 3 repeats).
- GT numerics의 S 값은 evaluator-only oracle 정합성 확인이며 모델 성능이 아니다.

| Prior/SiPhy | Condition | Mass MnRE↓ | Mass Acc↑ | Suction Acc↑ | Suction PF↑ | Friction MAE↓ | Clearance RelErr↓ | Feas. Acc↑ | DA↑ | Crit.↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior | Name-only | TBD | 0.8889 ± 0.0385 (S; n=3) | 0.5667 ± 0.0577 (S; n=3) | 0.1333 ± 0.1155 (S; n=3) | — | — | 0.5111 ± 0.0385 (S; n=3) | 0.8000 ± 0.0000 (S; n=3) | 1.0000 ± 0.0000 (S; n=3) |
| Prior | + Affordance labels | TBD | 0.9333 ± 0.0000 (S; n=3) | 0.6000 ± 0.0000 (S; n=3) | 0.2000 ± 0.0000 (S; n=3) | — | — | 0.4667 ± 0.0000 (S; n=3) | 0.5333 ± 0.1155 (S; n=3) | 1.0000 ± 0.0000 (S; n=3) |
| SiPhy | SiPhy (as adopted) | 0.9085 ± 0.0046 (R) | 0.8889 ± 0.0385 (S; n=3) | — | — | — | — | 0.6000 ± 0.0000 (S; n=3) | 0.2000 ± 0.2000 (S; n=3) | 0.0000 ± 0.0000 (S; n=3) |
| SiPhy | + Geometric grounding (ours, §III-A) | 0.9085 ± 0.0046 (R; shared) | 0.8889 ± 0.0385 (S; n=3) | 0.2667 ± 0.0577 (S; n=3) | 0.0667 ± 0.1155 (S; n=3) | — | 0.1485 ± 0.1414 (S; n=3) | 0.5333 ± 0.0000 (S; n=3) | 0.2667 ± 0.1155 (S; n=3) | 0.6667 ± 0.5774 (S; n=3) |
|  | Ours (full: + friction, + spatial) | 0.9085 ± 0.0046 (R; shared) | 0.8889 ± 0.0385 (S; n=3) | 0.4333 ± 0.0577 (S; n=3) | 0.0000 ± 0.0000 (S; n=3) | 0.1550 ± 0.0048 (R-static) | 0.1990 ± 0.0557 (S; n=3) | 0.5333 ± 0.0667 (S; n=3) | 0.6667 ± 0.1155 (S; n=3) | 0.3333 ± 0.5774 (S; n=3) |
|  | GT numerics (upper bound) | — | 1.0000 (S; oracle) | 1.0000 (S; oracle) | 1.0000 (S; oracle) | — | 0.0000 (S; oracle) | 1.0000 (S; oracle) | 1.0000 (S; oracle) | 1.0000 (S; oracle) |

## 집계 및 주의

- 모델 S 값은 3개 repeat의 macro score 평균과 sample std(ddof=1)이다.
- GT numerics는 1회 deterministic oracle이라 std는 N/A이며 `(S; oracle)`로 구분했다.
- `—`는 frozen condition에서 해당 metric을 출력하지 않는 구조적 미지원이다.
- R Mass MnRE와 R-static Friction MAE는 실제 EV-RealPhys 정적 결과이며 S proxy와 혼합하지 않았다.
