# D_SIM evaluator report (separate from Real-5 Table III)

| Condition | Mass Acc | Suction Acc | Suction PF | Clearance RelErr | Feas. Acc | DA | Crit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Name-only | 0.7333 ± 0.0000 | - | - | - | - | - | 0.3333 ± 0.0000 |
| + Affordance labels | INCOMPLETE | - | - | - | - | - | INCOMPLETE |
| SiPhy (as adopted) | 0.8444 ± 0.0385 | - | - | - | - | - | 0.3333 ± 0.0000 |
| + Geometric grounding | 0.8444 ± 0.0385 | - | - | 0.0000 ± 0.0000 | 0.7778 ± 0.0385 | 0.1333 ± 0.1155 | 0.3333 ± 0.0000 |
| Ours (full) | 0.8444 ± 0.0385 | INCOMPLETE | INCOMPLETE | 1.3667 ± 1.5535 | 0.6889 ± 0.0770 | 0.4000 ± 0.0000 | 0.3333 ± 0.0000 |

Scores are simulator-only and use 3 repeats (mean ± ddof=1 repeat std). They must not be inserted into the official EV-RealPhys Real-5 table. Suction metrics remain incomplete because the adapter emitted sample-level rather than pose-level predictions.
