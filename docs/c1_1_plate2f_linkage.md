# Fixed-EE grasp verification

Coverage uses every task object as its denominator. Partial matrices are not final coverage results.
Inference-conditioned counts are separate. Unknown inference is not a negative verdict.
Historical Table II counts are not comparable when the current scene object set differs.

## c1_1 (seed=0, inference=reasoner)

| EE | PASS / all objects | tested cases | grasp attempted | inferred PASS / inferred | matrix complete |
|---|---|---|---|---|---|
| 2F | 0/5 | 1 | 1 | 0/0 | False |

| object | EE | status | integration | attachment phase | detail |
|---|---|---|---|---|---|
| plate | 2F | FAIL | EXPERIMENTAL | none/unknown | plate: GRASP: controller did not converge before the next contact-sensitive motion; GRASP: controller did not converge before the next contact-sensitive motion |

NOT_REGISTERED/EE_MISMATCH means unexecuted, not physically impossible. FAIL describes this motion/grasp attempt only.
ERROR/TIMEOUT is separate from grasp failure. Attachment-assisted PASS is not proof of unassisted retention.
This checks grasp, lift and hold; it does not evaluate completion of the full task.
