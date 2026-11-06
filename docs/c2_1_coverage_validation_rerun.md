# Fixed-EE grasp verification

Coverage uses every task object as its denominator. Partial matrices are not final coverage results.
Inference-conditioned counts are separate. Unknown inference is not a negative verdict.
Historical Table II counts are not comparable when the current scene object set differs.

## c2_1 (seed=0, inference=reasoner)

| EE | PASS / all objects | tested cases | grasp attempted | inferred PASS / inferred | matrix complete |
|---|---|---|---|---|---|
| 2F | 3/5 | 5 | 5 | 3/4 | True |
| 3F | 3/5 | 5 | 5 | 3/4 | True |
| vac | 3/5 | 5 | 5 | 1/2 | True |

| object | EE | status | integration | attachment phase | detail |
|---|---|---|---|---|---|
| apple | 2F | FAIL | EXPERIMENTAL | none/unknown | apple: None: HOLD_VALIDATION_FAILED; None: HOLD_VALIDATION_FAILED |
| apple | 3F | PASS | VALIDATED | POST_VALIDATED_GRASP |  |
| apple | vac | PASS | EXPERIMENTAL | ACQUISITION |  |
| bread | 2F | PASS | EXPERIMENTAL | POST_VALIDATED_GRASP |  |
| bread | 3F | PASS | VALIDATED | POST_VALIDATED_GRASP |  |
| bread | vac | FAIL | EXPERIMENTAL | none/unknown | bread: CLOSE: GRASP_CONTACT_NOT_STABLE; CLOSE: GRASP_CONTACT_NOT_STABLE |
| mug | 2F | PASS | EXPERIMENTAL | POST_VALIDATED_GRASP |  |
| mug | 3F | PASS | VALIDATED | POST_VALIDATED_GRASP |  |
| mug | vac | PASS | EXPERIMENTAL | ACQUISITION |  |
| plate | 2F | FAIL | EXPERIMENTAL | none/unknown | plate: PLAN_PRE_CLEARANCE: IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds; PLAN_PRE_CLEARANCE: IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds |
| plate | 3F | FAIL | EXPERIMENTAL | none/unknown | plate: PLAN_PRE_CLEARANCE: IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds; PLAN_PRE_CLEARANCE: IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds |
| plate | vac | PASS | EXPERIMENTAL | ACQUISITION |  |
| spoon | 2F | PASS | VALIDATED | POST_VALIDATED_GRASP |  |
| spoon | 3F | FAIL | VALIDATED | none/unknown | spoon: None: HOLD_VALIDATION_FAILED; None: HOLD_VALIDATION_FAILED |
| spoon | vac | FAIL | EXPERIMENTAL | none/unknown | spoon: CLOSE: GRASP_CONTACT_NOT_STABLE; CLOSE: GRASP_CONTACT_NOT_STABLE |

NOT_REGISTERED/EE_MISMATCH means unexecuted, not physically impossible. FAIL describes this motion/grasp attempt only.
ERROR/TIMEOUT is separate from grasp failure. Attachment-assisted PASS is not proof of unassisted retention.
This checks grasp, lift and hold; it does not evaluate completion of the full task.
