# Fixed-EE grasp verification

Coverage uses every task object as its denominator. Partial matrices are not final coverage results.
Inference-conditioned counts are separate. Unknown inference is not a negative verdict.
Historical Table II counts are not comparable when the current scene object set differs.

## c1_1 (seed=0, inference=reasoner)

| EE | PASS / all objects | tested cases | grasp attempted | inferred PASS / inferred | matrix complete |
|---|---|---|---|---|---|
| 2F | 1/5 | 4 | 4 | 1/4 | False |

| object | EE | status | integration | attachment phase | detail |
|---|---|---|---|---|---|
| ladle | 2F | FAIL | EXPERIMENTAL | none/unknown | ladle: CLOSE: GRASP_CONTACT_NOT_STABLE; CLOSE: GRASP_CONTACT_NOT_STABLE |
| fork | 2F | FAIL | EXPERIMENTAL | none/unknown | fork: CLOSE: GRASP_CONTACT_NOT_STABLE; CLOSE: GRASP_CONTACT_NOT_STABLE |
| scissors | 2F | FAIL | EXPERIMENTAL | none/unknown | scissors: PLAN_GRASP: CARTESIAN_COLLISION: GRASP invalid swept sample 1/1:  [{'geoms': ['table_collision', 'gripper0_right_left_outer_knuckle_collision'], 'penetration_m': 0.0126434899979464}]; PLAN_GRASP: CARTESIAN_COLLISION: GRASP invalid |
| bottle_distractor | 2F | PASS | EXPERIMENTAL | POST_VALIDATED_GRASP |  |

NOT_REGISTERED/EE_MISMATCH means unexecuted, not physically impossible. FAIL describes this motion/grasp attempt only.
ERROR/TIMEOUT is separate from grasp failure. Attachment-assisted PASS is not proof of unassisted retention.
This checks grasp, lift and hold; it does not evaluate completion of the full task.
