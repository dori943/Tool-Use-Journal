# Fixed-EE grasp verification

Coverage uses every task object as its denominator. Partial matrices are not final coverage results.
Inference-conditioned counts are separate. Unknown inference is not a negative verdict.
Historical Table II counts are not comparable when the current scene object set differs.

## c1_1 (seed=0, inference=reasoner)

| EE | PASS / all objects | tested cases | grasp attempted | inferred PASS / inferred | matrix complete |
|---|---|---|---|---|---|
| 2F | 0/5 | 5 | 5 | 0/4 | True |
| 3F | 0/5 | 5 | 5 | 0/4 | True |
| vac | 2/5 | 5 | 5 | 1/2 | True |

| object | EE | status | integration | attachment phase | detail |
|---|---|---|---|---|---|
| plate | 2F | FAIL | EXPERIMENTAL | none/unknown | plate: GRASP: controller did not converge before the next contact-sensitive motion; GRASP: controller did not converge before the next contact-sensitive motion |
| plate | 3F | FAIL | EXPERIMENTAL | none/unknown | plate: PLAN_PRE_CLEARANCE: IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds; PLAN_PRE_CLEARANCE: IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds |
| plate | vac | PASS | EXPERIMENTAL | ACQUISITION |  |
| ladle | 2F | ERROR | EXPERIMENTAL | none/unknown | AttributeError: 'C1_1_LegoSweep' object has no attribute 'catalog_hand_correction' |
| ladle | 3F | FAIL | EXPERIMENTAL | none/unknown | ladle: CLOSE: GRASP_CONTACT_NOT_STABLE; CLOSE: GRASP_CONTACT_NOT_STABLE |
| ladle | vac | FAIL | EXPERIMENTAL | none/unknown | ladle: CLOSE: GRASP_CONTACT_NOT_STABLE; CLOSE: GRASP_CONTACT_NOT_STABLE |
| fork | 2F | ERROR | EXPERIMENTAL | none/unknown | AttributeError: 'C1_1_LegoSweep' object has no attribute 'catalog_hand_correction' |
| fork | 3F | FAIL | EXPERIMENTAL | none/unknown | fork: CLOSE: GRASP_CONTACT_NOT_STABLE; CLOSE: GRASP_CONTACT_NOT_STABLE |
| fork | vac | FAIL | EXPERIMENTAL | none/unknown | fork: CLOSE: object 'fork' penetrates EE geometry by 0.0022 m; limit is 0.0020 m; closest geom pair 'gripper0_right_vac_cup' <-> 'fork_g29' measured by SIMULATOR_CONTACT; closest pairs: -0.0022 m  'gripper0_right_vac_cup' type=cylinder size |
| scissors | 2F | ERROR | EXPERIMENTAL | none/unknown | AttributeError: 'C1_1_LegoSweep' object has no attribute 'catalog_hand_correction' |
| scissors | 3F | FAIL | EXPERIMENTAL | none/unknown | scissors: CLOSE: GRASP_CONTACT_NOT_STABLE; CLOSE: GRASP_CONTACT_NOT_STABLE |
| scissors | vac | PASS | EXPERIMENTAL | ACQUISITION |  |
| bottle_distractor | 2F | ERROR | EXPERIMENTAL | none/unknown | AttributeError: 'C1_1_LegoSweep' object has no attribute 'catalog_hand_correction' |
| bottle_distractor | 3F | FAIL | EXPERIMENTAL | none/unknown | bottle_distractor: PLAN_PRE_CLEARANCE: IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds; PLAN_PRE_CLEARANCE: IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds |
| bottle_distractor | vac | FAIL | EXPERIMENTAL | none/unknown | bottle_distractor: CLOSE: GRASP_CONTACT_NOT_STABLE; CLOSE: GRASP_CONTACT_NOT_STABLE |

NOT_REGISTERED/EE_MISMATCH means unexecuted, not physically impossible. FAIL describes this motion/grasp attempt only.
ERROR/TIMEOUT is separate from grasp failure. Attachment-assisted PASS is not proof of unassisted retention.
This checks grasp, lift and hold; it does not evaluate completion of the full task.
