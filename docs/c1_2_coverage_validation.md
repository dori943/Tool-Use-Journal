# Fixed-EE grasp verification

Coverage uses every task object as its denominator. Partial matrices are not final coverage results.
Inference-conditioned counts are separate. Unknown inference is not a negative verdict.
Historical Table II counts are not comparable when the current scene object set differs.

## c1_2 (seed=0, inference=reasoner)

| EE | PASS / all objects | tested cases | grasp attempted | inferred PASS / inferred | matrix complete |
|---|---|---|---|---|---|
| 2F | 1/4 | 4 | 4 | 1/4 | True |
| 3F | 3/4 | 4 | 4 | 3/4 | True |
| vac | 0/4 | 4 | 4 | 0/1 | True |

| object | EE | status | integration | attachment phase | detail |
|---|---|---|---|---|---|
| bottle | 2F | FAIL | EXPERIMENTAL | none/unknown | bottle: PLAN_PRE_CLEARANCE: IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds; PLAN_PRE_CLEARANCE: IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds |
| bottle | 3F | PASS | VALIDATED | POST_VALIDATED_GRASP |  |
| bottle | vac | ERROR | EXPERIMENTAL | ACQUISITION | bottle: CLOSE: Invalid index -1. Valid indices from 0 to 1409; CLOSE: Invalid index -1. Valid indices from 0 to 1409 |
| spatula | 2F | FAIL | EXPERIMENTAL | none/unknown | spatula: PLAN_GRASP: CARTESIAN_COLLISION: GRASP invalid swept sample 1/1:  [{'geoms': ['gripper0_right_left_fingertip_collision', 'island_island_group_top_2'], 'penetration_m': 0.0016352155763650492}]; PLAN_GRASP: CARTESIAN_COLLISION: GRASP |
| spatula | 3F | PASS | VALIDATED | POST_VALIDATED_GRASP |  |
| spatula | vac | ERROR | EXPERIMENTAL | ACQUISITION | spatula: CLOSE: Invalid index -1. Valid indices from 0 to 1409; CLOSE: Invalid index -1. Valid indices from 0 to 1409 |
| spoon | 2F | PASS | VALIDATED | POST_VALIDATED_GRASP |  |
| spoon | 3F | PASS | VALIDATED | POST_VALIDATED_GRASP |  |
| spoon | vac | FAIL | EXPERIMENTAL | none/unknown | spoon: CLOSE: GRASP_CONTACT_NOT_STABLE; CLOSE: GRASP_CONTACT_NOT_STABLE |
| tongs | 2F | FAIL | EXPERIMENTAL | none/unknown | tongs: PLAN_GRASP: CARTESIAN_COLLISION: GRASP invalid swept sample 1/1:  [{'geoms': ['gripper0_right_left_fingertip_collision', 'island_island_group_top_2'], 'penetration_m': 0.0014439771222479261}]; PLAN_GRASP: CARTESIAN_COLLISION: GRASP i |
| tongs | 3F | FAIL | EXPERIMENTAL | none/unknown | tongs: CLOSE: GRIPPER_JOINT_LIMIT_EXCEEDED: physics substep 0.010124 rad (stage=CLOSE joint=gripper0_right_joint_index_distal lower q=-0.010124 range=[0.000000,2.000000] command=[-0.1593, -0.1791, -0.195] aperture_m=None); CLOSE: GRIPPER_JO |
| tongs | vac | FAIL | EXPERIMENTAL | none/unknown | tongs: CLOSE: UNEXPECTED_COLLISION: [{'geoms': ['gripper0_right_vac_mount_col', 'tongs_collision_20'], 'penetration_m': 0.0010383276736439665}]; CLOSE: UNEXPECTED_COLLISION: [{'geoms': ['gripper0_right_vac_mount_col', 'tongs_collision_20'], |

NOT_REGISTERED/EE_MISMATCH means unexecuted, not physically impossible. FAIL describes this motion/grasp attempt only.
ERROR/TIMEOUT is separate from grasp failure. Attachment-assisted PASS is not proof of unassisted retention.
This checks grasp, lift and hold; it does not evaluate completion of the full task.
