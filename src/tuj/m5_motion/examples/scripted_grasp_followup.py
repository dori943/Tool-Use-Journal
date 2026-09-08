"""Integration probe: a supplied keyframe plan goes through normal M5 after grasp.

The supplied test provider replaces only the network call, allowing an exact,
repeatable check of planning, contact collision proxies, control and release.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

from tuj.m5_motion.geometry import tool_rotation_from_axis
from tuj.m5_motion.schema import (
    ArtifactProvenance, KeyframePlanArtifact, KeyframePlanCandidate,
    MotionPlanRequest, MotionTask, MotionGoal, Pose, RelativeKeyframeSpec,
    StrategyGenerationProvenance,
)
from tuj.m5_motion.generic_runner import default_constraints
from tuj.m5_motion.scripted_grasps.frames import inverse, pose_dict
from tuj.m5_motion.scripted_grasps.live import ScriptedGraspSession
from tuj.m5_motion.scripted_grasps.runtime import save_json
from tuj.m5_motion.scripted_grasps.settings import REPOSITORY
from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalControllerTrajectoryPlayer


def check_followup(runtime, entry, output, seed):
    c = runtime.scripted_grasp_retention.context
    start, body = c.grip_pose(), c.body_pose()
    target = start.copy()
    target[2, 3] += .02

    class SuppliedKeyframes:
        def generate(self, request):
            keys = []
            for i, fraction in enumerate((.5, 1.)):
                goal = start.copy()
                goal[2, 3] += fraction * .02
                anchor = "integration_transfer_" + str(i)
                request.world.objects[entry.object_id]["anchors"][anchor] = (inverse(body) @ goal)[:3, 3].tolist()
                z, x = goal[:3, 2], goal[:3, 0]
                base = tool_rotation_from_axis(z, 0.)
                keys.append(RelativeKeyframeSpec(keyframe_id=anchor, keyframe_type="TRANSFER",
                    frame_ref="object:" + entry.object_id, anchor=anchor,
                    approach_axis_xyz=tuple(body[:3, :3].T @ z),
                    roll_rad=math.atan2(x @ base[:, 1], x @ base[:, 0]), planner="CARTESIAN",
                    metadata={"hold_duration_after_s": 2. if i == 1 else 0.}))
            return KeyframePlanArtifact(artifact_id="integration-keyframes", provenance=request.provenance.model_copy(
                update={"artifact_id": "integration-keyframes", "artifact_type": "KeyframePlanArtifact"}),
                scene_signature=request.world.scene.signature, subgoal_id=request.task.subgoal_id,
                candidates=[KeyframePlanCandidate(strategy_id="supplied-transfer", keyframes=keys,
                    provenance=StrategyGenerationProvenance(generator_kind="TEMPLATE",
                        generator_id="INTEGRATION_TEST", input_hash="fixed-two-centimetre-transfer"))])

    session = ScriptedGraspSession(runtime, REPOSITORY, output, seed=seed, provider=SuppliedKeyframes())
    request = MotionPlanRequest(request_id="integration-transfer",
        provenance=ArtifactProvenance(artifact_id="integration-transfer-request", artifact_type="MotionPlanRequest",
            produced_by="TASK_PLANNER", invocation_id="integration"), world=session.world,
        task=MotionTask(task_id="integration", subgoal_id="transfer", action_type="MOVE", ee=entry.ee,
            tool=entry.object_id, goal=MotionGoal(goal_type="POSE", target_pose=Pose(frame_id="world", **pose_dict(target)))),
        constraints=default_constraints(session.world))
    record = session.execute_request(request, completed_subgoal="transfer")
    save_json(output / "retention.json", runtime.scripted_grasp_retention.samples)
    held_height = float(c.body_pose()[2, 3])
    if runtime.attachment is not None:
        runtime.detach_object(entry.object_id)
    runtime.command_gripper(engaged=False, suction=entry.ee == "vac")
    action = np.zeros(c.robot.action_dim)
    lo, hi = c.robot.composite_controller._action_split_indexes["right"]
    action[lo:hi] = c.data.qpos[c.arm_ids]
    lo, hi = c.robot.composite_controller._action_split_indexes["right_gripper"]
    action[lo:hi] = -1.
    player = ToolUseJournalControllerTrajectoryPlayer(runtime)
    for _ in range(100):
        player._advance_controller(action)
    drop = held_height - float(c.body_pose()[2, 3])
    result = {"move_status": record["status"], "release_drop_m": drop,
        "attachment_after_release": runtime.attached_object_id, "held_tool_after_release": runtime.held_tool_id}
    save_json(output / "release.json", result)
    if drop < .05 or runtime.held_tool_id is not None or runtime.attachment is not None:
        raise RuntimeError("FOLLOWUP_RELEASE_DID_NOT_FREE_OBJECT")
    return result
