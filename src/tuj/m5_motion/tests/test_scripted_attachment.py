"""Validate the boundary between physical acquisition and runtime attachment."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tuj.m5_motion.scripted_grasps import context, retention
from tuj.m5_motion.scripted_grasps.runtime import GraspFailure
from tuj.m5_motion.tool_use_journal_runtime import AttachmentMode


def setup_grasp(monkeypatch, tmp_path, *, acquired=True, already_attached=False):
    attachment = SimpleNamespace(object_id="object", mode=AttachmentMode.KINEMATIC)
    runtime = SimpleNamespace(attachment=attachment if already_attached else None,
                             finish_attachment_step=Mock(), command_gripper=Mock(),
                             mark_attached_object_as_tool=Mock())
    events = []

    def attach(object_id):
        events.append("attach")
        assert object_id == "object"
        runtime.command_gripper.assert_called_once()
        runtime.attachment = attachment
        return attachment

    runtime.attach_object = Mock(side_effect=attach)
    result = {"status": "SUCCESS" if acquired else "FAILED",
              "attachment_used": already_attached, "failure_reason": "HOLD_FAILED"}

    def acquire(c):
        events.append("acquire")
        return result

    entry = SimpleNamespace(object_id="object", ee="vac" if already_attached else "3F",
                            function=lambda: acquire)
    c = SimpleNamespace(output=tmp_path, data=SimpleNamespace(qpos=np.array([.2, .4])),
                        arm_ids=[0, 1], grip_pose=lambda: np.eye(4), body_pose=lambda: np.eye(4))
    monkeypatch.setattr(context, "bind_context", lambda *args, **kwargs: c)
    retained = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(retention, "GraspRetention", retained)
    return runtime, entry, events, retained


def test_attach_only_after_validated_acquisition(monkeypatch, tmp_path):
    runtime, entry, events, retained = setup_grasp(monkeypatch, tmp_path)
    result = context.execute_grasp(runtime, entry, tmp_path)
    assert events == ["acquire", "attach"]
    runtime.mark_attached_object_as_tool.assert_called_once_with("object")
    retained.assert_called_once()
    assert result["acquisition_status"] == "SUCCESS"
    assert result["acquisition_attachment_used"] is False
    assert result["attachment_used"] is True
    assert result["attachment_mode"] == "KINEMATIC"
    assert result["attachment_phase"] == "POST_VALIDATED_GRASP"
    assert json.loads((tmp_path / "result.json").read_text())["attachment_used"] is True


def test_failed_acquisition_never_attaches(monkeypatch, tmp_path):
    runtime, entry, events, retained = setup_grasp(monkeypatch, tmp_path, acquired=False)
    with pytest.raises(GraspFailure, match="HOLD_FAILED"):
        context.execute_grasp(runtime, entry, tmp_path)
    runtime.attach_object.assert_not_called()
    runtime.command_gripper.assert_not_called()
    retained.assert_not_called()
    assert events == ["acquire"]


def test_existing_vac_attachment_is_reused(monkeypatch, tmp_path):
    runtime, entry, _, _ = setup_grasp(monkeypatch, tmp_path, already_attached=True)
    original = runtime.attachment
    result = context.execute_grasp(runtime, entry, tmp_path)
    runtime.attach_object.assert_not_called()
    assert runtime.attachment is original
    assert result["acquisition_attachment_used"] is True
    assert result["attachment_phase"] == "ACQUISITION"


def test_rejected_attachment_is_a_failed_integration(monkeypatch, tmp_path):
    runtime, entry, _, retained = setup_grasp(monkeypatch, tmp_path)
    runtime.attach_object.side_effect = RuntimeError("attach distance exceeds limit")
    with pytest.raises(RuntimeError, match="attach distance"):
        context.execute_grasp(runtime, entry, tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "FAILED"
    assert result["acquisition_status"] == "SUCCESS"
    assert result["failure_stage"] == "POST_GRASP_ATTACHMENT"
    assert result["attachment_used"] is False
    retained.assert_not_called()
    runtime.mark_attached_object_as_tool.assert_not_called()


def test_scripted_planning_failure_preserves_rejected_edge_evidence(monkeypatch, tmp_path):
    from tuj.m5_motion.scripted_grasps import live
    from tuj.m5_motion.tests.test_scripted_grasps import request_for, ENTRIES
    from tuj.m5_motion.pipeline import MotionPlanningPipelineError

    request = request_for(ENTRIES[4], action="MOVE")
    world = request.world
    monkeypatch.setattr(live, "snapshot", lambda *args: world.model_copy(deep=True))
    edge = SimpleNamespace(source_keyframe_id="transfer", target_keyframe_id="place",
        source_branch_id="a", target_branch_id="b", failure_code="COLLISION",
        detail="held object collides with destination rim")
    attempt = SimpleNamespace(strategy_id="place", failure_code="NO_CONNECTED_SEQUENCE",
        detail="no connected branches", resolved_keyframes=(), ik_diagnostics=(),
        selection=SimpleNamespace(rejected_edges=(edge,)))
    error = MotionPlanningPipelineError("planning failed",
        compilation=SimpleNamespace(attempts=(attempt,)))

    def planner(_request):
        raise error

    session = live.ScriptedGraspSession(SimpleNamespace(env=object()), tmp_path, tmp_path,
        planner_factory=lambda *args, **kwargs: planner)
    with pytest.raises(MotionPlanningPipelineError) as caught:
        session.execute_request(request)
    assert caught.value is error
    manifest = json.loads((tmp_path / "live-execution-manifest.json").read_text())
    step = manifest["steps"][0]
    assert step["status"] == "FAILED"
    from pathlib import Path
    saved = json.loads(Path(step["planning_failure"]).read_text())
    assert saved["attempts"][0]["rejected_edges"][0]["detail"] == edge.detail
