from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.planning_attachment import planning_attachment_transform
from tuj.m5_motion.tool_use_journal_planning import attached_object_transform_from_state
from tuj.m5_motion.tool_use_journal_runtime import (
    AttachedObjectState, AttachmentMode, BreakableWeldConfig,
)


def setup(mode=AttachmentMode.BREAKABLE_WELD):
    reference_rotation = Rotation.from_euler('xyz', [20, -35, 60], degrees=True).as_matrix()
    reference_position = np.array([.22, -.17, .91])
    measured_rotation = Rotation.from_euler('xyz', [14, 8, -22], degrees=True).as_matrix()
    measured_position = np.array([.012, -.005, .029])
    attachment = AttachedObjectState(
        object_id='tool', free_joint_name='tool_free', reference_kind='site',
        reference_name='grip_site', position_in_reference_m=(.01, -.004, 0.),
        rotation_in_reference=tuple(tuple(row) for row in np.eye(3)),
        attach_distance_m=.001, mode=mode,
        breakable_weld=BreakableWeldConfig() if mode is AttachmentMode.BREAKABLE_WELD else None,
    )
    data = SimpleNamespace(
        xpos=np.array([reference_position + reference_rotation @ measured_position]),
        xmat=np.array([(reference_rotation @ measured_rotation).ravel()]),
        qpos=np.arange(7, dtype=float), qvel=np.arange(6, dtype=float),
    )
    runtime = SimpleNamespace(
        attachment=attachment, env=object(),
        _reference_pose=lambda *args: (reference_position, reference_rotation),
    )
    adapter = SimpleNamespace(object_body_ids={'tool': 0}, data=data)
    return runtime, adapter, measured_position, measured_rotation


def test_measured_nonidentity_reference_replaces_stale_planning_pose_only():
    runtime, adapter, position, rotation = setup()
    saved_attachment = asdict(runtime.attachment)
    saved_arrays = {name: getattr(adapter.data, name).copy() for name in ('xpos', 'xmat', 'qpos', 'qvel')}
    measured, provenance = planning_attachment_transform(runtime, adapter)
    assert measured.position_in_reference_m == pytest.approx(position)
    assert Rotation.from_quat(measured.orientation_in_reference_xyzw).as_matrix() == pytest.approx(rotation)
    assert provenance['formation_rest_transform'] == attached_object_transform_from_state(runtime.attachment).model_dump(mode='json')
    assert provenance['measured_planning_transform'] == measured.model_dump(mode='json')
    assert provenance['position_deviation_from_rest_m'] > .028
    assert provenance['runtime_rest_transform_unchanged'] is True
    assert asdict(runtime.attachment) == saved_attachment
    for name, before in saved_arrays.items():
        np.testing.assert_array_equal(getattr(adapter.data, name), before)


def test_kinematic_rack_transform_uses_rest_without_reading_live_pose():
    runtime, _, _, _ = setup(AttachmentMode.KINEMATIC)
    expected = attached_object_transform_from_state(runtime.attachment)
    measured, provenance = planning_attachment_transform(runtime, object())
    assert measured == expected
    assert provenance is None


def test_no_attachment_leaves_contact_friction_snapshot_path_unchanged():
    measured, provenance = planning_attachment_transform(SimpleNamespace(attachment=None), object())
    assert measured is None and provenance is None


def test_nonfinite_measured_pose_fails_without_falling_back_to_stale_rest():
    runtime, adapter, _, _ = setup()
    adapter.data.xpos[0, 2] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        planning_attachment_transform(runtime, adapter)


@pytest.mark.parametrize('path', ['scripted', 'generic'])
def test_both_snapshot_paths_export_measured_transform_with_actual_object_pose(monkeypatch, path):
    from tuj.m5_motion import tool_use_journal, tool_use_journal_execution
    from tuj.m5_motion.scripted_grasps.live import snapshot
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalKinematicTrajectoryPlayer
    from tuj.m5_motion.tests.test_tool_use_journal_execution import _request_and_plan
    request, _ = _request_and_plan()
    runtime, measured_adapter, position, _ = setup()
    runtime.attached_object_id = 'tool'
    original = asdict(runtime.attachment)
    request.world.robot_state.attached_object_id = 'tool'
    request.world.robot_state.held_tool_id = 'tool'
    object_pose = {'position_m': measured_adapter.data.xpos[0].tolist(),
                   'orientation_xyzw': Rotation.from_matrix(measured_adapter.data.xmat[0].reshape(3, 3)).as_quat().tolist()}
    class Adapter:
        def __init__(self, env):
            self.object_body_ids = measured_adapter.object_body_ids
            self.data = measured_adapter.data
        def world_snapshot(self, **kwargs):
            world = request.world.model_copy(deep=True)
            transform = kwargs['attached_object_transform']
            world.metadata['attached_object_transforms'] = {'tool': transform.model_dump(mode='json')}
            world.objects['tool'] = {'pose': object_pose.copy()}
            return world
    monkeypatch.setattr(tool_use_journal, 'ToolUseJournalEnvironmentAdapter', Adapter)
    monkeypatch.setattr(tool_use_journal_execution, 'ToolUseJournalEnvironmentAdapter', Adapter)
    monkeypatch.setattr(ToolUseJournalKinematicTrajectoryPlayer, '_final_state', staticmethod(lambda runtime: request.world.robot_state))
    if path == 'scripted':
        world = snapshot(runtime)
    else:
        adapter = object.__new__(tool_use_journal_execution.ToolUseJournalExecutionAdapter)
        adapter.runtime = runtime
        world = adapter.world_snapshot(request, SimpleNamespace(final_robot_state=None, metadata={}))
    assert world.metadata['attached_object_transforms']['tool']['position_in_reference_m'] == pytest.approx(position)
    assert world.objects['tool']['pose'] == object_pose
    assert world.metadata['attachment_planning_geometry']['tool']['runtime_rest_transform_unchanged']
    assert asdict(runtime.attachment) == original
