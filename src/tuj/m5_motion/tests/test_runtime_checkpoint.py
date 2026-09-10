import json

import pytest

from tuj.m5_motion.runtime_checkpoint import (
    RuntimeCheckpointError, _attachment_from_payload, _attachment_payload,
)
from tuj.m5_motion.tool_use_journal_runtime import (
    AttachedObjectState, AttachmentMode, BreakableWeldConfig,
)


def payload(groups=()):
    state = AttachedObjectState(
        object_id='payload', free_joint_name='payload_free', reference_kind='site',
        reference_name='grip', position_in_reference_m=(0., 0., .01),
        rotation_in_reference=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
        attach_distance_m=.002, mode=AttachmentMode.BREAKABLE_WELD,
        breakable_weld=BreakableWeldConfig(required_contact_groups=groups),
    )
    return state, json.loads(json.dumps(_attachment_payload(state)))


@pytest.mark.parametrize('groups', [(), ('left_finger', 'right_finger')])
def test_breakable_attachment_survives_json_roundtrip(groups):
    state, raw = payload(groups)
    assert _attachment_from_payload(raw) == state


@pytest.mark.parametrize('groups', [['left', 'left'], 'left', [1], ['']])
def test_invalid_contact_groups_remain_rejected(groups):
    _, raw = payload()
    raw['breakable_weld']['required_contact_groups'] = groups
    with pytest.raises(RuntimeCheckpointError):
        _attachment_from_payload(raw)


def test_unknown_weld_fields_remain_rejected():
    _, raw = payload()
    raw['breakable_weld']['unexpected'] = 1
    with pytest.raises(RuntimeCheckpointError):
        _attachment_from_payload(raw)
