"""Short goal corrections must be real side pushes, not validation exemptions."""
from __future__ import annotations

import numpy as np
import pytest

from tuj.m5_motion.contact_keyframe_validation import (
    ContactKeyframeGeometryError,
    _retarget_tcp_world_xy,
    validate_sweep_keyframe_strategy,
)
from tuj.m5_motion.geometry import RelativePoseResolver
from tuj.m5_motion.native_side_push import generate_native_side_push_keyframes
from tuj.m5_motion.tests.test_native_side_push import _attitude, _request


def _near_goal():
    request = _request()
    request.world.objects['collect_zone']['pose']['orientation_xyzw'] = [0., 0., 0., 1.]
    target = request.world.objects['block_a']
    target['pose']['orientation_xyzw'] = [0., 0., 0., 1.]
    # Four millimetres outside the region, seven from its existing 3 mm inset.
    target['pose']['position_m'] = [.056, .1, .8058]
    frames = generate_native_side_push_keyframes(request, eef_orientation_xyzw=_attitude(request))
    return request, frames


def test_short_side_push_to_existing_goal_inset_is_valid():
    request, frames = _near_goal()
    before = request.model_dump_json()
    resolver = RelativePoseResolver(request.world)
    start, contact, end = [np.asarray(resolver.resolve(f).position_m) for f in frames[1:4]]
    assert np.linalg.norm(end[:2] - start[:2]) == pytest.approx(.067)
    assert np.linalg.norm(end[:2] - contact[:2]) == pytest.approx(.007)
    validate_sweep_keyframe_strategy(request, frames)
    assert request.model_dump_json() == before


@pytest.mark.parametrize('extra_x', [-.007, .1])
def test_native_marker_does_not_authorize_collapsed_or_forged_end(extra_x):
    request, frames = _near_goal()
    resolver = RelativePoseResolver(request.world)
    end = resolver.resolve(frames[3])
    frames[3] = _retarget_tcp_world_xy(
        frames[3], resolver=resolver,
        target_xy=np.asarray(end.position_m[:2]) + [extra_x, 0.],
        target_z=end.position_m[2], packing_xyzw=end.orientation_xyzw,
        metadata_flag='native_side_push_geometry',
    )
    assert frames[3] is not None
    with pytest.raises(ContactKeyframeGeometryError):
        validate_sweep_keyframe_strategy(request, frames)
