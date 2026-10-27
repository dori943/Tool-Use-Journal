from copy import deepcopy
from itertools import product
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.tests.test_contact_manipulation import _request
from tuj.m5_motion.flatten_contact import FlattenContactProvider, compression_trial_height, flattening_outcome
from tuj.m5_motion.geometry import RelativePoseResolver


CONFIG = {'material': {'shear_modulus_pa': 500, 'yield_stress_pa': 1000},
          'height_reduction_fraction': [.3, .5], 'unloaded_observation_s': 1.,
          'volume_tolerance_fraction': .01}


def box_faces(points):
    corners = np.asarray(points)
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    return np.asarray([corners[list(face)] for a, b, c, d in quads
                       for face in ((a, b, c), (a, c, d))])


def set_faces(record, triangles):
    record['collision_surface_triangles_m'] = np.asarray(triangles).tolist()
    record['collision_surface_source'] = 'COMPILED_MUJOCO_COLLISION_FACES'


def request():
    r = _request()
    r.task.contact.primitive = 'flatten'
    r.world.objects['plate']['collision_points_m'] = list(product([-.03, .03], [-.03, .03], [-.1, .1]))
    set_faces(r.world.objects['plate'], box_faces(r.world.objects['plate']['collision_points_m']))
    r.world.metadata['attached_object_transforms'] = {'plate': {
        'position_in_reference_m': [.01, -.02, -.08],
        'orientation_in_reference_xyzw': Rotation.from_rotvec([.1, .2, -.1]).as_quat().tolist()}}
    r.world.objects['block'].update(deformable_metrics={
        'bounds_min_m': [.1, .1, .02], 'bounds_max_m': [.16, .16, .08], 'initial_height_m': .06},
        flattening_configuration=deepcopy(CONFIG))
    return r


def test_generated_contact_uses_tool_surface_and_actual_grasp_transform():
    r = request()
    artifact = FlattenContactProvider().generate(r)
    resolver = RelativePoseResolver(r.world)
    raw = r.world.metadata['attached_object_transforms']['plate']
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_quat(raw['orientation_in_reference_xyzw']).as_matrix()
    grasp[:3, 3] = raw['position_in_reference_m']
    points = np.asarray(r.world.objects['plate']['collision_points_m'])
    for candidate in artifact.candidates:
        for key in candidate.keyframes:
            pose = resolver.resolve(key)
            eef = np.eye(4)
            eef[:3, :3] = Rotation.from_quat(pose.orientation_xyzw).as_matrix()
            eef[:3, 3] = pose.position_m
            tool = eef @ grasp
            lowest = (points @ tool[:3, :3].T + tool[:3, 3])[:, 2].min()
            stage = key.metadata['flatten_stage']
            if stage == 'engage':
                np.testing.assert_allclose(lowest, .08, atol=1e-10)
            if stage == 'press':
                np.testing.assert_allclose(lowest, .02 + compression_trial_height(.06, CONFIG), atol=1e-10)


def test_success_requires_measured_contact_unloading_and_residual_shape():
    record = {'flattening_configuration': deepcopy(CONFIG), 'deformable_metrics': {
        'height_m': .036, 'initial_height_m': .06, 'volume_ratio': .998,
        'tool_contact': {'start_height_m': .048, 'impulse_ns': 1., 'current_force_n': 0.,
                         'peak_force_n': 4., 'force_limit_n': 10., 'unloaded_duration_s': 1.1}}}
    assert flattening_outcome(record)[0]
    for field, bad in [('impulse_ns', 0), ('unloaded_duration_s', .1), ('peak_force_n', 11), ('current_force_n', 1)]:
        invalid = deepcopy(record)
        invalid['deformable_metrics']['tool_contact'][field] = bad
        assert not flattening_outcome(invalid)[0]
    for field, bad in [('height_m', .055), ('volume_ratio', .8)]:
        invalid = deepcopy(record)
        invalid['deformable_metrics'][field] = bad
        assert not flattening_outcome(invalid)[0]

def test_ee_material_contact_requires_opt_in_and_is_scoped_to_contact_edges():
    from tuj.m5_motion.schema import CollisionContext
    from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalCollisionContextFactory
    r = request()
    base = CollisionContext(context_id='base', active_ee=r.task.ee, collision_model_version='test')
    factory = object.__new__(ToolUseJournalCollisionContextFactory)
    artifact = FlattenContactProvider().generate(r)
    _, contexts = factory._bind_default(r, artifact, base)
    assert len(contexts) == 1
    r.world.objects['block']['flattening_configuration']['allow_active_ee_contact'] = True
    bound, contexts = factory._bind_default(r, artifact, base)
    expected = tuple(sorted((r.task.ee, 'block')))
    assert base.allowed_collision_pairs == []
    for candidate in bound.candidates:
        for key in candidate.keyframes:
            pairs = contexts[key.collision_context_id].allowed_collision_pairs
            assert (expected in pairs) == (key.metadata['flatten_stage'] in {'engage', 'press', 'unload'})
            assert all('table' not in pair and 'arm' not in pair for pair in pairs)


def test_ee_load_prevents_unloaded_success_and_combined_overload_fails():
    record = {'flattening_configuration': deepcopy(CONFIG), 'deformable_metrics': {
        'height_m': .036, 'initial_height_m': .06, 'volume_ratio': .998,
        'tool_contact': {'start_height_m': .048, 'impulse_ns': 1., 'current_force_n': 0.,
                         'peak_force_n': 4., 'force_limit_n': 10., 'unloaded_duration_s': 1.1,
                         'combined_current_force_n': 1., 'combined_peak_force_n': 5.}}}
    assert not flattening_outcome(record)[0]
    record['deformable_metrics']['tool_contact'].update(combined_current_force_n=0., combined_peak_force_n=11.)
    assert not flattening_outcome(record)[0]


def test_asymmetric_paddle_uses_broad_face_instead_of_handle_center():
    from tuj.m5_motion.flatten_contact import contact_face_center
    handle = list(product([-.008, .008], [-.125, .05], [-.003, .003]))
    paddle = list(product([-.024, .024], [.07, .125], [-.003, .003]))
    points = np.asarray(handle + paddle)
    face = contact_face_center(points, normal_axis=2)
    assert .07 < face[1] < .125
    np.testing.assert_allclose(face[[0, 2]], [0., 0.], atol=1e-9)
    # Vertex ordering and a local-frame translation must not choose a new patch.
    shift = np.array([.3, -.4, .2])
    np.testing.assert_allclose(contact_face_center(points[::-1] + shift, 2), face + shift, atol=1e-8)

    r = request()
    r.world.objects['plate']['collision_points_m'] = points.tolist()
    set_faces(r.world.objects['plate'], np.concatenate((box_faces(handle), box_faces(paddle))))
    artifact = FlattenContactProvider().generate(r)
    resolver = RelativePoseResolver(r.world)
    raw = r.world.metadata['attached_object_transforms']['plate']
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_quat(raw['orientation_in_reference_xyzw']).as_matrix()
    grasp[:3, 3] = raw['position_in_reference_m']
    for candidate in artifact.candidates:
        for key in candidate.keyframes:
            pose = resolver.resolve(key)
            eef = np.eye(4)
            eef[:3, :3] = Rotation.from_quat(pose.orientation_xyzw).as_matrix()
            eef[:3, 3] = pose.position_m
            tool = eef @ grasp
            measured_face = np.asarray(candidate.metadata['contact_face_center_local_m'])
            face_world = tool[:3, :3] @ measured_face + tool[:3, 3]
            np.testing.assert_allclose(face_world[:2], [.13, .13], atol=1e-9)
            # The narrow handle midpoint must no longer occupy the target center.
            assert np.linalg.norm(tool[:2, 3] - [.13, .13]) > .07


def test_uniform_face_keeps_center_and_degenerate_face_is_rejected():
    import pytest
    from tuj.m5_motion.flatten_contact import contact_face_center
    for points, normal in [(list(product([-.03, .03], [-.1, .1], [-.005, .005])), 2),
                           (list(product([-.005, .005], [-.1, .1], [-.03, .03])), 0)]:
        np.testing.assert_allclose(contact_face_center(points, normal), [0., 0., 0.], atol=1e-8)
    with pytest.raises(ValueError, match='nondegenerate'):
        contact_face_center([[0., 0., 0.], [0., .1, 0.], [0., .2, 0.]], 2)


def tool_pose_for_key(request, key):
    pose = RelativePoseResolver(request.world).resolve(key)
    eef = np.eye(4)
    eef[:3, :3] = Rotation.from_quat(pose.orientation_xyzw).as_matrix()
    eef[:3, 3] = pose.position_m
    raw = request.world.metadata['attached_object_transforms']['plate']
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_quat(raw['orientation_in_reference_xyzw']).as_matrix()
    grasp[:3, 3] = raw['position_in_reference_m']
    return eef @ grasp


def test_tilted_actual_face_sets_both_contact_height_and_normal_at_any_altitude():
    tilt = Rotation.from_euler('x', .15).as_matrix()
    handle = np.asarray(list(product([-.008, .008], [-.125, .05], [-.003, .003])))
    paddle = np.asarray(list(product([-.024, .024], [.07, .125], [-.002, .002]))) @ tilt.T
    for altitude in (0., 2.3):
        r = request()
        record = r.world.objects['plate']
        record['collision_points_m'] = np.concatenate((handle, paddle)).tolist()
        set_faces(record, np.concatenate((box_faces(handle), box_faces(paddle))))
        metrics = r.world.objects['block']['deformable_metrics']
        metrics['bounds_min_m'][2] += altitude
        metrics['bounds_max_m'][2] += altitude
        before_config = deepcopy(r.world.objects['block']['flattening_configuration'])
        for candidate in FlattenContactProvider().generate(r).candidates:
            face = np.asarray(candidate.metadata['contact_face_center_local_m'])
            normal = np.asarray(candidate.metadata['contact_face_normal_local'])
            np.testing.assert_allclose(abs(normal @ tilt[:, 2]), 1., atol=1e-10)
            np.testing.assert_allclose(abs(face @ tilt[:, 2]), .002, atol=1e-10)
            for key in candidate.keyframes:
                if key.metadata['flatten_stage'] not in ('engage', 'press'):
                    continue
                tool = tool_pose_for_key(r, key)
                expected_z = (altitude + .08 if key.metadata['flatten_stage'] == 'engage'
                              else altitude + .02 + compression_trial_height(.06, CONFIG))
                np.testing.assert_allclose(tool[:3, :3] @ face + tool[:3, 3],
                                           [.13, .13, expected_z], atol=1e-9)
                np.testing.assert_allclose(tool[:3, :3] @ normal, [0., 0., -1.], atol=1e-10)
        assert r.world.objects['block']['flattening_configuration'] == before_config


def test_measured_faces_do_not_fill_holes_between_components():
    import pytest
    from tuj.m5_motion.flatten_contact import contact_face_center, measured_contact_face
    boxes = [np.asarray(list(product(x, y, [-.005, .005]))) for x, y in
             [([-.10, -.08], [-.10, .10]), ([.08, .10], [-.10, .10]),
              ([-.10, .10], [-.10, -.08])]]
    center = contact_face_center(np.concatenate(boxes), 2)
    np.testing.assert_allclose(center, [0., 0., 0.], atol=1e-9)
    with pytest.raises(ValueError, match='no measured collision face'):
        measured_contact_face(np.concatenate([box_faces(box) for box in boxes]), center, 2, -1.)
    r = request()
    r.world.objects['plate'].pop('collision_surface_triangles_m')
    with pytest.raises(ValueError, match='measured collision surface triangles'):
        FlattenContactProvider().generate(r)


def test_collision_face_snapshot_uses_nested_geometry_and_excludes_visuals():
    import mujoco
    from tuj.m5_motion.tool_use_journal import _object_record
    from tuj.m5_motion.scene_context import spatial_record
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="object" pos="1 2 3" euler="0 0 35"><freejoint/>
        <body pos=".02 .03 .04" euler="20 0 0">
          <geom type="box" size=".1 .13 .006"/>
          <geom type="box" size="1 1 1" contype="0" conaffinity="0"/>
        </body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    record = _object_record(model, data, 'object', model.body('object').id)
    triangles = np.asarray(record['collision_surface_triangles_m'])
    assert triangles.shape == (12, 3, 3)
    expected = np.asarray(list(product([-.1, .1], [-.13, .13], [-.006, .006])))
    expected = expected @ Rotation.from_euler('x', 20, degrees=True).as_matrix().T + [.02, .03, .04]
    for point in triangles.reshape(-1, 3):
        assert np.min(np.linalg.norm(expected - point, axis=1)) < 1e-12
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    assert 'collision_surface_triangles_m' not in spatial_record(record)


def test_actual_spatula_face_height_and_normal_are_measured_not_axis_assumed():
    import mujoco
    from tuj.m5_motion.tool_use_journal import _object_record
    from environments.objects.xml_asset import make_resolved_object_xml
    from environments.objects.spatula_object import SPATULA_ASSET_DIR
    model = mujoco.MjModel.from_xml_path(make_resolved_object_xml(SPATULA_ASSET_DIR))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    r = request()
    r.world.objects['plate'] = _object_record(model, data, 'plate', model.body('object').id)
    for candidate in FlattenContactProvider().generate(r).candidates:
        face = np.asarray(candidate.metadata['contact_face_center_local_m'])
        normal = np.asarray(candidate.metadata['contact_face_normal_local'])
        # The full tool silhouette includes beveled panels. Its disk center
        # (y=.101716) is not the center of this exposed planar load face.
        assert face[1] == pytest.approx(.089654283, abs=1e-8)
        assert abs(face[2]) == pytest.approx(.001961371, abs=1e-8)
        assert np.degrees(np.arccos(abs(normal[2]))) == pytest.approx(1.90071, abs=1e-4)
        assert candidate.metadata['legacy_contact_face_height_error_m'] == pytest.approx(.000592292, abs=1e-8)
        assert candidate.metadata['contact_planar_patch_radius_m'] == pytest.approx(.017857544, abs=1e-7)
        press = next(key for key in candidate.keyframes if key.metadata['flatten_stage'] == 'press')
        tool = tool_pose_for_key(r, press)
        np.testing.assert_allclose(tool[:3, :3] @ normal, [0., 0., -1.], atol=1e-9)
        np.testing.assert_allclose((tool[:3, :3] @ face + tool[:3, 3])[2],
                                   .02 + compression_trial_height(.06, CONFIG), atol=1e-9)


def test_planar_patch_center_excludes_disconnected_coplanar_handle_and_is_rigid_invariant():
    from tuj.m5_motion.flatten_contact import planar_contact_center
    paddle = np.array([[[-.02, 0., 0.], [.02, 0., 0.], [.02, .06, 0.]],
                       [[-.02, 0., 0.], [.02, .06, 0.], [-.02, .06, 0.]]])
    handle = paddle + [0., -.2, 0.]
    triangles = np.concatenate((handle, paddle))
    for rotation, shift in [(np.eye(3), np.zeros(3)),
                            (Rotation.from_rotvec([.3, -.4, .8]).as_matrix(), np.array([2., -3., 4.]))]:
        transformed = triangles @ rotation.T + shift
        center, radius = planar_contact_center(transformed[::-1],
            rotation @ [0., .04, 0.] + shift, rotation @ [0., 0., -1.])
        np.testing.assert_allclose(center, rotation @ [0., .03, 0.] + shift, atol=1e-10)
        assert radius == pytest.approx(.02, abs=1e-10)


def test_planar_patch_hull_cannot_fill_a_concavity():
    from tuj.m5_motion.flatten_contact import planar_contact_center
    # Three unit squares joined by full edges make a connected L, not a rectangle.
    vertices = np.array([[x, y, 0.] for x, y in [(0, 0), (1, 0), (2, 0),
                                               (0, 1), (1, 1), (2, 1), (0, 2), (1, 2)]])
    faces = [(0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4), (3, 4, 7), (3, 7, 6)]
    with pytest.raises(ValueError, match='connected convex planar'):
        planar_contact_center(vertices[faces], np.array([.5, .5, 0.]), np.array([0., 0., -1.]))
