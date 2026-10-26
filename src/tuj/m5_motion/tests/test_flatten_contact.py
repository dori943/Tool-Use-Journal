from copy import deepcopy
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation

from tuj.m5_motion.tests.test_contact_manipulation import _request
from tuj.m5_motion.flatten_contact import FlattenContactProvider, compression_trial_height, flattening_outcome
from tuj.m5_motion.geometry import RelativePoseResolver


CONFIG = {'material': {'shear_modulus_pa': 500, 'yield_stress_pa': 1000},
          'height_reduction_fraction': [.3, .5], 'unloaded_observation_s': 1.,
          'volume_tolerance_fraction': .01}


def request():
    r = _request()
    r.task.contact.primitive = 'flatten'
    r.world.objects['plate']['collision_points_m'] = list(product([-.03, .03], [-.03, .03], [-.1, .1]))
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
            face_world = tool[:3, :3] @ face + tool[:3, 3]
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
