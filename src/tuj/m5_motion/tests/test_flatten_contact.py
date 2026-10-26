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
                np.testing.assert_allclose(lowest, .02 + key.metadata['trial_press_height_m'], atol=1e-10)


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
            centers = np.asarray(candidate.metadata['contact_centers_world_xy_m'])
            assert np.linalg.norm(centers - face_world[:2], axis=1).min() < 1e-9
            # The narrow handle midpoint must no longer occupy the target center.
            assert np.linalg.norm(tool[:2, 3] - face_world[:2]) > .07


def test_uniform_face_keeps_center_and_degenerate_face_is_rejected():
    import pytest
    from tuj.m5_motion.flatten_contact import contact_face_center
    for points, normal in [(list(product([-.03, .03], [-.1, .1], [-.005, .005])), 2),
                           (list(product([-.005, .005], [-.1, .1], [-.03, .03])), 0)]:
        np.testing.assert_allclose(contact_face_center(points, normal), [0., 0., 0.], atol=1e-8)
    with pytest.raises(ValueError, match='nondegenerate'):
        contact_face_center([[0., 0., 0.], [0., .1, 0.], [0., .2, 0.]], 2)



def test_contact_cells_cover_observed_footprint_and_translate_with_target():
    from tuj.m5_motion.flatten_contact import compression_contact_centers
    lower, upper, radius = np.array([.1, -.2, .02]), np.array([.161, -.139, .08]), .023
    centers = np.array(compression_contact_centers(lower, upper, radius))
    assert len(centers) > 1
    samples = np.array(list(product(np.linspace(lower[0], upper[0], 21),
                                   np.linspace(lower[1], upper[1], 21))))
    distances = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
    assert np.max(np.min(distances, axis=1)) <= radius
    shift = np.array([.7, -.9, .3])
    translated = np.array(compression_contact_centers(lower + shift, upper + shift, radius))
    np.testing.assert_allclose(translated, centers + shift[:2])


def test_coverage_progresses_to_exact_original_trial_depth_without_changing_criteria():
    r = request()
    before_config = deepcopy(r.world.objects['block']['flattening_configuration'])
    before_contact = r.task.contact.model_dump()
    artifact = FlattenContactProvider().generate(r)
    for candidate in artifact.candidates:
        heights = candidate.metadata['trial_heights_m']
        assert all(a > b for a, b in zip(heights, heights[1:]))
        assert heights[-1] == compression_trial_height(.06, CONFIG)
        centers = candidate.metadata['contact_centers_world_xy_m']
        presses = [k for k in candidate.keyframes if k.metadata['flatten_stage'] == 'press']
        assert len(presses) == len(centers) * len(heights)
        for height in heights:
            assert sum(k.metadata['trial_press_height_m'] == height for k in presses) == len(centers)
        assert candidate.keyframes[-1].metadata['flatten_stage'] == 'unload'
        assert candidate.keyframes[-1].metadata['hold_duration_after_s'] == CONFIG['unloaded_observation_s']
    assert r.world.objects['block']['flattening_configuration'] == before_config
    assert r.task.contact.model_dump() == before_contact
