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
