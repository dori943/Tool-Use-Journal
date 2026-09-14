"""Bind packing orientation probes to the ordinary transport collision policy."""
from functools import partial


def needs_packing_transport_filter(request, retention):
    from .transport import _is_transport
    if retention is None or not _is_transport(request.task):
        return False
    region = request.world.objects.get(request.task.goal.target_region_id, {})
    record = request.world.objects.get(retention.entry.object_id, {})
    metadata = record.get('packing_metadata', {})
    return (region.get('packing_metadata', {}).get('kind') == 'CONTAINER'
            and metadata.get('stable_face_policy') != 'MINIMUM_HEIGHT'
            and bool(metadata.get('orientation_candidates')))


def bind_packing_transport_filter(request, retention, planner):
    """Rebuild the request binding; never cache a prior scene or hand state."""
    factory = planner.collision_context_factory
    factory._validate_environment(request)
    active = factory._active_ee(request)
    if active != request.task.ee:
        raise ValueError('PACKING_FILTER_ACTIVE_EE_MISMATCH')
    context = factory._base_context(request, active_ee=active)
    context = factory._bind_generic_touch_policy(request, context)
    registry = factory.compiler.build_collision_registry(
        {context.context_id: context},
        collision_margin_m=request.constraints.collision_margin_m,
        allowed_collision_pairs=request.constraints.allowed_collision_pairs,
        default_active_ee=active)
    retention.packing_transport_state_check = partial(registry.check, context=context)
    request.task.metadata['packing_transport_collision_binding'] = {
        'context_id': context.context_id,
        'collision_model_version': context.collision_model_version,
        'scene_state_id': context.scene_state_id,
        'collision_margin_m': request.constraints.collision_margin_m,
        'source': 'OFFICIAL_TRANSPORT_COLLISION_REGISTRY'}
