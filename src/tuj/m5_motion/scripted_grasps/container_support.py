"""Fresh contact-force support paths for layered container placements."""
import hashlib
import json
import math
from tuj.evaluation.task_goals import (
    SUPPORT_TOLERANCE_M, _containment, _released, evaluate_placement)
from .placement_contacts import placement_contacts


def world_digest(world):
    return hashlib.sha256(json.dumps(world, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def measure_container_support(runtime, world):
    payload = world.model_dump(mode='json')
    objects = payload['objects']
    regions = [name for name, record in objects.items()
               if record.get('packing_metadata', {}).get('kind') == 'CONTAINER']
    pairs = []
    for region in regions:
        packed = [name for name, record in objects.items()
                  if record.get('packing_metadata', {}).get('kind') == 'PACKABLE_OBJECT'
                  and _containment(objects, name, region, vertical=True)[0]
                  and _released(payload, [name])[0]]
        pairs.extend({'target': target, 'region': support}
                     for target in packed for support in [region, *packed] if target != support)
    return {'world_sha256': world_digest(payload),
            'basis': 'CURRENT_RUNTIME_CONTACT_FORCES',
            'pairs': placement_contacts(runtime, world, pairs)} if pairs else None


def contact_support_path(world, target, region, evidence):
    """Contacts need measured upward load; floating peers and robot support cannot ground this path."""
    if (not isinstance(evidence, dict)
            or evidence.get('world_sha256') != world_digest(world)
            or evidence.get('basis') != 'CURRENT_RUNTIME_CONTACT_FORCES'
            or not isinstance(evidence.get('pairs'), list)):
        return None, {'reason': 'missing or stale measured container support contacts'}
    objects = world['objects']
    eligible = {name for name, record in objects.items()
                if record.get('packing_metadata', {}).get('kind') == 'PACKABLE_OBJECT'
                and _containment(objects, name, region, vertical=True)[0]
                and _released(world, [name])[0]}
    edges = {}
    for pair in evidence['pairs']:
        a, b = pair['target'], pair['region']
        if a not in eligible or b not in eligible | {region}:
            continue
        contacts = [c for c in pair.get('contacts', [])
                    if all(math.isfinite(c[k]) for k in ('distance_m', 'upward_force_n'))
                    and -SUPPORT_TOLERANCE_M <= c['distance_m'] <= 0.
                    and c['upward_force_n'] > 0.]
        if contacts:
            edges.setdefault(a, []).append((b, contacts))
    pending = [(target, [target], [])]
    while pending:
        name, path, contacts = pending.pop()
        for base, measured in edges.get(name, []):
            if base in path:
                continue
            edge = {'target': name, 'support': base, 'contacts': measured}
            if base == region:
                return True, {'support_path': path + [base], 'edges': contacts + [edge],
                              'basis': evidence['basis'], 'tolerance_m': SUPPORT_TOLERANCE_M}
            pending.append((base, path + [base], contacts + [edge]))
    return False, {'reason': 'no measured upward contact-force path to container',
                   'target': target, 'support': region, 'tolerance_m': SUPPORT_TOLERANCE_M}


def evaluate_preserved_placement(world, target, region, evidence=None):
    result = evaluate_placement(world, target, region)
    if world['objects'].get(region, {}).get('packing_metadata', {}).get('kind') != 'CONTAINER':
        return result
    # Explicit 3D containers permit layers. Require 3D inclusion and a measured
    # upward support path, rather than geometric contact with the bottom only.
    for check in result['checks']:
        try:
            if check['name'] == 'contained':
                check['passed'], check['evidence'] = _containment(world['objects'], target, region, vertical=True)
            elif check['name'] == 'supported':
                check['passed'], check['evidence'] = contact_support_path(world, target, region, evidence)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            check['passed'], check['evidence'] = None, {'reason': f'insufficient evidence: {error}'}
    values = [c['passed'] for c in result['checks']]
    result['passed'] = False if False in values else None if None in values else True
    return result
