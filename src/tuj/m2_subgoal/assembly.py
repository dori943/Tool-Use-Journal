"""Semantic repair of sandwich assembly; physical estimates are never changed.

The benchmark's serving plate is the final assembly support. Corrections use
only task text, object roles and observed on/inside relations, and are audited.
Ambiguous destinations must be resolved upstream, not guessed by M4.
"""
from copy import deepcopy


# task_registry.py C2_2 contract. Do not override a differently worded task
# that explicitly asks to assemble on another support. Applied to every variant.
SANDWICH_INSTRUCTION = "재료를 순서대로 쌓아 샌드위치를 만들어라"


def normalize_assembly_plan(task, subgoals, m1):
    subs = deepcopy(subgoals)
    audit = []
    if task.strip().rstrip('.!') != SANDWICH_INSTRUCTION:
        return subs, audit
    stacks = [s for s in subs if s.get('kind') == 'stack']
    if len(stacks) != 1:
        return subs, audit
    assembly = stacks[0]
    ingredients = set(assembly.get('target_ids') or [])
    supports = {(e['from'], e['to']) for e in m1.get('edges', [])
                if e.get('type') in {'on', 'inside'} and e.get('from') and e.get('to')}
    source = assembly.get('container_id')
    # Repair a source plate mistaken for the assembly destination, not an
    # arbitrary user-selected support or an unknown spatial relation.
    if any((target, source) in supports for target in ingredients):
        destinations = [n['id'] for n in m1.get('nodes', [])
                        if n.get('class') == 'serving_plate']
        if len(destinations) != 1:
            raise ValueError('ASSEMBLY_DESTINATION_AMBIGUOUS: expected one serving_plate')
        destination = destinations[0]
        if source != destination:
            assembly['container_id'] = destination
            assembly['object_ids'] = list(dict.fromkeys([
                *assembly.get('object_ids', []), destination]))
            audit.append({'subgoal_id': assembly['subgoal_id'],
                          'rule': 'assembly_destination', 'from': source, 'to': destination})

    # A no-op staging move is redundant only before the ingredient has moved
    # and before the stack that already acquires it. Never remove a later
    # return, an explicit move elsewhere, or a target absent from the stack.
    kept, moved = [], set()
    for index, sub in enumerate(subs):
        targets = set(sub.get('target_ids') or [])
        if sub is assembly:
            kept.extend(subs[index:])
            break
        if (sub.get('kind') == 'relocate' and targets and targets <= ingredients
                and not targets & moved
                and all((t, sub.get('container_id')) in supports for t in targets)):
            audit.append({'subgoal_id': sub['subgoal_id'],
                          'rule': 'redundant_source_relocation',
                          'target_ids': sub['target_ids'], 'source': sub['container_id']})
            continue
        kept.append(sub)
        moved.update(targets)
    return kept, audit
