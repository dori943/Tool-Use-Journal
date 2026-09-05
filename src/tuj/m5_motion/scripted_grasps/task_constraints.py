"""Expose each scripted recipe's EE requirement to upstream task planning."""
from .registry import ALIASES, ENTRIES, ALTERNATIVE_ENTRIES


def scene_id_aliases(scene):
    """Use M1's declared full class prefix, preserving underscores in names."""
    aliases = {}
    for node in scene.get('nodes', []):
        identifier, kind = node.get('id'), node.get('class')
        if not isinstance(identifier, str) or not isinstance(kind, str):
            continue
        prefix = 'obj_' + kind + '_'
        if identifier.startswith(prefix) and len(identifier) > len(prefix):
            aliases[identifier] = identifier[len(prefix):]
    return aliases


def constrain_task_request(request, environment):
    result=request.model_copy(deep=True)
    entries={e.object_id:e for e in ENTRIES if e.environment==environment}
    changes=[]
    for subgoal in result.task_graph.subgoals:
        target=subgoal.tool_id
        if target is None and len(subgoal.target_ids)==1:
            target=subgoal.target_ids[0]
        entry=entries.get(ALIASES.get(target,target))
        if entry is None:
            continue
        before=list(subgoal.feasible_ee)
        # Preserve a grounded, explicitly selected alternative (e.g. plate vac).
        # When the default EE remains feasible, existing task selection is kept.
        if before and entry.ee not in before:
            alternatives=[e for e in ALTERNATIVE_ENTRIES
                if e.environment==environment and e.object_id==entry.object_id and e.ee in before]
            if alternatives:
                entry=alternatives[0]
        if before and entry.ee not in before:
            raise ValueError(f'SCRIPTED_GRASP_EE_INFEASIBLE: {subgoal.subgoal_id}: {entry.object_id} requires {entry.ee}, grounded feasible EEs={before}')
        subgoal.feasible_ee=[entry.ee]
        changes.append({'subgoal_id':subgoal.subgoal_id,'object_id':entry.object_id,
            'grounded_feasible_ee':before,'scripted_feasible_ee':[entry.ee]})
    return result,changes
