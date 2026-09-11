"""Expose scripted recipe compatibility to upstream task planning."""
from .registry import ALIASES, ENABLED_ENTRIES, ALTERNATIVE_ENTRIES


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
    changes=[]
    for subgoal in result.task_graph.subgoals:
        target=subgoal.tool_id
        if target is None and len(subgoal.target_ids)==1:
            target=subgoal.target_ids[0]
        object_id=ALIASES.get(target,target)
        entries=[e for e in ENABLED_ENTRIES + ALTERNATIVE_ENTRIES
            if e.environment==environment and e.object_id==object_id]
        if not entries:
            continue
        before=list(subgoal.feasible_ee)
        supported=list(dict.fromkeys(e.ee for e in entries))
        allowed=[ee for ee in before if ee in supported] if before else supported
        if not allowed:
            raise ValueError(
                f'SCRIPTED_GRASP_EE_INFEASIBLE: {subgoal.subgoal_id}: '
                f'{object_id} supports {supported}, grounded feasible EEs={before}'
            )
        subgoal.feasible_ee=allowed
        contract={
            'source':'scripted_grasp_registry',
            'environment':environment,
            'object_id':object_id,
            'supported_ee':supported,
            'selected_feasible_ee':allowed,
        }
        subgoal.action_parameters={
            **subgoal.action_parameters,
            'execution_compatibility':contract,
        }
        changes.append({
            'subgoal_id':subgoal.subgoal_id,
            'object_id':object_id,
            'grounded_feasible_ee':before,
            'scripted_feasible_ee':allowed,
            'source':'scripted_grasp_registry',
        })
    return result,changes
