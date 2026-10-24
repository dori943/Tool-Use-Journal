"""Recheck completed placements against each measured live state.

No pose is restored or edited. A broken or unobservable relation stops the
sequence; it cannot silently be counted as a successful earlier subgoal.
"""
from tuj.evaluation.task_goals import evaluate_placement
from tuj.m5_motion.task_semantics import is_acquire_task, task_operation


class PlacementPreservationError(RuntimeError):
    def __init__(self, code, checks):
        self.checks = checks
        broken = [f"{c['target']} -> {c['region']}" for c in checks if c['passed'] is not True]
        super().__init__(f"{code}: {', '.join(broken)}")


class PlacementPreservation:
    def __init__(self):
        self.relations = {}

    def observe(self, task, world):
        """Called after every executed request, including grasp and EE swaps."""
        world = world.model_dump(mode='json') if hasattr(world, 'model_dump') else world
        moving = set(task.target_ids) if is_acquire_task(task) else set()
        retained = {target: region for target, region in self.relations.items()
                    if target not in moving}
        checks = [evaluate_placement(world, target, region)
                  for target, region in retained.items()]
        if any(c['passed'] is not True for c in checks):
            raise PlacementPreservationError('PLACEMENT_PRESERVATION_FAILED', checks)
        # Relocate/place promises containment. place_on/cover has a different
        # geometric meaning, and conceptual tool_rest is not a measured region.
        region = task.goal.target_region_id
        if task_operation(task) == 'PLACE' and region and region != 'tool_rest':
            placed = [evaluate_placement(world, target, region) for target in task.target_ids]
            checks.extend(placed)
            if any(c['passed'] is not True for c in placed):
                raise PlacementPreservationError('PLACEMENT_GOAL_NOT_SATISFIED', checks)
            retained.update({target: region for target in task.target_ids})
        self.relations = retained
        return checks
