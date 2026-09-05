"""Explicit, scene-scoped dispatch. Importing this module creates no simulator."""
from dataclasses import dataclass
from importlib import import_module

from tuj.m5_motion.task_semantics import is_acquire_task


@dataclass(frozen=True)
class GraspEntry:
    object_id: str
    environment: str
    ee: str
    driver: str

    def recipe(self):
        module = import_module(f"{__package__}.objects.{self.object_id}")
        if self.driver == "catalog":
            return getattr(module, self.object_id + "_recipe")()
        return getattr(module, self.object_id.title() + "Recipe")()

    def function(self):
        module = import_module(f"{__package__}.objects.{self.object_id}")
        return getattr(module, "grasp_" + self.object_id)


ENTRIES = tuple(GraspEntry(*row) for row in (
    ("plate", "C1_1_LegoSweep", "2F", "plate"),
    ("bottle", "C1_2_DoughFlatten", "3F", "bottle"),
    ("spatula", "C1_2_DoughFlatten", "3F", "spatula"),
    ("spoon", "C1_2_DoughFlatten", "2F", "spoon"),
    ("apple", "C2_1_ObjectSorting", "3F", "catalog"),
    ("bread", "C2_1_ObjectSorting", "3F", "catalog"),
    ("mug", "C2_1_ObjectSorting", "3F", "catalog"),
    ("knife", "C2_2_SandwichAssembly", "2F", "catalog"),
    ("rolling_pin", "C4_2_DiagonalFitPacking", "2F", "catalog"),
    ("baguette", "C4_2_DiagonalFitPacking", "2F", "catalog"),
    ("whisk", "C4_2_DiagonalFitPacking", "2F", "catalog"),
    ("cereal", "C4_2_DiagonalFitPacking", "2F", "catalog"),
    ("milk", "C4_2_DiagonalFitPacking", "3F", "catalog"),
    ("lid", "C4_2_DiagonalFitPacking", "vac", "catalog"),
))

# Exact M1 identifiers only; no substring or fuzzy matching of object names.
ALIASES = {f"obj_{e.object_id}_{e.object_id}": e.object_id for e in ENTRIES}

# Experimental entries are routed through their object function and remain
# visibly distinct from validated entries in every execution artifact. A failed
# experimental grasp still stops the task; it is never replaced by an LLM grasp.
EXPERIMENTAL_INTEGRATION = {
    "plate": "2F recipe is connected for C1_1; current-main physical validation is pending",
}
PENDING_INTEGRATION = {}
ENABLED_ENTRIES = tuple(entry for entry in ENTRIES if entry.object_id not in PENDING_INTEGRATION)


def integration_status(entry):
    if entry.object_id in PENDING_INTEGRATION:
        return "BLOCKED"
    if entry.object_id in EXPERIMENTAL_INTEGRATION:
        return "EXPERIMENTAL"
    return "VALIDATED"


class ScriptedGraspUnavailable(RuntimeError):
    pass


def resolve(request):
    """Return a validated recipe for a supported acquire, otherwise None.

    A known object with a different EE is an input error, not permission to
    silently execute a different grasp. Unsupported objects retain M5 routing.
    """
    if not is_acquire_task(request.task):
        return None
    task = request.task
    target = task.goal.target_object_id
    if target is None and len(task.target_ids) == 1:
        target = task.target_ids[0]
    if target is None:
        target = task.tool
    target = ALIASES.get(target, target)
    environment = request.world.metadata.get("environment_name")
    for entry in ENTRIES:
        if (entry.object_id, entry.environment) == (target, environment):
            if task.ee != entry.ee:
                raise ValueError(f"SCRIPTED_GRASP_EE_MISMATCH: {target} requires {entry.ee}, got {task.ee}")
            if target in PENDING_INTEGRATION:
                raise ScriptedGraspUnavailable(f"SCRIPTED_GRASP_NOT_VALIDATED: {target}: {PENDING_INTEGRATION[target]}")
            return entry
    return None
