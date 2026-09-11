"""Explicit, scene-scoped dispatch. Importing this module creates no simulator."""
from dataclasses import dataclass, replace
from importlib import import_module

from tuj.m5_motion.task_semantics import is_acquire_task


@dataclass(frozen=True)
class GraspEntry:
    object_id: str
    environment: str
    ee: str
    driver: str
    module_name: str | None = None
    # When set, MuJoCo body / attachment id for multi-instance scenes
    # (``plate_b``). Recipe modules still key off ``object_id`` (``plate``).
    body_object_id: str | None = None
    recipe_name: str | None = None

    @property
    def scene_object_id(self) -> str:
        return self.body_object_id or self.object_id

    def recipe(self):
        name = self.module_name or self.object_id
        module = import_module(f"{__package__}.objects.{name}")
        if self.driver == "catalog":
            return getattr(module, self.recipe_name or name + "_recipe")()
        if self.driver == "spoon":
            asset = "c3_2" if self.environment.startswith("C3_2") else "default"
            return module.spoon_recipe(ee_id=self.ee, asset=asset)
        return getattr(module, self.object_id.title() + "Recipe")()

    def function(self):
        name = self.module_name or self.object_id
        module = import_module(f"{__package__}.objects.{name}")
        return getattr(module, "grasp_" + name)


ENTRIES = tuple(GraspEntry(*row) for row in (
    ("plate", "C1_1_LegoSweep", "2F", "plate"),
    ("bottle", "C1_2_DoughFlatten", "3F", "bottle"),
    ("spatula", "C1_2_DoughFlatten", "3F", "spatula"),
    ("spoon", "C1_2_DoughFlatten", "2F", "spoon"),
    # 0909: C3_1 also picks the spoon with the 2F gripper. Without an entry for
    # this environment the generic path plans the grasp, and its pose put the
    # object 10.6 cm from the grip site: the fingers closed on air (0 contacts,
    # 0.08 mm lift against a 50 mm requirement).
    ("spoon", "C3_1_ObjectSorting", "2F", "spoon"),
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
    # C3_2 breakfast tray: type-keyed recipes; *_a/*_b resolve via instance
    # suffix → type, with body_object_id = request target.
    ("plate", "C3_2_BreakfastTrayPreparation", "vac", "catalog", "plate_vac"),
    ("fork", "C3_2_BreakfastTrayPreparation", "3F", "catalog"),
    ("bread", "C3_2_BreakfastTrayPreparation", "vac", "catalog", "bread_vac"),
    ("fruit", "C3_2_BreakfastTrayPreparation", "3F", "catalog"),
    ("spoon", "C3_2_BreakfastTrayPreparation", "3F", "spoon"),
    ("mug", "C3_2_BreakfastTrayPreparation", "3F", "catalog", "mug_c3_2"),
))

# Optional alternatives remain an explicit extension point. The repository does
# not select object- or scene-specific alternatives in the generic M5 path.
ALTERNATIVE_ENTRIES: tuple[GraspEntry, ...] = ()

# Explicit validator-only feasibility probes. They do not alter M4 choices or
# the normal M5 registry surface.
VALIDATOR_EXPERIMENTAL_ENTRIES: tuple[GraspEntry, ...] = (
    GraspEntry("bread", "C3_2_BreakfastTrayPreparation", "2F", "catalog",
               "bread", recipe_name="bread_2f_c3_2_recipe"),
    GraspEntry("spoon", "C3_2_BreakfastTrayPreparation", "2F", "spoon"),
    GraspEntry("fork", "C3_2_BreakfastTrayPreparation", "2F", "catalog",
               "fork", recipe_name="fork_2f_c3_2_recipe"),
)

# Exact M1 identifiers only; no substring or fuzzy matching of object names.
ALIASES = {f"obj_{e.object_id}_{e.object_id}": e.object_id for e in ENTRIES}

# Experimental entries are routed through their object function and remain
# visibly distinct from validated entries in every execution artifact. A failed
# experimental grasp still stops the task; it is never replaced by an LLM grasp.
EXPERIMENTAL_INTEGRATION = {
    "plate": "2F recipe is connected for C1_1; C3_2 vac plate physical validation is pending",
    "fork": "C3_2 3F fork recipe connected; physical validation is pending",
    "fruit": "C3_2 3F fruit recipe connected; physical validation is pending",
}
# Object ids that already have validated non-C3_2 recipes stay out of
# EXPERIMENTAL_INTEGRATION; only the C3_2 registrations are experimental.
EXPERIMENTAL_ENTRY_KEYS = {
    ("bread", "C3_2_BreakfastTrayPreparation"),
    ("spoon", "C3_2_BreakfastTrayPreparation"),
    ("mug", "C3_2_BreakfastTrayPreparation"),
}
PENDING_INTEGRATION = {}
ENABLED_ENTRIES = tuple(entry for entry in ENTRIES if entry.object_id not in PENDING_INTEGRATION)


def integration_status(entry):
    if entry.object_id in PENDING_INTEGRATION:
        return "BLOCKED"
    if entry in ALTERNATIVE_ENTRIES:
        return "VALIDATED"
    if entry in VALIDATOR_EXPERIMENTAL_ENTRIES:
        return "EXPERIMENTAL"
    if (entry.object_id, entry.environment) in EXPERIMENTAL_ENTRY_KEYS:
        return "EXPERIMENTAL"
    if entry.object_id in EXPERIMENTAL_INTEGRATION:
        return "EXPERIMENTAL"
    return "VALIDATED"


class ScriptedGraspUnavailable(RuntimeError):
    pass


def _recipe_keys_for_target(target: str, environment: str) -> tuple[str, ...]:
    """Exact id first; optional ``type_a`` / ``type_b`` → ``type`` for that env."""

    keys = [target]
    if len(target) > 2 and target[-2:] in {"_a", "_b"}:
        base = target[:-2]
        if any(
            entry.object_id == base and entry.environment == environment
            for entry in ENTRIES + ALTERNATIVE_ENTRIES + VALIDATOR_EXPERIMENTAL_ENTRIES
        ):
            keys.append(base)
    return tuple(keys)


def resolve(request):
    """Return a validated recipe for a supported acquire, otherwise None.

    A known object with a different EE is an input error, not permission to
    silently execute a different grasp. Unsupported objects retain M5 routing.

    Multi-instance scenes (``plate_b``) may resolve a type-keyed entry
    (``plate``) while preserving the request target as ``body_object_id``.
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
    candidates = ENTRIES + ALTERNATIVE_ENTRIES
    if task.metadata.get("scripted_grasp_validator_experimental", False):
        candidates += VALIDATOR_EXPERIMENTAL_ENTRIES
    matched_key = None
    matches: list[GraspEntry] = []
    for key in _recipe_keys_for_target(target, environment):
        matches = [
            e
            for e in candidates
            if (e.object_id, e.environment) == (key, environment)
        ]
        if matches:
            matched_key = key
            break
    for entry in matches:
        if task.ee == entry.ee:
            if entry.object_id in PENDING_INTEGRATION:
                raise ScriptedGraspUnavailable(
                    f"SCRIPTED_GRASP_NOT_VALIDATED: {entry.object_id}: "
                    f"{PENDING_INTEGRATION[entry.object_id]}"
                )
            if matched_key != target:
                return replace(entry, body_object_id=target)
            return entry
    if matches:
        supported = "/".join(e.ee for e in matches)
        raise ValueError(
            f"SCRIPTED_GRASP_EE_MISMATCH: {target} requires {supported}, got {task.ee}"
        )
    return None
