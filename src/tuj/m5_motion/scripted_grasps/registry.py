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
    module_name: str | None = None

    def recipe(self):
        name = self.module_name or self.object_id
        module = import_module(f"{__package__}.objects.{name}")
        if self.driver == "catalog":
            if name == "plate_vac":
                return module.plate_vac_recipe(self.environment, self.object_id)
            return getattr(module, name + "_recipe")()
        if self.driver == "spoon":
            # Spoon has independently calibrated 2F and 3F recipes.  The EE
            # chosen by M4 is part of the dispatch key, so recipe selection
            # must preserve that choice instead of falling back to the 2F
            # dataclass default.
            return module.spoon_recipe(self.ee,self.environment)
        return getattr(module, self.object_id.title() + "Recipe")()

    def function(self):
        name = self.module_name or self.object_id
        module = import_module(f"{__package__}.objects.{name}")
        return getattr(module, "grasp_" + name)


ENTRIES = tuple(GraspEntry(*row) for row in (
    ("plate", "C1_1_LegoSweep", "2F", "plate"),
    # 0911: C1_1 에서 M2 가 접시를 쓸어 담는 도구로 고르는데, 접지값상 접시는
    # 폭 182mm 라 2F(개구 85mm)/3F(140mm)로는 안 잡히고 vac 만 가능하다. 2F 항목만
    # 있으면 M4 의 constrain_task_request 가 SCRIPTED_GRASP_EE_INFEASIBLE 로 멈춘다.
    # C2_1 에서 검증된 흡착 동작을 같은 크기의 접시에 재사용한다. resolve() 가
    # 환경과 EE 로 정확히 매칭하므로 아래 2F 항목과 공존한다.
    ("plate", "C1_1_LegoSweep", "vac", "catalog", "plate_vac"),
    ("bottle", "C1_2_DoughFlatten", "3F", "bottle"),
    ("spatula", "C1_2_DoughFlatten", "3F", "spatula"),
    ("spoon", "C1_2_DoughFlatten", "2F", "spoon"),
    ("spoon", "C1_2_DoughFlatten", "3F", "spoon"),
    # C2_1 reuses the validated object-frame 2F spoon recipe so the tabletop
    # sorting pick uses the tuned scripted grasp (reliable formation/lift/
    # retention) instead of a per-run LLM contact-friction grasp.  The plate is
    # grasped with the vacuum EE (M4 mounts vac for the flat disc in C2_1): the
    # per-run LLM vacuum grasp kept missing the surface (CONTACT_COUNT=0), so it
    # uses a dedicated catalog vacuum recipe (objects/plate_vac.py) that presses
    # the cup onto the top face.  A separate module name keeps it distinct from
    # the 2F ``plate`` driver used at C1_1 (objects/plate.py / grasp_plate).
    ("spoon", "C2_1_ObjectSorting", "2F", "spoon"),
    ("spoon", "C2_1_ObjectSorting", "3F", "spoon"),
    ("plate", "C2_1_ObjectSorting", "vac", "catalog", "plate_vac"),
    ("plate", "C3_1_ObjectSorting", "vac", "catalog", "plate_vac"),
    ("spoon", "C3_1_ObjectSorting", "2F", "spoon"),
    ("spoon", "C3_1_ObjectSorting", "3F", "spoon"),
    ("plate_a", "C3_2_BreakfastTrayPreparation", "vac", "catalog", "plate_vac"),
    ("plate_b", "C3_2_BreakfastTrayPreparation", "vac", "catalog", "plate_vac"),
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

# Optional alternatives remain an explicit extension point. The repository does
# not select object- or scene-specific alternatives in the generic M5 path.
ALTERNATIVE_ENTRIES: tuple[GraspEntry, ...] = ()

# Exact M1 identifiers only; no substring or fuzzy matching of object names.
ALIASES = {f"obj_{e.object_id}_{e.object_id}": e.object_id for e in ENTRIES}

# Experimental entries are routed through their object function and remain
# visibly distinct from validated entries in every execution artifact. A failed
# experimental grasp still stops the task; it is never replaced by an LLM grasp.
EXPERIMENTAL_INTEGRATION = {
    "plate": "vacuum routes are task-scoped for C1_1, C2_1 and C3_1; the C1_1 "
             "2F grasp and suction-held sweep still require physical validation",
}
PENDING_INTEGRATION = {}
ENABLED_ENTRIES = tuple(entry for entry in ENTRIES if entry.object_id not in PENDING_INTEGRATION)


def integration_status(entry):
    if entry.object_id in PENDING_INTEGRATION:
        return "BLOCKED"
    if entry in ALTERNATIVE_ENTRIES:
        return "VALIDATED"
    if entry.object_id in EXPERIMENTAL_INTEGRATION:
        return "EXPERIMENTAL"
    return "VALIDATED"


class ScriptedGraspUnavailable(RuntimeError):
    pass


def resolve(request):
    """Return a validated recipe for a supported acquire, otherwise None.

    A scripted recipe is pinned to one EE, but M4 chooses the EE per run and
    per LLM model, so the mounted EE may differ from the recipe's.  When it
    does, this is NOT a fatal input error: fall back to the ordinary M5 grasp
    path (exactly what an unregistered object does) so a registry/M4 EE
    disagreement never hard-stops the task.  The scripted recipe is used only
    when the mounted EE matches; otherwise M5 plans the grasp with the mounted
    EE and the controller preview still physically validates it.
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
    matches = [e for e in ENTRIES + ALTERNATIVE_ENTRIES
               if (e.object_id, e.environment) == (target, environment)]
    for entry in matches:
        if task.ee == entry.ee:
            if target in PENDING_INTEGRATION:
                raise ScriptedGraspUnavailable(f"SCRIPTED_GRASP_NOT_VALIDATED: {target}: {PENDING_INTEGRATION[target]}")
            return entry
    if matches:
        supported = '/'.join(e.ee for e in matches)
        print(f"[M5][SCRIPTED_GRASP] EE mismatch for {target}: recipe expects "
              f"{supported}, M4 mounted {task.ee}; falling back to the M5 grasp "
              f"path for this object.", flush=True)
    return None
