"""Explicit, scene-scoped dispatch. Importing this module creates no simulator."""
from dataclasses import dataclass, replace
from importlib import import_module
from inspect import signature

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
            # 0912: plate_vac 만 특수 케이스로 환경을 넘기고 있었는데, 같은 자산을
            # 여러 환경에 등록하는 레시피가 늘면 특수 케이스도 같이 늘어난다.
            # 팩토리가 받는 인자만 골라 넘긴다. 인자를 안 받는 기존 레시피는
            # 그대로 호출되고, environment/object_id 를 선언한 레시피는 자기
            # 환경을 알게 되어 실행 기록의 task_id 가 실제 태스크와 맞는다.
            factory = getattr(module, self.recipe_name or name + "_recipe")
            accepted = signature(factory).parameters
            arguments = {key: value for key, value in
                         (("environment", self.environment),
                          ("object_id", self.object_id))
                         if key in accepted}
            return factory(**arguments)
        if self.driver == "spoon":
            # Spoon has independently calibrated 2F and 3F recipes.  The EE
            # chosen by M4 is part of the dispatch key, so recipe selection
            # must preserve that choice instead of falling back to the 2F
            # dataclass default. C3_2 also selects the thin-handle asset.
            asset = "c3_2" if self.environment.startswith("C3_2") else "default"
            return module.spoon_recipe(
                self.ee, self.environment, asset=asset)
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
    # C2_1 에서 검증된 흡착 레시피를 그대로 쓴다 (같은 접시 자산). resolve() 가
    # EE 로 매칭하므로 위 2F 항목과 공존한다.
    # 주의: C1_1 vac 는 plate_vac_c1_1_recipe (flat, zero seating immersion) 를
    # 쓴다. C2_1 의 -0.5 mm 좌석은 table_collision 위에서 접시를 눌러 LIFT 시
    # early-support 면제 한도(~2 mm)를 넘긴다. 기본 plate_vac_recipe() 는 C3_2 rim.
    ("plate", "C1_1_LegoSweep", "vac", "catalog", "plate_vac", None, "plate_vac_c1_1_recipe"),
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
    ("plate", "C2_1_ObjectSorting", "vac", "catalog", "plate_vac", None, "plate_vac_c2_1_recipe"),
    # 0912: C3_1 도 정렬 태스크라 M4 가 접시에 vac EE 를 물리는데, C3_1 용 plate
    # 항목이 없어 resolve() 가 매칭 실패(matches 빈 리스트) → generic M5 파지로
    # 폴백했고, 그 진공 파지가 접시 표면을 못 짚어 CONTACT_COUNT=0, normal_force=0
    # 으로 BREAKABLE_WELD 접촉 계약이 깨졌다 (C1_1/C2_1 주석의 그 실패 모드).
    # environment-aware plate_vac_recipe() 가 C3_1 flat seating 을 고른다.
    ("plate", "C3_1_ObjectSorting", "vac", "catalog", "plate_vac"),
    # 0909/0912: C3_1 spoon routes (kept once; merge had duplicated the pair).
    ("spoon", "C3_1_ObjectSorting", "2F", "spoon"),
    ("spoon", "C3_1_ObjectSorting", "3F", "spoon"),
    # Exact C3_2 instance ids (main) plus type-keyed plate entry below.
    ("plate_a", "C3_2_BreakfastTrayPreparation", "vac", "catalog", "plate_vac", None, "plate_vac_c3_2_recipe"),
    ("plate_b", "C3_2_BreakfastTrayPreparation", "vac", "catalog", "plate_vac", None, "plate_vac_c3_2_recipe"),
    ("apple", "C2_1_ObjectSorting", "3F", "catalog"),
    ("bread", "C2_1_ObjectSorting", "3F", "catalog"),
    ("mug", "C2_1_ObjectSorting", "3F", "catalog"),
    # 0912: C3_1 은 C2_1 과 같은 씬인데 이 세 물체의 항목이 없어서, M4 가 고른 EE
    # (교체 최소화로 2F)에 맞는 레시피가 없어 일반 파지 경로로 떨어졌다. 거기서는
    # GRIPPER_CLOSE 와 ATTACH_OBJECT 가 같은 시각에 발행돼 손가락이 닫히기 전에
    # 용접 접촉을 검사하므로 CONTACT_COUNT=0 으로 멈춘다.
    #
    # 2F 와 3F 를 모두 등록하는 이유는 실행을 통과시키기 위해서만이 아니다.
    # constrain_task_request 는 feasible_ee 를 등록된 EE 로 좁히므로, 한쪽만
    # 등록하면 M4 에게 선택지가 하나뿐이고 EE 선택 정확도를 측정할 수 없다.
    # c2_1 이 정답표와 5/5 일치한 것도 빵/사과/머그가 3F 단독 등록이라 선택의
    # 여지가 없었던 결과다 (실제로 고른 것은 스푼 하나뿐이다). 두 EE 를 모두
    # 등록해야 2F 시도가 진짜 시도가 되고, 미끄러져 실패하면 그것이 "2F 는
    # 차선" 의 물리적 근거가 된다.
    ("apple", "C3_1_ObjectSorting", "3F", "catalog"),
    ("bread", "C3_1_ObjectSorting", "2F", "catalog", "bread_2f"),
    ("bread", "C3_1_ObjectSorting", "3F", "catalog"),
    ("mug", "C3_1_ObjectSorting", "2F", "catalog", "mug_2f"),
    ("mug", "C3_1_ObjectSorting", "3F", "catalog"),
    # 0912: C3_1 mounts the 2F gripper for the apple (C2_1 used 3F). Without a
    # 2F entry the grasp fell back to the generic M5 path, which closed on the
    # apple but left the grip site 20.4 mm from centre - past the 20 mm attach
    # limit. A dedicated centred 2F recipe (objects/apple_2f.py) seats the grip
    # on the body centre. Distinct module name keeps it apart from the 3F
    # ``apple`` catalog recipe used at C2_1.
    ("apple", "C3_1_ObjectSorting", "2F", "catalog", "apple_2f"),
    ("knife", "C2_2_SandwichAssembly", "2F", "catalog"),
    # 0912: C2_2 는 칼만 등록돼 있어 재료 다섯 개가 일반 M5 파지로 떨어졌다.
    # 스크립트가 없으니 VLM 이 전략을 지어내는데, 흡착으로 4mm 두께 슬라이스를
    # 옆에서 집는 전략(strat_cheese_side_x_left 등)을 내고 조리대 높이에서
    # 옆으로 손을 뻗어 팔뚝이 아일랜드를 137mm 뚫었다. 흡착컵은 윗면에 수직
    # 하강해야 하므로 접근을 고정한다 (objects/ingredient_vac.py).
    ("bread_a", "C2_2_SandwichAssembly", "vac", "catalog", "ingredient_vac"),
    ("bread_b", "C2_2_SandwichAssembly", "vac", "catalog", "ingredient_vac"),
    ("turkey_1", "C2_2_SandwichAssembly", "vac", "catalog", "ingredient_vac"),
    ("cheese_1", "C2_2_SandwichAssembly", "vac", "catalog", "ingredient_vac"),
    ("tomato_slice", "C2_2_SandwichAssembly", "vac", "catalog", "ingredient_vac"),
    ("rolling_pin", "C4_2_DiagonalFitPacking", "2F", "catalog"),
    ("baguette", "C4_2_DiagonalFitPacking", "2F", "catalog"),
    ("whisk", "C4_2_DiagonalFitPacking", "2F", "catalog"),
    ("cereal", "C4_2_DiagonalFitPacking", "2F", "catalog"),
    ("milk", "C4_2_DiagonalFitPacking", "3F", "catalog"),
    ("lid", "C4_2_DiagonalFitPacking", "vac", "catalog"),
    # C3_2 breakfast tray: type-keyed recipes; *_a/*_b resolve via instance
    # suffix → type, with body_object_id = request target.
    ("plate", "C3_2_BreakfastTrayPreparation", "vac", "catalog", "plate_vac", None, "plate_vac_c3_2_recipe"),
    ("fork", "C3_2_BreakfastTrayPreparation", "3F", "catalog"),
    # Breakfast bread: M4 may mount 3F or vac. Both keep a scripted path —
    # 3F enclosure (C2_1 pattern + C3_2 AABB) and vac top-face attach.
    ("bread", "C3_2_BreakfastTrayPreparation", "3F", "catalog", "bread", None, "bread_3f_c3_2_recipe"),
    ("bread", "C3_2_BreakfastTrayPreparation", "vac", "catalog", "bread_vac"),
    ("fruit", "C3_2_BreakfastTrayPreparation", "3F", "catalog"),
    ("spoon", "C3_2_BreakfastTrayPreparation", "3F", "spoon"),
    ("mug", "C3_2_BreakfastTrayPreparation", "3F", "catalog", "mug_c3_2"),
))

# C1_1 plate+vac is a primary ENTRIES row (flat zero-immersion seating) so M4's
# EE feasibility check can select it alongside the validated 2F plate entry.
ALTERNATIVE_ENTRIES: tuple[GraspEntry, ...] = ()

# Explicit validator-only / greedy-only feasibility probes. They do not alter
# production M4 choices or the normal M5 registry surface unless a caller opts in.
_C3_2_2F_CORE: tuple[GraspEntry, ...] = (
    GraspEntry("bread", "C3_2_BreakfastTrayPreparation", "2F", "catalog",
               "bread", recipe_name="bread_2f_c3_2_recipe"),
    GraspEntry("spoon", "C3_2_BreakfastTrayPreparation", "2F", "spoon"),
    GraspEntry("fork", "C3_2_BreakfastTrayPreparation", "2F", "catalog",
               "fork", recipe_name="fork_2f_c3_2_recipe"),
)
VALIDATOR_EXPERIMENTAL_ENTRIES: tuple[GraspEntry, ...] = _C3_2_2F_CORE
# Greedy EE-order experiment only: same 2F core plus fruit 2F. C3_2 mug AABB
# (~88 mm) exceeds the Robotiq 85 stroke, so mug stays 3F-only even for greedy.
GREEDY_EXTRA_ENTRIES: tuple[GraspEntry, ...] = _C3_2_2F_CORE + (
    GraspEntry("fruit", "C3_2_BreakfastTrayPreparation", "2F", "catalog",
               "fruit", recipe_name="fruit_2f_c3_2_recipe"),
)

# Exact M1 identifiers only; no substring or fuzzy matching of object names.
ALIASES = {f"obj_{e.object_id}_{e.object_id}": e.object_id for e in ENTRIES}

# Experimental entries are routed through their object function and remain
# visibly distinct from validated entries in every execution artifact. A failed
# experimental grasp still stops the task; it is never replaced by an LLM grasp.
EXPERIMENTAL_INTEGRATION = {
    "plate": "vacuum routes are task-scoped for C1_1, C2_1, C3_1 and C3_2; the "
             "C1_1 2F grasp and suction-held sweep still require physical validation",
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
    if entry in VALIDATOR_EXPERIMENTAL_ENTRIES or entry in GREEDY_EXTRA_ENTRIES:
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
            for entry in (
                ENTRIES + ALTERNATIVE_ENTRIES
                + VALIDATOR_EXPERIMENTAL_ENTRIES + GREEDY_EXTRA_ENTRIES
            )
        ):
            keys.append(base)
    return tuple(keys)


def _opt_in_extra_entries(task_metadata, world_metadata) -> tuple[GraspEntry, ...]:
    """Return opted-in extra recipes without duplicating shared 2F core rows."""
    extras: list[GraspEntry] = []
    seen: set[tuple[str, str, str]] = set()

    def _extend(entries: tuple[GraspEntry, ...]) -> None:
        for entry in entries:
            key = (entry.object_id, entry.environment, entry.ee)
            if key in seen:
                continue
            seen.add(key)
            extras.append(entry)

    if task_metadata.get("scripted_grasp_validator_experimental", False):
        _extend(VALIDATOR_EXPERIMENTAL_ENTRIES)
    if (
        task_metadata.get("scripted_grasp_greedy_extra", False)
        or world_metadata.get("scripted_grasp_greedy_extra", False)
    ):
        _extend(GREEDY_EXTRA_ENTRIES)
    return tuple(extras)


def resolve(request):
    """Return a validated recipe for a supported acquire, otherwise None.

    A scripted recipe is pinned to one EE, but M4 chooses the EE per run and
    per LLM model, so the mounted EE may differ from the recipe's.  When it
    does, this is NOT a fatal input error: fall back to the ordinary M5 grasp
    path (exactly what an unregistered object does) so a registry/M4 EE
    disagreement never hard-stops the task.  The scripted recipe is used only
    when the mounted EE matches; otherwise M5 plans the grasp with the mounted
    EE and the controller preview still physically validates it.

    Multi-instance scenes (``plate_b``) may resolve a type-keyed entry
    (``plate``) while preserving the request target as ``body_object_id``.

    Production callers never set ``scripted_grasp_greedy_extra``; only the
    greedy EE-order experiment exposes C3_2 2F extras through that flag.
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
    candidates = (
        ENTRIES + ALTERNATIVE_ENTRIES
        + _opt_in_extra_entries(task.metadata, request.world.metadata)
    )
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
        supported = '/'.join(e.ee for e in matches)
        print(f"[M5][SCRIPTED_GRASP] EE mismatch for {target}: recipe expects "
              f"{supported}, M4 mounted {task.ee}; falling back to the M5 grasp "
              f"path for this object.", flush=True)
    return None
