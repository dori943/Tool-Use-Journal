"""Explicit, scene-scoped dispatch. Importing this module creates no simulator."""
from dataclasses import dataclass
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

    def recipe(self):
        name = self.module_name or self.object_id
        module = import_module(f"{__package__}.objects.{name}")
        if self.driver == "catalog":
            # 0912: plate_vac 만 특수 케이스로 환경을 넘기고 있었는데, 같은 자산을
            # 여러 환경에 등록하는 레시피가 늘면 특수 케이스도 같이 늘어난다.
            # 팩토리가 받는 인자만 골라 넘긴다. 인자를 안 받는 기존 레시피는
            # 그대로 호출되고, environment/object_id 를 선언한 레시피는 자기
            # 환경을 알게 되어 실행 기록의 task_id 가 실제 태스크와 맞는다.
            factory = getattr(module, name + "_recipe")
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
    # 0912: C3_1 도 정렬 태스크라 M4 가 접시에 vac EE 를 물리는데, C3_1 용 plate
    # 항목이 없어 resolve() 가 매칭 실패(matches 빈 리스트) → generic M5 파지로
    # 폴백했고, 그 진공 파지가 접시 표면을 못 짚어 CONTACT_COUNT=0, normal_force=0
    # 으로 BREAKABLE_WELD 접촉 계약이 깨졌다 (C1_1/C2_1 주석의 그 실패 모드).
    # 같은 접시 자산이므로 C2_1 에서 검증된 흡착 레시피(objects/plate_vac.py)를
    # 그대로 재사용한다. plate_vac_recipe().task_id 는 'c2_1' 고정이라 실행 기록에
    # C3_1 작업도 c2_1 로 남지만 동작에는 영향 없다 (C1_1 항목과 동일).
    ("plate", "C3_1_ObjectSorting", "vac", "catalog", "plate_vac"),
    # 0912: 아래 네 줄은 main 에서 사라져 있었다. 테스트는 그대로 남아 있어
    # (test_excluded_unknown_and_wrong_hand 의 스푼 6개 경로,
    # test_plate_vac_routes_are_task_and_instance_scoped 의 plate_a/plate_b)
    # main 단독으로도 빨간 상태였다. PR #71 이 스푼 4줄을 날린 것과 같은 사고다.
    ("spoon", "C3_1_ObjectSorting", "2F", "spoon"),
    ("spoon", "C3_1_ObjectSorting", "3F", "spoon"),
    ("plate_a", "C3_2_BreakfastTrayPreparation", "vac", "catalog", "plate_vac"),
    ("plate_b", "C3_2_BreakfastTrayPreparation", "vac", "catalog", "plate_vac"),
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
