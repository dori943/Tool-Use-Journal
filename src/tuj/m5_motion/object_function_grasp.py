"""Existing catalog functions followed by a persistent M5 attachment.

The catalog is imported as a library, never through its snapshot bootstrap.
One live runtime owns the scene before, during and after the function call.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
import os
from pathlib import Path
import sys

from tuj.m5_motion.tool_use_journal import (
    TOOL_USE_JOURNAL_BARE_HOME_QPOS,
    ToolUseJournalEnvironmentAdapter,
)
from tuj.m5_motion.tool_use_journal_planning import attached_object_transform_from_state


class ObjectFunctionGraspError(RuntimeError):
    pass


SCRIPTED_GRASP_LIBRARY_ENV = "TUJ_SCRIPTED_GRASP_LIBRARY"


def _catalog_package(library_root: str | Path) -> Path:
    root = Path(library_root).expanduser().resolve()
    package = root if root.name == "grasp_lab" else root / "grasp_lab"
    if not (package / "catalog.py").is_file():
        raise ObjectFunctionGraspError(
            f"{SCRIPTED_GRASP_LIBRARY_ENV} must name a grasp_lab package or "
            f"its parent directory: {root}"
        )
    return package


def catalog_library(
    repository: Path,
    *,
    library_root: str | Path | None = None,
):
    """Load the object-function catalog without assuming a sibling checkout.

    An explicit ``library_root`` or ``TUJ_SCRIPTED_GRASP_LIBRARY`` may point to
    either the directory containing ``grasp_lab`` or the package directory
    itself. With neither configured, ``grasp_lab`` must be importable from the
    active Python environment.
    """

    configured = library_root or os.environ.get(SCRIPTED_GRASP_LIBRARY_ENV)
    expected_package = _catalog_package(configured) if configured else None
    if expected_package is not None:
        import_root = str(expected_package.parent)
        if import_root not in sys.path:
            sys.path.insert(0, import_root)
    try:
        catalog = import_module("grasp_lab.catalog")
    except ModuleNotFoundError as error:
        if error.name not in {"grasp_lab", "grasp_lab.catalog"}:
            raise
        raise ObjectFunctionGraspError(
            "object-function catalog is unavailable; install grasp_lab or set "
            f"{SCRIPTED_GRASP_LIBRARY_ENV} to a catalog checkout "
            f"(repository: {repository.resolve()})"
        ) from error
    if expected_package is not None:
        actual_package = Path(catalog.__file__).resolve().parent
        if actual_package != expected_package:
            raise ObjectFunctionGraspError(
                "a different grasp_lab catalog is already loaded: "
                f"expected {expected_package}, got {actual_package}"
            )
    return catalog


def snapshot(runtime, *, previous=None):
    attachment = runtime.attachment
    world = ToolUseJournalEnvironmentAdapter(runtime.env).world_snapshot(
        attached_object_id=runtime.attached_object_id,
        attached_object_transform=(
            attached_object_transform_from_state(attachment) if attachment else None
        ),
        completed_subgoals=list(previous.scene.completed_subgoals) if previous else [],
        facts=list(previous.scene.facts) if previous else [],
    )
    world.robot_state.held_tool_id = runtime.held_tool_id
    if previous is not None:
        from tuj.m5_motion.geometry_evidence import carry_observation_geometry

        world = carry_observation_geometry(previous, world)
    return world


@dataclass
class FunctionGraspResult:
    object_id: str
    function: str
    attachment_reused: bool
    function_result: dict
    attachment: dict
    world: object


def execute_object_function_grasp(
    runtime,
    *,
    object_id,
    resource_kind="object",
    output_dir,
    repository,
    video=False,
    on_control_step=None,
    seed=0,
):
    catalog = catalog_library(Path(repository))
    recipe = catalog.get_recipe(object_id)
    if runtime.active_ee != recipe.ee_id:
        raise ObjectFunctionGraspError(
            f"{object_id}: function requires {recipe.ee_id}, "
            f"M4/runtime selected {runtime.active_ee}"
        )
    if runtime.attached_object_id is not None or runtime.held_tool_id is not None:
        raise ObjectFunctionGraspError("release the held object before acquiring another")
    if resource_kind not in {"object", "tool"}:
        raise ValueError("resource_kind must be object or tool")
    from grasp_lab.catalog_runtime import CatalogContext
    from grasp_lab.runtime import save_json

    function = catalog.get_grasp_function(object_id)
    context = CatalogContext.from_runtime(
        runtime,
        recipe=recipe,
        output=output_dir,
        repository=repository,
        video=video,
        seed=seed,
    )
    context.on_control_step = on_control_step
    try:
        result = function(context, object_id=object_id)
        if result.get("status") != "SUCCESS":
            raise ObjectFunctionGraspError(
                f"FUNCTION_GRASP_FAILED: {object_id}: {result.get('failure_reason')}"
            )
        function_end_time = float(context.data.time)
        reused = runtime.attached_object_id == object_id
        if runtime.attached_object_id not in {None, object_id}:
            raise ObjectFunctionGraspError("ATTACHMENT_TARGET_MISMATCH")
        recipe_data = recipe.to_dict() if hasattr(recipe, "to_dict") else {}
        gripper_actuation = str(
            recipe_data.get(
                "gripper_actuation",
                "suction" if "vacuum_attachment_mode" in recipe_data else "grip",
            )
        ).strip().lower()
        if not reused:
            runtime.command_gripper(
                engaged=True, suction=gripper_actuation == "suction"
            )
            try:
                # Keep M5's existing geometry limits. The function has already
                # validated real contacts and a completed lift/hold.
                runtime.attach_object(object_id, attachment_mode="KINEMATIC")
            except Exception as error:
                raise ObjectFunctionGraspError(
                    f"POST_GRASP_ATTACH_FAILED: {error}"
                ) from error
        if gripper_actuation != "suction":
            runtime.capture_gripper_hold()
        if resource_kind == "tool":
            runtime.mark_attached_object_as_tool(object_id)
        world = snapshot(runtime)
        record = FunctionGraspResult(
            object_id,
            function.__name__,
            reused,
            result,
            asdict(runtime.attachment),
            world,
        )
        save_json(
            Path(output_dir) / "post_grasp_attachment.json",
            {
                "status": "SUCCESS",
                "object_id": object_id,
                "function": function.__name__,
                "function_end_time_s": function_end_time,
                "attachment_time_s": float(context.data.time),
                "attachment_reused": reused,
                "attachment": record.attachment,
                "runtime_shared": context.runtime is runtime,
                "environment_shared": context.env is runtime.env,
            },
        )
        (Path(output_dir) / "attached_world.json").write_text(
            world.model_dump_json(indent=2), encoding="utf-8"
        )
        return record
    except Exception as error:
        save_json(
            Path(output_dir) / "post_grasp_attachment.json",
            {
                "status": "FAILED",
                "object_id": object_id,
                "detail": str(error),
                "attached_object_id": runtime.attached_object_id,
            },
        )
        raise
    finally:
        context.close()


def release_object(runtime, object_id, *, gripper_actuation="grip"):
    """Share the release order with normal DETACH/open event semantics.

    Requires an attached matching object; callers must subsequently step physics
    to open the fingers and settle the released body.
    """
    attachment = runtime.detach_object(object_id)
    runtime.command_gripper(
        engaged=False,
        suction=str(gripper_actuation).strip().lower() == "suction",
    )
    return attachment


def prepare_catalog_environment(env, ee, repository):
    """Apply the catalog's hand/timing model before reset, never during a grasp."""
    catalog_library(Path(repository))
    from grasp_lab.catalog_timing import timing_xml
    from grasp_lab.spoon_hand_model import (
        repair_spoon_hand_xml,
        repair_spoon_parallel_2f_xml,
    )

    # RoboCasa defers model construction until reset; corrections must precede
    # compile. Borrowed contexts never change the live arm pose.
    env.robot_configs[0]["initial_qpos"] = (
        list(TOOL_USE_JOURNAL_BARE_HOME_QPOS)
        if ee is None
        else [0.0, -1.8, 1.2, -0.97, -1.57, 0.0]
    )
    env._load_model()
    env.set_xml_processor(lambda xml: timing_xml(xml, 0.001, "implicitfast"))
    if ee is not None:
        gripper = env.robots[0].gripper["right"]
        if ee == "2F":
            correction = repair_spoon_parallel_2f_xml(
                env.model.root, gripper.naming_prefix
            )
        elif ee == "3F":
            correction = repair_spoon_hand_xml(env.model.root, gripper.naming_prefix)
        else:
            correction = {"policy": "CONTACT_GATED_KINEMATIC_ATTACH"}
    else:
        correction = {"policy": "BARE_FLANGE"}
    env.catalog_hand_correction = correction
    env._initialize_sim()
    env.hard_reset = False
    env.model_timestep = float(env.sim.model.opt.timestep)
    original_reset = env.reset

    def reset(*args, **kwargs):
        observation = original_reset(*args, **kwargs)
        # Kitchen's reset runs an OSC initialization. Switch to joint control
        # only afterwards, exactly as the independent catalog does.
        from grasp_lab.kitchen_env import enable_joint_controller

        enable_joint_controller(env, 150.0)
        if ee == "2F":
            import numpy as np

            gripper = env.robots[0].gripper["right"]
            model, data = env.sim.model._model, env.sim.data._data
            ids = np.array([model.joint(name).id for name in gripper.joints])
            data.qpos[model.jnt_qposadr[ids]] = 0.0
            data.qvel[model.jnt_dofadr[ids]] = 0.0
            data.ctrl[[model.actuator(name).id for name in gripper.actuators]] = 0.0
            gripper.current_action = np.full(gripper.dof, -1.0)
            env.sim.forward()
        return observation

    env.reset = reset
    return env


def make_function_runtime(repository, environment, *, active_ee, seed=0, **options):
    from tuj.m5_motion.tool_use_journal import make_tool_use_journal_env
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

    def factory(ee):
        env = make_tool_use_journal_env(
            repository,
            environment,
            active_ee=ee,
            seed=seed,
            control_freq=50,
            **options,
        )
        try:
            return prepare_catalog_environment(env, ee, repository)
        except Exception:
            env.close()
            raise

    env = factory(active_ee)
    try:
        env.reset()
        return ToolUseJournalEERuntime(env, factory)
    except Exception:
        env.close()
        raise
