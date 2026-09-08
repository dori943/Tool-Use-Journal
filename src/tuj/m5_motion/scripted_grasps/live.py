"""M5 request-by-request execution with measured state between subgoals.

Scripted acquisitions have their own execution artifacts, not placeholder
MotionPlans. All other requests retain the ordinary M5 planner and player.
"""
import hashlib
import json
from pathlib import Path

from tuj.m5_motion.orchestration import (
    MotionPlanStore, SelectedPlanPlanningResult, _resource_transition_requests,
    _single_subgoal_plan, _source_value, _unwrap_plan,
)
from tuj.m5_motion.schema import MotionConstraints, PlannerOptions
from tuj.m5_motion.selected_plan_adapter import SelectedPlanMotionRequestAdapter
from .registry import ALIASES, integration_status, resolve
from .runtime import save_json


def snapshot(runtime, previous=None):
    from tuj.m5_motion.tool_use_journal import ToolUseJournalEnvironmentAdapter
    from tuj.m5_motion.tool_use_journal_planning import attached_object_transform_from_state
    from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalKinematicTrajectoryPlayer
    attachment = runtime.attachment
    world = ToolUseJournalEnvironmentAdapter(runtime.env).world_snapshot(
        completed_subgoals=previous.scene.completed_subgoals if previous else (),
        facts=previous.scene.facts if previous else (),
        attached_object_id=runtime.attached_object_id,
        attached_object_transform=attached_object_transform_from_state(attachment) if attachment else None)
    world.robot_state = ToolUseJournalKinematicTrajectoryPlayer._final_state(runtime)
    if previous is not None:
        world.metadata = {**previous.metadata, **world.metadata}
        # Keep semantic anchors/affordances supplied by M1/M4, while actual
        # geometry, poses, joints and EE declarations always come from runtime.
        for identifier, record in world.objects.items():
            old = previous.objects.get(identifier, {})
            world.objects[identifier] = {**old, **record,
                "anchors": {**old.get("anchors", {}), **record.get("anchors", {})}}
    world.metadata.pop("contact_friction_held_objects", None)
    retention = getattr(runtime, "scripted_grasp_retention", None)
    if retention is not None and attachment is None:
        world.metadata["contact_friction_held_objects"] = {
            retention.entry.object_id: retention.transform().model_dump(mode="json")}
    world.metadata["scripted_grasps"] = True
    return world


class ScriptedGraspSession:
    """Borrow a caller-owned runtime. No reset, close, EE replacement or LLM
    construction is performed by the acquisition dispatch itself.
    """
    def __init__(self, runtime, repository, output, *, seed=0, provider=None,
                 planner_factory=None, executor_factory=None, **planner_options):
        self.runtime, self.repository, self.output = runtime, Path(repository), Path(output)
        self.seed, self.provider, self.planner_options = seed, provider, planner_options
        self.planner_factory, self.executor_factory = planner_factory, executor_factory
        self.world = snapshot(runtime)
        self.records = []
        self.sequence_status = None
        self.failure = None
        self._planner = None
        self._planner_key = None
        self.output.mkdir(parents=True, exist_ok=True)

    def execute_request(self, request, *, completed_subgoal=None):
        from .context import execute_grasp
        from tuj.m5_motion.execution import SimulationArtifactStore
        from tuj.m5_motion.tool_use_journal_execution import ToolUseJournalExecutionAdapter
        from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalMotionRequestPlanner

        request = request.model_copy(deep=True)
        request.world = self.world.model_copy(deep=True)
        retention = getattr(self.runtime, "scripted_grasp_retention", None)
        if retention is not None:
            recipe = retention.context.recipe
            for name in ("velocity_scaling", "acceleration_scaling"):
                limit = getattr(recipe, "post_grasp_" + name, None)
                if limit is not None:
                    setattr(request.constraints, name, min(getattr(request.constraints, name), limit))
        # Resolve only the explicit M1 aliases known to this asset registry.
        task = request.task
        task.target_ids = [ALIASES.get(n, n) for n in task.target_ids]
        task.tool = ALIASES.get(task.tool, task.tool)
        task.goal.target_object_id = ALIASES.get(task.goal.target_object_id, task.goal.target_object_id)
        task.allowed_touch_objects = [ALIASES.get(n, n) for n in task.allowed_touch_objects]
        if retention is not None:
            from .transport import ground_held_transport
            ground_held_transport(request, retention)
        token = hashlib.sha256(request.model_dump_json().encode()).hexdigest()[:20]
        request.request_id = f"motion-request:live:{token}"
        request.provenance = request.provenance.model_copy(update={
            "artifact_id": f"motion-request-artifact:live:{token}",
            "metadata": {**request.provenance.metadata, "state_source": "LIVE_RUNTIME"}})
        index = len(self.records)
        directory = self.output / f"{index:04d}-{token}"
        store = MotionPlanStore(directory)
        request_path = store.save_request(request, index=0)
        record = {"request_id": request.request_id, "request": str(request_path), "status": "RUNNING"}
        self.records.append(record)
        try:
            entry = resolve(request)
            if entry is not None:
                record.update(route="SCRIPTED_GRASP", object_id=entry.object_id,
                    recipe_id=entry.recipe().recipe_id,
                    integration_status=integration_status(entry))
                result = execute_grasp(self.runtime, entry, directory / "grasp", seed=self.seed, request=request)
                record.update(result=str((directory / "grasp/result.json").resolve()),
                    metrics=result.get("metrics"))
            else:
                record["route"] = "M5_MOTION_PLAN"
                factory = self.planner_factory or ToolUseJournalMotionRequestPlanner.from_environment
                key = (id(self.runtime.env), id(self.provider))
                if self._planner is None or key != self._planner_key:
                    self._planner = factory(self.runtime.env, self.repository, provider=self.provider,
                        seed=self.seed, **self.planner_options)
                    self._planner_key = key
                else:
                    collision_factory = self._planner.collision_context_factory
                    collision_factory.compiler = collision_factory.compiler.with_reference_environment(self.runtime.env)
                planner = self._planner
                plan = _unwrap_plan(planner(request), request)
                store.save_request(request, index=0)
                record["plan"] = str(store.save_plan(plan, index=0))
                execution_factory = self.executor_factory or ToolUseJournalExecutionAdapter
                adapter = execution_factory(self.runtime, compiler=planner.collision_context_factory.compiler,
                    controller=True, random_seed=self.seed)
                adapter.world_snapshot = lambda req, report: snapshot(self.runtime, req.world)
                execution = adapter.execute(SelectedPlanPlanningResult((request,), (plan,), request.world),
                    store=SimulationArtifactStore(directory / "simulation"))
                record["execution_status"] = execution.status.value
                if not execution.successful:
                    raise RuntimeError(execution.detail)
            self.world = snapshot(self.runtime, self.world)
            if completed_subgoal and completed_subgoal not in self.world.scene.completed_subgoals:
                self.world.scene.completed_subgoals.append(completed_subgoal)
            record["status"] = "SUCCESS"
            record["final_robot_state"] = self.world.robot_state.model_dump(mode="json")
            retention = getattr(self.runtime, "scripted_grasp_retention", None)
            if retention is not None:
                save_json(directory / "retention.json", retention.samples)
            return record
        except Exception as error:
            record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            self.world = snapshot(self.runtime, self.world)
            raise
        finally:
            retention = getattr(self.runtime, "scripted_grasp_retention", None)
            if retention is not None:
                save_json(directory / "retention.json", retention.samples)
            self.save_manifest()

    def save_manifest(self):
        path = self.output / "live-execution-manifest.json"
        save_json(path, {"manifest_version": "scripted-live-1", "steps": self.records,
            "status": "FAILED" if self.failure or any(r["status"] != "SUCCESS" for r in self.records) else (self.sequence_status or "SUCCESS"),
            "failure": self.failure,
            "final_world": self.world.model_dump(mode="json")})
        return path

    def execute_selected_plan(self, selected, *, constraints, options=None, selected_plan_artifact_id=None):
        self.sequence_status = "IN_PROGRESS"
        try:
            self._execute_selected_plan(selected, constraints=constraints, options=options,
                selected_plan_artifact_id=selected_plan_artifact_id)
            self.sequence_status = "SUCCESS"
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"
            self.sequence_status = "FAILED"
            raise
        finally:
            self.save_manifest()
        return self.save_manifest()

    def _execute_selected_plan(self, selected, *, constraints, options=None, selected_plan_artifact_id=None):
        """Reuse M4/M5 grounding and resource transition construction."""
        adapter = SelectedPlanMotionRequestAdapter()
        options = options or PlannerOptions(random_seed=self.seed)
        artifact = selected_plan_artifact_id or "selected-plan:scripted-live"
        for index, subgoal_id in enumerate(selected.subgoal_order):
            limits = _source_value(constraints, subgoal_id, index, MotionConstraints, "constraints")
            option = _source_value(options, subgoal_id, index, PlannerOptions, "options")
            sliced = _single_subgoal_plan(selected, subgoal_id)
            transitions = _resource_transition_requests(parent_subgoal_id=subgoal_id, steps=sliced.steps,
                world=self.world, constraints=limits, options=option,
                selected_plan_artifact_id=artifact, fallback_ee=sliced.candidate_assignments[0].ee)
            for request in transitions:
                self.execute_request(request)
            request = adapter.convert(sliced, worlds=self.world, constraints=limits,
                options=option, selected_plan_artifact_id=artifact)[0]
            if (request.task.metadata.get("action_parameters", {}).get("target_pose") is None
                    and request.task.grasp is None):
                request.task.metadata["scripted_m4_implicit_object_pose"] = True
            self.execute_request(request, completed_subgoal=subgoal_id)
        terminal = [step for step in selected.steps if step.subgoal_id is None]
        if terminal:
            for request in _resource_transition_requests(parent_subgoal_id="terminal", steps=terminal,
                    world=self.world, constraints=limits, options=option,
                    selected_plan_artifact_id=artifact,
                    fallback_ee=self.runtime.active_ee or selected.candidate_assignments[-1].ee):
                self.execute_request(request)
        return self.save_manifest()
