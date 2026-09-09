"""Request-by-request planning execution with observed-world handoff.

The default motion path cannot predict free-object motion caused by contact.
This session owns one simulator runtime and executes each finalized plan before
the orchestrator constructs the next request.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tuj.m5_motion.execution import SimulationArtifactStore
from tuj.m5_motion.orchestration import SelectedPlanPlanningResult
from tuj.m5_motion.schema import MotionPlan, MotionPlanRequest, WorldSnapshot


class LivePlanExecutionError(RuntimeError):
    """A finalized plan failed execution or grounded goal verification."""


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if normalized:
        return normalized[:100]
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


class LivePlanExecutionSession:
    """Execute plans in one persistent runtime and return measured worlds."""

    def __init__(
        self,
        runtime: Any,
        repository: str | Path,
        output_dir: str | Path,
        *,
        seed: int,
        controller: bool,
        realtime_factor: float,
        render: bool,
        recorder: Any | None = None,
        adapter_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.repository = Path(repository)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.controller = controller
        self.realtime_factor = realtime_factor
        self.render = render
        self.recorder = recorder
        self.adapter_factory = adapter_factory
        self.records: list[dict[str, Any]] = []
        self.final_world: WorldSnapshot | None = None
        self.status = "IN_PROGRESS"
        self.failure: str | None = None
        self._closed = False
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_repository(
        cls,
        repository: str | Path,
        initial_world: WorldSnapshot,
        output_dir: str | Path,
        *,
        mode: str,
        seed: int,
        show_viewer: bool,
        realtime_factor: float,
        video: Path | None,
        camera: str,
        width: int,
        height: int,
        video_fps: float,
    ) -> "LivePlanExecutionSession":
        from tuj.m5_motion.generic_runner import (
            GenericMotionRunnerError,
            GenericSimulationVideoRecorder,
            _runtime_active_ee,
            _runtime_environment_name,
            _validate_runtime_start,
        )
        from tuj.m5_motion.tool_use_journal import apply_world_snapshot_state
        from tuj.m5_motion.tool_use_journal_runtime import ToolUseJournalEERuntime

        repository_path = Path(repository)
        record_video = video is not None
        suite_options = {
            "ignore_done": True,
            "use_camera_obs": False,
            "has_renderer": bool(show_viewer and not record_video),
            "has_offscreen_renderer": record_video,
            "render_camera": camera,
        }
        if record_video:
            suite_options.update(
                {
                    "camera_names": camera,
                    "camera_heights": height,
                    "camera_widths": width,
                }
            )
        environment_name = _runtime_environment_name(initial_world)
        active_ee = _runtime_active_ee(initial_world)
        if mode == "controller":
            runtime = ToolUseJournalEERuntime.from_repository_for_controller(
                repository_path,
                environment_name,
                active_ee=active_ee,
                seed=seed,
                **suite_options,
            )
        elif mode == "kinematic":
            runtime = ToolUseJournalEERuntime.from_repository(
                repository_path,
                environment_name,
                active_ee=active_ee,
                seed=seed,
                **suite_options,
            )
        else:
            raise GenericMotionRunnerError(
                f"unsupported simulation mode {mode!r}"
            )
        recorder = None
        try:
            apply_world_snapshot_state(runtime.env, initial_world)
            _validate_runtime_start(runtime, initial_world)
            if video is not None:
                recorder = GenericSimulationVideoRecorder(
                    runtime,
                    video,
                    camera=camera,
                    width=width,
                    height=height,
                    fps=video_fps,
                )
            session = cls(
                runtime,
                repository_path,
                output_dir,
                seed=seed,
                controller=mode == "controller",
                realtime_factor=realtime_factor,
                render=show_viewer or record_video,
                recorder=recorder,
            )
            session.final_world = initial_world.model_copy(deep=True)
            session.save_manifest()
            return session
        except Exception:
            if recorder is not None:
                recorder.close()
            runtime.close()
            raise

    def _adapter(self) -> Any:
        if self.adapter_factory is not None:
            return self.adapter_factory(
                self.runtime,
                self.repository,
                seed=self.seed,
                controller=self.controller,
                realtime_factor=self.realtime_factor,
                render=self.render,
            )
        from tuj.m5_motion.tool_use_journal_execution import (
            ToolUseJournalExecutionAdapter,
        )

        # Rebuild against the current runtime environment. EE exchanges replace
        # that environment, so keeping a compiler from a prior step is unsafe.
        return ToolUseJournalExecutionAdapter.from_repository(
            self.runtime,
            self.repository,
            seed=self.seed,
            controller=self.controller,
            realtime_factor=self.realtime_factor,
            render=self.render,
        )

    def __call__(
        self,
        request: MotionPlanRequest,
        plan: MotionPlan,
    ) -> WorldSnapshot:
        if self._closed:
            raise RuntimeError("live execution session is closed")
        if self.status != "IN_PROGRESS":
            raise RuntimeError("live execution session is not accepting plans")
        index = len(self.records)
        directory = self.output_dir / f"{index:04d}-{_slug(request.task.subgoal_id)}"
        record: dict[str, Any] = {
            "index": index,
            "request_id": request.request_id,
            "plan_id": plan.plan_id,
            "subgoal_id": request.task.subgoal_id,
            "action_type": request.task.action_type,
            "status": "RUNNING",
            "artifact_directory": str(directory.resolve()),
        }
        self.records.append(record)
        self.save_manifest()
        try:
            planning = SelectedPlanPlanningResult(
                requests=(request,),
                plans=(plan,),
                final_world=request.world.model_copy(deep=True),
            )
            execution = self._adapter().execute(
                planning,
                store=SimulationArtifactStore(directory),
            )
            record.update(
                {
                    "execution_status": execution.status.value,
                    "execution_detail": execution.detail,
                    "execution_manifest": (
                        str(execution.manifest_path.resolve())
                        if execution.manifest_path is not None
                        else None
                    ),
                    "simulation_run_ids": [item.run_id for item in execution.runs],
                    "report_ids": [item.report_id for item in execution.reports],
                    "goal_statuses": [
                        item.status.value for item in execution.goal_evaluations
                    ],
                }
            )
            self.final_world = execution.final_world.model_copy(deep=True)
            if not execution.successful:
                raise LivePlanExecutionError(execution.detail)
            record["status"] = "SUCCESS"
            return self.final_world.model_copy(deep=True)
        except Exception as error:
            record["status"] = "FAILED"
            record["error"] = f"{type(error).__name__}: {error}"
            self.status = "FAILED"
            self.failure = record["error"]
            raise
        finally:
            self.save_manifest()

    @property
    def run_count(self) -> int:
        return sum(len(item.get("simulation_run_ids", ())) for item in self.records)

    @property
    def report_count(self) -> int:
        return sum(len(item.get("report_ids", ())) for item in self.records)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "live-execution-manifest.json"

    def save_manifest(self) -> Path:
        payload = {
            "manifest_version": "generic-live-1",
            "status": self.status,
            "failure": self.failure,
            "steps": self.records,
            "final_world": (
                self.final_world.model_dump(mode="json")
                if self.final_world is not None
                else None
            ),
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)
        return self.manifest_path

    def complete(
        self,
        *,
        video_hold_seconds: float,
        viewer_hold_seconds: float,
    ) -> None:
        if self.status != "IN_PROGRESS":
            raise LivePlanExecutionError(
                self.failure or "live execution did not complete"
            )
        self.status = "SUCCESS"
        if self.recorder is not None:
            self.recorder.hold_final_frame(video_hold_seconds)
        elif self.render and viewer_hold_seconds > 0.0:
            deadline = time.monotonic() + viewer_hold_seconds
            while time.monotonic() < deadline:
                self.runtime.render()
                time.sleep(0.02)
        self.save_manifest()

    def mark_failure(self, error: BaseException) -> None:
        if self.status == "IN_PROGRESS":
            self.status = "FAILED"
            self.failure = f"{type(error).__name__}: {error}"
            self.save_manifest()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.save_manifest()
        finally:
            try:
                if self.recorder is not None:
                    self.recorder.close()
            finally:
                self.runtime.close()


__all__ = ["LivePlanExecutionError", "LivePlanExecutionSession"]
