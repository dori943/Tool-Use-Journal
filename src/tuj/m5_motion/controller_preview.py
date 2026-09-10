"""Reject dynamic candidates using an isolated, checkpointed controller rollout."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .live_execution import LivePlanExecutionError, LivePlanExecutionSession
from .plan_builder import MotionPlanBuildError
from .runtime_checkpoint import capture_runtime_checkpoint, restore_runtime_checkpoint
from .tool_use_journal_runtime import ToolUseJournalEERuntime


class ControllerPreviewError(RuntimeError):
    """A preview could not be trusted; do not treat it as a candidate rejection."""


class ControllerPlanPreview:
    """Keep the live runtime untouched; candidate failures retain their reports.

    The pipeline owns the finite candidate and collision-feedback retry budgets.
    A passing preview is only a planning filter, never task-completion evidence.
    """

    def __init__(self, live_session, output_dir: Path):
        if not live_session.controller:
            raise ValueError('controller preview requires a controller session')
        self.live = live_session
        self.output_dir = Path(output_dir)
        # Preview directories are created per candidate with exist_ok=False so a
        # duplicated index within a run is caught.  A prior run leaves numbered
        # directories behind (subgoals after an EE exchange restart the index at
        # 0000), so clear the tree once here to keep the within-run guard while
        # letting re-runs start clean instead of raising FileExistsError.
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.index = 0

    @staticmethod
    def _encoded(runtime):
        return json.dumps(capture_runtime_checkpoint(runtime), sort_keys=True)

    def __call__(self, request, plan):
        directory = self.output_dir / f'{self.index:04d}'
        self.index += 1
        directory.mkdir(parents=True, exist_ok=False)
        before = self._encoded(self.live.runtime)
        checkpoint = directory / 'checkpoint.json'
        checkpoint.write_text(before, encoding='utf-8')
        (directory / 'request.json').write_text(request.model_dump_json(indent=2), encoding='utf-8')
        (directory / 'plan.json').write_text(plan.model_dump_json(indent=2), encoding='utf-8')
        record = {'request_id': request.request_id, 'plan_id': plan.plan_id,
                  'checkpoint_sha256': hashlib.sha256(before.encode()).hexdigest(),
                  'status': 'IN_PROGRESS', 'purpose': 'CANDIDATE_VALIDATION_ONLY'}
        preview = None
        clone = None
        try:
            clone = ToolUseJournalEERuntime.from_repository_for_controller(
                self.live.repository, self.live.runtime.environment_name,
                active_ee=self.live.runtime.active_ee, seed=self.live.seed,
                ignore_done=True, use_camera_obs=False, has_renderer=False,
                has_offscreen_renderer=False, render_camera='agentview',
            )
            restore_runtime_checkpoint(checkpoint, clone)
            preview = LivePlanExecutionSession(
                clone, self.live.repository, directory / 'simulation',
                seed=self.live.seed, controller=True, realtime_factor=0, render=False,
            )
            try:
                preview(request, plan)
            except LivePlanExecutionError as error:
                record['status'] = 'REJECTED'
                record['detail'] = str(error)
                # Keep the validator-owned failure code for existing structured
                # collision feedback; never manufacture collision evidence.
                codes = []
                for path in (directory / 'simulation').glob('*/reports/*.json'):
                    report = json.loads(path.read_text(encoding='utf-8'))
                    for failure in [report.get('failure')]:
                        if not failure:
                            continue
                        observed = failure.get('observed') or {}
                        code = observed.get('failure_code') or failure.get('code')
                        if code:
                            codes.append(str(code))
                record['failure_codes'] = codes
                raise MotionPlanBuildError(
                    f"CONTROLLER_PREVIEW_REJECTED {' '.join(codes)}: {error}; evidence={directory}"
                ) from error
            record['status'] = 'ACCEPTED'
        except MotionPlanBuildError:
            raise
        except Exception as error:
            record.update(status='ERROR', error_type=type(error).__name__)
            raise ControllerPreviewError('controller preview unavailable; see ' + str(directory)) from error
        finally:
            try:
                if preview is not None:
                    preview.close()
                elif clone is not None:
                    clone.close()
            finally:
                unchanged = self._encoded(self.live.runtime) == before
                record['live_state_unchanged'] = unchanged
                (directory / 'preview.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
                if not unchanged:
                    raise ControllerPreviewError('controller preview changed live state')
