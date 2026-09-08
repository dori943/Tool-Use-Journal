"""Run M5 and preserve an honest video even if planning fails partway."""
from pathlib import Path
import json
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output' / 'c2_1'
VIDEO = OUT / 'c2_1_result.mp4'

def main():
    import run_m5
    result = {'stage': 'M5', 'status': 'STARTED', 'video': str(VIDEO)}
    try:
        code = run_m5.main(['c2_1', '--seed', '0', '--simulate', 'controller',
            '--headless', '--video', str(VIDEO), '--width', '1280', '--height', '720',
            '--ee-attach-policy', 'precomputed-or-plan'])
        result.update(status='SUCCESS' if code == 0 else 'EXECUTION_FAILED', exit_code=code)
    except Exception as exc:
        detail = re.sub(r'AIza[\w-]+', '[REDACTED]', ''.join(traceback.format_exception(exc)))
        (OUT / 'm5_exception.txt').write_text(detail, encoding='utf-8')
        result.update(status='PLANNING_FAILED', error_type=type(exc).__name__, detail=str(exc))
        print(detail, flush=True)
        from run import dump_motion_failure
        dump_motion_failure(exc, OUT / 'm5')
        # Plans are saved only after validation. Replay the completed prefix,
        # if any, without executing the rejected trajectory.
        plan_files = sorted((OUT / 'm5' / 'plans').glob('*.json'))
        request_files = sorted((OUT / 'm5' / 'requests').glob('*.json'))
        if plan_files and len(plan_files) == len(request_files):
            try:
                from tuj.m5_motion.generic_runner import execute_planning_result
                from tuj.m5_motion.orchestration import SelectedPlanPlanningResult, _predicted_world
                from tuj.m5_motion.schema import MotionPlan, MotionPlanRequest, WorldSnapshot
                load = lambda p: json.loads(p.read_text(encoding='utf-8'))
                initial = WorldSnapshot.model_validate(load(OUT / 'm5' / 'initial_world.json'))
                plans = tuple(MotionPlan.model_validate(load(p)) for p in plan_files)
                requests = tuple(MotionPlanRequest.model_validate(load(p)) for p in request_files)
                final = initial.model_copy(deep=True)
                for request, plan in zip(requests, plans, strict=True):
                    if plan.request_id != request.request_id:
                        raise ValueError('Saved prefix request/plan identity mismatch')
                    final = _predicted_world(final, request, plan, completed_subgoal=None)
                prefix = SelectedPlanPlanningResult(requests, plans, final)
                execution = execute_planning_result(prefix, repository=ROOT,
                    initial_world=initial, output_dir=OUT / 'm5' / 'partial_replay',
                    mode='controller', seed=0, show_viewer=False, realtime_factor=0,
                    hold_seconds=0, video=VIDEO, camera='agentview',
                    width=1280, height=720, video_fps=24, video_hold_seconds=3)
                result['video_scope'] = 'validated prefix only'
                result['prefix_status'] = execution.status.value
            except Exception as replay_error:
                result['prefix_error'] = str(replay_error)
    if not VIDEO.exists() or VIDEO.stat().st_size < 1024:
        reason = 'M5 FAILED: ' + result.get('error_type', result['status'])
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'record_c2_1_scene.py'),
            '--output', str(VIDEO), '--reason', reason], cwd=ROOT, check=True)
        result['video_scope'] = 'initial scene; motion planning failed before execution'
    (OUT / 'run_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result['status'] == 'SUCCESS' else 2

if __name__ == '__main__':
    raise SystemExit(main())
