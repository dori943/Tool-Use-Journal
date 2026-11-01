"""Bounded pose-only follow-up to the first C4_2 coverage matrix.

All candidates are declared before execution, including unsuccessful attempts.
No feasibility estimate, force, contact gate, attachment policy or success
threshold is changed. Entries exist only inside each validator worker process.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from verify_table2_coverage import save_json
from tuj.m5_motion.scripted_grasps.objects.c4_2_probes import ENVIRONMENT, probe_recipe
from tuj.m5_motion.scripted_grasps.registry import GraspEntry

# Contact locations stay inside the original handle region. Lid variants
# rotate the same thin-rim proposal to the three remaining edges.
CANDIDATES = [
    {'id': f'lid_{ee}_{edge}', 'object': 'lid', 'ee': ee,
     'changes': {'offset_fraction': offset, 'rotation_xyz_deg': (0., 90., yaw)}}
    for ee in ('2F', '3F')
    for edge, offset, yaw in (
        ('plus_y', (0., .46, 0.), 90.),
        ('minus_x', (-.46, 0., 0.), 180.),
        ('minus_y', (0., -.46, 0.), 270.))
] + [
    {'id': 'whisk_3F_yaw90', 'object': 'whisk', 'ee': '3F',
     'changes': {'rotation_xyz_deg': (0., 0., 90.)}},
    {'id': 'whisk_3F_handle_end', 'object': 'whisk', 'ee': '3F',
     'changes': {'offset_fraction': (0., -.40, 0.)}},
    {'id': 'whisk_3F_raised', 'object': 'whisk', 'ee': '3F',
     'changes': {'offset_m': (0., 0., .020)}},
] + [
    {'id': f'whisk_vac_handle_{int(abs(y)*100):02}', 'object': 'whisk', 'ee': 'vac',
     'changes': {'offset_fraction': (0., y, .5)}}
    for y in (-.38, -.44, -.18)
]
ALLOWED_FIELDS = frozenset({'offset_fraction', 'offset_m', 'rotation_xyz_deg'})


def candidate_recipe(candidate):
    if not set(candidate['changes']) <= ALLOWED_FIELDS:
        raise ValueError('POSE_ONLY_CANDIDATE_REQUIRED')
    changes = {k: tuple(v) for k, v in candidate['changes'].items()}
    return replace(probe_recipe(candidate['object'], candidate['ee']), **changes)


def install_candidate(candidate):
    """Isolated worker override; normal production registration is untouched."""
    from tuj.m5_motion.scripted_grasps import registry
    recipe = candidate_recipe(candidate)
    class CandidateEntry(GraspEntry):
        def recipe(self):
            return recipe
    matches = [e for e in registry.VALIDATOR_EXPERIMENTAL_ENTRIES
               if (e.environment, e.object_id, e.ee) == (ENVIRONMENT, candidate['object'], candidate['ee'])]
    if len(matches) != 1:
        raise ValueError('EXPECTED_ONE_EXPERIMENTAL_ENTRY')
    original = matches[0]
    registry.VALIDATOR_EXPERIMENTAL_ENTRIES = tuple(
        CandidateEntry(**asdict(e)) if e is original else e
        for e in registry.VALIDATOR_EXPERIMENTAL_ENTRIES)


def snapshot(output, args):
    patch = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)
    (output / 'source.patch').write_bytes(patch)
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    names = [n for n in names if n.endswith('.py') and (len(Path(n).parts) == 1 or Path(n).parts[0] in {'scripts', 'src', 'tests', 'environments'})]
    hashes = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    for name in untracked:
        if name in hashes:
            target = output / 'untracked_source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
    save_json(output / 'manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_hashes': hashes, 'source_patch_sha256': hashlib.sha256(patch).hexdigest(),
        'scope': 'BOUNDED_POSE_DIAGNOSTIC_NOT_WHOLE_TASK', 'api_calls': 0,
        'seed': args.seed, 'timeout_seconds': args.timeout_seconds,
        'candidates': [{**c, 'recipe': asdict(candidate_recipe(c))} for c in CANDIDATES],
        'stop_on_success': False, 'candidate_count': len(CANDIDATES)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout-seconds', type=float, default=600)
    parser.add_argument('--worker', choices=[c['id'] for c in CANDIDATES])
    args = parser.parse_args(argv)
    output = args.out_root.resolve()
    if args.worker:
        candidate = next(c for c in CANDIDATES if c['id'] == args.worker)
        install_candidate(candidate)
        from tuj.m5_motion.scripted_grasps.validate import controller_validate_case
        controller_validate_case(candidate['object'], candidate['ee'], output,
                                 environment=ENVIRONMENT, seed=args.seed,
                                 repository=ROOT, video=output/'trial.mp4')
        return 0
    if args.timeout_seconds <= 0:
        parser.error('timeout must be positive')
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output, args)
    trials = []
    for candidate in CANDIDATES:
        dest = output/candidate['id']
        dest.mkdir()
        save_json(dest/'candidate.json', candidate)
        start = time.monotonic()
        error = None
        print(f"[pose] {candidate['id']}", flush=True)
        with (dest/'console.log').open('w') as log:
            try:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    '--worker', candidate['id'], '--out-root', str(dest), '--seed', str(args.seed)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
                if result.returncode:
                    error = {'status': 'ERROR', 'detail': f'WORKER_EXIT:{result.returncode}'}
            except subprocess.TimeoutExpired:
                error = {'status': 'TIMEOUT', 'detail': f'WALL_TIME_LIMIT:{args.timeout_seconds}'}
        try:
            report = json.loads((dest/'validation_report.json').read_text())
            if not isinstance(report, dict) or 'controller_status' not in report:
                raise ValueError('INVALID_REPORT')
        except (ValueError, OSError) as exc:
            report = {}
            error = error or {'status': 'ERROR', 'detail': f'REPORT_ERROR:{type(exc).__name__}'}
        row = {**candidate, 'seconds': round(time.monotonic()-start, 2),
               'status': report.get('controller_status', 'ERROR'),
               'failures': report.get('failures', []), 'result': report.get('result', {}),
               'simulation_started': report.get('simulation_started', False),
               'grasp_attempted': report.get('grasp_attempted', False),
               'report': str(dest/'validation_report.json')}
        if error:
            row.update(error)
        trials.append(row)
        save_json(output/'summary.json', {'seed': args.seed, 'planned': len(CANDIDATES),
                                         'completed': len(trials), 'trials': trials})
        print(f"[pose] -> {row['status']} {row['failures'][:1]}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
