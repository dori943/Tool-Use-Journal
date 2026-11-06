"""통합 실행기 — 한 번의 실행으로 M1 → M2 → G_k → M4 → M5를 순서대로 돌린다.

현재 M1은 기존 Scene Abstraction과 Metric & Physical Grounding 기능을 통합한다.
따라서 별도의 active M3 stage 없이 M1에서 scene abstraction, geometry grounding,
physical grounding, Object Knowledge retrieval 및 EE evaluation까지 수행한다.

M1에서 생성한 산출물(m1.json / m1_points.npz / crops)을 이후 모듈에 그대로 전달한다.
--seed 로 배치 난수를 고정하므로 재실행 시 같은 장면이 재현되고,
M5가 자체적으로 다시 만드는 환경도 같은 시드로 맞춰진다.

사용법:
  python scripts/run.py c1_1
  python scripts/run.py c2_1
  python scripts/run.py c2_1 --model gpt-4o
  python scripts/run.py c1_1 --m1-json path.json
  python scripts/run.py c1_1 --start-from m5

실행 순서:
  1) M1  Scene + Physical Grounding
  2) M2  Subgoal Decomposition
  3) G_k Subgoal Graph Assembly
  4) M4  Task Planner
  5) M5  Motion Planner

산출물 (output/<task>/):
  m1.json
  m1_points.npz
  crops/*.png
  frame*.png
  m2.json
  gk_<SG>.json
  gk_bundle.json
  m4.json
  m5/
  m5.json
  timing_summary.json   # 모듈별 wall time / token / plan step counts
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


# 현재 active pipeline:
# M1 → M2 → G_k/M4 → M5
#
# --start-from / --stop-after에서는 사용자가 보는 module 이름 기준으로
# m1, m2, m4, m5를 유지한다.
STAGES = ("m1", "m2", "m4", "m5")
# Wall-time report includes G_k even though the CLI stage list collapses it
# into the m2→m4 span.
TIMING_STAGES = ("m1", "m2", "gk", "m4", "m5")


# 태스크 id <-> 환경 이름 단일 출처
from task_registry import TASK_ENVS as TASK_ENV

# ══════════════════════════════════════════════════════════════════════
# Timing / token summary (inlined so ours does not require tuj.ablations)
# ══════════════════════════════════════════════════════════════════════

def _llm_usage_rows(usage: dict | None) -> list[dict]:
    rows: list[dict] = []
    for name, entry in (usage or {}).items():
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "call": name,
                "prompt_tokens": int(entry.get("prompt_tokens") or 0),
                "completion_tokens": int(entry.get("completion_tokens") or 0),
                "total_tokens": int(
                    entry.get("tokens") or entry.get("total_tokens") or 0
                ),
                "seconds": float(entry.get("seconds") or 0.0),
                "calls": int(entry.get("calls") or 0),
            }
        )
    return rows


def _sum_llm_rows(rows: list[dict]) -> dict:
    return {
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "seconds": round(sum(row["seconds"] for row in rows), 2),
        "calls": sum(row["calls"] for row in rows),
    }


def build_timing_summary(
    *,
    stage_order: tuple[str, ...],
    stage_seconds: dict[str, float | None],
    stage_status: dict[str, str],
    llm_usage: dict | None,
    wall_seconds: float | None,
    extras: dict | None = None,
) -> dict:
    llm_calls = _llm_usage_rows(llm_usage)
    stages: dict[str, dict] = {}
    for name in stage_order:
        seconds = stage_seconds.get(name)
        stages[name] = {
            "seconds": None if seconds is None else round(float(seconds), 2),
            "status": stage_status.get(name, "not_run"),
        }
    measured = [
        stages[name]["seconds"]
        for name in stage_order
        if stages[name]["seconds"] is not None
    ]
    summary: dict = {
        "llm_calls": llm_calls,
        "llm_total": _sum_llm_rows(llm_calls),
        "stages": stages,
        "total_seconds": (
            None if wall_seconds is None else round(float(wall_seconds), 2)
        ),
        "measured_stage_sum_seconds": (
            None if not measured else round(sum(measured), 2)
        ),
    }
    if extras:
        summary.update(extras)
    return summary


def merge_stage_seconds(
    previous: dict | None,
    current: dict[str, float | None],
) -> dict[str, float | None]:
    merged = dict(current)
    if not previous:
        return merged
    prior_stages = previous.get("stages") or {}
    for name, entry in prior_stages.items():
        if not isinstance(entry, dict):
            continue
        if merged.get(name) is None and entry.get("seconds") is not None:
            merged[name] = float(entry["seconds"])
    return merged


def merge_stage_status(
    previous: dict | None,
    current: dict[str, str],
) -> dict[str, str]:
    merged = dict(current)
    if not previous:
        return merged
    prior_stages = previous.get("stages") or {}
    for name, entry in prior_stages.items():
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if merged.get(name, "not_run") in {"not_run", "resumed"} and status:
            if merged.get(name) == "not_run" and status not in {None, "not_run"}:
                merged[name] = str(status)
            elif merged.get(name) == "resumed" and status not in {
                None,
                "not_run",
                "resumed",
            }:
                merged[name] = f"resumed:{status}"
    return merged


def print_timing_summary(
    summary: dict,
    *,
    stage_order: tuple[str, ...],
) -> None:
    llm = summary.get("llm_total") or {}
    print(
        "[timing] LLM "
        f"prompt={llm.get('prompt_tokens', 0)} "
        f"completion={llm.get('completion_tokens', 0)} "
        f"total={llm.get('total_tokens', 0)} "
        f"({llm.get('seconds', 0):.2f}s)"
    )
    for row in summary.get("llm_calls") or []:
        print(
            f"  [{row['call']}] "
            f"{row['prompt_tokens']}+{row['completion_tokens']}="
            f"{row['total_tokens']}tok {row['seconds']:.2f}s"
        )
    stages = summary.get("stages") or {}
    for name in stage_order:
        entry = stages.get(name) or {}
        seconds = entry.get("seconds")
        label = "n/a" if seconds is None else f"{seconds:.2f}s"
        print(f"  [{name}] {label} ({entry.get('status', 'not_run')})")
    total = summary.get("total_seconds")
    if total is not None:
        print(f"  [total] {total:.2f}s")


# ══════════════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════════════

def banner(text):
    print()
    print("=" * 72)
    print(f"  {text}")
    print("=" * 72)
    sys.stdout.flush()


def load_script(name):
    """scripts/<name>.py 를 모듈로 읽는다."""
    path = SCRIPTS / f"{name}.py"

    if not path.exists():
        sys.exit(f"[err] {path} 없음")

    spec = importlib.util.spec_from_file_location(
        f"_tuj_{name}",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def call_main(module, argv, label):
    """각 scripts/run_*.py 의 main()을 동일 프로세스에서 호출한다."""
    import inspect

    fn = getattr(module, "main", None)

    if fn is None:
        sys.exit(f"[err] {label}: main() 이 없습니다")

    takes_argv = bool(inspect.signature(fn).parameters)

    saved = sys.argv

    try:
        if takes_argv:
            sys.argv = [label] + list(argv)
            rc = fn(list(argv))
        else:
            sys.argv = [label] + list(argv)
            rc = fn()

    except SystemExit as exc:
        code = exc.code

        if isinstance(code, str):
            print(code)
            sys.exit(
                f"\n[중단] {label} 단계에서 멈췄습니다. "
                "위 메시지를 확인하십시오."
            )

        rc = code or 0

    finally:
        sys.argv = saved

    if rc:
        sys.exit(
            f"\n[중단] {label} 단계가 exit={rc} 로 끝났습니다."
        )

    return rc


def seed_everything(seed):
    """씬 배치 난수 고정."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)

    except ImportError:
        pass


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _m2_llm_usage(out: Path) -> dict | None:
    path = out / "m2.json"
    if not path.is_file():
        return None
    return (read_json(path).get("m2_stats") or {}).get("llm_usage")


def _m1_physical_token_summary(out: Path) -> dict | None:
    path = out / "m0_retrieval.json"
    if not path.is_file():
        return None
    summary = read_json(path).get("token_summary")
    return summary if isinstance(summary, dict) else None


def _plan_execution_metrics(out: Path) -> dict | None:
    """Task-plan step / EE-exchange counts from a successful M4 selected plan."""

    path = out / "m4.json"
    if not path.is_file():
        return None
    plan = read_json(path).get("selected_plan")
    if not isinstance(plan, dict):
        return None
    order = plan.get("subgoal_order") or []
    steps = plan.get("steps") or []
    action_counts = plan.get("action_counts") or {}
    cost = plan.get("cost_vector") or {}
    metrics = {
        "subgoal_count": len(order),
        "plan_step_count": len(steps),
        "transition_step_count": sum(
            1 for step in steps if step.get("kind") == "transition"
        ),
        "subgoal_step_count": sum(
            1 for step in steps if step.get("kind") == "subgoal"
        ),
        "action_counts": action_counts,
        "n_ee_attaches": int(action_counts.get("n_ee_attaches") or 0),
        "n_ee_detaches": int(action_counts.get("n_ee_detaches") or 0),
        "planned_ee_switches": int(cost.get("ee_switches") or 0),
    }
    executed = _executed_ee_metrics(out)
    if executed is not None:
        metrics.update(executed)
    return metrics


def _executed_ee_metrics(out: Path) -> dict | None:
    """Live-executed EE swap counts (SReg actual), when an M5 live manifest exists."""

    try:
        from tuj.gt.ee_swap_metrics import (
            find_latest_live_manifest,
            load_executed_ee_metrics,
        )
    except ImportError:
        return None

    m5_dir = out / "m5"
    summary_path = m5_dir / "m5_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if isinstance(summary.get("executed_ee_metrics"), dict):
            metrics = dict(summary["executed_ee_metrics"])
            return {
                "executed_ee_switches": metrics.get("executed_ee_switches"),
                "executed_n_ee_attaches": metrics.get("executed_n_ee_attaches"),
                "executed_n_ee_detaches": metrics.get("executed_n_ee_detaches"),
                "executed_ee_source": "m5_summary.json",
            }
        if summary.get("executed_ee_switches") is not None:
            return {
                "executed_ee_switches": int(summary["executed_ee_switches"]),
                "executed_ee_source": "m5_summary.json",
            }

    manifest = find_latest_live_manifest(m5_dir)
    if manifest is None:
        return None
    metrics = load_executed_ee_metrics(manifest)
    if metrics is None:
        return None
    return {
        "executed_ee_switches": metrics.get("executed_ee_switches"),
        "executed_n_ee_attaches": metrics.get("executed_n_ee_attaches"),
        "executed_n_ee_detaches": metrics.get("executed_n_ee_detaches"),
        "executed_ee_source": str(manifest),
    }


def _token_totals(out: Path, llm_usage: dict | None) -> dict:
    """Per-module and overall token totals for the ours pipeline."""

    m1 = _m1_physical_token_summary(out)
    m1_total = int((m1 or {}).get("physical_total", {}).get("total_tokens") or 0)
    m2_rows = []
    if isinstance(llm_usage, dict):
        for name, entry in llm_usage.items():
            if not isinstance(entry, dict):
                continue
            m2_rows.append(
                {
                    "call": name,
                    "total_tokens": int(
                        entry.get("tokens") or entry.get("total_tokens") or 0
                    ),
                    "prompt_tokens": int(entry.get("prompt_tokens") or 0),
                    "completion_tokens": int(entry.get("completion_tokens") or 0),
                    "seconds": float(entry.get("seconds") or 0.0),
                    "calls": int(entry.get("calls") or 0),
                }
            )
    m2_total = sum(row["total_tokens"] for row in m2_rows)
    modules = {
        "m1_physical": m1,
        "m2_llm": {
            "calls": m2_rows,
            "total_tokens": m2_total,
        },
        # M4 is symbolic search (no LLM). M5 VLM usage is not aggregated yet.
        "m4_llm": {"total_tokens": 0, "note": "no_llm"},
        "m5_llm": {"total_tokens": None, "note": "not_instrumented"},
    }
    known = [m1_total, m2_total]
    return {
        "modules": modules,
        "total_tokens": sum(known),
        "total_tokens_note": (
            "sum of instrumented modules only "
            "(m1 physical + m2 llm; m5 llm not included)"
        ),
    }


def persist_ours_timing_summary(
    out: Path,
    *,
    stage_seconds: dict[str, float | None],
    stage_status: dict[str, str],
    previous_timing: dict | None,
    wall_seconds: float | None,
    incomplete: bool = False,
) -> dict:
    """Write output/<task>/timing_summary.json for the integrated ours run."""

    llm_usage = _m2_llm_usage(out)
    summary = build_timing_summary(
        stage_order=TIMING_STAGES,
        stage_seconds=merge_stage_seconds(previous_timing, stage_seconds),
        stage_status=merge_stage_status(previous_timing, stage_status),
        llm_usage=llm_usage,
        wall_seconds=wall_seconds,
        extras={
            "pipeline": "ours",
            "plan_metrics": _plan_execution_metrics(out),
            "token_totals": _token_totals(out, llm_usage),
            "session_wall_seconds": (
                None if wall_seconds is None else round(float(wall_seconds), 2)
            ),
        },
    )
    measured = summary.get("measured_stage_sum_seconds")
    if measured is not None:
        summary["total_seconds"] = measured
    if incomplete:
        summary["incomplete"] = True
    write_json(out / "timing_summary.json", summary)
    print_timing_summary(summary, stage_order=TIMING_STAGES)
    plan = summary.get("plan_metrics") or {}
    if plan:
        print(
            "[timing] plan "
            f"subgoals={plan.get('subgoal_count')} "
            f"steps={plan.get('plan_step_count')} "
            f"ee_attach={plan.get('n_ee_attaches')} "
            f"ee_detach={plan.get('n_ee_detaches')} "
            f"planned_ee_switches={plan.get('planned_ee_switches')} "
            f"executed_ee_switches={plan.get('executed_ee_switches')}"
        )
    tokens = summary.get("token_totals") or {}
    print(
        f"[timing] tokens total={tokens.get('total_tokens')} "
        f"({tokens.get('total_tokens_note')})"
    )
    print(f"[timing] -> {out / 'timing_summary.json'}")
    return summary


# ══════════════════════════════════════════════════════════════════════
# M1
# ══════════════════════════════════════════════════════════════════════

def stage_m1(task, out, args):
    """Scene + Physical Grounding.

    scripts/run_m1.py를 호출하여:
      - scene abstraction
      - geometry grounding
      - M0 Object Knowledge retrieval
      - physical grounding
      - EE evaluation
    을 수행한다.
    """

    if args.m1_json:
        src = Path(args.m1_json).resolve()

        print(
            f"[M1] {src} 사용 "
            "(씬 재로드 / physical grounding 없음)"
        )

        if src != (out / "m1.json").resolve():
            out.mkdir(
                parents=True,
                exist_ok=True,
            )

            (out / "m1.json").write_text(
                src.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            keepout = src.parent / "m1_ee_keepout.npz"

            if keepout.exists():
                (out / "m1_ee_keepout.npz").write_bytes(keepout.read_bytes())

            npz = src.parent / "m1_points.npz"

            if npz.exists():
                (out / "m1_points.npz").write_bytes(
                    npz.read_bytes()
                )
            else:
                print(
                    f"[M1] 경고: {npz} 없음 — "
                    "점군 기반 geometry를 재사용할 수 없습니다."
                )

        return

    seed_everything(args.seed)

    module = load_script("run_m1")

    if task not in TASK_ENV:
        sys.exit(
            f"[err] 등록되지 않은 태스크 {task!r}. "
            f"등록됨: {list(TASK_ENV)}"
        )

    argv = _stage_m1_argv(
        task,
        out,
        args,
    )

    call_main(
        module,
        argv,
        "run_m1",
    )


def _stage_m1_argv(task, out, args):
    """run_m1에 M0 / VLM 관련 설정을 전달한다."""

    argv = [
        task,

        "--output-dir",
        str(out),

        "--seed",
        str(args.seed),

        "--backend",
        args.backend,

        "--model",
        args.model,

        "--memory",
        args.memory,

        "--m0-bbox-threshold",
        str(args.m0_bbox_threshold),

        "--m0-density-threshold",
        str(args.m0_density_threshold),
    ]

    if getattr(args, "view", False):
        argv.append("--view")

    return argv


# ══════════════════════════════════════════════════════════════════════
# M2
# ══════════════════════════════════════════════════════════════════════

def stage_m2(task, out, args):
    """scripts/run_m2.py — 서브골 분해."""

    module = load_script("run_m2")

    argv = [
        task,
        "--output-dir",
        str(out),
    ]

    if args.m1_json:
        argv += [
            "--m1-json",
            str(out / "m1.json"),
        ]

    call_main(
        module,
        argv,
        "run_m2",
    )


# ══════════════════════════════════════════════════════════════════════
# G_k
# ══════════════════════════════════════════════════════════════════════

def _gk_files(out):
    return [
        p
        for p in sorted(out.glob("gk_*.json"))
        if p.name != "gk_bundle.json"
    ]


def stage_gk(task, out):
    """M1 + M2 결과를 이용하여 subgoal graph를 조립한다."""

    module = load_script("assemble_gk")

    call_main(
        module,
        [
            task,
            "--output-dir",
            str(out),
        ],
        "assemble_gk",
    )

    return _gk_files(out)


# Legacy compatibility wrapper.
#
# 현재 integrated pipeline에서는 active M3 stage를 사용하지 않는다.
# 외부 코드가 stage_m3를 호출하는 경우 G_k assembly로 연결한다.
def stage_m3(task, out, args, label="M3"):
    return stage_gk(task, out)


def build_gk_bundle(out, gk_paths=None):
    """gk_<SG>.json을 M4 입력 형식으로 묶는다."""

    paths = (
        list(gk_paths)
        if gk_paths is not None
        else _gk_files(out)
    )

    if not paths:
        sys.exit(
            "[err] gk_<SG>.json 이 하나도 없습니다 — "
            "G_k assembly를 먼저 수행하십시오."
        )

    loaded = [
        (p, read_json(p))
        for p in sorted(paths)
    ]

    split_parents = {
        r.get("split_from")
        for _, r in loaded
        if r.get("split_from")
    }

    records = []
    dropped = []

    for p, r in loaded:
        sid = r.get("subgoal_id")

        if sid in split_parents:
            dropped.append(
                f"{p.name}(분할된 부모)"
            )

        elif not r.get("details"):
            dropped.append(
                f"{p.name}(detail 0건)"
            )

        else:
            records.append(r)

    if dropped:
        print(
            f"[M4] 번들에서 제외: {dropped}"
        )

    if not records:
        sys.exit(
            "[err] 번들에 넣을 gk 레코드가 없습니다 — "
            "G_k 출력을 확인하십시오."
        )

    bundle = out / "gk_bundle.json"

    bundle.write_text(
        json.dumps(
            {
                "gk_by_subgoal": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[M4] gk_bundle.json 생성: 서브골 "
        f"{[r['subgoal_id'] for r in records]} "
        f"-> {bundle}"
    )

    return bundle


# ══════════════════════════════════════════════════════════════════════
# M4
# ══════════════════════════════════════════════════════════════════════

def stage_m4(
    task,
    out,
    args,
    gk_paths=None,
):
    """scripts/run_m4.py — gk_bundle + m2 + m1 → m4.json."""

    bundle = build_gk_bundle(
        out,
        gk_paths,
    )

    module = load_script("run_m4")

    argv = [
        "--gk",
        str(bundle),

        "--m2",
        str(out / "m2.json"),

        "--m1",
        str(out / "m1.json"),

        "--robot-spec",
        str(args.robot_spec),

        "--output",
        str(out / "m4.json"),
    ]

    if args.initial_state:
        argv += [
            "--initial-state",
            str(args.initial_state),
        ]

    # Controller execution uses the validated scripted grasp registry by
    # default.  Ground that same compatibility contract before M4 searches so
    # an unsupported EE is not selected and rejected only after M5 starts.
    if "--no-scripted-grasps" not in args.m5_args:
        environment = args.m5_environment or TASK_ENV.get(task)
        if environment:
            argv += [
                "--execution-environment",
                environment,
            ]

    call_main(
        module,
        argv,
        "run_m4",
    )


# ══════════════════════════════════════════════════════════════════════
# M5
# ══════════════════════════════════════════════════════════════════════

def dump_motion_failure(exc, m5_dir):
    """MotionPlanningPipelineError의 세부 거절 사유를 출력한다."""

    from collections import Counter

    comp = getattr(
        exc,
        "compilation",
        None,
    )

    print(
        f"\n[M5] 모션 계획 실패: {exc}"
    )

    if (
        comp is None
        or not getattr(comp, "attempts", None)
    ):
        print(
            "[M5] 세부 거절 사유가 "
            "예외에 실려 있지 않습니다."
        )
        return

    hint = {
        "COLLISION_MARGIN_VIOLATION":
            "경로가 물체/랙과 충돌하거나 여유거리를 못 지킴",

        "INTERPOLATED_STATE_INVALID":
            "양 끝은 유효하나 보간 중간 자세가 무효",

        "NO_IK_BRANCH":
            "해당 pose의 IK 해가 없음",

        "KINEMATIC_SINGULARITY":
            "특이점 부근",

        "JOINT_LIMIT_VIOLATION":
            "관절 한계 초과",

        "RRT_CONNECT_EXHAUSTED":
            "샘플링 계획 반복 소진",

        "RRT_CONNECT_TIMEOUT":
            "샘플링 계획 시간 초과",

        "CARTESIAN_INTERMEDIATE_IK_FAILED":
            "직선 경로 중간점 IK 실패",
    }

    report = []

    for attempt in comp.attempts:
        sel = getattr(
            attempt,
            "selection",
            None,
        )

        edges = list(
            getattr(
                sel,
                "rejected_edges",
                (),
            )
            or ()
        )

        code = (
            attempt.failure_code
            or getattr(
                sel,
                "failure_code",
                None,
            )
            or "?"
        )

        print(
            f"\n[M5] strategy {attempt.strategy_id}: "
            f"{code} — 거절 엣지 {len(edges)}건"
        )

        counts = Counter(
            e.failure_code
            for e in edges
        )

        for c, n in counts.most_common(6):
            ex = next(
                e
                for e in edges
                if e.failure_code == c
            )

            print(
                f"       {c:34s} {n:4d}건  "
                f"{ex.source_keyframe_id} "
                f"-> {ex.target_keyframe_id}"
            )

            if ex.detail:
                print(
                    f"         └ {ex.detail[:300]}"
                )

            if c in hint:
                print(
                    f"         └ {hint[c]}"
                )

        report.append({
            "strategy_id":
                attempt.strategy_id,

            "failure_code":
                code,

            "detail":
                attempt.detail
                or getattr(
                    sel,
                    "detail",
                    "",
                ),

            "rejected_edge_counts":
                dict(counts),

            "rejected_edges": [
                {
                    "from":
                        e.source_keyframe_id,

                    "to":
                        e.target_keyframe_id,

                    "from_branch":
                        e.source_branch_id,

                    "to_branch":
                        e.target_branch_id,

                    "failure_code":
                        e.failure_code,

                    "detail":
                        e.detail,
                }
                for e in edges
            ],
        })

    path = (
        m5_dir
        / "m5_failure.json"
    )

    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[M5] 거절 엣지 전체 -> {path}"
    )


def run_m5_runner(
    module,
    argv,
    label,
    m5_dir,
):
    """M5 러너 호출."""

    try:
        call_main(
            module,
            argv,
            label,
        )

    except Exception as exc:
        if (
            type(exc).__name__
            != "MotionPlanningPipelineError"
        ):
            raise

        dump_motion_failure(
            exc,
            m5_dir,
        )

        sys.exit(
            "\n[중단] M5 모션 계획 실패 — "
            "위 거절 사유를 확인하십시오."
        )


def stage_m5(task, out, args):
    """M5 Motion Planning."""

    m4 = out / "m4.json"

    if not m4.exists():
        sys.exit(
            f"[err] {m4} 없음 — "
            "M4 를 먼저 돌리십시오."
        )

    result = read_json(m4)

    if not result.get("selected_plan"):
        print(
            "[M5] M4가 계획을 선택하지 못해 "
            "M5를 생략합니다."
        )
        return

    m5_dir = out / "m5"

    m5_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env_name = (
        args.m5_environment
        or TASK_ENV.get(task)
    )

    seed_everything(
        args.seed
    )

    if args.m5_physical:
        module = load_script(
            "run_m5"
        )

        argv = [
            task,
            "--physical",

            "--task-planner",
            str(m4),

            "--output-dir",
            str(m5_dir),
        ]

        if args.m5_validate_only:
            argv.append(
                "--validate-input-only"
            )

        argv += args.m5_args

        run_m5_runner(
            module,
            argv,
            "run_m5(physical)",
            m5_dir,
        )

    else:
        if not env_name:
            sys.exit(
                f"[err] {task!r} 의 환경 이름을 모릅니다 — "
                "--m5-environment 로 지정하거나 "
                "TASK_ENV 에 등록하십시오."
            )

        module = load_script(
            "run_m5"
        )

        argv = [
            "--task-planner",
            str(m4),

            "--environment",
            env_name,

            "--output-dir",
            str(m5_dir),

            "--seed",
            str(args.seed),

            "--provider",
            os.environ["TUJ_LLM_PROVIDER"],

            "--model",
            args.model,
        ]

        if args.m5_validate_only:
            argv.append(
                "--validate-input-only"
            )

        elif args.m5_simulate:
            argv += [
                "--simulate",
                args.m5_simulate,
                "--headless",
            ]
            # 재생은 계획과 같이 한 번만 돈다 — 리플레이 경로가 없으므로 영상을
            # 그때 안 남기면 보려고 전체를 다시 돌려야 한다. 그래서 기본으로 남긴다.
            # `--m5-args --no-video` 로 끄면 오프스크린 녹화/카메라 캡처를 생략한다.
            m5_extra = [
                a for a in args.m5_args if a != "--no-video"
            ]
            if (
                "--video" not in m5_extra
                and "--no-video" not in args.m5_args
            ):
                argv += [
                    "--video",
                    str(m5_dir / f"{task}.mp4"),
                ]
            argv += m5_extra
        else:
            argv += args.m5_args

        run_m5_runner(
            module,
            argv,
            "run_m5",
            m5_dir,
        )

    summary = (
        m5_dir
        / "m5_summary.json"
    )

    if summary.exists():
        (out / "m5.json").write_text(
            summary.read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )

        print(
            f"[M5] -> {out}/m5.json "
            f"(+ {m5_dir}/)"
        )

    else:
        print(
            f"[M5] -> {m5_dir}/ "
            "(m5_summary.json 없음 — "
            "m5.json 미생성)"
        )


# ══════════════════════════════════════════════════════════════════════
# M5 실행 재시도 + 공통 결과 스키마 (ablation 공정 비교용)
# ══════════════════════════════════════════════════════════════════════

def _m5_summary_of(out):
    """<out>/m5/m5_summary.json 을 읽는다. 없으면 {}."""
    path = Path(out) / "m5" / "m5_summary.json"
    return read_json(path) if path.is_file() else {}


def _classified(summary, error=None):
    """m5_summary 에 분류가 없으면(generic 경로/예외 경로) 여기서 보충한다."""
    if summary.get("failure_stage"):
        return summary
    if summary.get("status") == "SUCCESS" and error is None:
        return summary
    from tuj.m5_motion.failure_taxonomy import classify_failure

    detail = summary.get("detail") or (str(error) if error is not None else "")
    summary = dict(summary)
    summary.update(classify_failure(detail, error=error))
    return summary


def _api_retry_log():
    try:
        from tuj.common.api_retry import api_retry_log
        return list(api_retry_log)
    except Exception:  # noqa: BLE001
        return []


def write_run_result(out, task, args, *, method, attempts, final_error=None,
                     plan_status=None, extra=None):
    """조건(Full / w/o ...)에 상관없이 같은 필드로 <out>/run_result.json 을 쓴다.

      plan_success   M4(또는 대체 할당)가 plan 을 냈는가
      exec_success   M5 가 마지막 subgoal 까지 예외 없이 실행했는가
      goal_success   측정된 최종 상태로 본 과제 전체 목표 달성 여부 (근거 부족이면 None)
      success        실행과 목표 모두 성공해야 True; 실행 완료/목표 미평가는 None
      failure_stage  planning | execution | goal | evaluation | none
    """
    out = Path(out)
    last = attempts[-1] if attempts else {}
    m5_status = last.get("m5_status")
    exec_success = m5_status == "SUCCESS"
    goal_status = last.get("task_goal_status")
    goal_success = (
        True if goal_status == "SATISFIED"
        else False if goal_status == "FAILED"
        else None
    )
    result = {
        "schema": "run_result-v1",
        "method": method,
        "task": task,
        "seed": getattr(args, "seed", None),
        "provider": os.environ.get("TUJ_LLM_PROVIDER"),
        "model": getattr(args, "model", None),
        "m5_simulate": getattr(args, "m5_simulate", None),
        "exec_retries_allowed": getattr(args, "exec_retries", 0),
        "plan_status": plan_status,
        "plan_success": (plan_status == "SUCCESS") if plan_status is not None else None,
        "m5_status": m5_status,
        "exec_success": exec_success,
        "task_goal_status": goal_status,
        "goal_success": goal_success,
        # 외부 API 장애로 끝난 시도는 성공/실패 어느 쪽 증거도 아니다 → None (Table III 집계 제외, 재실행 대상)
        "success": (None if (not exec_success and last.get("failure_stage") == "infrastructure")
                    else goal_success if exec_success else False),
        "rerun_required": (not exec_success and last.get("failure_stage") == "infrastructure"),
        "api_retry_log": _api_retry_log(),
        "failure_stage": (("none" if goal_success is True else "goal" if goal_success is False else "evaluation")
                          if exec_success else last.get("failure_stage") or "unknown"),
        "failure_code": (("GOAL_NOT_ACHIEVED" if goal_success is False else "GOAL_EVIDENCE_MISSING" if goal_success is None else None)
                         if exec_success else last.get("failure_code")),
        "task_goal_evaluation": last.get("task_goal_evaluation"),
        "task_goal_policy": last.get("task_goal_policy"),
        "physics_related": last.get("physics_related"),
        "failed_step": last.get("failed_step"),
        "detail": last.get("detail"),
        "m5_attempt_count": len(attempts),
        "m5_attempts": attempts,
        "final_error": (
            f"{type(final_error).__name__}: {final_error}" if final_error is not None else None
        ),
    }
    if extra:
        result.update(extra)
    (out / "run_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[result] success={result['success']} exec={exec_success} goal={goal_status} "
        f"stage={result['failure_stage']} code={result['failure_code']} "
        f"attempts={len(attempts)} -> {out / 'run_result.json'}"
    )
    return result


def run_m5_with_retries(task, out, args, *, method="full", plan_status=None, extra=None):
    """stage_m5 를 돌리고, '실행(물리) 실패'일 때만 --exec-retries 만큼 다시 돈다.

    계획 실패는 같은 입력으로 다시 돌려도 같은 결과이므로 재시도하지 않는다.
    이전 시도의 m5/ 는 m5_attempt_<k>/ 로 보존된다. 모든 시도는 run_result.json 의
    m5_attempts 에 남으므로 '재시도 없는 성공률'도 나중에 계산할 수 있다.
    마지막 시도가 실패하면 원래 예외(SystemExit 포함)를 그대로 올린다.
    """
    import shutil

    out = Path(out)
    import time as _time

    retries = max(0, int(getattr(args, "exec_retries", 0) or 0))
    api_reruns = max(0, int(getattr(args, "api_reruns", 2) or 0))
    api_wait = float(os.environ.get("TUJ_API_RERUN_WAIT_S", "180"))
    attempts = []
    last_error = None
    attempt = 0
    exec_used = api_used = 0
    while True:
        if attempt > 0:
            prev = out / "m5"
            if prev.exists():
                shutil.move(str(prev), str(out / f"m5_attempt_{attempt}"))
            print(f"\n[M5] 재시도 #{attempt} (실행 실패 {exec_used}/{retries}, API 장애 {api_used}/{api_reruns})")
        last_error = None
        try:
            stage_m5(task, out, args)
        except BaseException as exc:  # SystemExit 도 기록 후 재판단
            last_error = exc
        summary = _classified(_m5_summary_of(out), last_error)
        record = {
            "attempt": attempt + 1,
            "m5_dir": str(out / "m5"),
            "m5_status": summary.get("status"),
            "task_goal_status": summary.get("task_goal_status"),
            "task_goal_evaluation": summary.get("task_goal_evaluation"),
            "task_goal_policy": summary.get("task_goal_policy"),
            "failure_stage": summary.get("failure_stage"),
            "failure_code": summary.get("failure_code"),
            "physics_related": summary.get("physics_related"),
            "failed_step": summary.get("failed_step"),
            "detail": summary.get("detail"),
            "error": f"{type(last_error).__name__}: {last_error}" if last_error else None,
        }
        attempts.append(record)
        write_run_result(out, task, args, method=method, attempts=attempts,
                         final_error=last_error, plan_status=plan_status, extra=extra)
        attempt += 1
        if last_error is None and summary.get("status") == "SUCCESS":
            break
        if record["failure_stage"] == "infrastructure":
            # 요청 단위 재시도(api_retry)를 다 쓰고도 장애 → 잠시 뒤 M5 전체를 다시 돈다.
            # 실행 실패 재시도 예산(--exec-retries)과 별도로 센다.
            if api_used >= api_reruns:
                print(f"[M5] API 장애 지속: {api_reruns}회 재실행 후 중단 → run_result.rerun_required=true")
                break
            api_used += 1
            print(f"[M5] API 장애 → {api_wait:.0f}s 후 M5 재실행 ({api_used}/{api_reruns})")
            _time.sleep(api_wait)
            continue
        if record["failure_stage"] != "execution" and not record.get("physics_related"):
            break  # 순수 계획 실패는 재시도 대상이 아니다 (물리 접촉 코드가 있으면 재시도)
        if exec_used >= retries:
            break
        exec_used += 1
    if last_error is not None:
        raise last_error
    return attempts


# ══════════════════════════════════════════════════════════════════════
# Argument Parser
# ══════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        prog="run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "M1 → M2 → G_k → M4 → M5 통합 실행기"
        ),
        epilog=(
            "예) python scripts/run.py "
            "c2_1 --model gpt-4o"
        ),
    )

    p.add_argument("--grounding-mode", choices=("full", "without_grounding", "without_reasoner"),
                   default="full",
                   help=("full=Reasoner 측정값 사용; without_reasoner=논문 'w/o Reasoner' "
                         "(측정값만 VLM 추정으로 대체, 나머지 Full 동일); "
                         "without_grounding=구 상위 grounding ablation (VLM 이 EE 직접 선택)"))
    p.add_argument(
        "--planner-mode",
        choices=("full", "without-planner", "greedy-ee-order"),
        default="full",
        help=(
            "full=M4 joint search; without-planner=M2-order suitability baseline; "
            "greedy-ee-order=myopic min EE-switch among ready subgoals "
            "(exposes C3_2 2F scripted extras only on that path)"
        ),
    )
    p.add_argument("--scene-frame", type=Path,
                   help="Matching scene image for without_grounding when reusing M1")
    p.add_argument(
        "task",
        nargs="?",
        default="c1_1",
        help=(
            "태스크 id "
            f"(등록됨: {', '.join(TASK_ENV)})"
        ),
    )

    p.add_argument(
        "--m1-json",
        default=None,
        help=(
            "M1 JSON 경로 직접 지정 "
            "(지정 시 씬 재로드 없음)"
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "씬 배치 난수 시드 — "
            "M1과 M5 환경 생성에 동일 적용"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "별도 실행 폴더 "
            "(기본 output/<task>)"
        ),
    )

    p.add_argument(
        "--backend",
        default="siphy",
        choices=(
            "siphy",
            "mock",
        ),
        help=(
            "M1 physical grounding backend"
        ),
    )

    p.add_argument(
        "--model",
        default=None,
        help=(
            "M1/M2 공통 LLM 모델. "
            "예: gemini-3.6-flash, gpt-4o"
        ),
    )

    p.add_argument(
        "--provider",
        choices=(
            "gemini",
            "openai",
        ),
        default=None,
        help=(
            "LLM 제공자 "
            "(미지정 시 --model 이름으로 추론)"
        ),
    )

    p.add_argument(
        "--memory",
        default=str(
            ROOT
            / "output"
            / "memory.json"
        ),
        help=(
            "M0 Memory Region 경로 "
            "('none' 이면 사용 안 함)"
        ),
    )

    # ----------------------------------------------------------
    # M0 Object Knowledge Retrieval
    # ----------------------------------------------------------

    p.add_argument(
        "--m0-bbox-threshold",
        type=float,
        default=0.25,
        help=(
            "M0 cross-task retrieval의 "
            "BBox 상대 차이 threshold "
            "(기본 0.25)"
        ),
    )

    p.add_argument(
        "--m0-density-threshold",
        type=float,
        default=0.20,
        help=(
            "M0 cross-task retrieval의 "
            "density 상대 차이 threshold "
            "(기본 0.20)"
        ),
    )

    # ----------------------------------------------------------
    # M4
    # ----------------------------------------------------------

    p.add_argument(
        "--robot-spec",
        default=str(
            ROOT
            / "configs"
            / "robot_spec.json"
        ),
        help="M4 로봇/EE 스펙",
    )

    p.add_argument(
        "--initial-state",
        default=None,
        help=(
            "M4 초기 상태 JSON "
            "(미지정 시 robot_spec에서 유도)"
        ),
    )

    # legacy CLI compatibility
    p.add_argument(
        "--no-roundtrip",
        action="store_true",
        help=(
            "Legacy compatibility option. "
            "현재 integrated pipeline에는 "
            "M2↔M3 round-trip이 없음"
        ),
    )

    # ----------------------------------------------------------
    # Pipeline
    # ----------------------------------------------------------

    p.add_argument(
        "--start-from",
        choices=STAGES,
        default=None,
        help=(
            "해당 모듈부터 실행 "
            "(앞 단계는 기존 산출물 재사용)"
        ),
    )

    p.add_argument(
        "--stop-after",
        choices=STAGES,
        default=None,
        help="해당 모듈까지만 실행",
    )

    p.add_argument(
        "--skip-m4",
        action="store_true",
    )

    p.add_argument(
        "--skip-m5",
        action="store_true",
    )

    # ----------------------------------------------------------
    # M5
    # ----------------------------------------------------------

    p.add_argument(
        "--m5-environment",
        default=None,
        help=(
            "M5 초기 world 캡처에 쓸 환경 이름 "
            "(기본: 태스크 기본값)"
        ),
    )

    p.add_argument(
        "--api-reruns",
        type=int,
        default=2,
        help=(
            "M5 가 외부 API 장애(503/429/타임아웃, 요청 단위 재시도 소진)로 끝났을 때 "
            "TUJ_API_RERUN_WAIT_S(기본 180초) 대기 후 M5 를 다시 도는 횟수 (기본 2). "
            "그래도 실패하면 run_result 에 success=null, rerun_required=true 로 남는다"
        ),
    )

    p.add_argument(
        "--exec-retries",
        type=int,
        default=0,
        help=(
            "M5 가 '실행(물리) 실패'로 끝났을 때만 같은 plan 으로 다시 돌리는 횟수 "
            "(기본 0 = 재시도 없음; Full 과 ablation 에 동일하게 적용할 것). "
            "이전 시도는 m5_attempt_<k>/ 로 보존되고 run_result.json 에 모두 기록된다"
        ),
    )

    p.add_argument(
        "--m5-validate-only",
        action="store_true",
        help=(
            "M5를 입력 계약 검증만 수행"
        ),
    )

    p.add_argument(
        "--m5-simulate",
        choices=(
            "kinematic",
            "controller",
        ),
        default=None,
        help=(
            "M5 계획을 MuJoCo로 헤드리스 재생하고 "
            "영상을 <출력>/m5/<태스크>.mp4 로 저장. "
            "기본은 계획만 — 재생을 켜면 계획이 실행 상태를 "
            "따라가므로 계획만 돌릴 때와 결과가 달라진다"
        ),
    )

    p.add_argument(
        "--m5-physical",
        action="store_true",
        help="물리 실행 모드",
    )

    p.add_argument(
        "--m5-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "이 뒤의 인자는 "
            "M5 러너로 그대로 전달"
        ),
    )

    p.add_argument(
        "--view",
        action="store_true",
        help="M1 단계에서 뷰어 표시",
    )

    return p


# ══════════════════════════════════════════════════════════════════════
# LLM provider
# ══════════════════════════════════════════════════════════════════════

def _infer_provider(model):
    m = (
        model
        or ""
    ).lower()

    if m.startswith("gemini"):
        return "gemini"

    if m.startswith(
        (
            "gpt",
            "o1",
            "o3",
            "o4",
            "chatgpt",
            "text-",
        )
    ):
        return "openai"

    return None


def _resolve_llm(args):
    """M1/M2에서 동일 provider/model을 사용하도록 설정한다."""

    provider = (
        args.provider
        or _infer_provider(args.model)
        or os.environ.get(
            "TUJ_LLM_PROVIDER"
        )
        or "gemini"
    )

    os.environ[
        "TUJ_LLM_PROVIDER"
    ] = provider

    if args.model:
        os.environ[
            "TUJ_M2_MODEL"
        ] = args.model

    else:
        args.model = {
            "gemini":
                "gemini-3.6-flash",

            "openai":
                "gpt-4o-mini",

        }[provider]

    os.environ[
        "TUJ_M2_MODEL"
    ] = args.model

    print(
        f"[run] LLM provider={provider} "
        f"model={args.model}"
    )


# ══════════════════════════════════════════════════════════════════════
# Integrated pipeline
# ══════════════════════════════════════════════════════════════════════

def _run_integrated(
    task,
    out,
    args,
    start,
    stop,
):
    """Current integrated pipeline.

    M1
      ↓
    M2
      ↓
    G_k Assembly
      ↓
    M4
      ↓
    M5
    """

    out = Path(out)
    run_started = time.monotonic()
    previous_timing = (
        read_json(out / "timing_summary.json")
        if (out / "timing_summary.json").is_file()
        else None
    )
    stage_seconds: dict[str, float | None] = {
        name: None for name in TIMING_STAGES
    }
    stage_status: dict[str, str] = {
        name: "not_run" for name in TIMING_STAGES
    }
    gk_paths = None

    def persist(*, incomplete: bool = False) -> dict:
        return persist_ours_timing_summary(
            out,
            stage_seconds=stage_seconds,
            stage_status=stage_status,
            previous_timing=previous_timing,
            wall_seconds=time.monotonic() - run_started,
            incomplete=incomplete,
        )

    def mark_resumed_before(index: int) -> None:
        # CLI stages: m1=0, m2=1, m4=2, m5=3. G_k is timed between m2 and m4.
        mapping = {
            0: (),
            1: ("m1",),
            2: ("m1", "m2", "gk"),
            3: ("m1", "m2", "gk", "m4"),
        }
        for name in mapping.get(index, ()):
            if stage_status[name] == "not_run":
                stage_status[name] = "resumed"

    mark_resumed_before(start)

    try:
        # ----------------------------------------------------------
        # M1
        # ----------------------------------------------------------

        if start <= 0:
            banner(
                "M1  Scene + Physical Grounding"
            )
            t0 = time.monotonic()
            stage_m1(
                task,
                out,
                args,
            )
            stage_seconds["m1"] = time.monotonic() - t0
            stage_status["m1"] = "completed"

        if stop < 1:
            persist()
            return

        # ----------------------------------------------------------
        # M2
        # ----------------------------------------------------------

        if start <= 1:
            banner(
                "M2  Subgoal Decomposition"
            )
            t0 = time.monotonic()
            stage_m2(
                task,
                out,
                args,
            )
            stage_seconds["m2"] = time.monotonic() - t0
            stage_status["m2"] = "completed"

        if stop < 2:
            persist()
            return

        # ----------------------------------------------------------
        # G_k
        # ----------------------------------------------------------

        if start <= 2:
            banner(
                "G_k  Subgoal Graph Assembly"
            )
            t0 = time.monotonic()
            gk_paths = stage_gk(
                task,
                out,
            )
            stage_seconds["gk"] = time.monotonic() - t0
            stage_status["gk"] = "completed"

        # ----------------------------------------------------------
        # M4
        # ----------------------------------------------------------

        if (
            args.skip_m4
            or start > 2
        ):
            print(
                "\n[M4] "
                + (
                    "skipped"
                    if args.skip_m4
                    else "using existing m4.json"
                )
            )
            stage_status["m4"] = (
                "skipped" if args.skip_m4 else "resumed"
            )

        else:
            banner(
                "M4  Task Planner"
            )
            t0 = time.monotonic()
            stage_m4(
                task,
                out,
                args,
                gk_paths,
            )
            stage_seconds["m4"] = time.monotonic() - t0
            stage_status["m4"] = "completed"

        if stop < 3:
            persist()
            return

        # ----------------------------------------------------------
        # M5
        # ----------------------------------------------------------

        if args.skip_m5:
            print(
                "\n[M5] skipped"
            )
            stage_status["m5"] = "skipped"
            persist()
            return

        banner(
            "M5  Motion Planner"
        )
        t0 = time.monotonic()
        m4_status = (
            (read_json(out / "m4.json") or {}).get("status")
            if (out / "m4.json").is_file()
            else None
        )
        run_m5_with_retries(
            task,
            out,
            args,
            method="full",
            plan_status=m4_status,
        )
        stage_seconds["m5"] = time.monotonic() - t0
        stage_status["m5"] = "completed"
        persist()

    except SystemExit:
        persist(incomplete=True)
        raise
    except Exception:
        persist(incomplete=True)
        raise


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = (
        build_parser()
        .parse_args()
    )

    _resolve_llm(args)

    if args.planner_mode == "without-planner":
        from run_without_planner import run
        return run(args, sys.modules[__name__])

    if args.planner_mode == "greedy-ee-order":
        from run_greedy_ee_order import run
        return run(args, sys.modules[__name__])

    if args.grounding_mode == "without_grounding":
        from run_without_grounding import run
        return run(args, sys.modules[__name__])

    if args.grounding_mode == "without_reasoner":
        from run_without_reasoner import run
        return run(args, sys.modules[__name__])

    task = args.task

    out = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT
        / "output"
        / task
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (out / "ablation_manifest.json").exists():
        sys.exit(
            "[err] This output directory belongs to an ablation; "
            "choose a separate full output."
        )

    start = (
        STAGES.index(args.start_from)
        if args.start_from
        else 0
    )

    stop = (
        STAGES.index(args.stop_after)
        if args.stop_after
        else len(STAGES) - 1
    )

    if start > stop:
        sys.exit(
            f"[err] --start-from {STAGES[start]} 이 "
            f"--stop-after {STAGES[stop]} 보다 뒤입니다."
        )

    print(
        f"[run] task={task} "
        f"seed={args.seed} "
        f"out={out}"
    )

    print(
        "[run] pipeline: "
        "M1 -> M2 -> G_k -> M4 -> M5"
    )

    return _run_integrated(
        task,
        out,
        args,
        start,
        stop,
    )


if __name__ == "__main__":
    main()
