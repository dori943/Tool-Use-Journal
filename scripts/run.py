# -*- coding: utf-8 -*-
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
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
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


# 태스크 id <-> 환경 이름 단일 출처
from task_registry import TASK_ENVS as TASK_ENV  # noqa: E402


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

    except Exception as exc:  # noqa: BLE001
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
            if "--video" not in args.m5_args:
                argv += [
                    "--video",
                    str(m5_dir / f"{task}.mp4"),
                ]

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

    gk_paths = None

    # ----------------------------------------------------------
    # M1
    # ----------------------------------------------------------

    if start <= 0:
        banner(
            "M1  Scene + Physical Grounding"
        )

        stage_m1(
            task,
            out,
            args,
        )

    if stop < 1:
        return

    # ----------------------------------------------------------
    # M2
    # ----------------------------------------------------------

    if start <= 1:
        banner(
            "M2  Subgoal Decomposition"
        )

        stage_m2(
            task,
            out,
            args,
        )

    if stop < 2:
        return

    # ----------------------------------------------------------
    # G_k
    # ----------------------------------------------------------

    if start <= 2:
        banner(
            "G_k  Subgoal Graph Assembly"
        )

        gk_paths = stage_gk(
            task,
            out,
        )

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

    else:
        banner(
            "M4  Task Planner"
        )

        stage_m4(
            task,
            out,
            args,
            gk_paths,
        )

    if stop < 3:
        return

    # ----------------------------------------------------------
    # M5
    # ----------------------------------------------------------

    if args.skip_m5:
        print(
            "\n[M5] skipped"
        )
        return

    banner(
        "M5  Motion Planner"
    )

    stage_m5(
        task,
        out,
        args,
    )


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = (
        build_parser()
        .parse_args()
    )

    _resolve_llm(args)

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
