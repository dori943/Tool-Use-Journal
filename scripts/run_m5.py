# -*- coding: utf-8 -*-
"""M5 실행기 — 태스크 id 하나로 M5 모션 계획을 돌린다.

run_m1.py 와 같은 방식이다. 태스크 id(c1_1, c2_1, ...)만 주면
  입력: output/<task>/m4.json  (M4 SelectedPlan)
  환경: 아래 TASKS 에 등록된 robosuite 환경으로 초기 world 를 캡처
  출력: output/<task>/m5/
를 자동으로 채운다. 명시적 플래그를 주면 그 값이 우선한다.

두 가지 실행 엔진을 하나로 묶었다.
  (기본) 범용 계획 — generic_runner. 태스크 무관 in-process 파이프라인.
         --simulate {kinematic,controller} 로 MuJoCo 재생까지 가능.
  (--physical) 물리 실행 — 태스크 전용 예제 러너를 subprocess 로 띄운다.
         검증된 물리 grasp/contact 프로파일로 실제 grasp+sweep 를 수행한다.
         전용 예제 러너가 있는 태스크만 지원한다(아래 PHYSICAL 참고).

사용법:
  python scripts/run_m5.py c1_1
  python scripts/run_m5.py c2_1 --validate-input-only
  python scripts/run_m5.py c1_1 --simulate controller --headless
  python scripts/run_m5.py c1_1 --physical            # 물리 실행
  python scripts/run_m5.py c1_1 --physical --stop-after-pick
  python scripts/run_m5.py --task-planner plan.json \
      --environment C1_1_LegoSweep --output-dir out/   # 태스크 없이 직접 지정

태스크 추가:
  - 범용 계획만: TASKS 에 (task id -> 환경 이름) 한 줄.
  - 물리 실행도: PHYSICAL 에 (task id -> 전용 예제 러너/프로파일/환경) 한 항목.
범용 옵션(--seed, --simulate, ...)은 generic_runner 로, 물리 옵션
(--motion-profile, --sweep-provider, --pick-keyframes, --stop-after-pick 등)은
전용 예제 러너로 그대로 전달된다.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPOSITORY / "src",
    REPOSITORY.parent / "dain-m3" / "src",
    REPOSITORY.parent / "tuj-m3" / "src",
)

# 태스크 id -> robosuite 환경 이름 (run_m1.py 의 TASKS env_name 과 동일 명명).
TASKS = {
    "c1_1": "C1_1_LegoSweep",
    "c2_1": "C2_1_ObjectSorting",
    # c1_2, c2_2: 씬 파일(environments/*.py) 추가 시 여기에 한 항목 등록
}

# 물리 실행: 태스크 id -> 전용 예제 러너 + 검증된 물리 프로파일 + 환경.
# 물리 grasp/sweep 는 태스크 전용 기하(예: C1_1 rim-grasp 바인더)에 묶여 있어
# 전용 예제 러너가 있어야 한다. 새 태스크의 물리 예제가 생기면 여기에 등록한다.
PHYSICAL = {
    "c1_1": {
        "runner": "src/tuj/m5_motion/examples/c1_1_openai_motion_run.py",
        "profile": "src/tuj/m5_motion/examples/c1_1_physical_grasp_profile.json",
        "environment": "C1_1_LegoSweep",
    },
}


def _expand_task(argv: list[str]) -> list[str]:
    """(범용) 맨 앞 위치 인자가 태스크 id면 기본 경로/환경 플래그로 펼친다.

    - 위치 인자가 없거나(예: run.py 가 플래그를 모두 넘길 때) 첫 인자가 옵션이면
      그대로 둔다.
    - 사용자가 --task-planner / --environment(또는 --initial-world) / --output-dir 을
      직접 주면 그 값을 우선하고 해당 항목만 주입하지 않는다.
    """
    if not argv or argv[0].startswith("-"):
        return list(argv)
    task, rest = argv[0], list(argv[1:])
    if task not in TASKS:
        sys.exit(f"[err] 등록되지 않은 태스크 {task!r}. 등록된 태스크: {list(TASKS)}")
    out = REPOSITORY / "output" / task
    injected: list[str] = []
    if "--task-planner" not in rest:
        injected += ["--task-planner", str(out / "m4.json")]
    if "--environment" not in rest and "--initial-world" not in rest:
        injected += ["--environment", TASKS[task]]
    if "--output-dir" not in rest:
        injected += ["--output-dir", str(out / "m5")]
    return injected + rest


def _pythonpath_environment() -> dict[str, str]:
    """전용 예제 러너 subprocess 에 src 경로들을 PYTHONPATH 로 넘긴다."""
    environment = os.environ.copy()
    roots = [str(path) for path in SOURCE_ROOTS if path.is_dir()]
    existing = environment.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def _run_physical(argv: list[str]) -> int:
    """(--physical) 태스크 전용 물리 예제 러너를 subprocess 로 실행한다.

    기존 run_m5_c1_1_motion_planner.py 가 하던 일을 태스크 레지스트리(PHYSICAL)로
    일반화한 것이다. 태스크 id 로 러너/프로파일/기본 경로를 정하고, 명시적 플래그가
    우선한다. 인식하지 못한 옵션은 전용 러너로 그대로 전달된다.
    """
    if not argv or argv[0].startswith("-"):
        sys.exit(f"[err] --physical 은 태스크 id 가 필요합니다. 등록됨: {list(PHYSICAL)}")
    task, rest = argv[0], list(argv[1:])
    if task not in PHYSICAL:
        sys.exit(
            f"[err] {task!r} 에는 물리 실행용 전용 예제 러너가 없습니다. "
            f"등록됨: {list(PHYSICAL)}"
        )
    spec = PHYSICAL[task]
    out = REPOSITORY / "output" / task
    runner = (REPOSITORY / spec["runner"]).resolve()

    parser = argparse.ArgumentParser(
        prog=f"run_m5.py {task} --physical",
        description="태스크 전용 물리 grasp/sweep 실행",
    )
    parser.add_argument("--task-planner", type=Path, default=out / "m4.json",
                        help="M4 Task Planner 결과 JSON")
    parser.add_argument("--output-dir", type=Path, default=out / "m5",
                        help="M5 계획/시뮬레이션/요약 출력 폴더")
    parser.add_argument("--motion-profile", type=Path,
                        default=REPOSITORY / spec["profile"],
                        help="검증된 물리 grasp/contact/recovery 설정")
    parser.add_argument("--environment", default=spec["environment"])
    parser.add_argument("--sweep-provider", choices=("task-geometry", "openai"),
                        default="task-geometry")
    parser.add_argument("--pick-keyframes", type=Path,
                        help="기존 상대 PICK keyframe 아티팩트 재사용")
    parser.add_argument("--validate-input-only", action="store_true",
                        help="OpenAI/MuJoCo 실행 없이 M4->M5 계약만 검증")
    parser.add_argument("--stop-after-pick", action="store_true",
                        help="선택 도구 PICK 계획/검증 후 종료")
    parser.add_argument("--dry-run", action="store_true",
                        help="실행 없이 최종 명령만 출력")
    args, forwarded = parser.parse_known_args(rest)

    task_planner = args.task_planner.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    motion_profile = args.motion_profile.expanduser().resolve()
    pick_keyframes = (
        args.pick_keyframes.expanduser().resolve()
        if args.pick_keyframes is not None else None
    )

    if not runner.is_file():
        sys.exit(f"[err] 물리 예제 러너 없음: {runner}")
    if not task_planner.is_file():
        sys.exit(
            f"[err] M4 결과 없음: {task_planner} — 먼저 "
            f"scripts/run_m4_task_planner.py 를 돌리거나 --task-planner 로 지정하십시오."
        )
    if not motion_profile.is_file():
        sys.exit(f"[err] 물리 프로파일 없음: {motion_profile}")
    if pick_keyframes is not None and not pick_keyframes.is_file():
        sys.exit(f"[err] PICK keyframe 아티팩트 없음: {pick_keyframes}")

    needs_openai = (
        not args.dry_run
        and not args.validate_input_only
        and (pick_keyframes is None or args.sweep_provider == "openai")
    )
    if needs_openai and not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "[err] keyframe 생성에 OPENAI_API_KEY 가 필요합니다. 환경변수를 설정하거나, "
            "task-geometry sweep 과 함께 --pick-keyframes 를 주거나, "
            "--validate-input-only 를 쓰십시오."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(runner), str(REPOSITORY),
        "--task-planner", str(task_planner),
        "--output-dir", str(output_dir),
        "--environment", args.environment,
        "--sweep-provider", args.sweep_provider,
        "--motion-profile", str(motion_profile),
    ]
    if pick_keyframes is not None:
        command += ["--pick-keyframes", str(pick_keyframes)]
    if args.validate_input_only:
        command.append("--validate-input-only")
    if args.stop_after_pick:
        command.append("--stop-after-pick")
    command += forwarded

    print("[M5:physical] " + shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    try:
        completed = subprocess.run(
            command, cwd=str(REPOSITORY), env=_pythonpath_environment(), check=False
        )
    except KeyboardInterrupt:
        return 130
    if completed.returncode == 0:
        print(f"[M5:physical] output: {output_dir}")
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    for source_root in reversed(SOURCE_ROOTS):
        if source_root.is_dir() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    args = list(sys.argv[1:] if argv is None else argv)
    if "--physical" in args:
        args = [a for a in args if a != "--physical"]
        return _run_physical(args)

    from tuj.m5_motion.generic_runner import main as generic_main
    return generic_main(_expand_task(args), repository=REPOSITORY)


if __name__ == "__main__":
    raise SystemExit(main())
