# -*- coding: utf-8 -*-
"""M5 실행기 — 태스크 id 하나로 M5 모션 계획을 돌린다.

run_m1.py 와 같은 방식이다. 등록된 태스크 id 하나를 주면
  입력: output/<task>/m4.json  (M4 SelectedPlan)
  환경: task_registry 에 등록된 robosuite 환경으로 초기 world 를 캡처
  출력: output/<task>/m5/
를 자동으로 채운다. 명시적 플래그를 주면 그 값이 우선한다.

기본 실행은 M1/M4 입력과 EE capability로 전체 trajectory를 계획하는 범용
planner 경로다. 태스크 전용 실험 runner는 examples에서 직접 실행한다.

사용법:
  python scripts/run_m5.py <task_id>
  python scripts/run_m5.py <task_id> --validate-input-only
  python scripts/run_m5.py <task_id> --simulate controller
  python scripts/run_m5.py <task_id> --simulate controller --headless
  python scripts/run_m5.py --task-planner plan.json \
      --environment RegisteredEnvironment --output-dir out/

태스크 추가:
  - 범용 계획: task_registry.TASKS 에 한 줄.
  - 물리 PICK: 환경 object record에 grasp feature/물리 메타데이터를 제공.
범용 옵션(--seed, --simulate, ...)은 generic_runner 로 전달된다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (REPOSITORY / "src",)

# 태스크 id <-> 환경 이름은 단일 출처(task_registry)에서 가져온다.
sys.path.insert(0, str(REPOSITORY))
from task_registry import TASK_ENVS  # noqa: E402


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
    if task not in TASK_ENVS:
        sys.exit(f"[err] 등록되지 않은 태스크 {task!r}. 등록된 태스크: {list(TASK_ENVS)}")
    out = REPOSITORY / "output" / task
    injected: list[str] = []
    if "--task-planner" not in rest:
        injected += ["--task-planner", str(out / "m4.json")]
    if "--initial-world" not in rest and (out / "m1_world.json").is_file():
        injected += ["--initial-world", str(out / "m1_world.json")]
    if "--environment" not in rest and "--initial-world" not in rest:
        injected += ["--environment", TASK_ENVS[task]]
    if "--output-dir" not in rest:
        injected += ["--output-dir", str(out / "m5")]
    if "--scene-geometry" not in rest and (out / "m1.json").is_file():
        injected += ["--scene-geometry", str(out / "m1.json")]
    if "--id-aliases" not in rest and (out / "id_aliases.json").is_file():
        injected += ["--id-aliases", str(out / "id_aliases.json")]
    # Generic runs use common defaults or an explicitly supplied profile.
    # Never inject a scenario-specific grasp profile based on the task id.
    return injected + rest


def main(argv: Sequence[str] | None = None) -> int:
    for source_root in reversed(SOURCE_ROOTS):
        if source_root.is_dir():
            if str(source_root) in sys.path:
                sys.path.remove(str(source_root))
            sys.path.insert(0, str(source_root))

    args = list(sys.argv[1:] if argv is None else argv)
    from tuj.m5_motion.generic_runner import main as generic_main
    return generic_main(_expand_task(args), repository=REPOSITORY)


if __name__ == "__main__":
    raise SystemExit(main())
