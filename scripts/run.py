# -*- coding: utf-8 -*-
"""통합 실행기 — 한 번의 실행으로 M1~M5를 순서대로 돌린다.

모듈별로 run_m1 / run_m2 / run_m3 를 따로 돌리면 robosuite 씬이 매번 새로 로드되며
배치가 달라져, 모듈마다 다른 장면을 보게 된다. 이 스크립트는 씬을 한 번만 로드하고
그 산출물(m1.json / m1_points.npz / crops)을 아래 모듈에 그대로 물려준다. --seed 로
배치 난수를 고정하므로 재실행 시 같은 장면이 재현되고, M5 가 자체적으로 다시 만드는
환경도 같은 시드로 맞춰진다.

기존 실행기를 대체하지 않고 그대로 호출한다 — 로직은 각 모듈 실행기에 한 벌만 있고
이 파일은 순서와 시드, 모듈 간 파일 연결만 책임진다.

사용법:
  python scripts/run.py c1_1              # output/c1_1/m1.json 있으면 그걸로, 없으면 mock
  python scripts/run.py c2_1
  python scripts/run.py c1_1 --m1-json path.json   # M1 JSON 경로 직접 지정
  python scripts/run.py c1_1 --start-from m5       # 앞 단계 산출물 재사용, M5 만 다시

실행 순서 (--no-roundtrip 이면 3~4 생략):
  1) M1 씬 추상화            2) M2 서브골 분해(LLM)
  3) M3 접지 1차             4) M2 재분해 + 측정 반영 + 서브골 분할
  5) M3 접지 2차(최종)       6) M4 태스크 계획      7) M5 모션 계획

산출물 (output/<task>/):
  m1.json  m1_points.npz  crops/*.png  frame*.png   ← M1 Scene Abstraction
  m2.json                                           ← M2 Subgoal Decomposition
  m3.json  gk_<SG>.json  gk_bundle.json
           m3_intrinsic.json                        ← M3 Metric & Physical Grounding
  m4.json                                           ← M4 Task Planner
  m5/  (+ m5.json = m5_summary.json 복사본)          ← M5 Motion Planner

M4 결과 파일은 모듈별 명명을 맞추려고 m4.json 으로 쓴다 —
run_m4_task_planner.py 를 직접 돌리면 같은 내용이 task_planner.json 으로 나온다.

gk_bundle.json 은 M3 가 서브골별로 남긴 gk_<SG>.json 을 이 스크립트가 모아 만든다
(M4 입력 형식). 이번 실행에서 M3 가 새로 쓴 파일만 담고, 분할된 부모 서브골과
detail 이 없는 레코드는 제외한다 — 이전 실행의 잔여 gk 는 번들에 들어가지 않는다.
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

STAGES = ("m1", "m2", "m3", "m4", "m5")

# M5 는 자체적으로 환경을 다시 만들기 때문에 환경 이름이 필요하다.
# scripts/run_m1.py 의 TASKS 와 같은 내용이며, M1 을 건너뛰어도(=--m1-json)
# robosuite import 없이 환경 이름을 알 수 있도록 여기에도 둔다.
TASK_ENV = {
    "c1_1": "C1_1_LegoSweep",
    "c2_1": "C2_1_ObjectSorting",
}


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
    """scripts/<name>.py 를 모듈로 읽는다 (패키지가 아니라 실행 스크립트라서)."""
    path = SCRIPTS / f"{name}.py"
    if not path.exists():
        sys.exit(f"[err] {path} 없음")
    spec = importlib.util.spec_from_file_location(f"_tuj_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_main(module, argv, label):
    """모듈의 main()을 argv로 호출한다. sys.exit 로 중단되면 파이프라인도 멈춘다.

    run_m1~m3 은 sys.argv 를 직접 읽고, run_m4/m5 는 main(argv) 를 받는다.
    양쪽 모두 처리한다.
    """
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
    except SystemExit as exc:                 # sys.exit("메시지") / exit(code)
        code = exc.code
        if isinstance(code, str):
            print(code)
            sys.exit(f"\n[중단] {label} 단계에서 멈췄습니다. 위 메시지를 확인하십시오.")
        rc = code or 0
    finally:
        sys.argv = saved

    if rc:
        sys.exit(f"\n[중단] {label} 단계가 exit={rc} 로 끝났습니다.")
    return rc


def seed_everything(seed):
    """씬 배치 난수 고정 — robosuite 배치 샘플러가 전역 RNG를 쓴다."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════
# 각 단계
# ══════════════════════════════════════════════════════════════════════

def stage_m1(task, out, args):
    """scripts/run_m1.py — 씬 로드 1회, m1.json / m1_points.npz / crops 생성."""
    if args.m1_json:
        src = Path(args.m1_json).resolve()
        print(f"[M1] {src} 사용 (씬 재로드 없음)")
        if src != (out / "m1.json").resolve():
            out.mkdir(parents=True, exist_ok=True)
            (out / "m1.json").write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8")
            npz = src.parent / "m1_points.npz"
            if npz.exists():
                (out / "m1_points.npz").write_bytes(npz.read_bytes())
            else:
                print(f"[M1] 경고: {npz} 없음 — M3 접지가 점군을 찾지 못합니다.")
        return

    seed_everything(args.seed)
    module = load_script("run_m1")
    if task not in module.TASKS:
        sys.exit(f"[err] run_m1.TASKS 에 {task!r} 없음: {list(module.TASKS)}")
    argv = [task] + (["--view"] if args.view else [])
    call_main(module, argv, "run_m1")


def stage_m2(task, out, args, pass_no):
    """scripts/run_m2.py — 서브골 분해(LLM).

    1차: m3.json 이 없어야 순수 분해가 된다. 2차: m3.json 을 읽어 측정 반영 + 분할.
    """
    m3 = out / "m3.json"
    if pass_no == 1 and m3.exists():
        # 이전 실행의 응답이 남아 있으면 run_m2 가 그것을 이번 분해에 섞거나
        # 안전장치에 걸려 멈춘다. 1차는 항상 깨끗한 분해여야 한다.
        backup = out / "m3.prev.json"
        m3.replace(backup)
        print(f"[M2] 이전 m3.json 을 {backup.name} 으로 옮기고 새로 분해합니다.")

    module = load_script("run_m2")
    argv = [task]
    if args.m1_json:
        argv += ["--m1-json", str(out / "m1.json")]
    call_main(module, argv, "run_m2")


def _gk_files(out):
    return [p for p in sorted(out.glob("gk_*.json")) if p.name != "gk_bundle.json"]


def stage_m3(task, out, args, label="M3"):
    """scripts/run_m3.py — 물리/기하 접지, m3.json + gk_<SG>.json 생성.

    반환: 이번 호출에서 새로 쓰인 gk 파일 목록 (이전 실행의 잔여 파일 배제용).
    """
    before = {p: p.stat().st_mtime for p in _gk_files(out)}
    module = load_script("run_m3")
    argv = [task, "--backend", args.backend, "--model", args.model,
            "--memory", args.memory]
    call_main(module, argv, "run_m3")
    fresh = [p for p in _gk_files(out)
             if p not in before or p.stat().st_mtime > before[p]]
    stale = [p.name for p in _gk_files(out) if p not in fresh]
    if stale:
        print(f"[{label}] 이번 실행에서 갱신되지 않은 gk 파일: {stale}")
    return fresh


def build_gk_bundle(out, gk_paths=None):
    """M3 의 gk_<SG>.json 을 M4 입력 형식(gk_by_subgoal)으로 묶는다.

    gk_paths 가 주어지면 그 파일만 후보로 삼는다 (이번 실행이 쓴 것들).
    분할된 부모 서브골(자식의 split_from 으로 지목된 id)과 detail 이 없는
    레코드는 제외한다 — 부모까지 넣으면 같은 작업을 두 번 계획하게 된다.
    """
    paths = list(gk_paths) if gk_paths is not None else _gk_files(out)
    if not paths:
        sys.exit("[err] gk_<SG>.json 이 하나도 없습니다 — M3 를 먼저 돌리십시오.")
    loaded = [(p, read_json(p)) for p in sorted(paths)]
    split_parents = {r.get("split_from") for _, r in loaded if r.get("split_from")}

    records, dropped = [], []
    for p, r in loaded:
        sid = r.get("subgoal_id")
        if sid in split_parents:
            dropped.append(f"{p.name}(분할된 부모)")
        elif not r.get("details"):
            dropped.append(f"{p.name}(detail 0건)")
        else:
            records.append(r)
    if dropped:
        print(f"[M4] 번들에서 제외: {dropped}")
    if not records:
        sys.exit("[err] 번들에 넣을 gk 레코드가 없습니다 — M3 출력을 확인하십시오.")

    bundle = out / "gk_bundle.json"
    bundle.write_text(
        json.dumps({"gk_by_subgoal": records}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[M4] gk_bundle.json 생성: 서브골 "
          f"{[r['subgoal_id'] for r in records]} -> {bundle}")
    return bundle


def stage_m4(task, out, args, gk_paths=None):
    """scripts/run_m4.py — gk_bundle + m2 + m1 → m4.json."""
    bundle = build_gk_bundle(out, gk_paths)
    module = load_script("run_m4")
    argv = ["--gk", str(bundle),
            "--m2", str(out / "m2.json"),
            "--m1", str(out / "m1.json"),
            "--robot-spec", str(args.robot_spec),
            "--output", str(out / "m4.json")]
    if args.initial_state:
        argv += ["--initial-state", str(args.initial_state)]
    call_main(module, argv, "run_m4")


def dump_motion_failure(exc, m5_dir):
    """MotionPlanningPipelineError 가 물고 있는 거절 사유를 펼쳐 보여준다.

    예외 메시지는 "NO_CONNECTED_SEQUENCE (evaluated 196 branch edges)" 까지만
    말해준다. 왜 196개가 전부 거절됐는지는 compilation.attempts[].selection
    .rejected_edges 에 코드와 detail 로 들어 있다.
    """
    from collections import Counter

    comp = getattr(exc, "compilation", None)
    print(f"\n[M5] 모션 계획 실패: {exc}")
    if comp is None or not getattr(comp, "attempts", None):
        print("[M5] 세부 거절 사유가 예외에 실려 있지 않습니다.")
        return

    hint = {
        "COLLISION_MARGIN_VIOLATION":
            "경로가 물체/랙과 충돌하거나 여유거리를 못 지킴 — detail 의 geom 쌍을 확인",
        "INTERPOLATED_STATE_INVALID":
            "양 끝은 유효하나 보간 중간 자세가 무효 — 스텝을 줄이거나 경유 keyframe 필요",
        "NO_IK_BRANCH": "그 포즈에 대한 IK 해가 없음 — 도달 범위 밖이거나 자세가 과함",
        "KINEMATIC_SINGULARITY": "특이점 부근 — 접근 자세를 바꿔야 함",
        "JOINT_LIMIT_VIOLATION": "관절 한계 초과",
        "RRT_CONNECT_EXHAUSTED": "샘플링 계획 반복 소진 — --options 로 반복/시간 상향 여지",
        "RRT_CONNECT_TIMEOUT": "샘플링 계획 시간 초과 — --options 로 시간 상향 여지",
        "CARTESIAN_INTERMEDIATE_IK_FAILED":
            "직선 경로 중간점 IK 실패 — 샘플링 계획으로 우회 필요",
    }

    report = []
    for attempt in comp.attempts:
        sel = getattr(attempt, "selection", None)
        edges = list(getattr(sel, "rejected_edges", ()) or ())
        code = attempt.failure_code or getattr(sel, "failure_code", None) or "?"
        print(f"\n[M5] strategy {attempt.strategy_id}: {code} — 거절 엣지 {len(edges)}건")
        counts = Counter(e.failure_code for e in edges)
        for c, n in counts.most_common(6):
            ex = next(e for e in edges if e.failure_code == c)
            print(f"       {c:34s} {n:4d}건  "
                  f"{ex.source_keyframe_id} -> {ex.target_keyframe_id}")
            if ex.detail:
                print(f"         └ {ex.detail[:300]}")
            if c in hint:
                print(f"         └ {hint[c]}")
        report.append({
            "strategy_id": attempt.strategy_id,
            "failure_code": code,
            "detail": attempt.detail or getattr(sel, "detail", ""),
            "rejected_edge_counts": dict(counts),
            "rejected_edges": [
                {"from": e.source_keyframe_id, "to": e.target_keyframe_id,
                 "from_branch": e.source_branch_id, "to_branch": e.target_branch_id,
                 "failure_code": e.failure_code, "detail": e.detail}
                for e in edges
            ],
        })

    path = m5_dir / "m5_failure.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[M5] 거절 엣지 전체 -> {path}")


def run_m5_runner(module, argv, label, m5_dir):
    """M5 러너 호출 — 모션 계획 실패는 사유를 펼친 뒤 파이프라인을 멈춘다."""
    try:
        call_main(module, argv, label)
    except Exception as exc:                      # noqa: BLE001
        if type(exc).__name__ != "MotionPlanningPipelineError":
            raise
        dump_motion_failure(exc, m5_dir)
        sys.exit("\n[중단] M5 모션 계획 실패 — 위 거절 사유를 확인하십시오.")


def stage_m5(task, out, args):
    """M5 모션 계획.

    기본은 태스크 비의존 범용 러너(run_m5.py). --m5-physical
    이면 같은 러너의 물리 실행 모드(--physical)를 쓴다 — 태스크 전용 물리 예제
    러너를 subprocess 로 띄운다(현재 c1_1 만 지원).

    범용 러너는 --environment 로 환경을 다시 만들어 초기 WorldSnapshot 을 뜬다.
    같은 프로세스 안에서 M1 과 같은 시드를 다시 심어 배치를 맞춘다.
    """
    m4 = out / "m4.json"
    if not m4.exists():
        sys.exit(f"[err] {m4} 없음 — M4 를 먼저 돌리십시오.")
    result = read_json(m4)
    if not result.get("selected_plan"):
        print("[M5] M4 가 계획을 선택하지 못해 생략합니다.")
        return

    m5_dir = out / "m5"
    m5_dir.mkdir(parents=True, exist_ok=True)
    env_name = args.m5_environment or TASK_ENV.get(task)

    seed_everything(args.seed)
    if args.m5_physical:
        module = load_script("run_m5")
        argv = [task, "--physical",
                "--task-planner", str(m4), "--output-dir", str(m5_dir)]
        if args.m5_validate_only:
            argv.append("--validate-input-only")
        argv += args.m5_args
        run_m5_runner(module, argv, "run_m5(physical)", m5_dir)
    else:
        if not env_name:
            sys.exit(f"[err] {task!r} 의 환경 이름을 모릅니다 — "
                     f"--m5-environment 로 지정하거나 TASK_ENV 에 등록하십시오.")
        module = load_script("run_m5")
        argv = ["--task-planner", str(m4),
                "--environment", env_name,
                "--output-dir", str(m5_dir),
                "--seed", str(args.seed)]
        if args.m5_validate_only:
            argv.append("--validate-input-only")
        elif args.m5_simulate:
            argv += ["--simulate", args.m5_simulate, "--headless"]
        argv += args.m5_args
        run_m5_runner(module, argv, "run_m5", m5_dir)

    summary = m5_dir / "m5_summary.json"
    if summary.exists():
        (out / "m5.json").write_text(
            summary.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[M5] -> {out}/m5.json (+ {m5_dir}/)")
    else:
        print(f"[M5] -> {m5_dir}/ (m5_summary.json 없음 — m5.json 미생성)")


# ══════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        prog="run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="M1~M5 통합 실행기 (씬 1회 로드 → 전 모듈 순차 실행)",
        epilog="예) python scripts/run.py c1_1 --backend mock --m5-validate-only")
    p.add_argument("task", nargs="?", default="c1_1",
                   help=f"태스크 id (등록됨: {', '.join(TASK_ENV)})")
    p.add_argument("--m1-json", default=None,
                   help="M1 JSON 경로 직접 지정 (지정 시 씬 재로드 없음)")
    p.add_argument("--seed", type=int, default=0,
                   help="씬 배치 난수 시드 — M1 과 M5 환경 생성에 동일 적용")
    p.add_argument("--backend", default="siphy", choices=("siphy", "mock"),
                   help="M3 물성 백엔드")
    p.add_argument("--model", default="gpt-4o-mini", help="M3 백엔드 VLM 모델")
    p.add_argument("--memory", default=str(ROOT / "output" / "memory.json"),
                   help="M3 물성 메모리 경로 ('none' 이면 사용 안 함)")
    p.add_argument("--robot-spec", default=str(ROOT / "configs" / "robot_spec.json"),
                   help="M4 로봇/EE 스펙")
    p.add_argument("--initial-state", default=None,
                   help="M4 초기 상태 JSON (미지정 시 robot_spec 에서 유도)")
    p.add_argument("--no-roundtrip", action="store_true",
                   help="M3→M2 측정 반영 및 서브골 분할 왕복을 생략")
    p.add_argument("--start-from", choices=STAGES, default=None,
                   help="해당 모듈부터 실행 (앞 단계는 기존 산출물 재사용)")
    p.add_argument("--stop-after", choices=STAGES, default=None,
                   help="해당 모듈까지만 실행")
    p.add_argument("--skip-m4", action="store_true")
    p.add_argument("--skip-m5", action="store_true")
    p.add_argument("--m5-environment", default=None,
                   help="M5 초기 world 캡처에 쓸 환경 이름 (기본: 태스크 기본값)")
    p.add_argument("--m5-validate-only", action="store_true",
                   help="M5 를 입력 계약 검증만 수행 (OpenAI/MuJoCo 실행 없음)")
    p.add_argument("--m5-simulate", choices=("kinematic", "controller"),
                   default=None, help="M5 계획을 MuJoCo 로 헤드리스 재생")
    p.add_argument("--m5-physical", action="store_true",
                   help="물리 실행 모드 (태스크 전용 물리 예제 러너, 현재 c1_1)")
    p.add_argument("--m5-args", nargs=argparse.REMAINDER, default=[],
                   help="이 뒤의 인자는 M5 러너로 그대로 전달")
    p.add_argument("--view", action="store_true", help="M1 단계에서 뷰어 표시")
    return p


def main():
    args = build_parser().parse_args()
    task = args.task
    out = ROOT / "output" / task
    out.mkdir(parents=True, exist_ok=True)
    start = STAGES.index(args.start_from) if args.start_from else 0
    stop = STAGES.index(args.stop_after) if args.stop_after else len(STAGES) - 1
    if start > stop:
        sys.exit(f"[err] --start-from {STAGES[start]} 이 --stop-after {STAGES[stop]} 보다 뒤입니다.")
    gk_paths = None

    print(f"[run] task={task} seed={args.seed} out={out}")
    print(f"[run] 단계: {' -> '.join(STAGES[start:stop + 1])}"
          + ("" if not args.no_roundtrip else "  (M2<->M3 왕복 생략)")
          + ("" if start == 0 else f"  (m1~{STAGES[start - 1]} 는 기존 산출물 재사용)"))

    if start <= 0:
        banner("M1  Scene Abstraction")
        stage_m1(task, out, args)
    if stop < 1:
        return

    if start > 2:
        pass                                   # M2·M3 는 기존 산출물 사용
    elif start == 2:
        banner("M3  Metric & Physical Grounding")
        gk_paths = stage_m3(task, out, args)
    elif args.no_roundtrip:
        banner("M2  Subgoal Decomposition")
        stage_m2(task, out, args, pass_no=1)
        if stop < 2:
            return
        banner("M3  Metric & Physical Grounding")
        gk_paths = stage_m3(task, out, args)
    else:
        banner("M2  Subgoal Decomposition (1차 — 측정 전 분해)")
        stage_m2(task, out, args, pass_no=1)
        if stop < 2:
            return
        banner("M3  Metric & Physical Grounding (1차 — M2 반영용)")
        stage_m3(task, out, args, label="M3.1")
        banner("M2  재분해 + 측정 반영 + 서브골 분할 (2차)")
        stage_m2(task, out, args, pass_no=2)
        banner("M3  Metric & Physical Grounding (2차 — 최종)")
        gk_paths = stage_m3(task, out, args, label="M3.2")
    if stop < 3:
        return

    if args.skip_m4 or start > 3:
        print("\n[M4] " + ("--skip-m4 로 생략" if args.skip_m4 else "기존 m4.json 재사용"))
    else:
        banner("M4  Task Planner")
        stage_m4(task, out, args, gk_paths)
    if stop < 4:
        return

    if args.skip_m5:
        print("\n[M5] --skip-m5 로 생략")
        return
    banner("M5  Motion Planner")
    stage_m5(task, out, args)

    banner(f"DONE  -> {out}")


if __name__ == "__main__":
    main()
