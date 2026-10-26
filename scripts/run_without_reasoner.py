# -*- coding: utf-8 -*-
"""w/o Reasoner (논문 Table III) runner for scripts/run.py.

Full 파이프라인과 단 하나만 다르다: M1 의 Reasoner 측정값(치수·질량·재질·마찰·표면·틈)을
버리고 VLM 이 이미지+이름만 보고 추정한 값으로 갈아끼운 m1.json 을 쓴다. 물체 identity 와
center_mm/bbox_mm(검출·시뮬레이터 관측)은 그대로 둔다. EE 판정 규칙, M2, G_k, M4, M5 는
Full 코드 그대로 호출한다. 자세한 정책은 tuj.ablations.vlm_prior 참고.

실행:  python scripts/run.py c2_1 --grounding-mode without_reasoner --m5-simulate controller
출력:  output/without_reasoner/<task>/seed_<n>/
         execution/            Full M1 원본 (mock backend, memory 미사용 — 값은 버림)
         m1_full.json          Full M1 payload 사본 (사후 비교용)
         m1.json               VLM 추정으로 대체된 payload  ← 이후 단계 입력
         m1_ee_keepout.npz      공유 EE 교체 금지영역 (M1이 생성한 경우에만)
         vlm_prior.json        VLM 원응답 + feasible_ee 표 (실패 판정 강제 변경 없음)
         m2.json, gk_*.json, m4.json, m5/, run_result.json

v2는 VLM에 세로 높이와 바닥면 크기를 별도로 요청한다. v3는 Full과 동일한 EE
판정을 보존하며, 모든 EE가 실패한 경우의 강제 fallback을 제거한다. 이전 정책의
출력 폴더에 이어 쓰지 않는다. 저장된 v2 추정을 재사용할 때도 새 폴더에서 파생
판정을 재계산하고 M2/G_k/M4를 다시 생성해야 한다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

from tuj.ablations.timing_summary import (
    build_timing_summary,
    merge_stage_seconds,
    merge_stage_status,
    print_timing_summary,
)
from tuj.ablations.vlm_prior import (
    MODE,
    POLICY_VERSION,
    VLMPriorEstimator,
    build_prior_scene,
    feasible_ee_table,
    identity_view,
)

STAGE_ORDER = ("m1", "vlm_prior", "m2", "gk", "m4", "m5")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, pipeline):
    if args.planner_mode != "full":
        raise ValueError("without_reasoner and planner ablations are separate conditions")
    if args.m5_physical:
        raise ValueError("without_reasoner uses the common generic M5 runner")
    forbidden = {"--task-planner", "--output-dir", "--seed", "--provider", "--model", "--repository"}
    if any(t.split("=", 1)[0] in forbidden for t in args.m5_args):
        raise ValueError("Set plan/output/seed/provider/model on run.py, not through --m5-args")
    start = pipeline.STAGES.index(args.start_from) if args.start_from else 0
    stop = pipeline.STAGES.index(args.stop_after) if args.stop_after else 3
    if start > stop:
        raise ValueError("--start-from must not follow --stop-after")

    task = args.task
    out = (args.output_dir.resolve() if args.output_dir
           else pipeline.ROOT / "output" / MODE / task / f"seed_{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "ablation_manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if (manifest.get("method"), manifest.get("task"), manifest.get("seed")) != (MODE, task, args.seed):
            raise ValueError("output directory belongs to a different ablation/task/seed")
        if manifest.get("policy_version") != POLICY_VERSION:
            raise ValueError("policy_version changed; rerun M1/VLM in a fresh output directory")
        if not args.start_from:
            raise ValueError("existing run requires --start-from or a fresh output directory")
    else:
        if not args.start_from and any(out.iterdir()):
            raise ValueError("new without_reasoner run requires an empty output directory")
        manifest = {"method": MODE, "policy_version": POLICY_VERSION, "task": task,
                    "seed": args.seed, "stages": {}, "artifacts": {}}
    _write(manifest_path, manifest)
    print(f"[run] method={MODE} task={task} seed={args.seed} out={out}")
    print("[without_reasoner] Reasoner measurements: REPLACED by VLM prior; M2/G_k/M4/M5: full code")

    run_started = time.monotonic()
    previous_timing = _read(out / "timing_summary.json") if (out / "timing_summary.json").is_file() else None
    stage_seconds = {n: None for n in STAGE_ORDER}
    stage_status = {n: "not_run" for n in STAGE_ORDER}
    llm_usage = None
    m2_usage_mtimes_before = {}

    def checkpoint(name):
        manifest["artifacts"][name] = _digest(out / name)
        _write(manifest_path, manifest)

    def persist_timing(*, incomplete=False):
        usage = llm_usage
        if usage is None:
            for name in ("m2.json", "m2_rejected.json"):
                path = out / name
                if not path.is_file():
                    continue
                # A failed resumed invocation must not charge old accepted or
                # rejected proposals as new API calls. Keep those files intact.
                if (stage_status["m2"] == "failed"
                        and path.stat().st_mtime_ns == m2_usage_mtimes_before.get(name)):
                    continue
                usage = _read(path).get("m2_stats", {}).get("llm_usage")
                break
        prior = _read(out / "vlm_prior.json").get("usage") if (out / "vlm_prior.json").is_file() else None
        summary = build_timing_summary(
            stage_order=STAGE_ORDER,
            stage_seconds=merge_stage_seconds(previous_timing, stage_seconds),
            stage_status=merge_stage_status(previous_timing, stage_status),
            llm_usage=usage,
            wall_seconds=time.monotonic() - run_started,
            extras={"vlm_prior_usage": prior},
        )
        if incomplete:
            summary["incomplete"] = True
        _write(out / "timing_summary.json", summary)
        manifest["timing_summary"] = summary
        _write(manifest_path, manifest)
        print_timing_summary(summary, stage_order=STAGE_ORDER)
        return summary

    stage = "m1"
    try:
        if start == 0:
            manifest["artifacts"].clear()
            manifest["stages"].clear()
            _write(manifest_path, manifest)
            execution = out / "execution"
            execution.mkdir(exist_ok=True)
            # Full M1 은 identity/위치/edge 위상만 쓰려고 돌린다. 측정 백엔드 호출과
            # 메모리 갱신은 하지 않는다 (값을 버리므로 비용·오염 모두 피한다).
            m1_args = copy.deepcopy(args)
            m1_args.backend = "mock"
            m1_args.memory = "none"
            t0 = time.monotonic()
            pipeline.stage_m1(task, execution, m1_args)
            stage_seconds["m1"] = time.monotonic() - t0
            frame = args.scene_frame or (execution / "frame.png")
            if not Path(frame).is_file():
                raise ValueError("Matching frame.png is missing; provide --scene-frame")
            shutil.copyfile(frame, out / "frame.png")
            shutil.copyfile(execution / "m1.json", out / "m1_full.json")
            # Robot sweep geometry is shared by Full and this ablation. Keep
            # it beside the replacement m1.json, where M2 load_scene reads it.
            # Do not copy m1_points.npz: measured object geometry is ablated.
            keepout_name = "m1_ee_keepout.npz"
            keepout = execution / keepout_name
            if keepout.is_file():
                shutil.copyfile(keepout, out / keepout_name)
                checkpoint(keepout_name)
            else:
                (out / keepout_name).unlink(missing_ok=True)
            for name in ("execution/m1.json", "frame.png", "m1_full.json"):
                checkpoint(name)
            manifest["stages"]["m1"] = "prepared"
            stage_status["m1"] = "prepared"

            stage = "vlm_prior"
            t0 = time.monotonic()
            full_m1 = _read(out / "m1_full.json")
            robot = _read(args.robot_spec)
            from task_registry import instruction as task_instruction
            estimator = VLMPriorEstimator(args.model, out / "frame.png")
            est = estimator.estimate(identity_view(full_m1), task_instruction(task))
            prior = build_prior_scene(full_m1, est, robot)
            _write(out / "m1.json", prior)
            table = feasible_ee_table(prior)
            _write(out / "vlm_prior.json", {
                "mode": MODE, "policy_version": POLICY_VERSION, "model": args.model,
                "estimate": est.model_dump(by_alias=True),
                "feasible_ee": table,
                "ee_fallback": prior["ablation"]["ee_fallback"],
                "usage": estimator.usage,
            })
            stage_seconds["vlm_prior"] = time.monotonic() - t0
            for name in ("m1.json", "vlm_prior.json"):
                checkpoint(name)
            manifest["stages"]["vlm_prior"] = "completed"
            stage_status["vlm_prior"] = "completed"
            _write(manifest_path, manifest)
            print("[without_reasoner] feasible_ee (VLM prior → ee_rules):")
            for nid, ees in table.items():
                fb = prior["ablation"]["ee_fallback"].get(nid)
                print(f"    {nid}: {ees}" + (f"  (fallback={fb})" if fb else ""))
        else:
            names = ["m1.json", "vlm_prior.json"]
            keepout_name = "m1_ee_keepout.npz"
            if keepout_name in manifest["artifacts"]:
                names.append(keepout_name)
            elif (out / keepout_name).exists():
                raise ValueError(f"resume found unexpected {keepout_name} from another run")
            for name in names:
                if (name not in manifest["artifacts"] or not (out / name).is_file()
                        or _digest(out / name) != manifest["artifacts"][name]):
                    raise ValueError(f"resume requires unchanged {name} from this run")
            stage_status["m1"] = stage_status["vlm_prior"] = "resumed"
        if stop == 0:
            persist_timing()
            return

        stage = "m2"
        if start <= 1:
            m2_usage_mtimes_before = {
                name: (out / name).stat().st_mtime_ns
                for name in ("m2.json", "m2_rejected.json") if (out / name).is_file()
            }
            for name in ("m2.json", "m4.json", "gk_bundle.json"):
                manifest["artifacts"].pop(name, None)
            for name in ("m2", "gk", "m4", "m5"):
                manifest["stages"].pop(name, None)
            t0 = time.monotonic()
            m2_args = copy.deepcopy(args)
            m2_args.m1_json = None  # run_m2 picks <out>/m1.json (the VLM-prior payload)
            pipeline.stage_m2(task, out, m2_args)
            stage_seconds["m2"] = time.monotonic() - t0
            llm_usage = _read(out / "m2.json").get("m2_stats", {}).get("llm_usage")
            checkpoint("m2.json")
            manifest["stages"]["m2"] = "completed"
            stage_status["m2"] = "completed"
            _write(manifest_path, manifest)
        else:
            if ("m2.json" not in manifest["artifacts"]
                    or not (out / "m2.json").is_file()
                    or _digest(out / "m2.json") != manifest["artifacts"]["m2.json"]):
                raise ValueError("resume requires unchanged accepted m2.json from this run")
            stage_status["m2"] = "resumed"
        if stop == 1:
            persist_timing()
            return

        gk_paths = None
        if start <= 2 and not args.skip_m4:
            stage = "gk"
            t0 = time.monotonic()
            gk_paths = pipeline.stage_gk(task, out)
            stage_seconds["gk"] = time.monotonic() - t0
            stage_status["gk"] = "completed"
            stage = "m4"
            t0 = time.monotonic()
            pipeline.stage_m4(task, out, args, gk_paths)
            stage_seconds["m4"] = time.monotonic() - t0
            m4 = _read(out / "m4.json")
            manifest["stages"]["m4"] = m4.get("status")
            stage_status["m4"] = str(m4.get("status"))
            checkpoint("m4.json")
            _write(manifest_path, manifest)
            if not m4.get("selected_plan"):
                raise RuntimeError(f"M4 found no plan: {m4.get('status')}; see m4.json rejections")
        else:
            stage_status["gk"] = stage_status["m4"] = "skipped" if args.skip_m4 else "resumed"
        if stop == 2 or args.skip_m5:
            persist_timing()
            return

        stage = "m5"
        manifest["stages"]["m5"] = "running"
        _write(manifest_path, manifest)
        t0 = time.monotonic()
        try:
            pipeline.run_m5_with_retries(
                task, out, args, method=MODE,
                plan_status=(_read(out / "m4.json") or {}).get("status"),
                extra={"policy_version": POLICY_VERSION,
                       "ee_fallback": _read(out / "vlm_prior.json").get("ee_fallback")},
            )
        finally:
            stage_seconds["m5"] = time.monotonic() - t0
            summary = out / "m5" / "m5_summary.json"
            manifest["stages"]["m5"] = _read(summary).get("status", "returned") if summary.is_file() else "returned"
            stage_status["m5"] = manifest["stages"]["m5"]
            _write(manifest_path, manifest)
        persist_timing()
    except BaseException as exc:
        manifest["stages"][stage] = "failed"
        if stage in stage_status:
            stage_status[stage] = "failed"
        manifest["failure"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:800],
                               "causal_attribution": "not_established"}
        rr = out / "run_result.json"
        if rr.is_file():
            r = _read(rr)
            manifest["failure"]["failure_stage"] = r.get("failure_stage")
            manifest["failure"]["failure_code"] = r.get("failure_code")
        _write(manifest_path, manifest)
        try:
            persist_timing(incomplete=True)
        except Exception:  # noqa: BLE001
            pass
        raise
