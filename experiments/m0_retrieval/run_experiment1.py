"""M0 Experiment 1 v3 — SiPhy-prompt-extended single-call ablation (Gemini).

Usage (PowerShell examples):
  python experiments/m0_retrieval/run_experiment1.py --object bottle --condition C2 --dry-run
  python experiments/m0_retrieval/run_experiment1.py --capture-observations --object bottle
  python experiments/m0_retrieval/run_experiment1.py --object bottle --condition C1

Outputs go to output/m0_retrieval/experiment1_v3/ (does not modify experiment1_v2/).
Live runs require --object and --condition (or --all).
Does NOT call OpenAI / SiPhyBackend. Uses GEMINI_API_KEY + Gemini SDK only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from conditions import CONDITIONS
from evaluator import aggregate_condition, evaluate_inference
from gemini_backend import DEFAULT_GEMINI_MODEL, PROVIDER, detect_sdk
from gt_loader import load_all_gt
from objects import OBJECTS
from result_writer import (
    RAW_FIELDS,
    SUMMARY_FIELDS,
    merge_raw_results,
    print_object_table,
    read_csv,
    write_csv,
    write_json,
)
from siphy_runner import (
    ConditionedSiPhyRunner,
    assert_no_gt_leakage,
    format_dry_run_log,
    format_input_factors_log,
    prediction_target_labels,
)

# Separate from v2 / legacy GT-factor outputs (do not modify those dirs)
OUT_DIR = ROOT / "output" / "m0_retrieval" / "experiment1_v3"
LEGACY_V2_DIR = ROOT / "output" / "m0_retrieval" / "experiment1_v2"
M1_MANIFEST = OUT_DIR / "m1_bbox_manifest.json"
EXPERIMENT_VERSION = "experiment1_v3"


def load_m1_manifest() -> dict[str, Any]:
    """Prefer v3 manifest; fall back to read-only v2 for crops already captured."""
    for path in (M1_MANIFEST, LEGACY_V2_DIR / "m1_bbox_manifest.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def m1_bbox_for(object_key: str, manifest: dict[str, Any]) -> list[float] | None:
    entry = manifest.get(object_key) or {}
    bbox = entry.get("m1_bbox_mm")
    if bbox is None:
        return None
    return [float(x) for x in bbox]


def crop_image_path_for(object_key: str) -> Path:
    """Write target is always v3; read falls back to existing v2 crop if needed."""
    v3 = OUT_DIR / "observations" / f"{object_key}_crop.png"
    if v3.exists():
        return v3
    v2 = LEGACY_V2_DIR / "observations" / f"{object_key}_crop.png"
    if v2.exists():
        return v2
    return v3


def _print_gt_table(gts: dict) -> None:
    print("\n===== GT AVAILABILITY (eval only; never prompt) =====")
    hdr = (
        f"{'Object':8s} {'BBox':5s} {'Mat':5s} {'Dens':5s} "
        f"{'Mass':5s} {'Mu':5s} {'Young':5s}"
    )
    print(hdr)
    for key, gt in gts.items():
        def mark(field: str) -> str:
            return "OK" if gt[field]["available"] else "N/A"

        print(
            f"{key:8s} {mark('bbox_mm'):5s} {mark('material'):5s} "
            f"{mark('density_kgm3'):5s} {mark('mass_kg'):5s} "
            f"{mark('mu'):5s} {mark('youngs_gpa'):5s}"
        )


def _tok(v: int | None) -> str:
    return "unavailable" if v is None else str(v)


def _print_unit_banner(
    *,
    object_label: str,
    condition,
    model: str,
    provided: dict[str, Any],
    image_used: bool,
    dry_run: bool,
    result=None,
) -> None:
    print("=" * 66)
    print(f"[Experiment 1 v3]  output={OUT_DIR.name}")
    print()
    print("Object:")
    print(object_label)
    print()
    print("Condition:")
    print(condition.name)
    print()
    print("Provider:")
    print("Gemini")
    print()
    print("Model:")
    print(model)
    print()
    print("Input Factors:")
    print(format_input_factors_log(provided, condition))
    print()
    print("Image (object crop):")
    print("USED" if image_used else "NOT USED / MISSING")
    print()
    print("Prediction Targets:")
    for lab in prediction_target_labels(condition):
        print(lab)
    print()
    if dry_run:
        print("Expected Model Calls:")
        print("1")
        print()
        print("API Call:")
        print("SKIPPED (dry-run)")
    elif result is not None:
        print("Calling Gemini...")
        print(f"Model Call Count: {result.model_call_count}")


def _row_from_eval(
    *,
    obj_key: str,
    cid: str,
    cond,
    ev: dict,
    result,
) -> dict:
    unit_failed = bool(result.error) or bool(
        result.failure_reason
        and str(result.failure_reason).startswith(("api_", "json_"))
    )
    status = ev.get("evaluation_status")
    pred = ev["prediction"]
    err = ev["error"]
    evaluated = ev["evaluated"]
    if unit_failed:
        pred = "unavailable"
        err = "failed"
        evaluated = False
        status = "prediction_missing"
    return {
        "object": obj_key,
        "condition": cid,
        "condition_name": cond.name,
        "input_factors": cond.input_factors_label,
        "property": ev["property"],
        "gt": ev["gt"],
        "prediction": pred,
        "error": err,
        "evaluated": evaluated,
        "evaluation_status": status,
        "gt_source": ev["gt_source"],
        "provider": result.provider,
        "model": result.model,
        "input_tokens": result.input_tokens if result.input_tokens is not None else "unavailable",
        "output_tokens": result.output_tokens if result.output_tokens is not None else "unavailable",
        "total_tokens": result.total_tokens if result.total_tokens is not None else "unavailable",
        "model_call_count": result.model_call_count,
        "image_used_in_inference": result.image_used,
        "object_name_in_prompt": result.object_name_in_prompt,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
        "error_message": result.error,
        "failure_reason": result.failure_reason
        or (
            f"prediction_missing:{ev['property']}"
            if status == "prediction_missing"
            else None
        ),
    }


def run_unit(
    *,
    object_key: str,
    condition_id: str,
    model: str,
    dry_run: bool,
    gts: dict | None = None,
    m1_manifest: dict | None = None,
) -> tuple[list[dict], Any]:
    gts = gts or load_all_gt()
    m1_manifest = m1_manifest if m1_manifest is not None else load_m1_manifest()
    if object_key not in OBJECTS:
        raise SystemExit(f"unknown object: {object_key}; choose from {list(OBJECTS)}")
    if condition_id not in CONDITIONS:
        raise SystemExit(f"unknown condition: {condition_id}; choose from {list(CONDITIONS)}")

    obj = OBJECTS[object_key]
    cond = CONDITIONS[condition_id]
    gt = gts[object_key]
    m1_bbox = m1_bbox_for(object_key, m1_manifest)
    crop_path = crop_image_path_for(object_key)

    runner = ConditionedSiPhyRunner(model=model, dry_run=dry_run)
    result = runner.infer(
        cond,
        m1_bbox_mm=m1_bbox,
        crop_image_path=crop_path if crop_path.exists() else None,
        object_key=object_key,
        object_label=obj.label,
        gt_for_leak_check=gt if dry_run else None,
    )

    _print_unit_banner(
        object_label=obj.label,
        condition=cond,
        model=model,
        provided=result.provided_factors,
        image_used=result.image_used,
        dry_run=dry_run,
        result=None if dry_run else result,
    )

    if dry_run:
        info = format_dry_run_log(
            cond,
            crop_path=crop_path if crop_path.exists() else None,
            m1_bbox_mm=m1_bbox,
            prompt=result.prompt_user,
        )
        stages = info["logical_stages"]
        print()
        print(f"Condition: {cond.id}")
        print(f"Production SiPhy prompt base reused?: {info['production_siphy_prompt_reused']}")
        print(f"Logical Stage 1 input: {stages['stage1_input']}")
        print(f"Logical Stage 1 inference: {stages['stage1_inference']}")
        print(f"Logical Stage 2 input: {stages['stage2_input']}")
        print(
            "Final predicted properties: "
            + ", ".join(stages["final_predicted_properties"])
        )
        print(f"BBox provided?: {info['bbox_provided']}")
        print(f"BBox allowed during Stage 1?: {info['bbox_allowed_during_stage1']}")
        print(f"BBox used during Stage 2?: {info['bbox_used_during_stage2']}")
        print(f"GT included?: {info['gt_used_in_inference']}")
        print(f"Object name/class included?: {result.object_name_in_prompt}")
        print(f"Expected LLM API calls: {info['expected_gemini_api_calls']}")
        print("(SiPhy-style Stage 1 is NOT a separate API / SiPhyBackend call)")
        print()
        print(f"Image attached: {info['image_attached']}")
        if cond.uses_bbox:
            print(f"M1 BBox: {m1_bbox if m1_bbox is not None else 'MISSING'}")
        print()
        print("crop_image_path:")
        print(info["crop_image_path"])
        print()
        print("Prompt (user text only; image bytes not printed):")
        print(result.prompt_user)
        print()
        leaks = assert_no_gt_leakage(result.prompt_user, gt)
        print(f"GT leakage check: {'FAIL ' + str(leaks) if leaks else 'OK (none)'}")
        if result.skipped:
            print(f"SKIPPED: {result.skip_reason}")
        elif result.skip_reason:
            print(result.skip_reason)
        print("=" * 66)
        return [], result

    print()
    print("Prediction:")
    print(json.dumps(result.prediction, ensure_ascii=False, indent=2))
    print()
    print("Token Usage:")
    print(f"Input: {_tok(result.input_tokens)}")
    print(f"Output: {_tok(result.output_tokens)}")
    print(f"Total: {_tok(result.total_tokens)}")
    print()

    evals = evaluate_inference(cond, result.prediction, gt)
    print("Ground Truth:")
    for ev in evals:
        print(f"  {ev['property']}: {ev['gt']}  ({ev.get('evaluation_status')})")
    print()
    print("Error:")
    for ev in evals:
        print(
            f"  {ev['property']}: status={ev.get('evaluation_status')} "
            f"pred={ev['prediction']} error={ev['error']} evaluated={ev['evaluated']}"
        )
    if result.failure_reason:
        print(f"  failure_reason: {result.failure_reason}")

    raw_rows = [
        _row_from_eval(obj_key=object_key, cid=condition_id, cond=cond, ev=ev, result=result)
        for ev in evals
    ]

    unit_path = OUT_DIR / "units" / f"{object_key}_{condition_id}.json"
    write_json(
        unit_path,
        {
            "experiment_version": EXPERIMENT_VERSION,
            "provider": result.provider,
            "model": result.model,
            "sdk": result.sdk,
            "object": object_key,
            "condition": condition_id,
            "condition_name": cond.name,
            "reasoning_mode": cond.reasoning_mode,
            "reasoning_family": getattr(cond, "reasoning_family", "siphy_prompt_extended_single_call"),
            "prompt_mode": cond.reasoning_mode,
            "logical_first_inference": cond.logical_first_inference,
            "downstream_uses": cond.downstream_uses,
            "bbox_stage": cond.bbox_stage,
            "bbox_used": bool(cond.uses_bbox and result.bbox_mm_used is not None),
            "production_siphy_prompt_reused": True,
            "input_factors": cond.input_factors_label,
            "expected_gemini_api_calls": 1,
            "crop_image_path": result.crop_image_path,
            "bbox_mm": result.bbox_mm_used,
            "prompt_user": result.prompt_user,
            "parsed_prediction": result.prediction,
            "prediction": result.prediction,
            "raw_response": result.raw_response,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "model_call_count": result.model_call_count,
            "image_used": result.image_used,
            "object_name_in_prompt": result.object_name_in_prompt,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "error": result.error,
            "failure_reason": result.failure_reason,
            "evaluations": evals,
        },
    )
    print()
    print("Result saved:")
    print(str(unit_path))
    print("=" * 66)
    return raw_rows, result


def run_dry_run_validation(*, object_key: str | None, condition_id: str | None, model: str) -> int:
    print("=" * 66)
    print("[Experiment 1 v3] DRY-RUN / STATIC VALIDATION")
    print("=" * 66)
    print(f"Provider: {PROVIDER}")
    print(f"Default model: {DEFAULT_GEMINI_MODEL}")
    print(f"Selected model: {model}")
    print(f"Output dir: {OUT_DIR}")
    try:
        sdk = detect_sdk()
        print(f"Gemini SDK import: OK ({sdk})")
    except ImportError as exc:
        print(f"Gemini SDK import: MISSING\n{exc}")

    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)
    _print_gt_table(gts)
    m1_manifest = load_m1_manifest()
    print(f"\nM1 bbox manifest: {M1_MANIFEST} ({'found' if M1_MANIFEST.exists() else 'MISSING'})")

    obj_keys = [object_key] if object_key else list(OBJECTS)
    cond_ids = [condition_id] if condition_id else list(CONDITIONS)
    if object_key and object_key not in OBJECTS:
        raise SystemExit(f"unknown --object: {object_key}")
    if condition_id and condition_id not in CONDITIONS:
        raise SystemExit(f"unknown --condition: {condition_id}")

    for ok in obj_keys:
        for cid in cond_ids:
            run_unit(
                object_key=ok,
                condition_id=cid,
                model=model,
                dry_run=True,
                gts=gts,
                m1_manifest=m1_manifest,
            )

    meta = {
        "experiment": EXPERIMENT_VERSION,
        "mode": "dry-run",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": PROVIDER,
        "model": model,
        "image_used_in_inference": True,
        "object_name_in_prompt": False,
        "notes": [
            "v3: production SiPhy SYS_MSG reused as system-prompt base; mass/mu are LLM predictions.",
            "Logical Stage1→Stage2 inside one Gemini call; no SiPhyBackend / second LLM call.",
            "Material/Density GT and object name/class never enter prompts.",
            "M1 bbox_mm only (never GT bbox); C4/C5/C7 Stage1 is IMAGE ONLY.",
            "Outputs isolated under experiment1_v3/ (v2 untouched).",
        ],
    }
    write_json(OUT_DIR / "run_metadata.json", meta)
    return 0


def run_live(
    *,
    object_keys: list[str],
    condition_ids: list[str],
    model: str,
) -> int:
    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)
    m1_manifest = load_m1_manifest()
    raw_rows: list[dict] = []
    failed = 0

    for ok in object_keys:
        for cid in condition_ids:
            rows, result = run_unit(
                object_key=ok,
                condition_id=cid,
                model=model,
                dry_run=False,
                gts=gts,
                m1_manifest=m1_manifest,
            )
            raw_rows.extend(rows)
            if result.error or result.failure_reason:
                failed += 1

    if raw_rows:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        csv_path = OUT_DIR / f"raw_results_{stamp}.csv"
        write_csv(csv_path, raw_rows, RAW_FIELDS)

        rolling_path = OUT_DIR / "raw_results.csv"
        existing = read_csv(rolling_path)
        merged = merge_raw_results(existing, raw_rows)
        write_csv(rolling_path, merged, RAW_FIELDS)

        for ok in sorted({r["object"] for r in merged}):
            print_object_table(ok, merged)

        summary_rows = []
        for cid in sorted({r["condition"] for r in merged}):
            cond_rows = [r for r in merged if r["condition"] == cid]
            if cond_rows:
                summary_rows.append(aggregate_condition(cond_rows))
        if summary_rows:
            write_csv(OUT_DIR / "condition_summary.csv", summary_rows, SUMMARY_FIELDS)
        print(f"\n[CSV] snapshot (this run only): {csv_path}")
        print(f"[CSV] merged rolling: {rolling_path} ({len(merged)} property rows)")

    meta = {
        "experiment": EXPERIMENT_VERSION,
        "mode": "live",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": PROVIDER,
        "model": model,
        "objects": object_keys,
        "conditions": condition_ids,
        "failed_units": failed,
        "image_used_in_inference": True,
    }
    write_json(OUT_DIR / "run_metadata.json", meta)
    return 0 if failed == 0 else 1


def recompute_from_units() -> int:
    units_dir = OUT_DIR / "units"
    if not units_dir.exists():
        raise SystemExit(f"units dir not found: {units_dir}")

    gts = load_all_gt()
    paths = sorted(units_dir.glob("*_C*.json"))
    if not paths:
        raise SystemExit(f"no unit JSON files under {units_dir}")

    class _StoredResult:
        def __init__(self, payload: dict):
            self.provider = payload.get("provider", PROVIDER)
            self.model = payload.get("model")
            self.input_tokens = payload.get("input_tokens")
            self.output_tokens = payload.get("output_tokens")
            self.total_tokens = payload.get("total_tokens")
            self.model_call_count = payload.get("model_call_count", 1)
            self.image_used = payload.get("image_used", True)
            self.object_name_in_prompt = payload.get("object_name_in_prompt", False)
            self.skipped = payload.get("skipped", False)
            self.skip_reason = payload.get("skip_reason")
            self.error = payload.get("error")
            self.failure_reason = payload.get("failure_reason")

    all_rows: list[dict] = []
    print("=" * 66)
    print("[Experiment 1 v3] RECOMPUTE FROM UNIT JSON (no Gemini)")
    print("=" * 66)

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        obj_key = payload["object"]
        cid = payload["condition"]
        if obj_key not in OBJECTS or cid not in CONDITIONS:
            print(f"SKIP unknown unit: {path.name}")
            continue
        cond = CONDITIONS[cid]
        gt = gts[obj_key]
        pred = payload.get("parsed_prediction") or payload.get("prediction") or {}
        evals = evaluate_inference(cond, pred, gt)
        payload["evaluations"] = evals
        missing = [e["property"] for e in evals if e.get("evaluation_status") == "prediction_missing"]
        if missing and not payload.get("failure_reason"):
            payload["property_prediction_failures"] = missing
        write_json(path, payload)

        result = _StoredResult(payload)
        rows = [
            _row_from_eval(obj_key=obj_key, cid=cid, cond=cond, ev=ev, result=result)
            for ev in evals
        ]
        all_rows.extend(rows)
        print(
            f"  {obj_key} {cid}: "
            + str({e["property"]: e["evaluation_status"] for e in evals})
        )

    write_csv(OUT_DIR / "raw_results.csv", all_rows, RAW_FIELDS)
    summary_rows = []
    for cid in sorted({r["condition"] for r in all_rows}):
        cond_rows = [r for r in all_rows if r["condition"] == cid]
        summary_rows.append(aggregate_condition(cond_rows))
    write_csv(OUT_DIR / "condition_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_json(
        OUT_DIR / "run_metadata.json",
        {
            "experiment": EXPERIMENT_VERSION,
            "mode": "recompute_from_units",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_property_rows": len(all_rows),
            "note": "Rebuilt from units/*.json without Gemini API calls",
        },
    )
    print(f"\n[CSV] {OUT_DIR / 'raw_results.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="M0 Experiment 1 v3 (SiPhy-prompt-extended Gemini)")
    ap.add_argument("--dry-run", action="store_true", help="Validate without Gemini API calls")
    ap.add_argument(
        "--recompute-from-units",
        action="store_true",
        help="Rebuild CSV/summary from units/*.json without calling Gemini",
    )
    ap.add_argument(
        "--capture-observations",
        action="store_true",
        help="Save raw/crop/bbox PNGs + M1 bbox (sim; no LLM)",
    )
    ap.add_argument("--object", dest="objects", action="append", default=None)
    ap.add_argument("--condition", dest="conditions", action="append", default=None)
    ap.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run all objects × conditions (49). Prefer single --object/--condition.",
    )
    args = ap.parse_args()

    if args.recompute_from_units:
        return recompute_from_units()

    def _do_capture(keys: list[str] | None) -> None:
        from observation_capture import capture_all_observations, print_observation_table

        if keys:
            unknown = [k for k in keys if k not in OBJECTS]
            if unknown:
                raise SystemExit(f"unknown --object values: {unknown}")
        gts = load_all_gt()
        write_json(OUT_DIR / "gt_manifest.json", gts)
        _print_gt_table(gts)
        obs = capture_all_observations(
            OUT_DIR / "observations",
            save_images=True,
            object_keys=keys,
            gt_by_object=gts,
        )
        print_observation_table(obs)
        write_json(OUT_DIR / "observation_manifest.json", obs)
        existing = load_m1_manifest()
        for r in obs:
            if r.get("raw_status") == "OK":
                existing[r["object"]] = {
                    "m1_bbox_mm": r.get("m1_bbox_mm"),
                    "m1_center_mm": r.get("m1_center_mm"),
                    "gt_bbox_mm": r.get("gt_bbox_mm"),
                    "bbox_difference": r.get("bbox_difference"),
                    "siphy_bbox_input": r.get("siphy_bbox_input"),
                    "crop_path": r.get("crop_path"),
                    "m1_source": r.get("m1_source"),
                }
        write_json(M1_MANIFEST, existing)

    if args.capture_observations and not args.dry_run and not args.conditions and not args.all:
        _do_capture(args.objects)
        return 0

    if args.dry_run:
        if args.capture_observations:
            _do_capture(args.objects)
        ok = args.objects[0] if args.objects else None
        cid = args.conditions[0] if args.conditions else None
        return run_dry_run_validation(object_key=ok, condition_id=cid, model=args.model)

    if args.all:
        object_keys = list(OBJECTS)
        condition_ids = list(CONDITIONS)
    else:
        if not args.objects or not args.conditions:
            raise SystemExit(
                "Live Gemini run requires --object and --condition "
                "(example: --object bottle --condition C1), "
                "or pass --all for the full 49-call matrix."
            )
        object_keys = args.objects
        condition_ids = args.conditions
        for k in object_keys:
            if k not in OBJECTS:
                raise SystemExit(f"unknown --object: {k}")
        for c in condition_ids:
            if c not in CONDITIONS:
                raise SystemExit(f"unknown --condition: {c}")

    if args.capture_observations:
        _do_capture(object_keys)

    return run_live(object_keys=object_keys, condition_ids=condition_ids, model=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
