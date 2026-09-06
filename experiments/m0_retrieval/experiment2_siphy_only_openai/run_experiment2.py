"""Experiment 2 OpenAI — Production M3 / Full SiPhy only (provider=openai).

Identical production path as Gemini Experiment2:

  Materializer.query_intrinsic
    -> ground_intrinsic(node, crop, SiPhyBackend, FrictionHead)
      -> SiPhyBackend.estimate(crop, class, points_mm=_points)
      -> FrictionHead.estimate(material, surface_rms)

No selective Stage1. No Experiment-1 Stage-2. No production edits.
No modifications to experiments/m0_retrieval/experiment2_siphy_only/.

Usage:
  python experiments/m0_retrieval/experiment2_siphy_only_openai/run_experiment2.py --dry-run
  python experiments/m0_retrieval/experiment2_siphy_only_openai/run_experiment2.py ^
    --object bottle --provider openai --model gpt-5.6
  python experiments/m0_retrieval/experiment2_siphy_only_openai/run_experiment2.py ^
    --all-core --provider openai --model gpt-5.6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXP2 = Path(__file__).resolve().parent
PARENT_EXP = EXP2.parent
GEMINI_EXP2 = PARENT_EXP / "experiment2_siphy_only"
ROOT = EXP2.parents[2]

for p in (ROOT, ROOT / "src", PARENT_EXP, GEMINI_EXP2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
# Local OpenAI experiment modules must shadow sibling Gemini experiment2 names.
sys.path.insert(0, str(EXP2))

# Reuse Gemini Experiment2 M1/crop bundle builder (import only; do not edit).
from m1_bundle import CORE_OBJECT_KEYS, prepare_object_bundle  # noqa: E402
from objects import OBJECTS  # noqa: E402
from gt_loader import load_all_gt  # noqa: E402
from eval_summary import (  # noqa: E402
    aggregate_batch,
    evaluate_prediction,
    summary_row_from_unit,
)
from token_meter import MeteredOpenAIClient, TokenMeter, sanitize_create_kwargs  # noqa: E402

OUT_DIR = ROOT / "output" / "m0_retrieval" / "experiment2_siphy_only_openai"
GEMINI_OUT = ROOT / "output" / "m0_retrieval" / "experiment2_siphy_only"
METHOD = "M3 / SiPhy only (OpenAI)"
DEFAULT_PROVIDER = "openai"

SUMMARY_CSV_FIELDS = [
    "object",
    "material_gt",
    "material_pred",
    "material_match",
    "density_gt",
    "density_pred",
    "density_error",
    "mass_gt",
    "mass_pred",
    "mass_error",
    "mu_gt",
    "mu_pred",
    "mu_error",
    "youngs_gt",
    "youngs_pred",
    "overall_error",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "llm_call_count",
    "api_attempt_count",
    "failure_reason",
]

RESULT_CSV_FIELDS = [
    "method",
    "object",
    "material",
    "density_kgm3",
    "mass_kg",
    "mu",
    "youngs_gpa",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "overall_error",
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})


def _mu_scalar(mu_field: Any) -> Any:
    if isinstance(mu_field, dict):
        return mu_field.get("mu")
    return mu_field


def apply_provider_env(provider: str) -> None:
    """Set TUJ_LLM_PROVIDER for production SiPhyBackend._make_client (no prod edits)."""
    os.environ["TUJ_LLM_PROVIDER"] = str(provider).strip().lower()


def print_report(result: dict[str, Any], *, saved_paths: list[Path] | None = None) -> None:
    pred = result["prediction"]
    tok = result["token_usage"]
    print()
    print("=" * 66)
    print("Experiment 2 OpenAI - Production M3 / SiPhy Only")
    print("=" * 66)
    print(f"Object: {result['object']}")
    print(f"Provider: {result.get('provider')}")
    print(f"Model: {result['model']}")
    if result.get("failure_reason"):
        print(f"FAILURE: {result['failure_reason']}")
    print()
    print("Prediction")
    print(f"  Material : {pred.get('material')}")
    print(f"  Density  : {pred.get('density_kgm3')} kg/m3")
    print(f"  Mass     : {pred.get('mass_kg')} kg")
    print(f"  Mu       : {pred.get('mu')}")
    print(f"  Young's  : {pred.get('youngs_gpa')} GPa")
    print()
    print("Token Usage")
    print(f"  Input Tokens  : {tok.get('input_tokens')}")
    print(f"  Output Tokens : {tok.get('output_tokens')}")
    print(f"  Total Tokens  : {tok.get('total_tokens')}")
    print(f"  LLM Calls     : {tok.get('llm_call_count')}")
    if tok.get("api_attempt_count") is not None:
        print(f"  API Attempts  : {tok.get('api_attempt_count')}  (retries counted)")
    ev = result.get("evaluation") or {}
    if ev.get("overall_error") is not None:
        print()
        print(f"Overall Error (excl. Young's): {ev.get('overall_error')}")
    print()
    print("Saved:")
    for p in saved_paths or []:
        print(f"  {p}")
    print("=" * 66)


def _result_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    pred = result["prediction"]
    tok = result["token_usage"]
    return {
        "method": result.get("method", METHOD),
        "object": result["object"],
        "material": pred.get("material"),
        "density_kgm3": pred.get("density_kgm3"),
        "mass_kg": pred.get("mass_kg"),
        "mu": pred.get("mu"),
        "youngs_gpa": pred.get("youngs_gpa"),
        "llm_calls": tok.get("llm_call_count"),
        "input_tokens": tok.get("input_tokens"),
        "output_tokens": tok.get("output_tokens"),
        "total_tokens": tok.get("total_tokens"),
        "overall_error": (result.get("evaluation") or {}).get("overall_error"),
    }


def run_dry_run(*, object_keys: list[str], model: str, provider: str) -> int:
    print("=" * 66)
    print("[Experiment 2 OpenAI] DRY-RUN (no OpenAI API)")
    print("=" * 66)
    print(f"Output dir: {OUT_DIR}")
    print(f"Must differ from Gemini Exp2: {GEMINI_OUT}")
    if OUT_DIR.resolve() == GEMINI_OUT.resolve():
        print("FAIL: OpenAI OUT_DIR equals Gemini Experiment2 output")
        return 1

    apply_provider_env(provider)
    print(f"Provider (planned): {os.environ.get('TUJ_LLM_PROVIDER')}")
    print(f"Model (planned, CLI passthrough): {model}")
    print("Production path:")
    print("  Materializer.query_intrinsic")
    print("    -> tuj.m3_grounding.intrinsic.ground_intrinsic")
    print("      -> tuj.m3_grounding.siphy_backend.SiPhyBackend.estimate")
    print("      -> tuj.m3_grounding.intrinsic.FrictionHead.estimate")
    print()

    from tuj.m3_grounding import Materializer, SiPhyBackend

    errors: list[str] = []
    warnings: list[str] = []
    per_object: dict[str, Any] = {}
    resolved_model = model
    backend = None
    mat_ok = False

    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append(
            "OPENAI_API_KEY not set: skipped live client construct "
            "(observation/GT checks still run)"
        )
        print(f"[WARN] {warnings[-1]}")
    else:
        try:
            raw_client, resolved_model = SiPhyBackend._make_client(None, ROOT, model)
            burl = str(getattr(raw_client, "base_url", "") or "")
            if "generativelanguage" in burl:
                errors.append(
                    "SiPhyBackend client resolved to Gemini base_url (expected OpenAI)"
                )
            print(f"SiPhyBackend._make_client OK: resolved_model={resolved_model}")
            print(f"  client.base_url={burl or '(default OpenAI)'}")
            meter = TokenMeter()
            metered = MeteredOpenAIClient(raw_client, meter)
            backend = SiPhyBackend(
                model=resolved_model,
                client=metered,
                repo_root=ROOT,
                verbose=False,
            )
            mat_ok = True
            print(
                f"SiPhyBackend construct OK: model={backend.model} "
                f"is_gemini={backend._is_gemini}"
            )
            if backend._is_gemini:
                errors.append("SiPhyBackend._is_gemini unexpectedly True for OpenAI run")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"SiPhyBackend/Materializer wiring failed: {exc}")
            print(f"[FAIL] backend wiring: {exc}")

    # Always exercise experiment-only compat layer (no network).
    sample = {"model": model, "max_tokens": 500, "temperature": 0, "messages": []}
    sanitized, audit = sanitize_create_kwargs(sample)
    print(f"Compat sanitize audit: {audit}")
    if "temperature" in sanitized:
        errors.append("sanitize failed to drop temperature")
    if str(model).startswith("gpt-5") and "max_tokens" in sanitized:
        errors.append("sanitize failed to remap max_tokens for gpt-5*")


    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)

    for ok in object_keys:
        print("-" * 66)
        try:
            bundle = prepare_object_bundle(ok, out_dir=OUT_DIR, prefer_v4_crop=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{ok}: bundle failed: {exc}")
            print(f"[FAIL] {ok}: {exc}")
            continue
        node = bundle["node"]
        crop = bundle["crop_path"]
        print(f"Object: {ok}")
        print(f"  Crop: {crop}  source={bundle['crop_source']} exists={crop.is_file()}")
        print(f"  Node id={node['id']} class={node['class']!r}")
        print(f"  bbox_mm={list(node['bbox_mm'])}  center_mm={list(node['center_mm'])}")
        print(f"  n_points={len(node['_points'])}")
        if not crop.is_file():
            errors.append(f"{ok}: crop missing")
        if len(node["_points"]) < 3:
            errors.append(f"{ok}: insufficient _points")
        if ok not in gts:
            errors.append(f"{ok}: GT load failed")
        else:
            print(f"  GT keys: {list(gts[ok].keys())}")

        if mat_ok and backend is not None:
            try:
                m1 = {"nodes": [node], "edges": []}
                _ = Materializer(m1, backend=backend, memory=None)
                print("  Materializer construct: OK")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{ok}: Materializer construct failed: {exc}")

        per_object[ok] = {
            "crop_path": str(crop),
            "crop_source": bundle["crop_source"],
            "m1_node": {
                "id": node["id"],
                "class": node["class"],
                "bbox_mm": list(node["bbox_mm"]),
                "center_mm": list(node["center_mm"]),
                "n_points": len(node["_points"]),
            },
            "v4_bbox_comparison": bundle.get("v4_bbox_comparison"),
            "gt_loaded": ok in gts,
        }

    print()
    print("LLM input (production SiPhy): crop RGB image ONLY.")
    print("  GT used in inference = false")
    print("  object name in VLM prompt = false (cls_hint errors only)")
    print("  BBox text in VLM prompt = false (points_mm local mass only)")
    print("  Selective Stage1 = false (Full SiPhyBackend.estimate)")
    print(f"  TUJ_LLM_PROVIDER={os.environ.get('TUJ_LLM_PROVIDER')}")
    print(f"  CLI model passthrough={model!r}")

    write_json(
        OUT_DIR / "dry_run_metadata.json",
        {
            "experiment": "experiment2_siphy_only_openai",
            "mode": "dry-run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "resolved_model": resolved_model,
            "objects": object_keys,
            "per_object": per_object,
            "production_entry": "Materializer.query_intrinsic -> ground_intrinsic",
            "no_api_call": True,
            "output_dir": str(OUT_DIR),
            "gemini_exp2_output_untouched": str(GEMINI_OUT),
            "errors": errors,
            "warnings": warnings,
            "leakage_checks": {
                "gt_used_in_inference": False,
                "object_name_in_vlm_prompt": False,
                "bbox_text_in_vlm_prompt": False,
            },
        },
    )
    print(f"\nSaved dry-run metadata: {OUT_DIR / 'dry_run_metadata.json'}")
    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print(f"OK: dry-run prepared {len(object_keys)} object(s)")
    return 0


def run_one_object(
    *,
    object_key: str,
    backend: Any,
    meter: TokenMeter,
    resolved_model: str,
    provider: str,
    gt: dict[str, Any],
) -> dict[str, Any]:
    from tuj.m3_grounding import Materializer, new_gk

    bundle = prepare_object_bundle(object_key, out_dir=OUT_DIR, prefer_v4_crop=True)
    node = bundle["node"]
    crop_path = bundle["crop_path"]
    if not crop_path.is_file():
        raise RuntimeError(f"crop missing: {crop_path}")

    m1 = {"nodes": [node], "edges": []}
    mat = Materializer(m1, backend=backend, memory=None)
    gk = new_gk(f"exp2_openai_{object_key}_intrinsic")

    failure_reason = None
    response: dict[str, Any] = {}
    print(f"[exp2-openai] {object_key}: Materializer.query_intrinsic ...", flush=True)
    try:
        response = mat.query_intrinsic(
            gk,
            node_id=node["id"],
            queried_by="exp2_siphy_only_openai",
            crop_rgb=str(crop_path),
        )
    except Exception as exc:  # noqa: BLE001
        failure_reason = f"query_intrinsic_failed: {exc}"
        print(f"[exp2-openai] FAIL {object_key}: {failure_reason}", flush=True)

    logical_calls = 1
    prediction = {
        "material": response.get("material"),
        "density_kgm3": response.get("density_kgm3"),
        "mass_kg": response.get("mass_kg"),
        "mu": _mu_scalar(response.get("mu")),
        "youngs_gpa": response.get("youngs_gpa"),
    }
    evaluation = evaluate_prediction(prediction, gt)
    token_usage = {
        "input_tokens": meter.input_tokens if meter.usage_events else None,
        "output_tokens": meter.output_tokens if meter.usage_events else None,
        "total_tokens": meter.total_tokens if meter.usage_events else None,
        "llm_call_count": logical_calls,
        "api_attempt_count": meter.api_attempt_count,
        "sanitized_kwargs_log": list(meter.sanitized_kwargs_log),
        "notes": [
            "llm_call_count = logical SiPhy VLM calls (1 object x estimate).",
            "api_attempt_count = MeteredOpenAIClient create() count (includes retries).",
            "total_tokens summed from API usage.total_tokens (not recomputed as in+out).",
            "temperature never forwarded (compat layer strips if present).",
        ],
    }
    result = {
        "object": object_key,
        "method": METHOD,
        "model": resolved_model,
        "provider": provider,
        "prediction": prediction,
        "evaluation": evaluation,
        "token_usage": token_usage,
        "failure_reason": failure_reason,
        "production_path": {
            "entry": "tuj.m3_grounding.materialize.Materializer.query_intrinsic",
            "intrinsic": "tuj.m3_grounding.intrinsic.ground_intrinsic",
            "backend": "tuj.m3_grounding.siphy_backend.SiPhyBackend.estimate",
            "friction": "tuj.m3_grounding.intrinsic.FrictionHead.estimate",
            "mass": (
                "SiPhyBackend.shell_mass_integral(points_mm) when points provided; "
                "else ground_intrinsic bbox*density fallback"
            ),
        },
        "observation": {
            "crop_path": str(crop_path),
            "crop_source": bundle["crop_source"],
            "m1_bbox_mm": list(node["bbox_mm"]),
            "m1_center_mm": list(node["center_mm"]),
            "n_points": len(node["_points"]),
            "node_class": node["class"],
            "v4_bbox_comparison": bundle.get("v4_bbox_comparison"),
        },
        "mu_record": response.get("mu"),
        "geometry": response.get("geometry"),
        "confidence": response.get("confidence"),
        "materials_topk": response.get("materials_topk"),
        "gk": gk,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gt_used_in_inference": False,
        "stage2_llm": False,
        "selective_siphy": False,
    }
    return result


def run_live(*, object_keys: list[str], model: str, provider: str) -> int:
    apply_provider_env(provider)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Set it before live Experiment 2 OpenAI runs."
        )

    from tuj.m3_grounding import SiPhyBackend

    raw_client, resolved_model = SiPhyBackend._make_client(None, ROOT, model)
    burl = str(getattr(raw_client, "base_url", "") or "")
    if "generativelanguage" in burl:
        raise SystemExit(
            f"Resolved Gemini base_url unexpectedly: {burl}. "
            "Set --provider openai and OPENAI_API_KEY."
        )

    master_meter = TokenMeter()
    metered = MeteredOpenAIClient(raw_client, master_meter)
    backend = SiPhyBackend(
        model=resolved_model,
        client=metered,
        repo_root=ROOT,
        verbose=True,
    )

    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)

    units_dir = OUT_DIR / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []

    print(f"[exp2-openai] provider={provider} model={resolved_model}")
    print(f"[exp2-openai] units -> {units_dir}")

    for ok in object_keys:
        master_meter.api_attempt_count = 0
        master_meter.input_tokens = 0
        master_meter.output_tokens = 0
        master_meter.total_tokens = 0
        master_meter.usage_events = []
        master_meter.sanitized_kwargs_log = []

        result = run_one_object(
            object_key=ok,
            backend=backend,
            meter=master_meter,
            resolved_model=resolved_model,
            provider=provider,
            gt=gts.get(ok) or {},
        )
        gk = result.pop("gk")
        unit_json = units_dir / f"{ok}.json"
        write_json(unit_json, result)
        write_json(OUT_DIR / f"gk_{ok}.json", gk)

        srow = summary_row_from_unit(result, gts.get(ok) or {})
        summary_rows.append(srow)
        result_rows.append(_result_csv_row(result))
        all_results.append(result)
        write_csv_rows(units_dir / f"{ok}.csv", [srow], SUMMARY_CSV_FIELDS)
        print_report(result, saved_paths=[unit_json, units_dir / f"{ok}.csv"])

    aggregates = aggregate_batch(summary_rows)
    write_csv_rows(OUT_DIR / "summary.csv", summary_rows, SUMMARY_CSV_FIELDS)
    write_csv_rows(OUT_DIR / "result.csv", result_rows, RESULT_CSV_FIELDS)
    write_json(
        OUT_DIR / "result.json",
        {
            "experiment": "experiment2_siphy_only_openai",
            "method": METHOD,
            "provider": provider,
            "model": resolved_model,
            "objects": object_keys,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aggregates": aggregates,
            "results": all_results,
        },
    )
    if len(all_results) == 1:
        write_json(OUT_DIR / "result_single.json", all_results[0])

    print()
    print(f"[exp2-openai] Summary ({len(summary_rows)} objects): {OUT_DIR / 'summary.csv'}")
    print(f"[exp2-openai] Aggregates: {json.dumps(aggregates, ensure_ascii=False)}")
    print(f"[exp2-openai] Combined: {OUT_DIR / 'result.json'} / {OUT_DIR / 'result.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 2 OpenAI: Production M3 / Full SiPhy only"
    )
    ap.add_argument(
        "--model",
        required=False,
        default=None,
        help="OpenAI model id (CLI passthrough; no hardcode). Required for live.",
    )
    ap.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help="LLM provider for production SiPhyBackend (default: openai)",
    )
    ap.add_argument(
        "--object",
        dest="objects",
        action="append",
        default=None,
        help="Object key (repeatable).",
    )
    ap.add_argument(
        "--all-core",
        action="store_true",
        help=f"Run CORE objects: {list(CORE_OBJECT_KEYS)}",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare M1+crop; verify OpenAI wiring; no API call",
    )
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    provider = str(args.provider).strip().lower()
    if provider != "openai":
        print(
            f"[exp2-openai] WARNING: this package is OpenAI-focused; "
            f"got --provider {provider!r}"
        )

    if args.all_core:
        object_keys = list(CORE_OBJECT_KEYS)
    elif args.objects:
        object_keys = list(args.objects)
    elif args.dry_run:
        object_keys = list(CORE_OBJECT_KEYS)
    else:
        raise SystemExit(
            "Live run requires --object (repeatable) or --all-core. "
            "Example: --object bottle --provider openai --model gpt-5.6"
        )

    for ok in object_keys:
        if ok not in OBJECTS:
            raise SystemExit(f"unknown --object: {ok}; choose from {list(OBJECTS)}")

    model = args.model
    if args.dry_run:
        model = model or "gpt-4o-mini"  # dry-run display only; not sent to API
        return run_dry_run(object_keys=object_keys, model=model, provider=provider)

    if not model:
        raise SystemExit("Live run requires explicit --model <OPENAI_MODEL>")
    return run_live(object_keys=object_keys, model=model, provider=provider)


if __name__ == "__main__":
    raise SystemExit(main())
