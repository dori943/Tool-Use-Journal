"""Experiment 2 — Production M3 / SiPhy only (multi-object).

Runs the real production intrinsic grounding path per object:

  Materializer.query_intrinsic
    → ground_intrinsic(node, crop, SiPhyBackend, FrictionHead)
      → SiPhyBackend.estimate(crop, class, points_mm=_points)
      → FrictionHead.estimate(material, surface_rms)

No Experiment-1 Stage-2 Gemini. No GT in prompts.
Does not modify src/tuj or experiment1_*.

Usage:
  python experiments/m0_retrieval/experiment2_siphy_only/run_experiment2.py --dry-run
  python experiments/m0_retrieval/experiment2_siphy_only/run_experiment2.py \\
    --object bottle --object spoon --object ladle --object plate --object mug \\
    --model gemini-3.6-flash
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
ROOT = EXP2.parents[2]

for p in (ROOT, ROOT / "src", PARENT_EXP, EXP2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from m1_bundle import CORE_OBJECT_KEYS, prepare_object_bundle  # noqa: E402
from objects import OBJECTS  # noqa: E402
from token_meter import MeteredOpenAIClient, TokenMeter  # noqa: E402

OUT_DIR = ROOT / "output" / "m0_retrieval" / "experiment2_siphy_only"
DEFAULT_MODEL = "gemini-3.6-flash"
METHOD = "M3 / SiPhy only"

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


def print_report(result: dict[str, Any], *, saved_paths: list[Path] | None = None) -> None:
    pred = result["prediction"]
    tok = result["token_usage"]
    print()
    print("=" * 66)
    print("Experiment 2 - Production M3 / SiPhy Only")
    print("=" * 66)
    print(f"Object: {result['object']}")
    print(f"Model: {result['model']}")
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
    print()
    print("Saved:")
    for p in saved_paths or []:
        print(f"  {p}")
    print("=" * 66)


def _csv_row_from_result(result: dict[str, Any]) -> dict[str, Any]:
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
    }


def run_dry_run(*, object_keys: list[str], model: str) -> int:
    """No API: verify observation + production call wiring for each object."""
    print("=" * 66)
    print("[Experiment 2] DRY-RUN (no Gemini API)")
    print("=" * 66)
    print(f"Model (planned): {model}")
    print(f"Objects: {object_keys}")
    print("Production path (object-independent):")
    print("  Materializer.query_intrinsic")
    print("    → tuj.m3_grounding.intrinsic.ground_intrinsic")
    print("      → tuj.m3_grounding.siphy_backend.SiPhyBackend.estimate")
    print("      → tuj.m3_grounding.intrinsic.FrictionHead.estimate")
    print()

    per_object: dict[str, Any] = {}
    errors: list[str] = []

    for ok in object_keys:
        print("-" * 66)
        try:
            bundle = prepare_object_bundle(ok, out_dir=OUT_DIR, prefer_v4_crop=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{ok}: bundle failed: {exc}")
            print(f"[FAIL] {ok}: {exc}")
            continue
        node = bundle["node"]
        print(f"Object: {ok}")
        print(f"  Crop: {bundle['crop_path']}  source={bundle['crop_source']}")
        print(f"  Node id={node['id']} class={node['class']!r}")
        print(f"  bbox_mm={list(node['bbox_mm'])}  center_mm={list(node['center_mm'])}")
        print(f"  n_points={len(node['_points'])}")
        if bundle.get("v4_bbox_comparison"):
            print(f"  v4 bbox compare: {bundle['v4_bbox_comparison']}")
        if not bundle["crop_path"].is_file():
            errors.append(f"{ok}: crop missing")
        if len(node["_points"]) < 3:
            errors.append(f"{ok}: insufficient _points")
        per_object[ok] = {
            "crop_path": str(bundle["crop_path"]),
            "crop_source": bundle["crop_source"],
            "m1_node": {
                "id": node["id"],
                "class": node["class"],
                "bbox_mm": list(node["bbox_mm"]),
                "center_mm": list(node["center_mm"]),
                "n_points": len(node["_points"]),
            },
            "v4_bbox_comparison": bundle.get("v4_bbox_comparison"),
        }

    print()
    print("LLM input (production SiPhy): crop RGB image ONLY.")
    print("  - GT: not used")
    print("  - object name: not in VLM messages (cls_hint only for errors)")
    print("  - points_mm: local shell_mass_integral only (not sent to VLM)")
    print("  - Stage2 Gemini: NONE")
    print("Token meter: fresh TokenMeter per object (same wrapper).")

    write_json(
        OUT_DIR / "dry_run_metadata.json",
        {
            "experiment": "experiment2_siphy_only",
            "mode": "dry-run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "objects": object_keys,
            "per_object": per_object,
            "production_entry": "Materializer.query_intrinsic → ground_intrinsic",
            "no_api_call": True,
            "errors": errors,
        },
    )
    print(f"\nSaved dry-run metadata: {OUT_DIR / 'dry_run_metadata.json'}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print(f"OK: dry-run prepared {len(object_keys)} object(s)")
    return 0


def run_one_object(
    *,
    object_key: str,
    model: str,
    backend: Any,
    meter: TokenMeter,
    resolved_model: str,
) -> dict[str, Any]:
    """Run production M3 intrinsic for one object; meter is reset by caller."""
    from tuj.m3_grounding import Materializer, new_gk

    bundle = prepare_object_bundle(object_key, out_dir=OUT_DIR, prefer_v4_crop=True)
    node = bundle["node"]
    crop_path = bundle["crop_path"]
    if not crop_path.is_file():
        raise RuntimeError(f"crop missing: {crop_path}")

    m1 = {"nodes": [node], "edges": []}
    mat = Materializer(m1, backend=backend, memory=None)
    gk = new_gk(f"exp2_{object_key}_intrinsic")

    print(f"[exp2] {object_key}: Materializer.query_intrinsic ...", flush=True)
    response = mat.query_intrinsic(
        gk,
        node_id=node["id"],
        queried_by="exp2_siphy_only",
        crop_rgb=str(crop_path),
    )

    logical_calls = 1
    prediction = {
        "material": response.get("material"),
        "density_kgm3": response.get("density_kgm3"),
        "mass_kg": response.get("mass_kg"),
        "mu": _mu_scalar(response.get("mu")),
        "youngs_gpa": response.get("youngs_gpa"),
    }
    token_usage = {
        "input_tokens": meter.input_tokens if meter.usage_events else None,
        "output_tokens": meter.output_tokens if meter.usage_events else None,
        "total_tokens": meter.total_tokens if meter.usage_events else None,
        "llm_call_count": logical_calls,
        "api_attempt_count": meter.api_attempt_count,
        "notes": [
            "llm_call_count = logical SiPhy VLM calls (1 object × estimate).",
            "api_attempt_count = MeteredOpenAIClient create() count (includes retries).",
            "total_tokens summed from API usage.total_tokens (not recomputed as in+out).",
        ],
    }
    result = {
        "object": object_key,
        "method": METHOD,
        "model": resolved_model,
        "provider": "gemini" if "gemini" in str(resolved_model) else "auto",
        "prediction": prediction,
        "token_usage": token_usage,
        "production_path": {
            "entry": "tuj.m3_grounding.materialize.Materializer.query_intrinsic",
            "intrinsic": "tuj.m3_grounding.intrinsic.ground_intrinsic",
            "backend": "tuj.m3_grounding.siphy_backend.SiPhyBackend.estimate",
            "friction": "tuj.m3_grounding.intrinsic.FrictionHead.estimate",
            "mass": (
                "SiPhyBackend.shell_mass_integral(points_mm) when points provided; "
                "else ground_intrinsic bbox×density fallback"
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
        "stage2_gemini": False,
    }
    return result


def run_live(*, object_keys: list[str], model: str) -> int:
    if str(model).startswith("gemini") and not os.environ.get("TUJ_LLM_PROVIDER"):
        os.environ["TUJ_LLM_PROVIDER"] = "gemini"

    from tuj.m3_grounding import SiPhyBackend

    raw_client, resolved_model = SiPhyBackend._make_client(None, ROOT, model)
    # One shared client; per-object TokenMeter snapshots via reset fields.
    master_meter = TokenMeter()
    metered = MeteredOpenAIClient(raw_client, master_meter)
    backend = SiPhyBackend(
        model=resolved_model,
        client=metered,
        repo_root=ROOT,
        verbose=True,
    )

    units_dir = OUT_DIR / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []

    for ok in object_keys:
        # Isolate token accounting per object.
        master_meter.api_attempt_count = 0
        master_meter.input_tokens = 0
        master_meter.output_tokens = 0
        master_meter.total_tokens = 0
        master_meter.usage_events = []

        result = run_one_object(
            object_key=ok,
            model=model,
            backend=backend,
            meter=master_meter,
            resolved_model=resolved_model,
        )
        gk = result.pop("gk")
        unit_json = units_dir / f"{ok}.json"
        write_json(unit_json, result)
        write_json(OUT_DIR / f"gk_{ok}.json", gk)
        write_csv_rows(
            units_dir / f"{ok}.csv",
            [_csv_row_from_result(result)],
            RESULT_CSV_FIELDS,
        )
        summary_rows.append(_csv_row_from_result(result))
        all_results.append(result)
        print_report(
            result,
            saved_paths=[unit_json, units_dir / f"{ok}.csv"],
        )

    # Summary over all objects in this batch.
    write_csv_rows(OUT_DIR / "summary.csv", summary_rows, RESULT_CSV_FIELDS)
    write_csv_rows(OUT_DIR / "result.csv", summary_rows, RESULT_CSV_FIELDS)
    write_json(
        OUT_DIR / "result.json",
        {
            "experiment": "experiment2_siphy_only",
            "method": METHOD,
            "model": resolved_model,
            "objects": object_keys,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": all_results,
        },
    )
    # Backward-compatible single-object paths when only one object was run.
    if len(all_results) == 1:
        write_json(OUT_DIR / "result_single.json", all_results[0])

    print()
    print(f"[exp2] Summary ({len(summary_rows)} objects): {OUT_DIR / 'summary.csv'}")
    print(f"[exp2] Combined: {OUT_DIR / 'result.json'} / {OUT_DIR / 'result.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 2: Production M3 / SiPhy only (multi-object)"
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    ap.add_argument(
        "--object",
        dest="objects",
        action="append",
        default=None,
        help=(
            "Object key (repeatable). Default dry-run: all CORE objects "
            f"{list(CORE_OBJECT_KEYS)}. Default live: requires --object or --all-core."
        ),
    )
    ap.add_argument(
        "--all-core",
        action="store_true",
        help=f"Run CORE objects: {list(CORE_OBJECT_KEYS)}",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare M1+crop per object; no Gemini API call",
    )
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_core:
        object_keys = list(CORE_OBJECT_KEYS)
    elif args.objects:
        object_keys = list(args.objects)
    elif args.dry_run:
        object_keys = list(CORE_OBJECT_KEYS)
    else:
        raise SystemExit(
            "Live run requires --object (repeatable) or --all-core. "
            "Example: --object bottle --object spoon ..."
        )

    for ok in object_keys:
        if ok not in OBJECTS:
            raise SystemExit(f"unknown --object: {ok}; choose from {list(OBJECTS)}")
        if ok not in CORE_OBJECT_KEYS and ok not in OBJECTS:
            raise SystemExit(f"unknown object: {ok}")

    if args.dry_run:
        return run_dry_run(object_keys=object_keys, model=args.model)
    return run_live(object_keys=object_keys, model=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
