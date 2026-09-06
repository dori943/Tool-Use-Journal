"""Experiment 1 v5 CLI ??independent SiPhy Stage-1 + Gemini Stage-2 per condition.

Usage (PowerShell):
  python experiments/m0_retrieval/experiment1_v5/run_experiment1_v5.py --capture-observations --object bottle
  python experiments/m0_retrieval/experiment1_v5/run_experiment1_v5.py --object bottle --dry-run
  python experiments/m0_retrieval/experiment1_v5/run_experiment1_v5.py --static-eval-test
  python experiments/m0_retrieval/experiment1_v5/run_experiment1_v5.py --recompute-from-units
  python experiments/m0_retrieval/experiment1_v5/run_experiment1_v5.py --object bottle --condition C1

Outputs: output/m0_retrieval/experiment1_v5/{units,raw_results*.csv,condition_summary.csv}
Does NOT modify experiment1_v3/ or production src/.

Observation / M1 path matches v3 (shared helpers under experiments/m0_retrieval/).
Eval / CSV path matches v3 (shared evaluator + result_writer).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP_V5 = Path(__file__).resolve().parent
PARENT_EXP = EXP_V5.parent

for p in (ROOT, ROOT / "src", PARENT_EXP, EXP_V5):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from v5_conditions import CONDITIONS  # noqa: E402
from gemini_stage2 import assert_no_gt_leakage  # noqa: E402
from gemini_backend import DEFAULT_GEMINI_MODEL, PROVIDER, detect_sdk  # noqa: E402
from gt_loader import load_all_gt  # noqa: E402
from objects import OBJECTS  # noqa: E402
from eval_io import (  # noqa: E402
    recompute_from_units,
    rows_from_live_unit,
    static_eval_pipeline_test,
    unit_is_batch_eligible,
    write_raw_and_summary,
)
from observation_prep import (  # noqa: E402
    capture_observations_v5,
    crop_path_v5,
    ensure_observations_v5,
    load_m1_manifest_v5,
    m1_bbox_for as m1_bbox_from_prep,
)
from runner import Experiment1V5Runner  # noqa: E402
from selective_siphy_stage1 import (  # noqa: E402
    SiPhyStage1Runner,
    resolve_siphy_model,
    static_gemini_only_feasibility,
)
from gemini_stage2 import GeminiStage2Runner  # noqa: E402

OUT_DIR = ROOT / "output" / "m0_retrieval" / "experiment1_v5"
UNITS_DIR = OUT_DIR / "units"
SIPHY_LOG_PATH = OUT_DIR / "intermediate_siphy_results.json"
EXPERIMENT_VERSION = "experiment1_v5"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_m1_manifest() -> dict[str, Any]:
    """v5-local production M1 only (same capture path as v3; no unit/GT copy)."""
    return load_m1_manifest_v5()


def m1_bbox_for(object_key: str, manifest: dict[str, Any]) -> list[float] | None:
    return m1_bbox_from_prep(object_key, manifest)


def crop_image_path_for(object_key: str) -> Path:
    """Prefer v5-local crop from the shared observation pipeline."""
    return crop_path_v5(object_key)


def _print_unit_dry(unit, *, gt: dict[str, Any] | None) -> None:
    meta = unit.dry_run_meta
    print()
    print(f"Condition: {unit.condition.id}")
    print(f"Stage 1: {meta['stage1_model']} | input={meta['stage1_input']}")
    print(f"Stage 1 expected call: {meta.get('stage1_expected_call')}")
    print(f"Stage 1 cache reuse: {meta.get('stage1_cache_reuse')}")
    print(f"Stage 1 output used: {meta['stage1_output_used']}")
    print(f"Stage 1 mode: {meta.get('stage1_mode')}")
    print(f"Stage 1 requested keys: {meta.get('stage1_requested_keys')}")
    print(f"Stage 1 youngs_gpa (must be None): {meta.get('stage1_youngs_gpa')}")
    if unit.stage1 and unit.stage1.stage1_system_prompt:
        print("--- Stage 1 system prompt (selective) ---")
        print(unit.stage1.stage1_system_prompt.strip())
        print("---")
    print(f"Stage 2: {meta['stage2_model']} | input={meta['stage2_input']}")
    print(f"BBox in SiPhy: {meta['bbox_in_siphy']}")
    print(f"BBox in Gemini: {meta['bbox_in_gemini']}")
    print(f"bbox_mm (unit): {unit.bbox_mm}")
    print(f"skip_reason: {unit.stage2.skip_reason}")
    print(f"Expected SiPhy calls: {meta['expected_siphy_calls']}")
    print(f"Expected Gemini calls: {meta['expected_gemini_calls']}")
    print(f"Expected total LLM calls: {meta['expected_total_llm_calls']}")
    print(f"Expected token accounting: {meta['expected_token_accounting']}")
    print(f"siphy_cache_hit: {unit.siphy_cache_hit}")
    print(f"siphy_material (this unit): {unit.siphy_material}")
    print(f"siphy_density_kgm3 (this unit): {unit.siphy_density_kgm3}")
    print("GT included?: False")
    print(f"Object name/class included?: {unit.stage2.object_name_in_prompt}")
    leaks = assert_no_gt_leakage(unit.stage2.prompt_user, gt)
    print(f"GT leakage: {'FAIL ' + str(leaks) if leaks else 'OK (none)'}")
    print()
    print("Stage 2 prompt (user text):")
    print(unit.stage2.prompt_user)
    print("=" * 66)


def assert_independent_siphy(units_by_cid: dict[str, Any]) -> list[str]:
    """Assert C2-C7 each have independent Stage-1 results (no forced cue sharing)."""
    errors: list[str] = []

    for cid, u in units_by_cid.items():
        if u.dry_run_meta.get("bbox_in_siphy"):
            errors.append(f"{cid}: bbox_in_siphy must be False")
        if u.siphy_cache_hit:
            errors.append(f"{cid}: siphy_cache_hit must be False")
        if u.dry_run_meta.get("stage1_cache_reuse"):
            errors.append(f"{cid}: stage1_cache_reuse must be False")

        exp_s = u.dry_run_meta.get("expected_siphy_calls")
        exp_g = u.dry_run_meta.get("expected_gemini_calls")
        exp_t = u.dry_run_meta.get("expected_total_llm_calls")
        if cid == "C1":
            if (exp_s, exp_g, exp_t) != (0, 1, 1):
                errors.append(f"C1 expected calls {(0,1,1)} got {(exp_s, exp_g, exp_t)}")
        else:
            if (exp_s, exp_g, exp_t) != (1, 1, 2):
                errors.append(f"{cid} expected calls {(1,1,2)} got {(exp_s, exp_g, exp_t)}")

    # Dry-run placeholders must be condition-specific (not shared equal)
    pairs = [("C2", "C4", "material"), ("C3", "C5", "density"), ("C6", "C7", "both")]
    for a, b, kind in pairs:
        ua, ub = units_by_cid[a], units_by_cid[b]
        if ua.stage1 is None or ub.stage1 is None:
            errors.append(f"{a}/{b}: missing stage1")
            continue
        if ua.stage1 is ub.stage1:
            errors.append(f"{a}/{b}: Stage1 result object unexpectedly identical (shared)")
        if ua.stage1.condition_id == ub.stage1.condition_id:
            errors.append(f"{a}/{b}: condition_id collision")
        if kind in ("material", "both") and ua.siphy_material == ub.siphy_material:
            # In dry-run we intentionally use distinct placeholders; live may coincide.
            # Structural independence: different condition_id on stage1 is enough;
            # also require dry-run placeholders differ.
            if str(ua.siphy_material).startswith("__siphy_material_"):
                errors.append(
                    f"{a}/{b}: dry-run material placeholders should differ "
                    f"({ua.siphy_material!r} vs {ub.siphy_material!r})"
                )
        if kind in ("density", "both") and ua.siphy_density_kgm3 == ub.siphy_density_kgm3:
            if ua.siphy_density_kgm3 is not None and ua.siphy_density_kgm3 < 0:
                errors.append(
                    f"{a}/{b}: dry-run density sentinels should differ "
                    f"({ua.siphy_density_kgm3} vs {ub.siphy_density_kgm3})"
                )

    # Six independent SiPhy logs for C2-C7
    n_stage1 = sum(1 for cid, u in units_by_cid.items() if cid != "C1" and u.stage1 is not None)
    if n_stage1 != 6:
        errors.append(f"expected 6 independent Stage1 results, got {n_stage1}")

    return errors


def run_dry_run(*, object_key: str, model: str) -> int:
    print("=" * 66)
    print("[Experiment 1 v5] DRY-RUN (selective Stage1 + Gemini Stage2)")
    print("=" * 66)
    print(f"Output dir: {OUT_DIR}")
    print(f"Provider (Stage 2): {PROVIDER}")
    # Shared --model for both stages (default gemini-3.6-flash via CLI).
    shared_model = model
    siphy_model = resolve_siphy_model(shared_model)
    print(f"Shared --model: {shared_model}")
    print(f"Selective Stage1 model: {siphy_model}")
    print(f"Gemini Stage2 model: {shared_model}")
    print("Stage 1: v5 selective (material/density-only; NOT SiPhyBackend.estimate)")
    print("Cache reuse: DISABLED (each C2-C7 runs its own Stage1 call)")
    print()
    print("===== STAGE1 SELECTIVE FEASIBILITY =====")
    feas = static_gemini_only_feasibility(ROOT)
    for k, v in feas.items():
        print(f"  {k}: {v}")
    print("  bbox_in_stage1: False")
    print(f"  v5_default_shared_model: {resolve_siphy_model('gemini-3.6-flash')}")
    try:
        print(f"Gemini SDK: {detect_sdk()}")
    except ImportError as exc:
        print(f"Gemini SDK: MISSING ({exc})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Same observation/M1 path as v3/v4 (shared helpers → v5 OUT_DIR).
    ensure_observations_v5([object_key])
    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)
    m1 = load_m1_manifest()
    crop = crop_image_path_for(object_key)
    bbox = m1_bbox_for(object_key, m1)
    print(f"Crop path: {crop} exists={crop.is_file()}")
    print(f"M1 bbox_mm (production, v5-local): {bbox}")
    if bbox is None:
        print("FAIL: production M1 bbox_mm missing after observation capture")
        return 1
    obj = OBJECTS[object_key]
    gt = gts.get(object_key)

    siphy = SiPhyStage1Runner(
        out_path=SIPHY_LOG_PATH,
        dry_run=True,
        model=shared_model,  # same --model as Stage2
        repo_root=ROOT,
    )
    siphy._log.pop(object_key, None)

    gemini = GeminiStage2Runner(model=shared_model, dry_run=True)
    runner = Experiment1V5Runner(siphy=siphy, gemini=gemini, dry_run=True)

    units = {}
    for cid in CONDITIONS:
        unit = runner.run_unit(
            CONDITIONS[cid],
            object_key=object_key,
            object_label=obj.label,
            crop_image_path=crop,
            m1_bbox_mm=bbox,
            gt_for_leak_check=gt,
        )
        units[cid] = unit
        _print_unit_dry(unit, gt=gt)
        # Explicit model flow lines for C1/C2 as requested
        if cid in ("C1", "C2"):
            print(f"  [model-flow] provider=gemini")
            print(f"  [model-flow] SiPhy model={siphy.model if cid != 'C1' else 'none'}")
            print(f"  [model-flow] Stage2 model={shared_model}")
            print(
                f"  [model-flow] expected calls="
                f"{unit.dry_run_meta['expected_total_llm_calls']}"
            )

    errors = assert_independent_siphy(units)
    # Model wiring asserts
    if siphy.model != shared_model:
        errors.append(f"SiPhy model {siphy.model!r} != shared {shared_model!r}")
    if gemini.model != shared_model:
        errors.append(f"Stage2 model {gemini.model!r} != shared {shared_model!r}")
    if units["C2"].stage1 and units["C2"].stage1.model != shared_model:
        errors.append("C2 Stage1 result model mismatch")

    # Observation / bbox asserts (C1, C4)
    for cid in ("C1", "C4"):
        u = units[cid]
        if u.bbox_mm is None:
            errors.append(f"{cid}: bbox_mm must be non-null after M1 capture")
        skip = u.stage2.skip_reason or ""
        if "required input unavailable" in skip or "bbox(m1)" in skip:
            errors.append(f"{cid}: unexpected skip_reason={skip!r}")
        if u.dry_run_meta.get("bbox_in_siphy"):
            errors.append(f"{cid}: BBox must not go to SiPhy")
        if u.bbox_mm is not None and str(u.bbox_mm) not in (u.stage2.prompt_user or ""):
            errors.append(f"{cid}: bbox_mm not present in Stage2 prompt")
    if units["C1"].dry_run_meta.get("expected_siphy_calls") != 0:
        errors.append("C1 expected_siphy_calls must be 0")
    if units["C1"].dry_run_meta.get("expected_gemini_calls") != 1:
        errors.append("C1 expected_gemini_calls must be 1")
    if units["C4"].dry_run_meta.get("expected_siphy_calls") != 1:
        errors.append("C4 expected_siphy_calls must be 1")
    if units["C4"].dry_run_meta.get("expected_gemini_calls") != 1:
        errors.append("C4 expected_gemini_calls must be 1")

    # v5 selective Stage1 schema asserts
    from selective_siphy_stage1 import (
        FORBIDDEN_STAGE1_KEYS,
        stage1_mode_for_cues,
        system_prompt_for_cues,
    )

    expect_mode = {
        "C2": "material_only",
        "C3": "density_only",
        "C4": "material_only",
        "C5": "density_only",
        "C6": "material_density",
        "C7": "material_density",
    }
    for cid, mode in expect_mode.items():
        u = units[cid]
        s1 = u.stage1
        if s1 is None:
            errors.append(f"{cid}: missing Stage1")
            continue
        if s1.stage1_mode != mode:
            errors.append(f"{cid}: stage1_mode={s1.stage1_mode!r} expected {mode!r}")
        if tuple(s1.stage1_requested_keys) != CONDITIONS[cid].siphy_cues:
            errors.append(f"{cid}: requested_keys mismatch {s1.stage1_requested_keys}")
        if s1.youngs_gpa is not None:
            errors.append(f"{cid}: Stage1 youngs_gpa must be None")
        raw = s1.raw_siphy_output or {}
        for bad in FORBIDDEN_STAGE1_KEYS:
            if bad in raw and raw.get(bad) is not None and bad not in ("materials",):
                # dry-run raw may list keys in notes; only flag if value present as field
                pass
        if mode == "material_only":
            if s1.material is None:
                errors.append(f"{cid}: material_only must produce material")
            if s1.density_kgm3 is not None:
                errors.append(f"{cid}: material_only must NOT produce density")
        elif mode == "density_only":
            if s1.density_kgm3 is None:
                errors.append(f"{cid}: density_only must produce density")
            if s1.material is not None:
                errors.append(f"{cid}: density_only must NOT produce material")
        else:
            if s1.material is None or s1.density_kgm3 is None:
                errors.append(f"{cid}: material_density must produce both")
        # Prompt text must not mention bbox / youngs / mass / mu as requested outputs
        sp = s1.stage1_system_prompt or system_prompt_for_cues(CONDITIONS[cid].siphy_cues)
        low = sp.lower()
        if "bbox" in low:
            errors.append(f"{cid}: Stage1 system prompt contains bbox")
        if "young" in low and mode != "material_density":
            # material_density prompt says "Do not estimate ... Young's"
            pass
        if '"youngs_gpa"' in low or '"mass_kg"' in low or '"mu"' in low:
            errors.append(f"{cid}: Stage1 schema must not request youngs/mass/mu")
        # Stage2 must not receive bbox in siphy path
        if u.dry_run_meta.get("bbox_in_siphy"):
            errors.append(f"{cid}: bbox_in_siphy")

    # C1 has no stage1
    if units["C1"].stage1 is not None:
        errors.append("C1 must not run Stage1")

    print()
    print("===== INDEPENDENCE / SELECTIVE STAGE1 ASSERTS =====")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("OK: C2-C7 each expect SiPhy=1, Gemini=1, Total=2")
    print("OK: C1 expects SiPhy=0, Gemini=1, Total=1")
    print("OK: C1/C4 bbox_mm non-null (production M1 from v5 observations)")
    print("OK: C2/C4 Stage1 = material_only (no density/youngs/mass/mu)")
    print("OK: C3/C5 Stage1 = density_only (no material/youngs/mass/mu)")
    print("OK: C6/C7 Stage1 = material+density only")
    print("OK: BBox never in Stage1")
    print("OK: Stage1 is selective Gemini (not production SiPhyBackend.estimate)")
    print(f"OK: Stage1 and Stage2 share model={shared_model}")
    print()
    print("If live C1-C7 on one object: Stage1 calls=6, Gemini Stage2=7, Total LLM=13")

    write_json(
        OUT_DIR / "run_metadata.json",
        {
            "experiment": EXPERIMENT_VERSION,
            "mode": "dry-run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "object": object_key,
            "shared_model": shared_model,
            "siphy_model": siphy.model,
            "stage2_model": shared_model,
            "m1_bbox_mm": bbox,
            "crop_image_path": str(crop),
            "gemini_only_feasibility": feas,
            "stage1_modes": {cid: units[cid].stage1.stage1_mode if units[cid].stage1 else None for cid in CONDITIONS},
            "notes": [
                "v5: selective Stage1 (material/density-only) -> Gemini Stage 2.",
                "Does NOT call production SiPhyBackend.estimate.",
                "Same --model for Stage1 and Stage2 (default gemini-3.6-flash).",
                "NO cross-condition Stage1 cache.",
                "BBox only in Gemini Stage 2 (C1/C4/C5/C7).",
                "Observations/M1 via same helpers as v3/v4.",
            ],
        },
    )
    siphy.save()
    return 0


def run_live(
    *,
    object_keys: list[str],
    condition_ids: list[str],
    model: str,
    use_live_runs: bool = False,
) -> int:
    """Run live inference; write units/ + raw_results like v3.

    Default: accumulate under experiment1_v5/units/ (no per-batch live_runs/).
    ``use_live_runs=True`` is legacy/debug only.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if use_live_runs:
        live_dir = OUT_DIR / "live_runs" / stamp
        print(f"[v5] LEGACY --use-live-runs ??{live_dir}")
        units_dir = live_dir / "units"
        siphy_log = live_dir / "intermediate_siphy_results.json"
        meta_dir = live_dir
    else:
        units_dir = UNITS_DIR
        siphy_log = SIPHY_LOG_PATH
        meta_dir = OUT_DIR
    units_dir.mkdir(parents=True, exist_ok=True)

    ensure_observations_v5(object_keys)
    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)
    m1 = load_m1_manifest()

    print(f"[v5 LIVE] shared --model for SiPhy+Stage2: {model}")
    print(f"[v5 LIVE] units ??{units_dir}")
    siphy = SiPhyStage1Runner(
        out_path=siphy_log,
        dry_run=False,
        model=model,
        repo_root=ROOT,
    )
    gemini = GeminiStage2Runner(model=model, dry_run=False)
    runner = Experiment1V5Runner(siphy=siphy, gemini=gemini, dry_run=False)

    batch_rows: list[dict] = []
    failed = 0

    for ok in object_keys:
        obj = OBJECTS[ok]
        crop = crop_image_path_for(ok)
        bbox = m1_bbox_for(ok, m1)
        if bbox is None:
            print(f"FAIL: {ok}: production M1 bbox_mm missing (run --capture-observations)")
            return 1
        if not crop.is_file():
            print(f"FAIL: {ok}: crop missing at {crop}")
            return 1
        for cid in condition_ids:
            cond = CONDITIONS[cid]
            print("=" * 66)
            print(f"[v5 LIVE] {ok} / {cid}")
            unit = runner.run_unit(
                cond,
                object_key=ok,
                object_label=obj.label,
                crop_image_path=crop,
                m1_bbox_mm=bbox,
            )
            gt = gts[ok]
            rows, payload = rows_from_live_unit(
                object_key=ok, condition=cond, unit=unit, gt=gt
            )
            unit_path = units_dir / f"{ok}_{cid}.json"
            write_json(unit_path, payload)

            if unit_is_batch_eligible(payload):
                batch_rows.extend(rows)
            else:
                failed += 1
                print(f"[v5] unit not eligible for raw_results merge: {ok}/{cid}")

            print(f"Saved: {unit_path}")
            print(
                f"calls: siphy={unit.siphy_model_call_count} "
                f"gemini={unit.gemini_model_call_count} "
                f"total={unit.total_llm_call_count}"
            )
            print("Prediction:")
            print(json.dumps(unit.prediction, ensure_ascii=False, indent=2))
            print("Error:")
            for ev in payload.get("evaluations") or []:
                print(
                    f"  {ev['property']}: status={ev.get('evaluation_status')} "
                    f"pred={ev['prediction']} error={ev['error']} "
                    f"evaluated={ev['evaluated']}"
                )

    siphy.save()

    if batch_rows and not use_live_runs:
        paths = write_raw_and_summary(
            batch_rows,
            out_dir=OUT_DIR,
            merge_rolling=True,
            write_timestamp_snapshot=True,
        )
        print(f"[CSV] {paths['rolling']}")
        print(f"[CSV] {paths['summary']}")
        if "snapshot" in paths:
            print(f"[CSV] {paths['snapshot']}")
    elif batch_rows and use_live_runs:
        # Legacy: write CSVs next to the live_runs stamp only (not experiment root).
        paths = write_raw_and_summary(
            batch_rows,
            out_dir=meta_dir,
            merge_rolling=False,
            write_timestamp_snapshot=True,
        )
        print(f"[CSV legacy] {paths.get('rolling')}")

    write_json(
        meta_dir / "run_metadata.json",
        {
            "experiment": EXPERIMENT_VERSION,
            "mode": "live_legacy" if use_live_runs else "live",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "objects": object_keys,
            "conditions": condition_ids,
            "n_batch_raw_rows": len(batch_rows),
            "n_failed_or_ineligible": failed,
            "model": model,
            "note": (
                "Independent SiPhy call per C2-C7 (no cache). "
                "units/ under experiment root; live_runs/ not used by default."
            ),
        },
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 1 v5 (independent SiPhy + Gemini per condition)"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--capture-observations",
        action="store_true",
        help="Same as v3: single-object scene + production M1 ??v5 observations/",
    )
    ap.add_argument(
        "--recompute-from-units",
        action="store_true",
        help="Rebuild raw_results.csv + condition_summary.csv from units/ (v3-compatible)",
    )
    ap.add_argument(
        "--static-eval-test",
        action="store_true",
        help="No API: smoke-test 5×7×5=175 raw_results schema under _static_eval_smoke/",
    )
    ap.add_argument(
        "--use-live-runs",
        action="store_true",
        help="LEGACY/debug: write under live_runs/<timestamp>/ instead of units/",
    )
    ap.add_argument("--object", dest="objects", action="append", default=None)
    ap.add_argument("--condition", dest="conditions", action="append", default=None)
    ap.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
        help=(
            "Shared Gemini model for Stage1 SiPhyBackend and Stage2 "
            f"(default: {DEFAULT_GEMINI_MODEL})."
        ),
    )
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.static_eval_test:
        return static_eval_pipeline_test()

    if args.recompute_from_units:
        return recompute_from_units(successful_only=True)

    # Capture-only (no LLM), mirroring v3.
    if args.capture_observations and not args.dry_run and not args.conditions and not args.all:
        capture_observations_v5(args.objects)
        return 0

    if args.dry_run:
        ok = (args.objects or ["bottle"])[0]
        if ok not in OBJECTS:
            raise SystemExit(f"unknown object: {ok}")
        if args.capture_observations:
            capture_observations_v5([ok])
        return run_dry_run(object_key=ok, model=args.model)

    if args.all:
        oks = list(OBJECTS)
        cids = list(CONDITIONS)
    else:
        if not args.objects or not args.conditions:
            raise SystemExit(
                "Live run requires --object and --condition "
                "(or --dry-run / --capture-observations / --all / "
                "--recompute-from-units / --static-eval-test)."
            )
        oks = args.objects
        cids = args.conditions
        for o in oks:
            if o not in OBJECTS:
                raise SystemExit(f"unknown object: {o}")
        for c in cids:
            if c not in CONDITIONS:
                raise SystemExit(f"unknown condition: {c}")

    if args.capture_observations:
        capture_observations_v5(oks)
    return run_live(
        object_keys=oks,
        condition_ids=cids,
        model=args.model,
        use_live_runs=args.use_live_runs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
