"""Experiment 1 v6 CLI — selective Stage-1 + OpenAI Stage-2 (v5 ablation clone).

Usage (PowerShell):
  python experiments/m0_retrieval/experiment1_v6/run_experiment1_v6.py --capture-observations --object bottle
  python experiments/m0_retrieval/experiment1_v6/run_experiment1_v6.py --all-core --dry-run --model gpt-4o-mini
  python experiments/m0_retrieval/experiment1_v6/run_experiment1_v6.py --static-eval-test
  python experiments/m0_retrieval/experiment1_v6/run_experiment1_v6.py --recompute-from-units
  python experiments/m0_retrieval/experiment1_v6/run_experiment1_v6.py --all-core --model gpt-5.6

Outputs: output/m0_retrieval/experiment1_v6/
Does NOT modify experiment1_v5/ or production src/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP_V6 = Path(__file__).resolve().parent
PARENT_EXP = EXP_V6.parent

for p in (ROOT, ROOT / "src", PARENT_EXP, EXP_V6):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from v6_conditions import CONDITIONS  # noqa: E402
from openai_stage2 import assert_no_gt_leakage  # noqa: E402
from openai_backend import DEFAULT_OPENAI_MODEL, PROVIDER, detect_sdk  # noqa: E402
from gt_loader import load_all_gt  # noqa: E402
from objects import OBJECTS  # noqa: E402
from eval_io import (  # noqa: E402
    CORE_OBJECT_KEYS,
    recompute_from_units,
    rows_from_live_unit,
    static_eval_pipeline_test,
    unit_is_batch_eligible,
    write_raw_and_summary,
)
from observation_prep import (  # noqa: E402
    OUT_DIR,
    capture_observations_v6,
    crop_path_v6,
    ensure_observations_v6,
    load_m1_manifest_v6,
    m1_bbox_for as m1_bbox_from_prep,
)
from runner import Experiment1V6Runner  # noqa: E402
from selective_siphy_stage1 import (  # noqa: E402
    SiPhyStage1Runner,
    resolve_siphy_model,
    static_openai_feasibility,
)
from openai_stage2 import OpenAIStage2Runner  # noqa: E402

UNITS_DIR = OUT_DIR / "units"
SIPHY_LOG_PATH = OUT_DIR / "intermediate_siphy_results.json"
EXPERIMENT_VERSION = "experiment1_v6"
V5_OUT = ROOT / "output" / "m0_retrieval" / "experiment1_v5"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_m1_manifest() -> dict[str, Any]:
    return load_m1_manifest_v6()


def m1_bbox_for(object_key: str, manifest: dict[str, Any]) -> list[float] | None:
    return m1_bbox_from_prep(object_key, manifest)


def crop_image_path_for(object_key: str) -> Path:
    return crop_path_v6(object_key)


def _print_unit_dry(unit, *, gt: dict[str, Any] | None) -> None:
    meta = unit.dry_run_meta
    print()
    print(f"Condition: {unit.condition.id}")
    print(f"Stage 1 required: {meta.get('stage1_required')}")
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
    print(f"BBox in SiPhy/Stage1: {meta['bbox_in_siphy']}")
    print(f"BBox in OpenAI Stage2: {meta.get('bbox_in_openai')}")
    print(f"bbox_mm (unit): {unit.bbox_mm}")
    print(f"skip_reason: {unit.stage2.skip_reason}")
    print(f"Expected SiPhy calls: {meta['expected_siphy_calls']}")
    print(f"Expected OpenAI Stage2 calls: {meta.get('expected_openai_calls')}")
    print(f"Expected total LLM calls: {meta['expected_total_llm_calls']}")
    print(f"Expected token accounting: {meta['expected_token_accounting']}")
    print(f"siphy_cache_hit: {unit.siphy_cache_hit}")
    print(f"siphy_material (this unit): {unit.siphy_material}")
    print(f"siphy_density_kgm3 (this unit): {unit.siphy_density_kgm3}")
    print(f"siphy_total_tokens (C1→0): {unit.siphy_total_tokens}")
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
        exp_o = u.dry_run_meta.get("expected_openai_calls")
        exp_t = u.dry_run_meta.get("expected_total_llm_calls")
        stage1_req = u.dry_run_meta.get("stage1_required")
        if cid == "C1":
            if stage1_req is not False:
                errors.append("C1 stage1_required must be False")
            if (exp_s, exp_o, exp_t) != (0, 1, 1):
                errors.append(f"C1 expected calls {(0,1,1)} got {(exp_s, exp_o, exp_t)}")
        else:
            if stage1_req is not True:
                errors.append(f"{cid}: stage1_required must be True")
            if (exp_s, exp_o, exp_t) != (1, 1, 2):
                errors.append(f"{cid} expected calls {(1,1,2)} got {(exp_s, exp_o, exp_t)}")

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

    n_stage1 = sum(1 for cid, u in units_by_cid.items() if cid != "C1" and u.stage1 is not None)
    if n_stage1 != 6:
        errors.append(f"expected 6 independent Stage1 results, got {n_stage1}")

    return errors


def _validate_object_inputs(object_key: str, errors: list[str]) -> tuple[Path, list[float] | None, dict]:
    crop = crop_image_path_for(object_key)
    m1 = load_m1_manifest()
    bbox = m1_bbox_for(object_key, m1)
    gts = load_all_gt()
    gt = gts.get(object_key)
    if not crop.is_file():
        errors.append(f"{object_key}: crop missing at {crop}")
    if bbox is None:
        errors.append(f"{object_key}: M1 bbox_mm load failed")
    if not gt:
        errors.append(f"{object_key}: GT load failed")
    else:
        for prop in ("material", "density_kgm3", "mass_kg", "mu", "youngs_gpa"):
            if prop not in gt:
                errors.append(f"{object_key}: GT missing key {prop}")
            elif not isinstance(gt[prop], dict):
                errors.append(f"{object_key}: GT {prop} malformed")
    return crop, bbox, gt or {}


def run_dry_run(*, object_keys: list[str], model: str) -> int:
    print("=" * 66)
    print("[Experiment 1 v6] DRY-RUN (selective Stage1 + OpenAI Stage2)")
    print("=" * 66)
    print(f"Output dir: {OUT_DIR}")
    print(f"Must NOT collide with v5: {V5_OUT}")
    if OUT_DIR.resolve() == V5_OUT.resolve():
        print("FAIL: v6 OUT_DIR equals v5")
        return 1
    print(f"Provider: {PROVIDER}")
    shared_model = model
    siphy_model = resolve_siphy_model(shared_model)
    print(f"Shared --model (passed through): {shared_model}")
    print(f"Selective Stage1 model: {siphy_model}")
    print(f"OpenAI Stage2 model: {shared_model}")
    print("Stage 1: v6 selective (material/density-only; NOT SiPhyBackend.estimate)")
    print("Cache reuse: DISABLED (each C2-C7 runs its own Stage1 call)")
    print()
    print("===== STAGE1 OPENAI FEASIBILITY =====")
    feas = static_openai_feasibility(ROOT)
    for k, v in feas.items():
        print(f"  {k}: {v}")
    print("  bbox_in_stage1: False")
    try:
        print(f"OpenAI SDK: {detect_sdk()}")
    except ImportError as exc:
        print(f"OpenAI SDK: MISSING ({exc})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_observations_v6(object_keys)
    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)

    errors: list[str] = []
    logical_calls = 0

    siphy = SiPhyStage1Runner(
        out_path=SIPHY_LOG_PATH,
        dry_run=True,
        model=shared_model,
        repo_root=ROOT,
    )
    openai_runner = OpenAIStage2Runner(model=shared_model, dry_run=True)
    runner = Experiment1V6Runner(siphy=siphy, openai=openai_runner, dry_run=True)

    for object_key in object_keys:
        print()
        print("#" * 66)
        print(f"OBJECT: {object_key}")
        print("#" * 66)
        crop, bbox, gt = _validate_object_inputs(object_key, errors)
        print(f"Crop path: {crop} exists={crop.is_file()}")
        print(f"M1 bbox_mm: {bbox}")
        obj = OBJECTS[object_key]
        siphy._log.pop(object_key, None)

        units = {}
        for cid in CONDITIONS:
            cond = CONDITIONS[cid]
            # Stage2 input composition check (structural)
            expected_parts = ["Image"]
            if cond.uses_bbox:
                expected_parts.append("M1 BBox")
            for c in cond.siphy_cues:
                expected_parts.append(
                    "predicted Material" if c == "material" else "predicted Density"
                )
            unit = runner.run_unit(
                cond,
                object_key=object_key,
                object_label=obj.label,
                crop_image_path=crop,
                m1_bbox_mm=bbox,
                gt_for_leak_check=gt,
            )
            units[cid] = unit
            meta = unit.dry_run_meta
            if meta.get("stage2_input") != " + ".join(expected_parts) and cid != "C1":
                # C1 uses "Image + M1 BBox"
                pass
            if cid == "C1" and meta.get("stage2_input") != "Image + M1 BBox":
                errors.append(f"{object_key}/{cid}: bad stage2_input {meta.get('stage2_input')!r}")
            logical_calls += int(meta.get("expected_total_llm_calls") or 0)
            _print_unit_dry(unit, gt=gt)

        errors.extend([f"{object_key}: {e}" for e in assert_independent_siphy(units)])

        if siphy.model != shared_model:
            errors.append(f"{object_key}: SiPhy model {siphy.model!r} != shared {shared_model!r}")
        if openai_runner.model != shared_model:
            errors.append(
                f"{object_key}: Stage2 model {openai_runner.model!r} != shared {shared_model!r}"
            )

        for cid in ("C1", "C4"):
            u = units[cid]
            if u.bbox_mm is None:
                errors.append(f"{object_key}/{cid}: bbox_mm must be non-null")
            skip = u.stage2.skip_reason or ""
            if "required input unavailable" in skip or "bbox(m1)" in skip:
                errors.append(f"{object_key}/{cid}: unexpected skip_reason={skip!r}")
            if u.dry_run_meta.get("bbox_in_siphy"):
                errors.append(f"{object_key}/{cid}: BBox must not go to Stage1")
            if u.bbox_mm is not None and str(u.bbox_mm) not in (u.stage2.prompt_user or ""):
                errors.append(f"{object_key}/{cid}: bbox_mm not present in Stage2 prompt")

        from selective_siphy_stage1 import (
            FORBIDDEN_STAGE1_KEYS,
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
                errors.append(f"{object_key}/{cid}: missing Stage1")
                continue
            if s1.stage1_mode != mode:
                errors.append(f"{object_key}/{cid}: stage1_mode={s1.stage1_mode!r}")
            if tuple(s1.stage1_requested_keys) != CONDITIONS[cid].siphy_cues:
                errors.append(f"{object_key}/{cid}: requested_keys mismatch")
            if s1.youngs_gpa is not None:
                errors.append(f"{object_key}/{cid}: Stage1 youngs_gpa must be None")
            if mode == "material_only":
                if s1.material is None:
                    errors.append(f"{object_key}/{cid}: material_only missing material")
                if s1.density_kgm3 is not None:
                    errors.append(f"{object_key}/{cid}: material_only must NOT produce density")
            elif mode == "density_only":
                if s1.density_kgm3 is None:
                    errors.append(f"{object_key}/{cid}: density_only missing density")
                if s1.material is not None:
                    errors.append(f"{object_key}/{cid}: density_only must NOT produce material")
            else:
                if s1.material is None or s1.density_kgm3 is None:
                    errors.append(f"{object_key}/{cid}: material_density must produce both")
            sp = s1.stage1_system_prompt or system_prompt_for_cues(CONDITIONS[cid].siphy_cues)
            low = sp.lower()
            if "bbox" in low:
                errors.append(f"{object_key}/{cid}: Stage1 system prompt contains bbox")
            if '"youngs_gpa"' in low or '"mass_kg"' in low or '"mu"' in low:
                errors.append(f"{object_key}/{cid}: Stage1 schema must not request youngs/mass/mu")
            _ = FORBIDDEN_STAGE1_KEYS  # referenced for parity with v5 asserts

        if units["C1"].stage1 is not None:
            errors.append(f"{object_key}: C1 must not run Stage1")
        if units["C1"].siphy_total_tokens not in (0, None):
            # dry-run sets 0 for C1
            if units["C1"].siphy_total_tokens != 0:
                errors.append(f"{object_key}: C1 siphy_total_tokens must be 0")

        # Stage2 input composition vs condition matrix
        for cid, u in units.items():
            p = u.stage2.prompt_user or ""
            cond = CONDITIONS[cid]
            if cond.uses_bbox and u.bbox_mm is not None and "bbox_mm" not in p:
                errors.append(f"{object_key}/{cid}: Stage2 missing bbox text")
            if not cond.uses_bbox and "bbox_mm" in p:
                errors.append(f"{object_key}/{cid}: Stage2 must not include bbox")
            if "material" in cond.fixed_from_siphy and "material =" not in p.lower():
                errors.append(f"{object_key}/{cid}: Stage2 missing fixed material")
            if "density_kgm3" in cond.fixed_from_siphy and "density_kgm3 =" not in p:
                errors.append(f"{object_key}/{cid}: Stage2 missing fixed density")
            if u.stage2.object_name_in_prompt:
                errors.append(f"{object_key}/{cid}: object name/class leakage in Stage2")

    expected_logical = len(object_keys) * (1 + 6 * 2)  # C1:1 + C2-C7:2 each
    print()
    print("===== DRY-RUN VALIDATION SUMMARY =====")
    print(f"Objects: {object_keys}")
    print(f"Conditions: {list(CONDITIONS)}")
    print(f"Sum of expected_total_llm_calls: {logical_calls} (expected {expected_logical})")
    if logical_calls != expected_logical:
        errors.append(f"logical call sum {logical_calls} != {expected_logical}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("OK: crop / M1 / GT / condition config / Stage1 required / Stage2 inputs")
    print("OK: C1 Stage1 required=false; C2-C7 Stage1 required=true")
    print("OK: C2-C7 each expect SiPhy=1, OpenAI=1, Total=2")
    print("OK: C1 expects SiPhy=0, OpenAI=1, Total=1")
    print("OK: BBox never in Stage1; GT / object name not in prompts")
    print("OK: output path is experiment1_v6 (no v5 overwrite)")
    print(f"OK: Stage1 and Stage2 share model={shared_model}")
    print()
    print(
        f"If live all-core C1-C7: logical calls = {expected_logical} "
        f"(5 objects -> 65 when object_keys=CORE)"
    )

    write_json(
        OUT_DIR / "run_metadata.json",
        {
            "experiment": EXPERIMENT_VERSION,
            "mode": "dry-run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "objects": object_keys,
            "shared_model": shared_model,
            "siphy_model": siphy.model,
            "stage2_model": shared_model,
            "provider": PROVIDER,
            "expected_logical_calls": expected_logical,
            "openai_feasibility": feas,
            "notes": [
                "v6: selective Stage1 (material/density-only) -> OpenAI Stage 2.",
                "Does NOT call production SiPhyBackend.estimate.",
                "Same --model for Stage1 and Stage2 (CLI value passed through).",
                "NO cross-condition Stage1 cache.",
                "BBox only in OpenAI Stage 2 (C1/C4/C5/C7).",
                "Observations/M1 via same helpers as v3/v4/v5.",
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
    """Run live inference; write units/ + raw_results like v5."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if use_live_runs:
        live_dir = OUT_DIR / "live_runs" / stamp
        print(f"[v6] LEGACY --use-live-runs → {live_dir}")
        units_dir = live_dir / "units"
        siphy_log = live_dir / "intermediate_siphy_results.json"
        meta_dir = live_dir
    else:
        units_dir = UNITS_DIR
        siphy_log = SIPHY_LOG_PATH
        meta_dir = OUT_DIR
    units_dir.mkdir(parents=True, exist_ok=True)

    ensure_observations_v6(object_keys)
    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)
    m1 = load_m1_manifest()

    print(f"[v6 LIVE] shared --model for Stage1+Stage2: {model}")
    print(f"[v6 LIVE] units → {units_dir}")
    siphy = SiPhyStage1Runner(
        out_path=siphy_log,
        dry_run=False,
        model=model,
        repo_root=ROOT,
    )
    openai_runner = OpenAIStage2Runner(model=model, dry_run=False)
    runner = Experiment1V6Runner(siphy=siphy, openai=openai_runner, dry_run=False)

    batch_rows: list[dict] = []
    failed = 0
    logical_calls = 0

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
            print(f"[v6 LIVE] {ok} / {cid}")
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
                print(f"[v6] unit not eligible for raw_results merge: {ok}/{cid}")

            logical_calls += unit.total_llm_call_count
            print(f"Saved: {unit_path}")
            print(
                f"calls: siphy={unit.siphy_model_call_count} "
                f"openai={unit.openai_model_call_count} "
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
        print(f"[CSV] {paths['object_condition_summary']}")
        print(f"[CSV] {paths['condition_consistency_summary']}")
        print(f"[JSON] {paths['result_json']}")
        if "snapshot" in paths:
            print(f"[CSV] {paths['snapshot']}")
    elif batch_rows and use_live_runs:
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
            "provider": PROVIDER,
            "recorded_logical_calls": logical_calls,
            "note": (
                "Independent selective Stage1 call per C2-C7 (no cache). "
                "OpenAI for Stage1+Stage2. units/ under experiment1_v6 only."
            ),
        },
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 1 v6 (selective Stage1 + OpenAI Stage2; v5 clone)"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--capture-observations",
        action="store_true",
        help="Same as v3/v5: single-object scene + production M1 → v6 observations/",
    )
    ap.add_argument(
        "--recompute-from-units",
        action="store_true",
        help="Rebuild raw_results + summaries from units/",
    )
    ap.add_argument(
        "--static-eval-test",
        action="store_true",
        help="No API: smoke-test 5×7×5=175 raw_results + consistency CSVs",
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
        required=False,
        default=None,
        help="OpenAI model id passed through unchanged (e.g. gpt-4o-mini, gpt-5.6).",
    )
    ap.add_argument(
        "--all-core",
        action="store_true",
        help="Run bottle/spoon/ladle/plate/mug × all conditions",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run all objects in OBJECTS registry × all conditions",
    )
    args = ap.parse_args()

    if args.static_eval_test:
        return static_eval_pipeline_test()

    if args.recompute_from_units:
        return recompute_from_units(successful_only=True)

    # Capture-only (no LLM).
    if args.capture_observations and not args.dry_run and not args.conditions and not args.all and not args.all_core:
        capture_observations_v6(args.objects)
        return 0

    model = args.model or DEFAULT_OPENAI_MODEL
    if not args.model and not args.dry_run and not args.static_eval_test:
        print(
            f"[v6] WARNING: --model not set; using soft default {DEFAULT_OPENAI_MODEL!r}. "
            "Pass --model explicitly for live runs."
        )

    if args.dry_run:
        if args.all_core:
            oks = [k for k in CORE_OBJECT_KEYS if k in OBJECTS]
        elif args.objects:
            oks = args.objects
        else:
            oks = ["bottle"]
        for ok in oks:
            if ok not in OBJECTS:
                raise SystemExit(f"unknown object: {ok}")
        if args.capture_observations:
            capture_observations_v6(oks)
        return run_dry_run(object_keys=oks, model=model)

    if args.all_core:
        oks = [k for k in CORE_OBJECT_KEYS if k in OBJECTS]
        cids = list(CONDITIONS)
    elif args.all:
        oks = list(OBJECTS)
        cids = list(CONDITIONS)
    else:
        if not args.objects or not args.conditions:
            raise SystemExit(
                "Live run requires --object and --condition "
                "(or --dry-run / --capture-observations / --all-core / --all / "
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

    if not args.model:
        raise SystemExit("Live run requires explicit --model <OPENAI_MODEL>")

    if args.capture_observations:
        capture_observations_v6(oks)
    return run_live(
        object_keys=oks,
        condition_ids=cids,
        model=args.model,
        use_live_runs=args.use_live_runs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
