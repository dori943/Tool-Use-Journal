"""Experiment 1 v4 CLI — independent SiPhy Stage-1 + Gemini Stage-2 per condition.

Usage (PowerShell):
  python experiments/m0_retrieval/experiment1_v4/run_experiment1_v4.py --object bottle --dry-run
  python experiments/m0_retrieval/experiment1_v4/run_experiment1_v4.py --object bottle --condition C2

Outputs: output/m0_retrieval/experiment1_v4/
Does NOT modify experiment1_v3/ or production src/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP_V4 = Path(__file__).resolve().parent
PARENT_EXP = EXP_V4.parent

for p in (ROOT, ROOT / "src", PARENT_EXP, EXP_V4):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from v4_conditions import CONDITIONS  # noqa: E402
from gemini_stage2 import assert_no_gt_leakage  # noqa: E402
from gemini_backend import DEFAULT_GEMINI_MODEL, PROVIDER, detect_sdk  # noqa: E402
from gt_loader import load_all_gt  # noqa: E402
from objects import OBJECTS  # noqa: E402
from runner import Experiment1V4Runner  # noqa: E402
from siphy_stage1 import SiPhyStage1Runner  # noqa: E402
from gemini_stage2 import GeminiStage2Runner  # noqa: E402

OUT_DIR = ROOT / "output" / "m0_retrieval" / "experiment1_v4"
LEGACY_V2 = ROOT / "output" / "m0_retrieval" / "experiment1_v2"
LEGACY_V3 = ROOT / "output" / "m0_retrieval" / "experiment1_v3"
SIPHY_LOG_PATH = OUT_DIR / "intermediate_siphy_results.json"
EXPERIMENT_VERSION = "experiment1_v4"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_m1_manifest() -> dict[str, Any]:
    for path in (
        OUT_DIR / "m1_bbox_manifest.json",
        LEGACY_V3 / "m1_bbox_manifest.json",
        LEGACY_V2 / "m1_bbox_manifest.json",
    ):
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
    for base in (OUT_DIR, LEGACY_V3, LEGACY_V2):
        p = base / "observations" / f"{object_key}_crop.png"
        if p.exists():
            return p
    return OUT_DIR / "observations" / f"{object_key}_crop.png"


def _print_unit_dry(unit, *, gt: dict[str, Any] | None) -> None:
    meta = unit.dry_run_meta
    print()
    print(f"Condition: {unit.condition.id}")
    print(f"Stage 1: {meta['stage1_model']} | input={meta['stage1_input']}")
    print(f"Stage 1 expected call: {meta.get('stage1_expected_call')}")
    print(f"Stage 1 cache reuse: {meta.get('stage1_cache_reuse')}")
    print(f"Stage 1 output used: {meta['stage1_output_used']}")
    print(f"Stage 2: {meta['stage2_model']} | input={meta['stage2_input']}")
    print(f"BBox in SiPhy: {meta['bbox_in_siphy']}")
    print(f"BBox in Gemini: {meta['bbox_in_gemini']}")
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
    print("[Experiment 1 v4] DRY-RUN (independent SiPhy per condition)")
    print("=" * 66)
    print(f"Output dir: {OUT_DIR}")
    print(f"Provider (Stage 2): {PROVIDER}")
    print(f"Gemini model: {model}")
    print("Stage 1: production tuj.m3_grounding.siphy_backend.SiPhyBackend")
    print("Cache reuse: DISABLED (each C2-C7 runs its own SiPhy call)")
    try:
        print(f"Gemini SDK: {detect_sdk()}")
    except ImportError as exc:
        print(f"Gemini SDK: MISSING ({exc})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)
    m1 = load_m1_manifest()
    crop = crop_image_path_for(object_key)
    obj = OBJECTS[object_key]
    gt = gts.get(object_key)

    siphy = SiPhyStage1Runner(
        out_path=SIPHY_LOG_PATH,
        dry_run=True,
        model="gpt-4o-mini",
        repo_root=ROOT,
    )
    siphy._log.pop(object_key, None)

    gemini = GeminiStage2Runner(model=model, dry_run=True)
    runner = Experiment1V4Runner(siphy=siphy, gemini=gemini, dry_run=True)

    units = {}
    for cid in CONDITIONS:
        unit = runner.run_unit(
            CONDITIONS[cid],
            object_key=object_key,
            object_label=obj.label,
            crop_image_path=crop,
            m1_bbox_mm=m1_bbox_for(object_key, m1),
            gt_for_leak_check=gt,
        )
        units[cid] = unit
        _print_unit_dry(unit, gt=gt)

    errors = assert_independent_siphy(units)
    print()
    print("===== INDEPENDENCE ASSERTS =====")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("OK: C2-C7 each expect SiPhy=1, Gemini=1, Total=2")
    print("OK: C1 expects SiPhy=0, Gemini=1, Total=1")
    print("OK: no cache reuse / siphy_cache_hit always False")
    print("OK: C2/C4, C3/C5, C6/C7 have independent Stage1 placeholders")
    print("OK: BBox never in SiPhy Stage 1")
    print()
    print("If live C1-C7 on one object: SiPhy calls=6, Gemini calls=7, Total LLM=13")

    write_json(
        OUT_DIR / "run_metadata.json",
        {
            "experiment": EXPERIMENT_VERSION,
            "mode": "dry-run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "object": object_key,
            "notes": [
                "True two-stage: production SiPhyBackend.estimate -> Gemini Stage 2.",
                "NO cross-condition SiPhy cache: each C2-C7 runs its own SiPhy call.",
                "intermediate_siphy_results.json is a condition-keyed LOG only.",
                "BBox only in Gemini Stage 2 (C1/C4/C5/C7).",
                "Final mass/mu from Gemini, not shell_mass_integral/FrictionHead.",
            ],
        },
    )
    siphy.save()
    return 0


def run_live(*, object_keys: list[str], condition_ids: list[str], model: str) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # If prior units exist, write this live batch under a timestamped snapshot dir
    # so independent-call results are not mixed with older cache-based units.
    units_root = OUT_DIR / "units"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if units_root.exists() and any(units_root.glob("*.json")):
        live_dir = OUT_DIR / "live_runs" / stamp
        print(f"[v4] Existing units/ found; writing this live batch to {live_dir}")
    else:
        live_dir = OUT_DIR
    (live_dir / "units").mkdir(parents=True, exist_ok=True)

    gts = load_all_gt()
    write_json(live_dir / "gt_manifest.json", gts)
    m1 = load_m1_manifest()

    siphy_log = live_dir / "intermediate_siphy_results.json"
    siphy = SiPhyStage1Runner(
        out_path=siphy_log,
        dry_run=False,
        model="gpt-4o-mini",
        repo_root=ROOT,
    )
    gemini = GeminiStage2Runner(model=model, dry_run=False)
    runner = Experiment1V4Runner(siphy=siphy, gemini=gemini, dry_run=False)

    for ok in object_keys:
        obj = OBJECTS[ok]
        crop = crop_image_path_for(ok)
        bbox = m1_bbox_for(ok, m1)
        for cid in condition_ids:
            cond = CONDITIONS[cid]
            print("=" * 66)
            print(f"[v4 LIVE] {ok} / {cid}")
            unit = runner.run_unit(
                cond,
                object_key=ok,
                object_label=obj.label,
                crop_image_path=crop,
                m1_bbox_mm=bbox,
            )
            unit_path = live_dir / "units" / f"{ok}_{cid}.json"
            write_json(
                unit_path,
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "object": ok,
                    "condition": cid,
                    "condition_name": cond.name,
                    "prediction": unit.prediction,
                    "siphy_cache_hit": False,
                    "siphy_call_executed": unit.siphy_call_executed,
                    "siphy_model_call_count": unit.siphy_model_call_count,
                    "gemini_model_call_count": unit.gemini_model_call_count,
                    "total_llm_call_count": unit.total_llm_call_count,
                    "siphy_input_tokens": unit.siphy_input_tokens,
                    "siphy_output_tokens": unit.siphy_output_tokens,
                    "siphy_total_tokens": unit.siphy_total_tokens,
                    "gemini_input_tokens": unit.gemini_input_tokens,
                    "gemini_output_tokens": unit.gemini_output_tokens,
                    "gemini_total_tokens": unit.gemini_total_tokens,
                    "total_input_tokens": unit.total_input_tokens,
                    "total_output_tokens": unit.total_output_tokens,
                    "total_tokens": unit.total_tokens,
                    "total_tokens_combined": unit.total_tokens_combined,
                    "siphy_material": unit.siphy_material,
                    "siphy_density_kgm3": unit.siphy_density_kgm3,
                    "bbox_mm": unit.bbox_mm,
                    "crop_image_path": unit.crop_image_path,
                    "fixed_cues": unit.stage2.fixed_cues_applied,
                    "prompt_user": unit.stage2.prompt_user,
                    "raw_response": unit.stage2.raw_response,
                    "stage1_raw": unit.stage1.to_json() if unit.stage1 else None,
                    "error": unit.stage2.error,
                    "failure_reason": unit.stage2.failure_reason,
                    "skipped": unit.stage2.skipped,
                    "skip_reason": unit.stage2.skip_reason,
                },
            )
            print(f"Saved: {unit_path}")
            print(
                f"calls: siphy={unit.siphy_model_call_count} "
                f"gemini={unit.gemini_model_call_count} "
                f"total={unit.total_llm_call_count}"
            )
            print(json.dumps(unit.prediction, ensure_ascii=False, indent=2))
    siphy.save()
    write_json(
        live_dir / "run_metadata.json",
        {
            "experiment": EXPERIMENT_VERSION,
            "mode": "live",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "Independent SiPhy call per C2-C7 condition (no cache reuse).",
        },
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 1 v4 (independent SiPhy + Gemini per condition)"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--object", dest="objects", action="append", default=None)
    ap.add_argument("--condition", dest="conditions", action="append", default=None)
    ap.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        ok = (args.objects or ["bottle"])[0]
        if ok not in OBJECTS:
            raise SystemExit(f"unknown object: {ok}")
        return run_dry_run(object_key=ok, model=args.model)

    if args.all:
        oks = list(OBJECTS)
        cids = list(CONDITIONS)
    else:
        if not args.objects or not args.conditions:
            raise SystemExit(
                "Live run requires --object and --condition "
                "(or --dry-run / --all)."
            )
        oks = args.objects
        cids = args.conditions
        for o in oks:
            if o not in OBJECTS:
                raise SystemExit(f"unknown object: {o}")
        for c in cids:
            if c not in CONDITIONS:
                raise SystemExit(f"unknown condition: {c}")

    return run_live(object_keys=oks, condition_ids=cids, model=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
