"""Compare downloaded Berkeley YCB mesh extents with YCB-V model metadata.

The BOP ``models_info.json`` values are used only as a geometry/scale
reference.  No Real-5 mass, friction or test labels are read, and no automatic
rescaling is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ycb_robosuite_audit import audit


REFERENCE_URL = "https://www.rocq.inria.fr/archive_ylabbeprojectsdata/cosypose/bop_datasets/ycbv/models_bop-compat_eval/models_info.json"
# BOP YCB-V class ids differ from the original YCB names.
BOP_ID = {"003_cracker_box": "2", "006_mustard_bottle": "5", "019_pitcher_base": "11",
          "021_bleach_cleanser": "12", "025_mug": "14"}


def compare(ycb_root: Path, metadata: Path, robosuite: Path, output: Path) -> dict[str, object]:
    ref = json.loads(metadata.read_text(encoding="utf-8"))
    audit_report = audit(robosuite, ycb_root, output / "_asset_audit")
    rows = []
    for oid, bop_id in BOP_ID.items():
        mesh = audit_report["targets"][oid]["mesh_files"][0]
        bounds = mesh.get("bounds") or {}
        measured = list(bounds.get("extent_source_units", []))
        reference = [float(ref[bop_id][k]) for k in ("size_x", "size_y", "size_z")]
        measured_sorted, reference_sorted = sorted(measured), sorted(reference)
        ratios = [m / r for m, r in zip(measured_sorted, reference_sorted) if r > 0]
        uniform_scale = sum(m * r for m, r in zip(measured_sorted, reference_sorted)) / max(sum(m * m for m in measured_sorted), 1e-12)
        scaled = [m * uniform_scale for m in measured_sorted]
        errors = [abs(s - r) / r for s, r in zip(scaled, reference_sorted)]
        max_error = max(errors) if errors else float("inf")
        status = "WITHIN_10_PERCENT" if max_error <= 0.10 else "REVIEW_10_25_PERCENT" if max_error <= 0.25 else "MISMATCH_GT_25_PERCENT"
        rows.append({"object_id": oid, "bop_model_id": bop_id, "mesh_path": mesh["path"],
                     "mesh_sha256": mesh["sha256"], "mesh_extent_source_units_sorted": measured_sorted,
                     "reference_extent_mm_sorted": reference_sorted, "uniform_scale_fit": uniform_scale,
                     "uniform_scale_interpretation": "reference_mm_per_source_unit",
                     "mesh_extent_if_source_units_are_m": [m * 1000.0 for m in measured_sorted],
                     "max_error_after_uniform_fit": max_error, "status": status,
                     "source_units_status": "UNVERIFIED; extents are numerically metre-like", "auto_rescaled": False})
    output.mkdir(parents=True, exist_ok=True)
    (output / "ycb_scale_comparison.json").write_text(json.dumps({"schema_version": "c4_ycb_scale_comparison_v1",
        "reference_url": REFERENCE_URL, "reference_file": str(metadata.resolve()), "rows": rows,
        "policy": "sort axis extents; report uniform fit; never auto-rescale"}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# YCB mesh scale comparison", "", f"- reference: [{REFERENCE_URL}]({REFERENCE_URL})", "- axis order: sorted before comparison", "- auto-rescale: **disabled**", "- `uniform_scale_fit` means reference mm per source unit; values near 1000 are evidence that OBJ coordinates are metre-like, not a proof of unit metadata.", "", "| object | mesh extents (source) | mesh×1000 (mm assumption) | reference (mm) | max error after fit | status |", "|---|---|---|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r['object_id']} | {', '.join(f'{x:.4f}' for x in r['mesh_extent_source_units_sorted'])} | {', '.join(f'{x:.1f}' for x in r['mesh_extent_if_source_units_are_m'])} | {', '.join(f'{x:.1f}' for x in r['reference_extent_mm_sorted'])} | {r['max_error_after_uniform_fit']:.1%} | {r['status']} |")
    lines += ["", "수치가 일치하지 않는 경우에도 mesh를 자동 변형하지 않는다. 기준 치수는 geometry 검토용이며, EV 이미지와 동일한 실물 인스턴스라는 증거가 아니다. 최종 simulator scale은 calibration split에서 확정해야 한다."]
    (output / "ycb_scale_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"object_count": len(rows), "statuses": {r["object_id"]: r["status"] for r in rows}, "auto_rescaled": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ycb-root", type=Path, default=Path("data/external/ycb"))
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--robosuite", type=Path, default=Path("external/robosuite"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(compare(args.ycb_root.resolve(), args.metadata.resolve(), args.robosuite.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
