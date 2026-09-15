"""Join the locked Static Real-5 input manifest to external YCB mesh assets.

This is a provenance/mapping artifact for simulator preparation.  It does not
run inference, read evaluator-only GT, or claim that a YCB mesh is the exact
physical instance visible in an EV-RealPhys image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


TARGETS = {
    "003_cracker_box": "Cracker Box",
    "006_mustard_bottle": "Mustard Bottle",
    "019_pitcher_base": "Pitcher Base",
    "021_bleach_cleanser": "Bleach Cleanser",
    "025_mug": "Mug",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(static_manifest: Path, audit_json: Path, output: Path) -> dict[str, object]:
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(static_manifest.open(encoding="utf-8", newline="")))
    if len(rows) != 50:
        raise ValueError(f"expected 50 locked static inputs, found {len(rows)}")
    counts = {oid: 0 for oid in TARGETS}
    out_rows = []
    for row in rows:
        oid = row["object_id"]
        if oid not in TARGETS:
            raise ValueError(f"unexpected or excluded object id: {oid}")
        counts[oid] += 1
        target = audit["targets"].get(oid)
        if not target or not target.get("mesh_files"):
            raise ValueError(f"no audited mesh asset for {oid}")
        mesh = target["mesh_files"][0]
        # Validate the locked crop/context assets without changing the input.
        mass_crop = Path(row["mass_crop_path"])
        friction_context = Path(row["friction_context_path"])
        if not mass_crop.is_file() or not friction_context.is_file():
            raise ValueError(f"missing crop/context for {row['input_id']}")
        out_rows.append({
            "input_id": row["input_id"],
            "object_id": oid,
            "object_name": TARGETS[oid],
            "sequence_id": row["sequence_id"],
            "frame_id": row["frame_id"],
            "mass_crop_path": row["mass_crop_path"],
            "friction_context_path": row["friction_context_path"],
            "ycb_mesh_path": mesh["path"],
            "ycb_mesh_sha256": mesh["sha256"],
            "asset_mapping_status": "EXACT_YCB_ID",
            "image_instance_mapping": "UNVERIFIED",
            "unit_scale_status": "NEEDS_CALIBRATION",
            "sim_mass_source": "CALIBRATION_ONLY",
            "sim_friction_source": "CALIBRATION_ONLY",
            "real5_gt_in_simulator": "FORBIDDEN",
        })
    if counts != {oid: 10 for oid in TARGETS}:
        raise ValueError(f"locked manifest must contain 10 inputs per target: {counts}")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "ycb_static_mapping.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    summary = {
        "schema_version": "c4_ycb_static_mapping_v1",
        "source_manifest": str(static_manifest.resolve()),
        "source_manifest_sha256": sha256(static_manifest),
        "ycb_audit": str(audit_json.resolve()),
        "input_count": len(out_rows),
        "object_counts": counts,
        "tuna_included": any(r["object_id"] == "007_tuna_fish_can" for r in out_rows),
        "image_instance_mapping": "UNVERIFIED",
        "sim_mass_and_friction": "CALIBRATION_ONLY; Real-5 evaluator GT is not copied or used",
        "unit_scale": "UNVERIFIED; supply a calibration record before final benchmark runs",
    }
    (output / "mapping_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Static Real-5 → YCB mapping audit", "",
        f"- locked inputs: {len(out_rows)} (5 objects × 10)",
        f"- source manifest SHA-256: `{summary['source_manifest_sha256']}`", "",
        "| object_id | inputs | YCB asset | image instance mapping | unit scale |", "|---|---:|---|---|---|",
    ]
    for oid, count in counts.items():
        item = audit["targets"][oid]
        lines.append(f"| {oid} | {count} | {item['mapping_status']} | UNVERIFIED | NEEDS_CALIBRATION |")
    lines += [
        "", "## 사용 제한", "",
        "이 파일은 이미지 입력과 외부 YCB geometry의 provenance만 연결한다. 파일명이 같다는 이유로 EV-RealPhys와 동일한 실물 인스턴스라고 주장하지 않는다.",
        "YCB mesh에 모래 충전 질량 또는 Table 5 결합 마찰을 넣지 않는다. 시뮬레이터 질량·마찰은 calibration split에서 별도로 고정해야 한다.",
        "정적 EV 이미지의 pose와 simulator pose는 아직 자동 정합되지 않았으므로, 이 산출물만으로 Table III의 공식 실물 점수를 확장하지 않는다.",
    ]
    (output / "ycb_static_mapping.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-manifest", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.static_manifest, args.audit_json, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
