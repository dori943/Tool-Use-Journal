"""Recheck BOP identity from annotation IDs only; ignore all numeric properties."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_r_inputs import ROOT, sha256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.run_dir.resolve()
    source_file = ROOT / "data/external/kandukuri_ev_realphys/manifest.json"
    source = json.loads(source_file.read_text(encoding="utf-8"))
    results = []
    for row in source["sequences"]:
        scene = ROOT / "data/external/kandukuri_ev_realphys/ev-realphys" / row["sequence_id"]
        bop = int(row["bop_object_id"])
        prop_keys = set(json.loads((scene / "scene_properties.json").read_text(encoding="utf-8"))
                        ["object_properties"])
        # Only obj_id is accessed. No annotation pose or synthetic mass/friction is read.
        scene_gt = json.loads((scene / "scene_gt.json").read_text(encoding="utf-8"))
        frame_ids = {int(entry["obj_id"]) for entries in scene_gt.values() for entry in entries}
        if prop_keys != {str(bop)} or frame_ids != {bop}:
            raise ValueError(f"BOP ID mismatch {row['sequence_id']}")
        results.append({"sequence_id": row["sequence_id"], "object_id": row["object_id"],
                        "bop_object_id": bop, "properties_key_matches": True,
                        "scene_gt_object_ids_match": True,
                        "annotated_frame_count": len(scene_gt)})
    if len(results) != 25 or Counter(r["object_id"] for r in results) != {
            "003_cracker_box": 5, "006_mustard_bottle": 5,
            "019_pitcher_base": 5, "021_bleach_cleanser": 5, "025_mug": 5}:
        raise ValueError("not five matched Real-5 sequences per object")
    target = out / "mapping_verification.json"
    with target.open("x", encoding="utf-8") as f:
        json.dump({"source_manifest_sha256": sha256(source_file),
                   "numeric_gt_used": False, "annotation_pose_used": False,
                   "scene_properties_physical_values_used": False,
                   "rows": results}, f, indent=2)
        f.write("\n")
    print(f"25 verified sequence IDs; 5 physical objects; {target}")


if __name__ == "__main__":
    main()
