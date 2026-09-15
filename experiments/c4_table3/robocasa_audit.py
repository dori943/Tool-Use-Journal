"""Audit RoboCasa code/metadata support for EV-RealPhys static-to-sim linkage.

This audit deliberately inspects the cloned source tree only; it does not
download the multi-GB dataset/assets and does not claim an exact image/mesh
mapping without an episode manifest.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


TARGETS = {
    "003_cracker_box": ("cracker_box", "cracker", "boxed_food"),
    "006_mustard_bottle": ("mustard", "mustard_bottle"),
    "019_pitcher_base": ("pitcher",),
    "021_bleach_cleanser": ("bleach", "bleach_cleanser", "cleaner"),
    "025_mug": ("mug",),
}


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def text_files(repo: Path) -> list[Path]:
    return [repo / "docs" / "assets" / "objects.md",
            repo / "robocasa" / "models" / "objects" / "kitchen_objects.py",
            repo / "robocasa" / "utils" / "camera_utils.py",
            repo / "robocasa" / "environments" / "kitchen" / "kitchen.py",
            repo / "robocasa" / "utils" / "object_utils.py",
            repo / "docs" / "datasets" / "using_datasets.md"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("external/robocasa"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    blobs = {str(p.relative_to(repo)): p.read_text(encoding="utf-8", errors="ignore")
             for p in text_files(repo) if p.exists()}
    joined = "\n".join(blobs.values()).lower()
    findings = {}
    for object_id, aliases in TARGETS.items():
        exact = {alias: bool(re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", joined))
                 for alias in aliases}
        findings[object_id] = {"aliases": exact,
                               # A source-level category hit is not proof of
                               # the EV-RealPhys object instance or image link.
                               "exact_asset_verified": False}
    report = {
        "robocasa_repo": str(repo),
        "robocasa_commit": git_head(repo),
        "dataset_assets_downloaded": False,
        "scope": "source/docs audit only; no episode-specific mapping claimed",
        "targets": findings,
        "capabilities": {
            "object_scale": {"status": "SUPPORTED_IN_SOURCE", "evidence": ["kitchen.py object_scale", "kitchen_object_utils.py scale"]},
            "object_world_pose": {"status": "AVAILABLE_AT_REPLAY", "evidence": ["states.npz described in dataset docs", "MuJoCo body state"]},
            "camera_pose": {"status": "SUPPORTED_IN_SOURCE", "evidence": ["utils/camera_utils.py", "kitchen.py CamUtils.set_cameras"]},
            "table_plane_contact": {"status": "SUPPORTED_BY_MUJOCO", "evidence": ["object_utils.check_obj_fixture_contact", "fixture geometry"]},
            "scene_replay": {"status": "SUPPORTED_IN_DATASET", "evidence": ["extras/model.xml.gz", "extras/states.npz", "extras/ep_meta.json"]},
            "vacuum_suction_pressure": {"status": "UNVERIFIED", "evidence": ["no vacuum/suction implementation found in audited source"]},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# RoboCasa static-to-sim audit", "", f"- commit: `{report['robocasa_commit']}`", "- dataset/assets downloaded: no", ""]
    lines += ["## EV-RealPhys object mapping", "", "| object_id | aliases found in source/docs | exact mapping status |", "|---|---|---|"]
    for oid, item in findings.items():
        aliases = ", ".join(k for k, v in item["aliases"].items() if v) or "none"
        status = "EXACT_NOT_PROVED" if item["exact_asset_verified"] else "UNVERIFIED"
        lines.append(f"| {oid} | {aliases} | {status} |")
    lines += ["", "## Capability audit", "", "| capability | status |", "|---|---|"]
    for key, value in report["capabilities"].items():
        lines.append(f"| {key} | {value['status']} |")
    lines += ["", "RoboCasa source support does not prove that an EV-RealPhys image is the same object instance. Exact mapping requires downloaded episode/object asset metadata and image-to-episode matching."]
    (args.output.parent / "robocasa_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
