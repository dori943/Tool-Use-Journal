"""Audit and optionally prepare YCB meshes for use as robosuite objects.

The official robosuite checkout does not ship the YCB object set.  This
utility therefore treats YCB as an external, user-supplied asset and records
provenance before an object can be used in a simulator experiment.  It never
loads Real-5 measured mass/friction values and never claims an image-to-mesh
match from a filename alone.

Usage (audit only)::

    python experiments/c4_table3/ycb_robosuite_audit.py \
      --robosuite external/robosuite \
      --ycb-root data/external/ycb \
      --output output/c4_table3/<run>_ycb_audit

To emit MJCF wrappers after independently verifying units, provide an
explicit scale (meters per source unit).  A mass is deliberately not accepted
by this command: inertial parameters must be supplied by a separate simulator
calibration file rather than from the Real-5 evaluator GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


YCB_SOURCE_URL = "https://ycb-benchmarks.s3.amazonaws.com/index.html"
ROBOSUITE_SOURCE_URL = "https://github.com/ARISE-Initiative/robosuite"
TARGETS = {
    "003_cracker_box": "Cracker Box",
    "006_mustard_bottle": "Mustard Bottle",
    "019_pitcher_base": "Pitcher Base",
    "021_bleach_cleanser": "Bleach Cleanser",
    "025_mug": "Mug",
}
YCB_MESH_URLS = {
    oid: f"https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/{oid}/{oid}_berkeley_meshes.tgz"
    for oid in TARGETS
}
MESH_EXTENSIONS = {".obj", ".ply", ".stl", ".msh"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _normalised(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def candidate_dirs(root: Path, object_id: str) -> list[Path]:
    """Find directories whose basename contains the exact YCB id.

    The id check is intentionally strict enough to avoid accidentally binding
    a generic ``mug`` or ``bottle`` asset to a Real-5 object.
    """
    token = _normalised(object_id)
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_dir() and token in _normalised(p.name):
            out.append(p)
    return sorted(set(out))


def mesh_files(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        found.extend(
            p for p in path.rglob("*")
            if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in MESH_EXTENSIONS
        )
    # Prefer the textured OBJ (usually the easiest mesh to load in MuJoCo),
    # then non-textured collision formats.  Keep all alternatives in the
    # manifest for review rather than silently discarding them.
    priority = {".obj": 0, ".stl": 1, ".ply": 2, ".msh": 3}
    return sorted(set(found), key=lambda p: (priority.get(p.suffix.lower(), 9), str(p).lower()))


def parse_obj_bounds(path: Path) -> dict[str, object] | None:
    """Read OBJ vertices without requiring trimesh/open3d."""
    if path.suffix.lower() != ".obj":
        return None
    vertices: list[tuple[float, float, float]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] == "v":
                vertices.append(tuple(float(v) for v in parts[1:]))
    except (OSError, ValueError):
        return None
    if not vertices:
        return None
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    return {"vertex_count": len(vertices), "bounds_source_units": {"min": mins, "max": maxs},
            "extent_source_units": [maxs[i] - mins[i] for i in range(3)]}


def inspect_robosuite(repo: Path) -> dict[str, object]:
    files = list(repo.rglob("*.py")) if repo.exists() else []
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files[:500])
    lower = text.lower()
    return {
        "repo": str(repo),
        "commit": git_head(repo),
        "official_ycb_reference_found": "ycb" in lower,
        "custom_mjcf_object_api": "mujocoxmlobject" in lower,
        "mesh_asset_support": "<mesh" in lower or "mesh" in lower,
        "built_in_target_assets": {oid: _normalised(oid) in _normalised(text) for oid in TARGETS},
    }


def wrapper_xml(mesh_rel: str, object_id: str, scale: float) -> str:
    # The wrapper is a geometry container only.  No mass or friction is set;
    # callers must provide calibrated simulator parameters separately.
    # robosuite's MujocoXMLObject expects a top-level anonymous body with a
    # child named ``object``.  The collision geom is group 0 and the visual
    # geom is group 1, matching robosuite's object-model conventions.
    return f'''<mujoco model="{object_id}">\n  <asset>\n    <mesh name="{object_id}_mesh" file="{mesh_rel}" scale="{scale:g} {scale:g} {scale:g}"/>\n  </asset>\n  <worldbody>\n    <body>\n      <body name="object" pos="0 0 0">\n        <geom name="{object_id}_collision" type="mesh" mesh="{object_id}_mesh" group="0"/>\n        <geom name="{object_id}_visual" type="mesh" mesh="{object_id}_mesh" group="1" contype="0" conaffinity="0"/>\n      </body>\n    </body>\n  </worldbody>\n</mujoco>\n'''


def audit(root: Path, ycb_root: Path, output: Path, emit_mjcf: bool = False, unit_scale: float | None = None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    if emit_mjcf and (unit_scale is None or unit_scale <= 0):
        raise ValueError("--emit-mjcf requires a positive --unit-scale; source units must be verified first")
    targets: dict[str, object] = {}
    wrappers = output / "mjcf"
    if emit_mjcf:
        wrappers.mkdir(exist_ok=True)
    for object_id, object_name in TARGETS.items():
        dirs = candidate_dirs(ycb_root, object_id)
        meshes = mesh_files(dirs)
        mesh_records = []
        for mesh in meshes:
            record = {
                "path": str(mesh.resolve()),
                "sha256": sha256(mesh),
                "format": mesh.suffix.lower().lstrip("."),
                "bounds": parse_obj_bounds(mesh),
            }
            mesh_records.append(record)
        wrapper = None
        if emit_mjcf and meshes:
            # Pick the first deterministic mesh; the manifest still lists all
            # alternatives so a reviewer can select a collision-specific mesh.
            mesh = meshes[0]
            rel = Path(os.path.relpath(mesh.resolve(), wrappers.resolve())).as_posix()
            wrapper_path = wrappers / f"{object_id}.xml"
            wrapper_path.write_text(wrapper_xml(rel, object_id, float(unit_scale)), encoding="utf-8")
            wrapper = {"path": str(wrapper_path.resolve()), "mesh_path": str(mesh.resolve()),
                       "unit_scale_m_per_source_unit": float(unit_scale), "mass_friction": "UNSET"}
        targets[object_id] = {
            "object_name": object_name,
            "download_url": YCB_MESH_URLS[object_id],
            "archive_sha256": sha256(ycb_root / f"{object_id}.tgz") if (ycb_root / f"{object_id}.tgz").is_file() else None,
            "candidate_directories": [str(p.resolve()) for p in dirs],
            "mesh_files": mesh_records,
            "mapping_status": "EXACT_ASSET_FOUND" if meshes else "YCB_ASSET_NOT_FOUND",
            "image_instance_mapping": "UNVERIFIED",
            "sand_fill_mass_represented": False,
            "real5_friction_represented": False,
            "mjcf_wrapper": wrapper,
        }
    report = {
        "schema_version": "c4_ycb_robosuite_audit_v1",
        "ycb_source": {"url": YCB_SOURCE_URL, "license": "CC BY 4.0", "downloaded": ycb_root.exists() and any(ycb_root.iterdir())},
        "robosuite_source": {**inspect_robosuite(root), "url": ROBOSUITE_SOURCE_URL},
        "ycb_root": str(ycb_root.resolve()),
        "targets": targets,
        "excluded_object_ids": ["007_tuna_fish_can"],
        "provenance_rules": {
            "real5_gt_allowed_in_simulator": False,
            "mass_and_friction_required_from_calibration": True,
            "filename_only_image_mapping_allowed": False,
            "source_units_must_be_verified_before_mjcf": True,
        },
    }
    (output / "ycb_robosuite_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# YCB → robosuite 자산 감사", "", f"- YCB source: [{YCB_SOURCE_URL}]({YCB_SOURCE_URL})", f"- robosuite commit: `{report['robosuite_source']['commit']}`", f"- YCB root: `{ycb_root}`", "", "| YCB id | asset files | exact image mapping | simulator use |", "|---|---:|---|---|"]
    for oid, item in targets.items():
        lines.append(f"| {oid} | {len(item['mesh_files'])} | {item['image_instance_mapping']} | {item['mapping_status']} |")
    lines += ["", "## 해석", "", "robosuite 자체에는 YCB 모델이 포함되지 않으므로 외부 YCB mesh를 MJCF `MujocoXMLObject`로 연결해야 한다. 파일명이 같다는 사실만으로 EV-RealPhys 이미지와 동일한 실물 인스턴스라고 주장하지 않는다.", "", "YCB mesh에는 모래 충전 질량과 Table 5의 실측 결합 마찰이 들어 있지 않다. 해당 값은 simulator 입력으로 주입하지 않으며, 별도 calibration split에서 정한 시뮬레이터 파라미터만 사용한다.", "", "Tuna Fish Can(007)은 대상과 산출물에서 제외했다."]
    (output / "ycb_robosuite_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robosuite", type=Path, default=Path("external/robosuite"))
    parser.add_argument("--ycb-root", type=Path, default=Path("data/external/ycb"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-mjcf", action="store_true")
    parser.add_argument("--unit-scale", type=float)
    args = parser.parse_args()
    report = audit(args.robosuite.resolve(), args.ycb_root.resolve(), args.output.resolve(), args.emit_mjcf, args.unit_scale)
    print(json.dumps({"output": str(args.output.resolve()), "targets": {k: v["mapping_status"] for k, v in report["targets"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
