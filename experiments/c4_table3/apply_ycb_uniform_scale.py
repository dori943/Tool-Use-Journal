"""Create reversible, uniformly scaled YCB mesh copies for simulator calibration.

The source YCB assets are never modified.  Scale factors come only from the
geometry comparison artifact and are recorded alongside hashes and residual
errors.  This utility deliberately does not apply per-axis scaling because it
would change the physical shape of an object.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scale_obj(src: Path, dst: Path, factor: float) -> None:
    """Scale only vertex positions, preserving faces, normals and UVs."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", errors="replace") as inp, dst.open("w", encoding="utf-8", newline="\n") as out:
        for line in inp:
            stripped = line.lstrip()
            if stripped.startswith("v ") or stripped.startswith("v\t"):
                prefix_len = len(line) - len(stripped)
                parts = stripped.split()
                try:
                    xyz = [float(parts[i]) * factor for i in range(1, 4)]
                except (IndexError, ValueError):
                    out.write(line)
                    continue
                lead = line[:prefix_len]
                # Keep optional homogeneous coordinate, comments, and a stable
                # decimal representation for deterministic output.
                tail = " " + " ".join(parts[4:]) if len(parts) > 4 else ""
                out.write(f"{lead}v {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}{tail}\n")
            else:
                out.write(line)


def apply(comparison_json: Path, ycb_root: Path, output: Path) -> dict[str, object]:
    comparison = json.loads(comparison_json.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in comparison["rows"]:
        oid = row["object_id"]
        src_mesh = Path(row["mesh_path"])
        if not src_mesh.exists():
            # Permit moving the comparison artifact between machines while
            # retaining its relative YCB path.
            src_mesh = ycb_root / oid / oid / "poisson" / "textured.obj"
        if not src_mesh.exists():
            raise FileNotFoundError(f"mesh not found for {oid}: {src_mesh}")
        source_unit_scale = float(row["uniform_scale_fit"]) / 1000.0
        src_asset_root = src_mesh.parents[1]  # <ycb>/<oid>/<oid>
        dst_asset_root = output / oid / oid
        if dst_asset_root.exists():
            shutil.rmtree(dst_asset_root)
        shutil.copytree(src_asset_root, dst_asset_root)
        dst_mesh = dst_asset_root / src_mesh.relative_to(src_asset_root)
        # copytree copied the original OBJ; replace it with transformed vertices
        scale_obj(src_mesh, dst_mesh, source_unit_scale)
        measured = row["mesh_extent_source_units_sorted"]
        reference = row["reference_extent_mm_sorted"]
        scaled_mm = [float(x) * source_unit_scale * 1000.0 for x in measured]
        residual = [abs(x - y) / y for x, y in zip(scaled_mm, reference)]
        rows.append({
            "object_id": oid,
            "source_mesh": str(src_mesh.resolve()),
            "source_mesh_sha256": sha256(src_mesh),
            "scaled_mesh": str(dst_mesh.resolve()),
            "scaled_mesh_sha256": sha256(dst_mesh),
            "uniform_scale_factor": source_unit_scale,
            "scale_units": "dimensionless multiplier applied to metre-like OBJ coordinates",
            "predicted_extent_mm": scaled_mm,
            "reference_extent_mm": reference,
            "axis_residual_fraction": residual,
            "max_residual_fraction": max(residual),
            "source_preserved": True,
        })
    metadata = {
        "schema_version": "c4_ycb_uniform_scaled_assets_v1",
        "source_comparison": str(comparison_json.resolve()),
        "policy": "uniform scaling only; no per-axis deformation; source assets unchanged",
        "rows": rows,
    }
    (output / "scaled_assets.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# YCB uniformly scaled simulator assets", "",
        "원본 OBJ는 수정하지 않고 객체별 uniform scale 복사본을 생성했다.",
        "scale factor = uniform_scale_fit / 1000 (원본 좌표가 metre-like라는 가정).",
        "비균일 축 scaling은 형상 왜곡을 일으키므로 적용하지 않았다.", "",
        "| object | factor | predicted extent (mm) | max residual | status |",
        "|---|---:|---|---:|---|",
    ]
    for r in rows:
        status = "WITHIN_10_PERCENT" if r["max_residual_fraction"] <= .10 else "REVIEW_10_25_PERCENT" if r["max_residual_fraction"] <= .25 else "MISMATCH_GT_25_PERCENT"
        lines.append(f"| {r['object_id']} | {r['uniform_scale_factor']:.6f} | {', '.join(f'{x:.1f}' for x in r['predicted_extent_mm'])} | {r['max_residual_fraction']:.1%} | {status} |")
    lines += ["", "이 결과는 simulator calibration용 후보 asset이다. EV 이미지와 동일한 실물 인스턴스라는 검증이나 최종 pose/scale 승인을 의미하지 않는다."]
    (output / "scaled_assets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"object_count": len(rows), "output": str(output.resolve()), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-json", type=Path, required=True)
    ap.add_argument("--ycb-root", type=Path, default=Path("data/external/ycb"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(apply(args.comparison_json.resolve(), args.ycb_root.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
