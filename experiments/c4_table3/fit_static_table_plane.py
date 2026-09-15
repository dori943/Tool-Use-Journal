"""Estimate a common table plane from locked RGB-D frames.

The estimator is deliberately conservative: it excludes the locked object
bounding box, uses one global RANSAC threshold for all objects, and reports
the result in native depth units.  It does not read GT or choose thresholds
from prediction errors.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _fit_plane(points: np.ndarray, threshold: float = 5.0, iterations: int = 80) -> tuple[np.ndarray, float, float]:
    if len(points) < 3:
        raise ValueError("not enough valid points for plane fit")
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iterations):
        tri = points[rng.choice(len(points), size=3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ tri[0])
        dist = np.abs(points @ n + d)
        inliers = dist <= threshold
        score = (int(inliers.sum()), -float(np.median(dist[inliers])) if inliers.any() else -1e9)
        if best is None or score > best[0]:
            best = (score, n, d, inliers)
    if best is None:
        raise ValueError("degenerate depth sample")
    _, n, d, inliers = best
    # Refit using all inliers for a stable normal, keeping the same threshold.
    q = points[inliers]
    centroid = q.mean(axis=0)
    _, _, vh = np.linalg.svd(q - centroid, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = -float(n @ centroid)
    residual = np.abs(points @ n + d)
    return n, d, float(np.median(residual[inliers]))


def build(static_manifest: Path, qc_jsonl: Path, output: Path, repo_root: Path) -> dict[str, object]:
    rows = list(csv.DictReader(static_manifest.open(encoding="utf-8", newline="")))
    qc = {}
    for line in qc_jsonl.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        qc[(item["sequence_id"], str(item["frame"]))] = item
    out = []
    for row in rows:
        depth_path = Path(row["depth_path"])
        if not depth_path.is_absolute():
            depth_path = repo_root / depth_path
        depth = np.asarray(Image.open(depth_path), dtype=np.float64)
        camera_path = depth_path.parent.parent / "scene_camera.json"
        camera = json.loads(camera_path.read_text(encoding="utf-8"))[str(int(row["frame_id"]))]
        K = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
        yy, xx = np.mgrid[0:depth.shape[0]:8, 0:depth.shape[1]:8]
        zz = depth[::8, ::8]
        q = qc.get((row["sequence_id"], str(int(row["frame_id"]))))
        if q is None:
            raise ValueError(f"missing QC bbox for {row['input_id']}")
        x0, y0, x1, y1 = [int(v) for v in q["metrics"]["bbox_xyxy"]]
        keep = ~((xx >= x0 - 12) & (xx <= x1 + 12) & (yy >= y0 - 12) & (yy <= y1 + 12))
        valid = keep & np.isfinite(zz) & (zz > 0)
        u, v, z = xx[valid].astype(float), yy[valid].astype(float), zz[valid].astype(float)
        if len(z) > 3500:
            # Deterministic, uniform subsample bounds runtime on large frames.
            idx = np.linspace(0, len(z) - 1, 3500, dtype=int)
            u, v, z = u[idx], v[idx], z[idx]
        pts = np.column_stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))
        try:
            n, d, med = _fit_plane(pts)
            distances = np.abs(pts @ n + d)
            inlier_ratio = float(np.mean(distances <= 5.0))
            status = "ESTIMATED" if inlier_ratio >= 0.35 and med <= 5.0 else "UNRESOLVED"
        except ValueError:
            n, d, med, inlier_ratio, status = np.zeros(3), float("nan"), float("nan"), 0.0, "UNRESOLVED"
        out.append({
            "input_id": row["input_id"], "object_id": row["object_id"], "sequence_id": row["sequence_id"], "frame_id": row["frame_id"],
            "plane_normal_camera": json.dumps([float(v) for v in n]), "plane_offset_native": d,
            "median_residual_native": med, "inlier_ratio": inlier_ratio, "threshold_native": 5.0,
            "table_plane_status": status, "depth_scale_metadata": camera.get("depth_scale"),
            "metric_scale_status": "NEEDS_CALIBRATION", "gt_access": False,
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "table_plane_estimates.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0])); writer.writeheader(); writer.writerows(out)
    estimated = sum(r["table_plane_status"] == "ESTIMATED" for r in out)
    summary = {"schema_version": "c4_static_table_plane_v1", "input_count": len(out), "estimated_count": estimated,
               "unresolved_count": len(out) - estimated, "threshold_native": 5.0, "threshold_policy": "global_fixed",
               "metric_scale_status": "NEEDS_CALIBRATION", "gt_access": False}
    (output / "table_plane_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "table_plane_report.md").write_text(
        "# Static RGB-D table-plane audit\n\n"
        f"- inputs: {len(out)}\n- estimated: {estimated}\n- unresolved: {len(out)-estimated}\n"
        "- threshold: 5 native depth units (one global fixed rule)\n"
        "- metric scale: `NEEDS_CALIBRATION`\n\n"
        "객체 bbox를 동일한 12 px padding으로 제외한 뒤 depth centroid와 RANSAC 평면을 계산했다. GT나 예측 오차는 사용하지 않았다. 평면 추정 성공은 simulator world-frame 정합이나 최종 접촉 GT를 의미하지 않는다.\n",
        encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-manifest", type=Path, required=True)
    ap.add_argument("--qc-jsonl", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(build(args.static_manifest, args.qc_jsonl, args.output, args.repo_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
