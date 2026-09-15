"""Lock static EV-RealPhys inputs without reading physical GT or old predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/c4_real5")]
from real_rgbd import load_frame, object_crop_and_points  # noqa: E402

SOURCE = ROOT / "data/external/kandukuri_ev_realphys"
POLICY = {
    "version": "static_real5_v1_20260914",
    "initial_frames": [10, 40],
    "early_fallback_frames": [5, 15, 20, 25, 0, 30],
    "late_fallback_frames": [35, 45, 50, 55, 30, 59],
    "minimum_frame_separation": 20,
    "mass_bbox_padding_px": 8,
    "context_left_right_bbox_fraction": 0.75,
    "context_top_bbox_fraction": 0.25,
    "context_bottom_bbox_fraction": 1.0,
    "minimum_crop_width_height_px": 35,
    "minimum_crop_foreground_fraction": 0.08,
    "minimum_valid_depth_fraction_in_mask": 0.5,
    "minimum_visible_support_context_pixels": 300,
    "minimum_laplacian_variance": 15.0,
    "exclusion_criteria": ["decode failure", "unidentifiable/absent object", "severe truncation/occlusion",
                           "mass crop mostly background", "support not visible", "identical RGB duplicate"],
    "warning_only": ["weak background contamination", "robot visible"],
    "forbidden_for_static_qc": ["physical GT", "old predictions", "free-slide interval",
                                "acceleration", "timestamp", "rotation"],
}
FIELDS = ["input_id", "object_id", "sequence_id", "frame_id", "rgb_path", "depth_path",
          "mass_crop_path", "friction_context_path", "mass_qc_status", "friction_qc_status",
          "warning_reason", "selection_rule_version", "input_sha256"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def candidate(scene: Path, frame: int) -> dict:
    bgr, depth, _, _ = load_frame(scene, frame)
    crop, points_mm, mask, plane = object_crop_and_points(scene, frame)
    yy, xx = np.nonzero(mask)
    x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
    h, w = bgr.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    px = round(POLICY["context_left_right_bbox_fraction"] * bw)
    top = round(POLICY["context_top_bbox_fraction"] * bh)
    bottom = round(POLICY["context_bottom_bbox_fraction"] * bh)
    cx0, cx1 = max(0, x0 - px), min(w, x1 + px)
    cy0, cy1 = max(0, y0 - top), min(h, y1 + bottom)
    context = bgr[cy0:cy1, cx0:cx1]
    crop_fg = float(np.mean(np.any(crop > 0, axis=2)))
    valid_depth = float(np.mean(depth[mask > 0] > 0))
    blur = float(cv2.Laplacian(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())
    # A lower-image support-plane footprint is only a visibility cue, never a friction measurement.
    core = plane[3][cy0:cy1, cx0:cx1] > 0
    outside = np.ones(core.shape, dtype=bool)
    outside[y0-cy0:y1-cy0, x0-cx0:x1-cx0] = False
    support_pixels = int(np.count_nonzero(core & outside))
    checks = []
    if min(bw, bh) < POLICY["minimum_crop_width_height_px"]:
        checks.append("object/mask too small")
    if min(x0, y0, w-x1, h-y1) < 3:
        checks.append("severe image-edge truncation")
    if crop_fg < POLICY["minimum_crop_foreground_fraction"]:
        checks.append("mass crop mostly background")
    if valid_depth < POLICY["minimum_valid_depth_fraction_in_mask"]:
        checks.append("insufficient object depth")
    if support_pixels < POLICY["minimum_visible_support_context_pixels"] or cy1 <= y1 + 12:
        checks.append("support surface not visible")
    if blur < POLICY["minimum_laplacian_variance"]:
        checks.append("severe blur")
    warning = []
    if crop_fg < 0.23:
        warning.append("weak background/mask contamination; visual review required")
    if min(x0, y0, w-x1, h-y1) < 12:
        warning.append("object near image edge; visual review required")
    return {"frame": frame, "bgr": bgr, "depth": depth, "crop": crop, "points_mm": points_mm,
            "mask": mask, "context": context, "rgb_sha256": sha(scene / "rgb" / f"{frame:06d}.png"),
            "metrics": {"bbox_xyxy": [x0, y0, x1, y1], "crop_foreground_fraction": crop_fg,
                        "valid_depth_fraction_in_mask": valid_depth, "laplacian_variance": blur,
                        "visible_support_pixels": support_pixels, "context_xyxy": [cx0, cy0, cx1, cy1],
                        "point_count": len(points_mm)},
            "checks": checks, "warnings": warning}


def tile(row: dict, run: Path) -> np.ndarray:
    pics = [cv2.imread(str(ROOT / row[k])) for k in ("rgb_path", "mass_crop_path", "friction_context_path")]
    canvas = np.full((172, 750, 3), 255, np.uint8)
    for j, img in enumerate(pics):
        if img is None:
            continue
        ih, iw = img.shape[:2]
        scale = min(235 / iw, 145 / ih)
        small = cv2.resize(img, (max(1, int(iw * scale)), max(1, int(ih * scale))))
        x = j * 250 + (250-small.shape[1]) // 2
        canvas[2:2+small.shape[0], x:x+small.shape[1]] = small
    cv2.putText(canvas, f"{row['sequence_id'][-6:]} f{row['frame_id']} {row['object_id']}",
                (8, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_static_real5_prep"
    run = (args.output_dir or ROOT / "output/c4_table3" / name).resolve()
    run.mkdir(parents=True, exist_ok=False)
    for sub in ("inputs/mass", "inputs/friction", "inputs/points", "inputs/mask_overlays", "smoke_test"):
        (run / sub).mkdir(parents=True)
    (run / "selection_policy.json").write_text(json.dumps(POLICY, indent=2) + "\n", encoding="utf-8")
    source_file = SOURCE / "manifest.json"
    source = json.loads(source_file.read_text(encoding="utf-8"))
    if sha(SOURCE / source["source_archive"]) != source["archive_sha256"]:
        raise ValueError("official source archive SHA-256 mismatch")
    sequences = source["sequences"]
    counts = Counter(r["object_id"] for r in sequences)
    if len(sequences) != 25 or sorted(counts.values()) != [5] * 5 or any("tuna" in x for x in counts):
        raise ValueError("not the official five-object, 25-sequence mapping")
    rows, attempts, seen_rgb = [], [], set()
    for source_row in sequences:
        sid, oid = source_row["sequence_id"], source_row["object_id"]
        if not source_row.get("mapping_verified") or not source_row.get("rgb_depth_aligned"):
            raise ValueError(f"source mapping/alignment unverified: {sid}")
        scene = SOURCE / "ev-realphys" / sid
        picked = []
        pools = [[10] + POLICY["early_fallback_frames"], [40] + POLICY["late_fallback_frames"]]
        for slot, options in enumerate(pools):
            selected = None
            for frame in options:
                if slot and picked and frame-picked[0]["frame"] < POLICY["minimum_frame_separation"]:
                    continue
                try:
                    c = candidate(scene, frame)
                    if c["rgb_sha256"] in seen_rgb:
                        c["checks"].append("identical RGB duplicate")
                    if any(p["rgb_sha256"] == c["rgb_sha256"] for p in picked):
                        c["checks"].append("same-sequence identical RGB duplicate")
                    attempts.append({"sequence_id": sid, "object_id": oid, "slot": slot,
                                     "frame": frame, "checks": c["checks"], "warnings": c["warnings"],
                                     "metrics": c["metrics"]})
                    if not c["checks"]:
                        selected = c
                        break
                except Exception as exc:
                    attempts.append({"sequence_id": sid, "object_id": oid, "slot": slot,
                                     "frame": frame, "checks": [f"{type(exc).__name__}: {exc}"]})
            if selected is None:
                raise RuntimeError(f"no valid static frame for {sid} slot {slot}")
            picked.append(selected)
            seen_rgb.add(selected["rgb_sha256"])
        for c in picked:
            frame = c["frame"]
            iid = f"{sid.split('/')[-1]}_{frame:06d}"
            mass = run / "inputs/mass" / f"{iid}.png"
            friction = run / "inputs/friction" / f"{iid}.png"
            points = run / "inputs/points" / f"{iid}.npz"
            overlay = run / "inputs/mask_overlays" / f"{iid}.png"
            cv2.imwrite(str(mass), cv2.cvtColor(c["crop"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(friction), c["context"])
            np.savez_compressed(points, points_mm=c["points_mm"])
            vis = c["bgr"].copy()
            vis[c["mask"] > 0] = (0.6 * vis[c["mask"] > 0] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(str(overlay), vis)
            rgb = scene / "rgb" / f"{frame:06d}.png"
            depth = scene / "depth" / f"{frame:06d}.png"
            combined = hashlib.sha256("".join(sha(p) for p in (rgb, depth, mass, friction, points)).encode()).hexdigest()
            rows.append({"input_id": iid, "object_id": oid, "sequence_id": sid,
                         "frame_id": frame, "rgb_path": rel(rgb), "depth_path": rel(depth),
                         "mass_crop_path": rel(mass), "friction_context_path": rel(friction),
                         "mass_qc_status": "WARNING" if c["warnings"] else "APPROVED",
                         "friction_qc_status": "WARNING" if c["warnings"] else "APPROVED",
                         "warning_reason": "; ".join(c["warnings"]),
                         "selection_rule_version": POLICY["version"], "input_sha256": combined})
    with (run / "selected_static_inputs.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    with (run / "candidate_qc.jsonl").open("x", encoding="utf-8") as f:
        for item in attempts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    grid = np.full((172 * 10, 750 * 5, 3), 225, np.uint8)
    for idx, row in enumerate(rows):
        grid[(idx // 5)*172:((idx // 5)+1)*172, (idx % 5)*750:((idx % 5)+1)*750] = tile(row, run)
    cv2.imwrite(str(run / "static_input_contact_sheet.png"), grid)
    report = ["# Static Real-5 입력 QC", "", "이 선택은 GT와 기존 예측을 읽지 않고 RGB, depth, 기존 M1/SiPhy crop 경로만 사용했다.",
              "자유 활주/감속도/시간축/회전은 정지 이미지 QC에 사용하지 않았다.",
              f"원본: {source['source_url']} · archive SHA-256 `{source['archive_sha256']}`",
              f"source manifest SHA-256 `{sha(source_file)}` · policy `{POLICY['version']}`", "",
              f"선정 {len(rows)}/50, 객체별 {dict(Counter(r['object_id'] for r in rows))}.",
              f"초기 프레임 교체 {sum(r['frame_id'] not in (10,40) for r in rows)}건."
              f" 후보 실패 {sum(bool(a['checks']) for a in attempts)}건.",
              f"경고 입력 {sum(bool(r['warning_reason']) for r in rows)}건. 경고는 자동 제외하지 않았다.", "",
              "동일 RGB 중복은 파일 SHA-256으로 검사했다. RGB와 depth는 같은 프레임 번호, 848×480 크기와 per-frame K를 사용한다.",
              "마스크는 depth 지지 평면 분할→RGB GrabCut→M1 점군 필터로 생성했다. 공개 GT mask, pose, 물성값은 사용하지 않았다.",
              "원본 RGB/depth와 추출 mask overlay를 별도 보존했다. crop/context는 모든 객체에 동일 padding 규칙을 적용했다.",
              "약한 배경 혼입과 로봇 노출은 contact sheet 수동 점검 후 WARNING에 기록한다.", "",
              "## 교체·경고", ""]
    for row in rows:
        if row["frame_id"] not in (10, 40) or row["warning_reason"]:
            report.append(f"- {row['input_id']}: " + ("대체 프레임; " if row["frame_id"] not in (10,40) else "")
                          + (row["warning_reason"] or "자동 QC 실패 후보를 대체"))
    (run / "static_input_qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (run / "run.log").write_text(f"PREP_CREATED {datetime.now(timezone.utc).isoformat()}\n"
                                   f"source_manifest_sha256={sha(source_file)}\n"
                                   f"selection_policy={POLICY['version']}\nselected_inputs={len(rows)}\n"
                                   f"initial_replacements={sum(r['frame_id'] not in (10,40) for r in rows)}\n",
                                   encoding="utf-8")
    print(run)


if __name__ == "__main__":
    main()
