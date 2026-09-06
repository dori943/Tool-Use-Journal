"""v5 observation / M1 preparation — same pipeline as Experiment 1 v3/v4.

Reuses shared helpers (no new bbox / crop logic).
Writes under output/m0_retrieval/experiment1_v5/.

If v4 already has valid crop + m1_bbox for an object, copies those into v5
so Stage1/Stage2 see the same observation bytes (fair v4–v5 comparison).
Does not modify v4 files.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

EXP_V5 = Path(__file__).resolve().parent
PARENT_EXP = EXP_V5.parent
ROOT = EXP_V5.parents[2]

for p in (ROOT, ROOT / "src", PARENT_EXP, EXP_V5):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gt_loader import load_all_gt  # noqa: E402
from objects import OBJECTS  # noqa: E402
from observation_capture import (  # noqa: E402
    capture_all_observations,
    print_observation_table,
)

OUT_DIR = ROOT / "output" / "m0_retrieval" / "experiment1_v5"
M1_MANIFEST = OUT_DIR / "m1_bbox_manifest.json"
OBS_DIR = OUT_DIR / "observations"

V4_OUT = ROOT / "output" / "m0_retrieval" / "experiment1_v4"
V4_OBS = V4_OUT / "observations"
V4_M1 = V4_OUT / "m1_bbox_manifest.json"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_m1_manifest_v5() -> dict[str, Any]:
    if M1_MANIFEST.exists():
        return json.loads(M1_MANIFEST.read_text(encoding="utf-8"))
    return {}


def crop_path_v5(object_key: str) -> Path:
    return OBS_DIR / f"{object_key}_crop.png"


def m1_bbox_for(object_key: str, manifest: dict[str, Any] | None = None) -> list[float] | None:
    manifest = manifest if manifest is not None else load_m1_manifest_v5()
    entry = manifest.get(object_key) or {}
    bbox = entry.get("m1_bbox_mm")
    if bbox is None:
        return None
    return [float(x) for x in bbox]


def has_valid_observation(object_key: str) -> bool:
    if not crop_path_v5(object_key).is_file():
        return False
    return m1_bbox_for(object_key) is not None


def _try_reuse_v4(object_key: str) -> bool:
    """Copy v4 crop + m1 entry into v5 when available. Returns True on success."""
    v4_crop = V4_OBS / f"{object_key}_crop.png"
    if not v4_crop.is_file() or not V4_M1.is_file():
        return False
    v4_man = json.loads(V4_M1.read_text(encoding="utf-8"))
    entry = v4_man.get(object_key)
    if not entry or entry.get("m1_bbox_mm") is None:
        return False
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(v4_crop, crop_path_v5(object_key))
    # Optional companion PNGs for parity
    for suffix in ("_raw.png", "_m1_bbox.png", "_gt_bbox.png", "_bbox.png"):
        src = V4_OBS / f"{object_key}{suffix}"
        if src.is_file():
            shutil.copy2(src, OBS_DIR / f"{object_key}{suffix}")
    existing = load_m1_manifest_v5()
    existing[object_key] = dict(entry)
    # Point crop_path at v5 location
    existing[object_key]["crop_path"] = str(crop_path_v5(object_key))
    existing[object_key]["reused_from"] = "experiment1_v4"
    write_json(M1_MANIFEST, existing)
    print(
        f"[v5 OBS] reused v4 crop+m1 for {object_key} → {crop_path_v5(object_key)}",
        flush=True,
    )
    return True


def capture_observations_v5(object_keys: list[str] | None = None) -> list[dict[str, Any]]:
    keys = object_keys or list(OBJECTS)
    unknown = [k for k in keys if k not in OBJECTS]
    if unknown:
        raise SystemExit(f"unknown --object values: {unknown}")

    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)

    print(
        f"[v5 OBS] same pipeline as v3/v4 observation_capture "
        f"(single_object_scene + production M1)\n"
        f"[v5 OBS] out={OBS_DIR}\n"
        f"[v5 OBS] objects={keys}\n"
        f"[v5 OBS] LLM calls: none",
        flush=True,
    )
    obs = capture_all_observations(
        OBS_DIR,
        save_images=True,
        object_keys=keys,
        gt_by_object=gts,
    )
    print_observation_table(obs)
    write_json(OUT_DIR / "observation_manifest.json", obs)

    existing = load_m1_manifest_v5()
    for r in obs:
        if r.get("raw_status") == "OK":
            existing[r["object"]] = {
                "m1_bbox_mm": r.get("m1_bbox_mm"),
                "m1_center_mm": r.get("m1_center_mm"),
                "gt_bbox_mm": r.get("gt_bbox_mm"),
                "bbox_difference": r.get("bbox_difference"),
                "siphy_bbox_input": r.get("siphy_bbox_input"),
                "crop_path": r.get("crop_path"),
                "m1_source": r.get("m1_source"),
            }
    write_json(M1_MANIFEST, existing)
    return obs


def ensure_observations_v5(object_keys: list[str], *, force: bool = False) -> None:
    need: list[str] = []
    for k in object_keys:
        if force:
            need.append(k)
            continue
        if has_valid_observation(k):
            continue
        if _try_reuse_v4(k) and has_valid_observation(k):
            continue
        need.append(k)
    if not need:
        print(
            f"[v5 OBS] reuse existing v5 observations for {object_keys} "
            f"(crop + m1_bbox_mm present)",
            flush=True,
        )
        return
    print(f"[v5 OBS] capturing missing/invalid observations: {need}", flush=True)
    capture_observations_v5(need)
