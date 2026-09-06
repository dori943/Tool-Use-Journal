"""v4 observation / M1 preparation — same pipeline as Experiment 1 v3.

Reuses shared helpers (no new bbox / crop logic):
  - observation_capture.capture_all_observations
  - single_object_scene (via capture)
  - m1_bbox_extract (via capture → production M1)
  - gt_loader (GT manifest only; never used as bbox input)

Writes under output/m0_retrieval/experiment1_v4/ only.
Does not read or copy bbox from v3 unit JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EXP_V4 = Path(__file__).resolve().parent
PARENT_EXP = EXP_V4.parent
ROOT = EXP_V4.parents[2]

for p in (ROOT, ROOT / "src", PARENT_EXP, EXP_V4):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gt_loader import load_all_gt  # noqa: E402
from objects import OBJECTS  # noqa: E402
from observation_capture import (  # noqa: E402
    capture_all_observations,
    print_observation_table,
)

OUT_DIR = ROOT / "output" / "m0_retrieval" / "experiment1_v4"
M1_MANIFEST = OUT_DIR / "m1_bbox_manifest.json"
OBS_DIR = OUT_DIR / "observations"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_m1_manifest_v4() -> dict[str, Any]:
    """Load only the v4-local M1 manifest (production M1 from this experiment's scenes)."""
    if M1_MANIFEST.exists():
        return json.loads(M1_MANIFEST.read_text(encoding="utf-8"))
    return {}


def crop_path_v4(object_key: str) -> Path:
    return OBS_DIR / f"{object_key}_crop.png"


def m1_bbox_for(object_key: str, manifest: dict[str, Any] | None = None) -> list[float] | None:
    manifest = manifest if manifest is not None else load_m1_manifest_v4()
    entry = manifest.get(object_key) or {}
    bbox = entry.get("m1_bbox_mm")
    if bbox is None:
        return None
    return [float(x) for x in bbox]


def has_valid_observation(object_key: str) -> bool:
    """True when v4 has crop PNG + non-null production M1 bbox_mm."""
    if not crop_path_v4(object_key).is_file():
        return False
    return m1_bbox_for(object_key) is not None


def capture_observations_v4(object_keys: list[str] | None = None) -> list[dict[str, Any]]:
    """Same capture path as v3 ``run_experiment1._do_capture``, writing to v4 OUT_DIR.

    Pipeline (identical to v3):
      GT load → single-object scene → RGB/depth/seg → production M1 bbox_mm
      → SiPhy-style crop → observation_manifest + m1_bbox_manifest
    """
    keys = object_keys or list(OBJECTS)
    unknown = [k for k in keys if k not in OBJECTS]
    if unknown:
        raise SystemExit(f"unknown --object values: {unknown}")

    gts = load_all_gt()
    write_json(OUT_DIR / "gt_manifest.json", gts)

    print(
        f"[v4 OBS] same pipeline as v3 observation_capture "
        f"(single_object_scene + production M1)\n"
        f"[v4 OBS] out={OBS_DIR}\n"
        f"[v4 OBS] objects={keys}\n"
        f"[v4 OBS] LLM calls: none",
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

    existing = load_m1_manifest_v4()
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


def ensure_observations_v4(object_keys: list[str], *, force: bool = False) -> None:
    """Ensure each object has v4-local crop + production M1 bbox_mm.

    Missing or incomplete → run the same capture path as v3 (per object).
    Does not copy bbox from v3 unit JSON / GT / hard-code.
    """
    need = [k for k in object_keys if force or not has_valid_observation(k)]
    if not need:
        print(
            f"[v4 OBS] reuse existing v4 observations for {object_keys} "
            f"(crop + m1_bbox_mm present)",
            flush=True,
        )
        return
    print(
        f"[v4 OBS] capturing missing/invalid observations: {need}",
        flush=True,
    )
    capture_observations_v4(need)
