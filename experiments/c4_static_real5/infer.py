"""Static C4 inference only: no evaluator import, numeric GT argument, or score computation.

One call per input per *backend*: SiPhy mass is shared by SiPhy/geometry/full;
the new visual object-table friction prompt belongs only to full. Old track
friction code is deliberately untouched and is not called here.
"""
from __future__ import annotations

import argparse
import builtins
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tuj.m1_scene.siphy_backend import SYS_MSG, SiPhyBackend, _to_b64_png  # noqa: E402

FRICTION_PROMPT = """You are estimating an object-table contact property from ONE still RGB image.
The image contains an object resting on a supporting table. Focus on the object's bottom contact
material and the visible support surface, not the robot or distant background. Estimate the
dimensionless COMBINED object-table kinetic friction coefficient as a visual prior only.
There is no measured motion, force, acceleration, or table tilt in this input, so do not claim a
physical measurement or high precision. Use no benchmark ground truth or object-specific lookup.
Return only JSON: {"combined_friction_coefficient": number, "visual_basis": "short phrase",
"uncertainty": "high or medium"}. The number must be positive and dimensionless.
"""
FORBIDDEN_GT_NAMES = {"evaluator_only_gt.yaml", "c4_real5_gt.json", "gt_snapshot.json"}


def deny_numeric_gt_reads() -> None:
    """Explicit runtime boundary in addition to the GT-free inference CLI/schema."""
    original_builtin, original_io, original_os = builtins.open, io.open, os.open

    def check(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            name = Path(path).name.lower()
            if name in FORBIDDEN_GT_NAMES or name.startswith("gt_snapshot"):
                raise PermissionError("NUMERIC_GT_FORBIDDEN_DURING_INFERENCE")

    def guarded_builtin(file, *args, **kwargs):
        check(file)
        return original_builtin(file, *args, **kwargs)

    def guarded_io(file, *args, **kwargs):
        check(file)
        return original_io(file, *args, **kwargs)

    def guarded_os(file, *args, **kwargs):
        check(file)
        return original_os(file, *args, **kwargs)

    builtins.open, io.open, os.open = guarded_builtin, guarded_io, guarded_os


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_friction(text: str) -> dict:
    obj = json.loads(text.replace("```json", "").replace("```", "").strip())
    value = obj["combined_friction_coefficient"]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 5:
        raise ValueError("invalid dimensionless combined friction value")
    return {"value": float(value), "visual_basis": str(obj.get("visual_basis", "")),
            "uncertainty": str(obj.get("uncertainty", ""))}


def base_record(row: dict, metric: str, repeat: int, model: str, prompt: str,
                requested_seed: int, provider_seed_supported: bool, temperature: float,
                manifest_sha256: str) -> dict:
    return {"input_id": row["input_id"], "object_id": row["object_id"],
            "sequence_id": row["sequence_id"], "frame_id": int(row["frame_id"]),
            "metric": metric, "condition_id": "siphy_adopted" if metric == "Mass_MnRE" else "ours_full",
            "shared_prediction_source": (["siphy_adopted", "geometric_grounding", "ours_full"]
                                         if metric == "Mass_MnRE" else ["ours_full"]),
            "repeat_id": repeat, "requested_seed": requested_seed,
            "provider_seed_supported": provider_seed_supported,
            "provider_seed_sent": provider_seed_supported,
            "model_version": model, "temperature": temperature,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_manifest_sha256": manifest_sha256, "input_sha256": row["input_sha256"],
            "unit": "kg" if metric == "Mass_MnRE" else "dimensionless_combined_object_table_visual_prior",
            "raw_response": None, "parsed_prediction": None, "parsing_status": "PENDING",
            "failure_reason": None, "inference_time_s": None, "provider_attempts": [],
            "started_at_utc": datetime.now(timezone.utc).isoformat()}


def infer_mass(row: dict, repeat: int, cfg: dict, manifest_sha256: str, backend: SiPhyBackend) -> dict:
    requested_seed = int(cfg["requested_seed_base"]) + repeat
    record = base_record(row, "Mass_MnRE", repeat, backend.model, SYS_MSG, requested_seed,
                         backend._supports_seed, float(cfg["temperature"]), manifest_sha256)
    t0 = time.perf_counter()
    try:
        image = cv2.imread(str(ROOT / row["mass_crop_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("mass crop decode failed")
        points_file = Path(cfg["prep_dir"]) / "inputs/points" / f"{row['input_id']}.npz"
        with np.load(points_file) as points:
            points_mm = points["points_mm"]
        backend.seed = requested_seed
        props = backend.estimate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), row["object_id"],
                                 points_mm=points_mm)
        value = props.get("mass_kg")
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid SiPhy mass: {value}")
        record["parsed_prediction"] = float(value)
        record["parsing_status"] = "PARSED"
        record["details"] = props
    except Exception as exc:
        record["parsing_status"] = "FAILED"
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        attempts = list(getattr(backend, "last_vlm_attempts", []))
        record["provider_attempts"] = attempts
        record["raw_response"] = [a.get("raw_response") for a in attempts]
        record["inference_time_s"] = time.perf_counter() - t0
        backend.last_vlm_attempts = []
    return record


def infer_friction(row: dict, repeat: int, cfg: dict, manifest_sha256: str,
                   backend: SiPhyBackend) -> dict:
    requested_seed = int(cfg["requested_seed_base"]) + repeat
    record = base_record(row, "StaticVisualFriction_MAE", repeat, backend.model,
                         FRICTION_PROMPT, requested_seed, backend._supports_seed,
                         float(cfg["temperature"]), manifest_sha256)
    t0 = time.perf_counter()
    try:
        image = ROOT / row["friction_context_path"]
        if cv2.imread(str(image), cv2.IMREAD_COLOR) is None:
            raise ValueError("friction context decode failed")
        kwargs = {"model": backend.model, "max_tokens": 500, "temperature": float(cfg["temperature"]),
                  "messages": [{"role": "system", "content": FRICTION_PROMPT},
                               {"role": "user", "content": [{"type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + _to_b64_png(image)}}]}]}
        if backend._supports_seed:
            kwargs["seed"] = requested_seed
        attempt = {"attempt": 1, "raw_response": None, "finish_reason": None,
                   "provider_seed_sent": requested_seed if backend._supports_seed else None,
                   "failure_reason": None}
        record["provider_attempts"] = [attempt]
        response = backend.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        raw = choice.message.content or ""
        record["raw_response"] = raw
        attempt["raw_response"] = raw
        attempt["finish_reason"] = getattr(choice, "finish_reason", None)
        if getattr(choice, "finish_reason", None) == "length":
            raise ValueError("friction response truncated")
        parsed = parse_friction(raw)
        record["parsed_prediction"] = parsed["value"]
        record["details"] = {k: v for k, v in parsed.items() if k != "value"}
        record["parsing_status"] = "PARSED"
    except Exception as exc:
        record["parsing_status"] = "FAILED"
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        if record["provider_attempts"]:
            record["provider_attempts"][-1]["failure_reason"] = record["failure_reason"]
    finally:
        record["inference_time_s"] = time.perf_counter() - t0
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--input-id", help="one input for an excluded smoke test")
    ap.add_argument("--repeat-id", type=int, required=True)
    ap.add_argument("--probe-denied-path", type=Path, help="smoke audit: verify runtime GT access denial")
    args = ap.parse_args()
    deny_numeric_gt_reads()
    if args.probe_denied_path:
        try:
            args.probe_denied_path.read_text(encoding="utf-8")
        except PermissionError:
            gt_access_denied = True
        else:
            raise RuntimeError("numeric GT access was not denied")
    else:
        gt_access_denied = None
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg["benchmark"] != "C4 Static Real-5" or int(args.repeat_id) not in range(5):
        raise ValueError("config/repeat mismatch")
    if (cfg["mass_prompt_sha256"] != hashlib.sha256(SYS_MSG.encode()).hexdigest() or
            cfg["friction_prompt_sha256"] != hashlib.sha256(FRICTION_PROMPT.encode()).hexdigest()):
        raise ValueError("PROMPT_HASH_CHANGED_AFTER_INPUT_LOCK")
    manifest_sha256 = sha(args.manifest)
    if manifest_sha256 != cfg["input_manifest_sha256"]:
        raise ValueError("locked input manifest SHA mismatch")
    with args.manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 50 or len({r["input_id"] for r in rows}) != 50:
        raise ValueError("static input manifest must contain 50 unique rows")
    if args.input_id:
        rows = [r for r in rows if r["input_id"] == args.input_id]
        if len(rows) != 1:
            raise ValueError("smoke input ID not found uniquely")
    elif len(rows) != 50:
        raise ValueError("full inference requires all 50 inputs")
    if args.output.exists():
        raise FileExistsError(f"refuse to overwrite predictions: {args.output}")
    # GT-free integrity check; same five image/geometry source hashes used by prepare.py.
    for row in rows:
        iid = row["input_id"]
        points_file = Path(cfg["prep_dir"]) / "inputs/points" / f"{iid}.npz"
        paths = [ROOT / row[k] for k in ("rgb_path", "depth_path", "mass_crop_path",
                                           "friction_context_path")] + [points_file]
        combined = hashlib.sha256("".join(sha(p) for p in paths).encode()).hexdigest()
        if combined != row["input_sha256"]:
            raise ValueError(f"input hash mismatch: {iid}")
    backend = SiPhyBackend(model=cfg["model"], seed=int(cfg["requested_seed_base"]),
                           repo_root=ROOT, temperature=float(cfg["temperature"]))
    records = []
    for row in rows:
        # Metric failures are independent. All same-backend mass conditions share one prediction.
        records.append(infer_mass(row, args.repeat_id, cfg, manifest_sha256, backend))
        records.append(infer_friction(row, args.repeat_id, cfg, manifest_sha256, backend))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"input_count": len(rows), "prediction_count": len(records),
               "actual_provider_calls": sum(len(r["provider_attempts"]) for r in records),
               "parsed": sum(r["parsing_status"] == "PARSED" for r in records),
               "failed": sum(r["parsing_status"] == "FAILED" for r in records),
               "gt_access_denied": gt_access_denied,
               "manifest_sha256": manifest_sha256, "smoke_excluded_from_evaluation": bool(args.input_id)}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                         encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
