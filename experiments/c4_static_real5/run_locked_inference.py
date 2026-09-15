"""Run only the 500 locked Static Real-5 inference slots; never load numeric GT.

The existing SiPhy material/density/shell-integral path supplies mass. A single
still-image VLM prompt supplies an object-table visual friction prior. The two
requests are independent, and Geometric/Ours mass rows reference SiPhy results.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tuj.m1_scene.siphy_backend import (SYS_MSG, SiPhyBackend, _parse_response,  # noqa: E402
                                        _to_b64_png)
from infer import FRICTION_PROMPT, deny_numeric_gt_reads  # noqa: E402

EXPECTED_MANIFEST_SHA256 = "89c476f3afd9f00223fff928e6d86c124905ad53d928d70b5b95e8e9f8726dbc"
EXPECTED_CONFIG_SHA256 = "cd511a6ee40eb8005f66c5a1896fdbbca60858cefcc965ab9d6bfb245f151f29"
EXPECTED_MATRIX_SHA256 = "35f27d874267d425f7e55461ff777e104d55f58685740306aa30f91574622c00"
EXPECTED_OBJECTS = {"003_cracker_box", "006_mustard_bottle", "019_pitcher_base",
                    "021_bleach_cleanser", "025_mug"}
METRICS = ("Mass_MnRE", "StaticVisualFriction_MAE")
REPEAT_IDS = (1, 2, 3, 4, 5)
MAX_ATTEMPTS = 3
RETRY_DELAYS_S = (2.0, 4.0)
MAX_WORKERS = 4
_local = threading.local()
_gt_access_attempts = 0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(text: str) -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[REDACTED_CREDENTIAL]")
    return text


def safe_path(raw: str) -> Path:
    p = (ROOT / raw).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("path escapes repository")
    if p.name.lower() in {"evaluator_only_gt.yaml", "c4_real5_gt.json", "gt_snapshot.json"}:
        raise PermissionError("GT_LEAKAGE_DETECTED")
    return p


def preflight(manifest: Path, config: Path) -> tuple[list[dict], dict, dict]:
    # Inspect the actual modules and environment before any provider client exists.
    forbidden_modules = {"evaluate_static", "scoring", "friction_from_rgbd",
                         "experiments.c4_real5.evaluate"}
    for source in (Path(__file__), Path(__file__).with_name("infer.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                     for a in n.names]
        if any(m in forbidden_modules for m in imported):
            raise PermissionError("GT_LEAKAGE_DETECTED: evaluator/trajectory import")
    if re.search(r"\b\d+\.\d+\b", SYS_MSG + FRICTION_PROMPT):
        raise PermissionError("GT_LEAKAGE_DETECTED: numeric few-shot/example in prompt")
    if any(any(name in str(value).lower() for name in
               ("evaluator_only_gt.yaml", "gt_snapshot", "c4_real5_gt.json"))
           for value in os.environ.values()):
        raise PermissionError("GT_LEAKAGE_DETECTED: GT path in environment")
    mhash, chash = sha(manifest), sha(config)
    if mhash != EXPECTED_MANIFEST_SHA256 or chash != EXPECTED_CONFIG_SHA256:
        raise ValueError("LOCKED_INPUT_MISMATCH: manifest/config SHA-256 differs from prep report")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    if (cfg["benchmark"] != "C4 Static Real-5" or cfg["input_manifest_sha256"] != mhash
            or cfg["model"] != "gemini-3.6-flash" or float(cfg["temperature"]) != 0.0
            or cfg["mass_prompt_sha256"] != hashlib.sha256(SYS_MSG.encode()).hexdigest()
            or cfg["friction_prompt_sha256"] != hashlib.sha256(FRICTION_PROMPT.encode()).hexdigest()):
        raise ValueError("LOCKED_INPUT_MISMATCH: model, prompt, temperature, or manifest changed")
    matrix = manifest.parent / "condition_matrix.csv"
    if sha(matrix) != EXPECTED_MATRIX_SHA256:
        raise ValueError("LOCKED_INPUT_MISMATCH: condition matrix changed")
    with matrix.open(encoding="utf-8", newline="") as f:
        condition_rows = {r["condition_id"]: r for r in csv.DictReader(f)}
    if (set(condition_rows) != {"name_only", "affordance_labels", "siphy_adopted",
                                "geometric_grounding", "ours_full", "gt_numerics"}
            or condition_rows["siphy_adopted"]["Mass_MnRE_adapter"] != "READY"
            or condition_rows["ours_full"]["StaticVisualFriction_MAE_adapter"] != "READY"
            or condition_rows["geometric_grounding"]["Mass_MnRE_adapter"] != "SHARED_BACKEND"
            or condition_rows["ours_full"]["Mass_MnRE_adapter"] != "SHARED_BACKEND"):
        raise ValueError("LOCKED_INPUT_MISMATCH: adapter matrix mismatch")
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["input_id"] for r in rows]
    hashes = [r["input_sha256"] for r in rows]
    objects = Counter(r["object_id"] for r in rows)
    sequences = defaultdict(list)
    for r in rows:
        sequences[r["sequence_id"]].append(r)
    if (len(rows) != 50 or set(objects) != EXPECTED_OBJECTS or set(objects.values()) != {10}
            or len(sequences) != 25 or any(len(s) != 2 for s in sequences.values())
            or len(set(ids)) != 50 or len(set(hashes)) != 50 or
            any("tuna" in r["object_id"].lower() for r in rows) or
            any("smoke_test" in str(v).lower() for r in rows for v in r.values()) or
            any("gt" in k.lower() or "prediction" in k.lower() for k in rows[0])):
        raise ValueError("LOCKED_INPUT_MISMATCH: 5x5x2, IDs, hashes, or GT/smoke isolation invalid")
    if any(abs(int(s[0]["frame_id"]) - int(s[1]["frame_id"])) < 20 for s in sequences.values()):
        raise ValueError("LOCKED_INPUT_MISMATCH: frames not temporally separated")
    if (sum(r["mass_qc_status"] == "WARNING" for r in rows) != 13 or
            sum(r["friction_qc_status"] == "WARNING" for r in rows) != 2 or
            any(r["mass_qc_status"] not in ("APPROVED", "WARNING") or
                r["friction_qc_status"] not in ("APPROVED", "WARNING") for r in rows)):
        raise ValueError("LOCKED_INPUT_MISMATCH: warning/exclusion state changed")
    by_seq = {sid.split("/")[-1]: rr for sid, rr in sequences.items()}
    if ([int(r["frame_id"]) for r in by_seq["000017"]] != [10, 35] or
            not all(r["mass_qc_status"] == "WARNING" for r in by_seq["000015"]) or
            len(by_seq["000009"]) != 2):
        raise ValueError("LOCKED_INPUT_MISMATCH: 000017/000015/000009 selection changed")
    # Verify every original/crop/context/point-file digest, not merely existence.
    for r in rows:
        points = manifest.parent / "inputs/points" / f"{r['input_id']}.npz"
        paths = [safe_path(r[k]) for k in ("rgb_path", "depth_path", "mass_crop_path",
                                            "friction_context_path")] + [points]
        if any(not p.is_file() for p in paths):
            raise ValueError(f"LOCKED_INPUT_MISMATCH: missing file for {r['input_id']}")
        if any(cv2.imread(str(p), cv2.IMREAD_UNCHANGED) is None for p in paths[:4]):
            raise ValueError(f"LOCKED_INPUT_MISMATCH: image decode failure for {r['input_id']}")
        combined = hashlib.sha256("".join(sha(p) for p in paths).encode()).hexdigest()
        if combined != r["input_sha256"]:
            raise ValueError(f"LOCKED_INPUT_MISMATCH: content changed for {r['input_id']}")
    # Smoke *input* may be reused, but smoke predictions are never read/imported.
    return rows, cfg, {"manifest_sha256": mhash, "config_sha256": chash,
                       "matrix_sha256": sha(matrix), "mass_warnings": 13,
                       "friction_warnings": 2, "input_count": 50, "sequence_count": 25,
                       "smoke_predictions_imported": 0, "gt_access_count": 0}


def worker_backend(cfg: dict) -> SiPhyBackend:
    if not hasattr(_local, "backend"):
        _local.backend = SiPhyBackend(model=cfg["model"], repo_root=ROOT,
                                      temperature=float(cfg["temperature"]), seed=0)
        # The SDK otherwise adds hidden retries on top of the locked policy.
        _local.backend.client = _local.backend.client.with_options(max_retries=0, timeout=90.0)
    return _local.backend


class InferenceError(Exception):
    def __init__(self, status: str, detail: str, *, retryable: bool = False, fatal: bool = False):
        super().__init__(detail)
        self.status, self.retryable, self.fatal = status, retryable, fatal


def provider_error(exc: Exception) -> InferenceError:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    message = redact(str(exc))
    if status in (401, 403, 404):
        return InferenceError("MODEL_FAILURE", f"{name}: HTTP {status}: {message}", fatal=True)
    if status == 429 or (isinstance(status, int) and 500 <= status <= 599):
        return InferenceError("MODEL_FAILURE", f"{name}: HTTP {status}: {message}", retryable=True)
    if name in {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout",
                "TimeoutException", "ConnectionError"}:
        return InferenceError("MODEL_FAILURE", f"{name}: {message}", retryable=True)
    return InferenceError("MODEL_FAILURE", f"{name}: {message}", fatal=True)


def parse_friction_response(raw: str) -> dict:
    try:
        obj = json.loads(raw.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError as exc:
        raise InferenceError("INVALID_FORMAT", str(exc), retryable=True) from exc
    if "combined_friction_coefficient" not in obj:
        raise InferenceError("MISSING_VALUE", "combined_friction_coefficient absent")
    unit = obj.get("unit")
    if unit is not None and str(unit).lower() not in ("dimensionless", "coefficient", "unitless"):
        raise InferenceError("UNIT_AMBIGUOUS", f"unexpected friction unit: {unit}")
    value = obj["combined_friction_coefficient"]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 5:
        raise InferenceError("OUT_OF_RANGE", f"friction value invalid: {value}")
    confidence = obj.get("friction_confidence", obj.get("confidence"))
    if confidence is not None and (type(confidence) not in (int, float) or
                                   not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise InferenceError("OUT_OF_RANGE", "friction confidence outside [0,1]")
    return {"value": float(value), "confidence": confidence,
            "visual_basis": str(obj.get("visual_basis", "")),
            "visible_object_material": obj.get("visible_object_material"),
            "visible_table_material": obj.get("visible_table_material"),
            "reasoning_summary": obj.get("reasoning_summary", obj.get("visual_basis", ""))}


def call_vlm(backend: SiPhyBackend, prompt: str, image, parse, attempts: list,
             max_tokens: int) -> tuple[object | None, InferenceError | None]:
    encoded = _to_b64_png(image)
    kwargs = {"model": backend.model, "max_tokens": max_tokens,
              "temperature": backend.temperature,
              "messages": [{"role": "system", "content": prompt},
                           {"role": "user", "content": [{"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + encoded}}]}]}
    for attempt_no in range(1, MAX_ATTEMPTS + 1):
        attempt = {"attempt": attempt_no, "started_at_utc": now(), "finished_at_utc": None,
                   "api_request_id": None, "response_model": None, "raw_response": None,
                   "finish_reason": None, "error_status": None, "failure_reason": None,
                   "provider_seed_sent": False}
        attempts.append(attempt)
        try:
            response = backend.client.chat.completions.create(**kwargs)
            attempt["api_request_id"] = getattr(response, "id", None)
            attempt["response_model"] = getattr(response, "model", None)
            choice = response.choices[0]
            raw = choice.message.content or ""
            attempt["raw_response"] = raw
            attempt["finish_reason"] = getattr(choice, "finish_reason", None)
            if attempt["finish_reason"] == "length":
                raise InferenceError("INVALID_FORMAT", "model response truncated", retryable=True)
            try:
                parsed = parse(raw)
            except json.JSONDecodeError as exc:
                raise InferenceError("INVALID_FORMAT", str(exc), retryable=True) from exc
            except KeyError as exc:
                raise InferenceError("MISSING_VALUE", f"missing JSON key: {exc}") from exc
            except ValueError as exc:
                raise InferenceError("OUT_OF_RANGE", str(exc)) from exc
            attempt["finished_at_utc"] = now()
            return parsed, None
        except InferenceError as exc:
            failure = exc
        except Exception as exc:  # provider transport/rate/server failures only retryable
            failure = provider_error(exc)
        attempt["error_status"] = failure.status
        attempt["failure_reason"] = str(failure)
        attempt["finished_at_utc"] = now()
        if not failure.retryable or attempt_no == MAX_ATTEMPTS:
            return None, failure
        time.sleep(RETRY_DELAYS_S[attempt_no - 1])
    raise AssertionError("retry loop escaped")


def prediction_key(condition: str, metric: str, repeat: int, input_id: str,
                   manifest_hash: str, prompt_hash: str, model: str) -> tuple:
    return (condition, metric, repeat, input_id, manifest_hash, prompt_hash, model)


def run_one(row: dict, repeat: int, metric: str, cfg: dict, identity: dict,
            prep_dir: Path) -> dict:
    backend = worker_backend(cfg)
    condition = "siphy_adopted" if metric == "Mass_MnRE" else "ours_full"
    prompt = SYS_MSG if metric == "Mass_MnRE" else FRICTION_PROMPT
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    key = prediction_key(condition, metric, repeat, row["input_id"],
                         identity["manifest_sha256"], prompt_hash, backend.model)
    prediction_id = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
    record = {"prediction_id": prediction_id, "condition": condition, "metric": metric,
              "repeat_id": repeat, "requested_seed": repeat - 1,
              "provider_seed_supported": False, "provider_seed_sent": False,
              "input_id": row["input_id"], "object_id": row["object_id"],
              "sequence_id": row["sequence_id"], "frame_id": int(row["frame_id"]),
              "input_sha256": row["input_sha256"], "manifest_hash": identity["manifest_sha256"],
              "config_hash": identity["config_sha256"], "prompt_hash": prompt_hash,
              "model_version": backend.model, "temperature": float(cfg["temperature"]),
              "input_modality": "static_rgbd" if metric == "Mass_MnRE" else "static_rgb",
              "estimation_method": "siphy_visual_material_rgbd_shell_integral" if metric == "Mass_MnRE"
                                   else "visual_property_estimation",
              "target": "mass" if metric == "Mass_MnRE" else "combined_object_table_friction",
              "unit": "kg" if metric == "Mass_MnRE" else "dimensionless",
              "api_request_id": None, "raw_response": None, "parsed_prediction": None,
              "parsing_status": "PENDING", "retry_count": 0, "failure_reason": None,
              "provider_attempts": [], "timestamp_utc": now(), "inference_time_s": None,
              "shared_prediction": False, "independent_model_call": True,
              "mass_confidence": None, "friction_confidence": None,
              "visible_object_material": None, "visible_table_material": None,
              "reasoning_summary": None}
    t0 = time.perf_counter()
    attempts = record["provider_attempts"]
    try:
        if metric == "Mass_MnRE":
            path = safe_path(row["mass_crop_path"])
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise InferenceError("MODEL_FAILURE", "mass crop decode failure")
            points_file = prep_dir / "inputs/points" / f"{row['input_id']}.npz"
            with np.load(points_file) as loaded:
                points_mm = loaded["points_mm"]
            parsed, failure = call_vlm(backend, SYS_MSG, cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                       _parse_response, attempts, backend._max_tokens)
            if failure:
                raise failure
            # Reuse the repository SiPhy mass computation without a second API call.
            backend._propose = lambda _: parsed
            props = backend.estimate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), row["object_id"],
                                     points_mm=points_mm)
            mass = props.get("mass_kg")
            if type(mass) not in (float, int) or not math.isfinite(mass) or mass <= 0:
                raise InferenceError("OUT_OF_RANGE", f"invalid mass_kg: {mass}")
            confidence = props.get("confidence")
            if confidence is not None and not 0 <= confidence <= 1:
                raise InferenceError("OUT_OF_RANGE", "mass confidence outside [0,1]")
            record["parsed_prediction"] = float(mass)
            record["mass_confidence"] = confidence
            record["visible_object_material"] = props.get("material")
            record["reasoning_summary"] = props.get("caption")
            record["details"] = props
        else:
            image = safe_path(row["friction_context_path"])
            parsed, failure = call_vlm(backend, FRICTION_PROMPT, image,
                                       parse_friction_response, attempts, 500)
            if failure:
                raise failure
            record["parsed_prediction"] = parsed["value"]
            record["friction_confidence"] = parsed["confidence"]
            record["visible_object_material"] = parsed["visible_object_material"]
            record["visible_table_material"] = parsed["visible_table_material"]
            record["reasoning_summary"] = parsed["reasoning_summary"]
            record["details"] = parsed
        record["parsing_status"] = "PARSED"
    except InferenceError as exc:
        record["parsing_status"] = exc.status
        record["failure_reason"] = str(exc)
        record["fatal_provider_error"] = exc.fatal
    except Exception as exc:
        record["parsing_status"] = "MODEL_FAILURE"
        record["failure_reason"] = redact(f"{type(exc).__name__}: {exc}")
    finally:
        record["retry_count"] = max(0, len(attempts) - 1)
        record["raw_response"] = [a["raw_response"] for a in attempts]
        record["api_request_id"] = next((a["api_request_id"] for a in reversed(attempts)
                                         if a["api_request_id"]), None)
        record["response_model_version"] = next((a["response_model"] for a in reversed(attempts)
                                                 if a["response_model"]), None)
        record["inference_time_s"] = time.perf_counter() - t0
        record["finished_at_utc"] = now()
    return record


def log(run: Path, message: str) -> None:
    with (run / "run.log").open("a", encoding="utf-8") as f:
        f.write(f"{now()} {message}\n")
    print(message, flush=True)


def identity_for(cfg: dict, hashes: dict) -> dict:
    return {**hashes, "model_version": cfg["model"], "temperature": cfg["temperature"],
            "prompt_hashes": {"Mass_MnRE": cfg["mass_prompt_sha256"],
                              "StaticVisualFriction_MAE": cfg["friction_prompt_sha256"]},
            "repeat_ids": list(REPEAT_IDS), "requested_seeds": list(range(5)),
            "retry_policy": {"max_attempts": MAX_ATTEMPTS,
                             "delays_s": list(RETRY_DELAYS_S),
                             "retry_causes": ["transport", "rate_limit", "server_5xx", "invalid_json"]}}


def prepare_run(run: Path, manifest: Path, config: Path, identity: dict) -> None:
    run.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config, run / "config_snapshot.yaml")
    shutil.copyfile(manifest, run / "manifest_snapshot.csv")
    shutil.copyfile(manifest.parent / "condition_matrix.csv", run / "condition_matrix_snapshot.csv")
    (run / "manifest_hash.txt").write_text(identity["manifest_sha256"] + "\n", encoding="utf-8")
    (run / "prompt_hashes.json").write_text(json.dumps(identity["prompt_hashes"], indent=2) + "\n",
                                            encoding="utf-8")
    (run / "run_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    for name in ("raw_predictions.jsonl", "request_metadata.jsonl", "checkpoint.jsonl"):
        (run / name).touch(exist_ok=False)
    (run / "failures.csv").write_text(
        "type,prediction_id,condition,metric,repeat_id,input_id,reason\n", encoding="utf-8")
    (run / "parsed_predictions.csv").write_text(
        "prediction_id,condition,metric,repeat_id,requested_seed,input_id,object_id,sequence_id,frame_id,parsed_prediction,unit,parsing_status,failure_reason\n",
        encoding="utf-8")
    (run / "shared_predictions.csv").write_text(
        "condition,metric,repeat_id,input_id,shared_prediction,shared_prediction_source,source_prediction_id,independent_model_call,source_parsing_status\n",
        encoding="utf-8")
    (run / "run.log").touch(exist_ok=False)
    base = (f"\"{ROOT / '.venv/Scripts/python.exe'}\" "
            f"\"{Path(__file__).resolve()}\" --manifest \"{manifest.resolve()}\" "
            f"--config \"{config.resolve()}\"")
    bat = (f"@echo off\r\ncd /d \"{ROOT}\"\r\n"
           "if /I \"%~1\"==\"/resume\" (\r\n"
           f"  {base} --resume \"{run}\"\r\n"
           "  if errorlevel 1 exit /b 1\r\n"
           "  exit /b 0\r\n"
           ")\r\n"
           "if not defined GEMINI_API_KEY (echo Set GEMINI_API_KEY in this shell first.& exit /b 2)\r\n"
           f"{base}\r\n")
    (run / "reproduce_inference.bat").write_text(bat, encoding="utf-8")
    log(run, "PREFLIGHT_PASS; 50 static inputs; no GT/evaluator/old prediction loaded")
    log(run, f"manifest_sha256={identity['manifest_sha256']} config_sha256={identity['config_sha256']}")
    log(run, f"model={identity['model_version']} temperature={identity['temperature']} "
             "requested_seeds=0,1,2,3,4 provider_seed_sent=false")


def key_from_record(r: dict) -> tuple:
    return prediction_key(r["condition"], r["metric"], int(r["repeat_id"]), r["input_id"],
                          r["manifest_hash"], r["prompt_hash"], r["model_version"])


def load_resume(run: Path, identity: dict, expected_keys: set[tuple]) -> dict[tuple, dict]:
    stored = json.loads((run / "run_identity.json").read_text(encoding="utf-8"))
    if (stored != identity or sha(run / "manifest_snapshot.csv") != identity["manifest_sha256"] or
            sha(run / "config_snapshot.yaml") != identity["config_sha256"]):
        raise ValueError("RESUME_CONFIG_MISMATCH")
    records = {}
    with (run / "raw_predictions.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = key_from_record(row)
            if key in records:
                log(run, f"DUPLICATE_PREDICTION_KEY {key}")
                with (run / "failures.csv").open("a", encoding="utf-8", newline="") as o:
                    writer = csv.DictWriter(o, fieldnames=["type", "prediction_id", "condition",
                                                      "metric", "repeat_id", "input_id", "reason"])
                    writer.writerow({"type": "DUPLICATE_PREDICTION_KEY",
                                     "prediction_id": row["prediction_id"], "condition": row["condition"],
                                     "metric": row["metric"], "repeat_id": row["repeat_id"],
                                     "input_id": row["input_id"], "reason": "duplicate checkpoint key"})
                raise ValueError("DUPLICATE_PREDICTION_KEY")
            if key not in expected_keys:
                raise ValueError("RESUME_CONFIG_MISMATCH: unknown prediction key")
            records[key] = row
    checkpoint_keys = set()
    with (run / "checkpoint.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            key = tuple(c["key"])
            if key in checkpoint_keys or key not in records:
                raise ValueError("RESUME_CHECKPOINT_CORRUPT")
            if c["record_sha256"] != hashlib.sha256(json.dumps(records[key], sort_keys=True).encode()).hexdigest():
                raise ValueError("RESUME_CHECKPOINT_CORRUPT")
            checkpoint_keys.add(key)
    # Crash between raw append and checkpoint append: preserve the raw prediction.
    missing = set(records) - checkpoint_keys
    if missing:
        with (run / "checkpoint.jsonl").open("a", encoding="utf-8") as f:
            for key in missing:
                f.write(json.dumps({"key": key, "record_sha256": hashlib.sha256(
                    json.dumps(records[key], sort_keys=True).encode()).hexdigest(),
                    "recovered_from_raw": True}) + "\n")
        log(run, f"RECOVERED_RAW_WITHOUT_CHECKPOINT count={len(missing)}")
    log(run, f"RESUME_VERIFIED completed_keys={len(records)}; no completed key will be recalled")
    return records


def append_record(run: Path, row: dict) -> None:
    encoded = json.dumps(row, ensure_ascii=False)
    with (run / "raw_predictions.jsonl").open("a", encoding="utf-8") as f:
        f.write(encoded + "\n"); f.flush(); os.fsync(f.fileno())
    with (run / "request_metadata.jsonl").open("a", encoding="utf-8") as f:
        for metadata in metadata_rows(row):
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    c = {"key": key_from_record(row),
         "record_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
         "parsing_status": row["parsing_status"], "timestamp_utc": now()}
    with (run / "checkpoint.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(c, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())


def metadata_rows(row: dict) -> list[dict]:
    base = {k: row[k] for k in ("prediction_id", "condition", "metric", "repeat_id",
                                 "input_id", "model_version", "prompt_hash", "manifest_hash",
                                 "requested_seed", "provider_seed_supported", "provider_seed_sent",
                                 "parsing_status")}
    return [{**base, **{k: a.get(k) for k in ("attempt", "started_at_utc", "finished_at_utc",
                                               "api_request_id", "response_model", "error_status",
                                               "failure_reason")}}
            for a in row["provider_attempts"]]


def finalize(run: Path, records: dict[tuple, dict], expected: set[tuple]) -> dict:
    failures = []
    parsed_rows, shared = [], []
    for row in sorted(records.values(), key=lambda r: (r["repeat_id"], r["input_id"], r["metric"])):
        parsed_rows.append({"prediction_id": row["prediction_id"], "condition": row["condition"],
                            "metric": row["metric"], "repeat_id": row["repeat_id"],
                            "requested_seed": row["requested_seed"], "input_id": row["input_id"],
                            "object_id": row["object_id"], "sequence_id": row["sequence_id"],
                            "frame_id": row["frame_id"], "parsed_prediction": row["parsed_prediction"],
                            "unit": row["unit"], "parsing_status": row["parsing_status"],
                            "failure_reason": row["failure_reason"]})
        if row["parsing_status"] != "PARSED":
            failures.append({"type": row["parsing_status"], "prediction_id": row["prediction_id"],
                             "condition": row["condition"], "metric": row["metric"],
                             "repeat_id": row["repeat_id"], "input_id": row["input_id"],
                             "reason": row["failure_reason"]})
        if row["metric"] == "Mass_MnRE":
            for condition in ("geometric_grounding", "ours_full"):
                shared.append({"condition": condition, "metric": "Mass_MnRE",
                               "repeat_id": row["repeat_id"], "input_id": row["input_id"],
                               "shared_prediction": True, "shared_prediction_source": "SiPhy",
                               "source_prediction_id": row["prediction_id"],
                               "independent_model_call": False,
                               "source_parsing_status": row["parsing_status"]})
    missing = expected - set(records)
    for key in sorted(missing):
        failures.append({"type": "MISSING_PREDICTION", "prediction_id": "",
                         "condition": key[0], "metric": key[1], "repeat_id": key[2],
                         "input_id": key[3], "reason": "request not completed"})
    # Regenerate the request-level index from immutable raw records. This also
    # reconciles a crash between raw/checkpoint and metadata writes on resume.
    with (run / "request_metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(records.values(), key=lambda r: (r["repeat_id"], r["input_id"], r["metric"])):
            for metadata in metadata_rows(row):
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    with (run / "parsed_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(parsed_rows[0]) if parsed_rows else
                                ["prediction_id", "condition", "metric", "repeat_id", "input_id",
                                 "parsed_prediction", "parsing_status"])
        writer.writeheader(); writer.writerows(parsed_rows)
    with (run / "shared_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["condition", "metric", "repeat_id", "input_id", "shared_prediction",
                  "shared_prediction_source", "source_prediction_id", "independent_model_call",
                  "source_parsing_status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(shared)
    with (run / "failures.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["type", "prediction_id", "condition", "metric", "repeat_id", "input_id", "reason"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(failures)
    by_repeat = []
    for repeat in REPEAT_IDS:
        sub = [r for r in records.values() if r["repeat_id"] == repeat]
        by_repeat.append({"repeat_id": repeat, "mass_parsed": sum(r["metric"] == "Mass_MnRE" and
                                                        r["parsing_status"] == "PARSED" for r in sub),
                          "friction_parsed": sum(r["metric"] == "StaticVisualFriction_MAE" and
                                                            r["parsing_status"] == "PARSED" for r in sub),
                          "failed": sum(r["parsing_status"] != "PARSED" for r in sub),
                          "retries": sum(r["retry_count"] for r in sub),
                          "slots_recorded": len(sub)})
    summary = {"status": "INFERENCE_COMPLETE" if not missing else "INFERENCE_PARTIAL",
               "prediction_slots_expected": 500, "prediction_slots_recorded": len(records),
               "mass_unique_predictions": sum(r["metric"] == "Mass_MnRE" for r in records.values()),
               "friction_unique_predictions": sum(r["metric"] == "StaticVisualFriction_MAE"
                                                   for r in records.values()),
               "provider_request_attempts": sum(len(r["provider_attempts"]) for r in records.values()),
               "shared_mass_aliases": len(shared), "missing_count": len(missing),
               "failed_count": sum(r["parsing_status"] != "PARSED" for r in records.values()),
               "gt_file_access_count": 0, "evaluator_executed": False,
               "smoke_predictions_imported": 0, "trajectory_predictions_imported": 0,
               "by_repeat": by_repeat}
    (run / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(run, f"FINAL status={summary['status']} slots={len(records)}/500 "
             f"success={len(records)-summary['failed_count']} failed={summary['failed_count']} "
             f"missing={len(missing)} provider_attempts={summary['provider_request_attempts']}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be 1..8")
    deny_numeric_gt_reads()
    try:
        rows, cfg, hashes = preflight(args.manifest.resolve(), args.config.resolve())
    except PermissionError as exc:
        print(f"GT_LEAKAGE_DETECTED: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"LOCKED_INPUT_MISMATCH: {exc}", file=sys.stderr)
        return 2
    identity = identity_for(cfg, hashes)
    expected_jobs = []
    for repeat in REPEAT_IDS:
        for row in rows:
            for metric in METRICS:
                prompt_hash = cfg["mass_prompt_sha256"] if metric == "Mass_MnRE" else cfg["friction_prompt_sha256"]
                condition = "siphy_adopted" if metric == "Mass_MnRE" else "ours_full"
                expected_jobs.append((prediction_key(condition, metric, repeat, row["input_id"],
                                                     hashes["manifest_sha256"], prompt_hash, cfg["model"]),
                                      row, repeat, metric))
    expected_keys = {j[0] for j in expected_jobs}
    if len(expected_keys) != 500:
        raise ValueError("duplicate expected prediction keys")
    if args.resume:
        run = args.resume.resolve()
        try:
            records = load_resume(run, identity, expected_keys)
        except Exception as exc:
            print(f"RESUME_CONFIG_MISMATCH: {exc}", file=sys.stderr)
            return 4
    else:
        run = ROOT / "output/c4_table3" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") +
                                            "_static_real5_inference")
        prepare_run(run, args.manifest.resolve(), args.config.resolve(), identity)
        records = {}
    pending_jobs = [j for j in expected_jobs if j[0] not in records]
    log(run, f"SCHEDULE pending={len(pending_jobs)} completed={len(records)} workers={args.workers}")
    fatal = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        cursor = 0
        while cursor < min(args.workers, len(pending_jobs)):
            key, row, repeat, metric = pending_jobs[cursor]
            futures[pool.submit(run_one, row, repeat, metric, cfg, hashes, args.manifest.parent)] = key
            cursor += 1
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                key = futures.pop(future)
                row = future.result()
                if key in records:
                    log(run, f"DUPLICATE_PREDICTION_KEY {key}")
                    fatal = True
                    continue
                append_record(run, row)
                records[key] = row
                if row.get("fatal_provider_error"):
                    fatal = True
                    log(run, f"FATAL_PROVIDER_ERROR {row['metric']} {row['input_id']}: "
                             f"{row['failure_reason'][:300]}")
                if len(records) % 25 == 0 or row["parsing_status"] != "PARSED":
                    log(run, f"PROGRESS slots={len(records)}/500 repeat={row['repeat_id']} "
                             f"metric={row['metric']} status={row['parsing_status']} "
                             f"retries={row['retry_count']}")
                if not fatal and cursor < len(pending_jobs):
                    next_key, next_row, next_repeat, next_metric = pending_jobs[cursor]
                    futures[pool.submit(run_one, next_row, next_repeat, next_metric,
                                        cfg, hashes, args.manifest.parent)] = next_key
                    cursor += 1
    summary = finalize(run, records, expected_keys)
    print(f"RUN_DIR={run}", flush=True)
    return 0 if summary["status"] == "INFERENCE_COMPLETE" and not fatal else 5


if __name__ == "__main__":
    raise SystemExit(main())
