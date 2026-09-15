"""Run D_SIM condition inference with strict input/GT isolation.

The runner uses only ``sim_manifest.yaml`` and observation assets.  It never
opens ``sim_gt.yaml`` and never imports an evaluator.  If no provider
credential is present, it writes a reproducible blocked run without making a
network request.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
try:
    from .adapters import (AdapterError, ConditionAdapter, ProviderResponse,
                           SharedPredictionCache, SiPhyAdapter, load_observation)
except ImportError:  # direct script execution
    from adapters import (AdapterError, ConditionAdapter, ProviderResponse,
                          SharedPredictionCache, SiPhyAdapter, load_observation)


CONDITION_ORDER = ("name_only", "affordance_labels", "siphy_adopted", "geometric_grounding", "ours_full")
METRICS = {
    "name_only": ["Mass_Acc", "Suction_Acc", "Suction_PF", "Feasibility_Acc", "DA", "Crit"],
    "affordance_labels": ["Mass_Acc", "Suction_Acc", "Suction_PF", "Feasibility_Acc", "DA", "Crit"],
    "siphy_adopted": ["Mass_Acc", "Feasibility_Acc", "DA", "Crit"],
    "geometric_grounding": ["Mass_Acc", "Suction_Acc", "Suction_PF", "Clearance_RelErr", "Feasibility_Acc", "DA", "Crit"],
    "ours_full": ["Mass_Acc", "Suction_Acc", "Suction_PF", "Clearance_RelErr", "Feasibility_Acc", "DA", "Crit"],
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _redact(text: str) -> str:
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return text


def provider_config() -> dict[str, Any]:
    provider = os.environ.get("TUJ_LLM_PROVIDER", "").lower().strip()
    if not provider:
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
    has_key = bool((provider == "openai" and os.environ.get("OPENAI_API_KEY")) or
                   (provider == "gemini" and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))))
    return {"provider": provider or None, "configured": has_key,
            "model": os.environ.get("C4_SIM_MODEL", "gemini-3.6-flash" if provider == "gemini" else "gpt-4o-mini"),
            "temperature": float(os.environ.get("C4_SIM_TEMPERATURE", "0.0")),
            "provider_seed_supported": False, "provider_seed_sent": False}


class OpenAIJsonProvider:
    def __init__(self, cfg: dict[str, Any]):
        from openai import OpenAI
        self.model_version = str(cfg["model"])
        self.temperature = float(cfg["temperature"])
        self.provider = cfg["provider"]
        if self.provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self.client = OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/", max_retries=1)
        else:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=1)

    def predict(self, *, prompt: str, image_path: str | None, input_id: str) -> ProviderResponse:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path:
            image_bytes = Path(image_path).read_bytes()
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")}})
        completion = self.client.chat.completions.create(
            model=self.model_version,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "You estimate physical properties from the visible observation only. Return JSON only."},
                      {"role": "user", "content": content}],
        )
        choice = completion.choices[0]
        text = choice.message.content or ""
        try:
            parsed = json.loads(text.replace("```json", "").replace("```", "").strip())
        except json.JSONDecodeError as error:
            raise AdapterError(f"INVALID_FORMAT:{error.msg}") from error
        if not isinstance(parsed, dict):
            raise AdapterError("INVALID_FORMAT")
        return ProviderResponse(parsed=parsed, raw_response=text, request_id=getattr(completion, "id", None))


def _load_manifest(prep: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = prep / "sim_manifest.yaml"
    if not manifest_path.exists():
        raise RuntimeError("LOCKED_MANIFEST_MISSING")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("panel") != "D_SIM" or manifest.get("status") != "READY_GT_ONLY":
        raise RuntimeError("LOCKED_MANIFEST_INVALID")
    if len(manifest.get("samples", [])) != 5:
        raise RuntimeError("LOCKED_MANIFEST_SAMPLE_COUNT")
    rows = []
    for row in manifest["samples"]:
        obs_path = prep / row["observation_path"]
        if not obs_path.exists() or sha256_file(obs_path) != row["observation_sha256"]:
            raise RuntimeError(f"INPUT_HASH_MISMATCH:{row['sample_id']}")
        obs = load_observation(obs_path)
        for key in ("mass_gt_kg", "trial_success", "allowed_ee_ids", "signed_margin_gt_mm"):
            if key in obs.payload:
                raise RuntimeError(f"GT_LEAKAGE_DETECTED:{row['sample_id']}:{key}")
        if obs.image_path and not Path(obs.image_path).exists():
            raise RuntimeError(f"RGB_INPUT_MISSING:{row['sample_id']}")
        rows.append({"manifest": row, "observation": obs})
    return manifest, rows


def _write_snapshot(run_dir: Path, prep: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "manifest_snapshot.yaml").write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for source, target in ((prep / "condition_matrix.csv", run_dir / "condition_matrix_snapshot.csv"),):
        if source.exists():
            target.write_bytes(source.read_bytes())
        else:
            target.write_text("condition_id,condition,status\n", encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _load_reusable_predictions(path: Path, manifest_sha256: str, model_version: str) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    """Load only successful rows compatible with the locked manifest/model.

    Rows with the old sample-level suction schema are deliberately rejected so
    they cannot masquerade as pose-level predictions after the adapter change.
    """
    indexed: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        condition = str(row.get("condition_id", row.get("condition", "")))
        metric = str(row.get("metric", ""))
        try:
            repeat_id = int(row.get("repeat_id", -1))
        except (TypeError, ValueError):
            continue
        key = (condition, repeat_id, str(row.get("input_id", "")), metric)
        if key in indexed:
            raise RuntimeError(f"REUSE_DUPLICATE_KEY:{key}")
        if row.get("manifest_sha256") not in (None, manifest_sha256):
            raise RuntimeError(f"REUSE_MANIFEST_MISMATCH:{key}")
        if row.get("model_version") not in (None, model_version):
            continue
        if row.get("parse_status") != "OK":
            continue
        value = row.get("value")
        valid = True
        if metric in {"Mass_Acc", "Crit", "Clearance_RelErr"}:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif metric in {"Suction_Acc", "Suction_PF"}:
            valid = isinstance(value, dict) and set(value) == {"pose_A", "pose_B"} and all(type(v) is bool for v in value.values())
        elif metric == "Feasibility_Acc":
            valid = isinstance(value, dict) and set(value) == {"2F", "3F", "vac"} and all(type(v) is bool for v in value.values())
        elif metric == "DA":
            valid = value in {"2F", "3F", "vac", "NO_FEASIBLE_EE"}
        if valid:
            indexed[key] = row
    return indexed


def _siphy_rows(obs, repeat_id: int, cfg: dict[str, Any], manifest_sha256: str, backend: Any) -> list[dict[str, Any]]:
    """Run the repository SiPhyBackend once and derive Mass Acc/Crit locally."""
    adapter = SiPhyAdapter(backend=backend)
    result = adapter.predict_mass_with_backend(obs)
    props = result["raw_backend_output"]
    from tuj.m1_scene.siphy_backend import SYS_MSG
    raw = {"backend_output": props, "vlm_attempts": getattr(backend, "last_vlm_attempts", [])}
    prediction_id = f"siphy_adopted:Mass_Acc:{repeat_id}:{obs.input_id}"
    base = {"input_id": obs.input_id, "sample_id": obs.sample_id,
            "object_id": obs.payload.get("object_instance_id", obs.sample_id),
            "condition_id": "siphy_adopted", "value": result["mass_kg"],
            "unit": "kg", "raw_response": raw, "request_id": None,
            "parse_status": "OK", "failure_reason": None,
            "prompt_sha256": hashlib.sha256(SYS_MSG.encode()).hexdigest(),
            "model_version": str(getattr(backend, "model", cfg["model"])),
            "temperature": float(cfg["temperature"]), "shared_prediction": False,
            "shared_prediction_source": None, "independent_model_call": True,
            "prediction_id": prediction_id, "repeat_id": repeat_id,
            "requested_seed": repeat_id, "provider_seed_supported": bool(getattr(backend, "_supports_seed", False)),
            "provider_seed_sent": bool(getattr(backend, "_supports_seed", False)), "manifest_sha256": manifest_sha256}
    crit = {**base, "metric": "Crit", "unit": "kg", "shared_prediction": True,
            "shared_prediction_source": "siphy_adopted_mass", "source_prediction_id": prediction_id,
            "independent_model_call": False, "prediction_id": f"{prediction_id}:Crit"}
    mass = {**base, "metric": "Mass_Acc"}
    return [mass, crit]


def run(prep: Path, run_dir: Path, repeats: int = 3, conditions: tuple[str, ...] = CONDITION_ORDER,
        shared_predictions: Path | None = None, reuse_predictions: Path | None = None) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest, inputs = _load_manifest(prep)
    _write_snapshot(run_dir, prep, manifest)
    cfg = provider_config()
    unknown = sorted(set(conditions) - set(CONDITION_ORDER))
    if unknown:
        raise RuntimeError(f"UNKNOWN_CONDITION:{','.join(unknown)}")
    config_snapshot = {"panel": "D_SIM", "prep_dir": str(prep), "manifest_sha256": sha256_file(prep / "sim_manifest.yaml"),
                       "conditions": list(conditions), "metrics": {k: METRICS[k] for k in conditions}, "repeats": repeats,
                       "provider": {k: v for k, v in cfg.items() if k != "configured"},
                       "shared_predictions": str(shared_predictions) if shared_predictions else None,
                       "reuse_predictions": str(reuse_predictions) if reuse_predictions else None,
                       "gt_access": {"sim_gt_read": False, "evaluator_imported": False}}
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config_snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "manifest_hash.txt").write_text(config_snapshot["manifest_sha256"] + "\n", encoding="utf-8")
    log_lines = [f"prep={prep}", f"manifest_sha256={config_snapshot['manifest_sha256']}", "sim_gt_read=0", "evaluator_imported=0"]
    if not cfg["configured"]:
        failure = {"status": "BLOCKED_PROVIDER_NOT_CONFIGURED", "provider": cfg["provider"],
                   "required_env": ["OPENAI_API_KEY or GEMINI_API_KEY/GOOGLE_API_KEY"],
                   "model_calls": 0, "reason": "No provider credential is available to this process"}
        (run_dir / "failures.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        with (run_dir / "failures.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["status", "provider", "model_calls", "reason"])
            writer.writeheader()
            writer.writerow({key: failure.get(key) for key in writer.fieldnames})
        (run_dir / "raw_predictions.jsonl").write_text("", encoding="utf-8")
        (run_dir / "parsed_predictions.jsonl").write_text("", encoding="utf-8")
        (run_dir / "checkpoint.jsonl").write_text("", encoding="utf-8")
        (run_dir / "request_metadata.jsonl").write_text("", encoding="utf-8")
        log_lines += ["status=BLOCKED_PROVIDER_NOT_CONFIGURED", "model_calls=0"]
        (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        (run_dir / "reproduce_inference.bat").write_text(
            f"@echo off\nsetlocal\nset PREP={prep}\n.venv\\Scripts\\python.exe experiments\\c4_table3\\sim_d\\run_inference.py --prep %PREP% --output {run_dir} --repeats {repeats} --conditions {' '.join(conditions)}\n",
            encoding="utf-8")
        failure["model_calls"] = 0
        return failure
    provider = OpenAIJsonProvider(cfg)
    cached_rows: list[dict[str, Any]] = []
    if shared_predictions:
        cached_rows = [json.loads(line) for line in shared_predictions.read_text(encoding="utf-8").splitlines() if line]
    reuse_index = _load_reusable_predictions(reuse_predictions, config_snapshot["manifest_sha256"], cfg["model"]) if reuse_predictions else {}
    counts = {"provider_calls": 0, "prediction_rows": 0, "failures": 0}
    raw_path, parsed_path, metadata_path, checkpoint_path = [run_dir / name for name in ("raw_predictions.jsonl", "parsed_predictions.jsonl", "request_metadata.jsonl", "checkpoint.jsonl")]
    for repeat_id in range(repeats):
        # A repeat-local cache allows Ours/Geo to reuse exactly the matching
        # SiPhy repeat without duplicating its provider call.
        cache = SharedPredictionCache()
        for cached in cached_rows:
            if (cached.get("condition") or cached.get("condition_id")) == "siphy_adopted" and cached.get("metric") == "Mass_Acc" and int(cached.get("repeat_id", -1)) == repeat_id and cached.get("parse_status") == "OK":
                cache.put(cached)
        for (cid, rid, _iid, metric), cached in reuse_index.items():
            if cid == "siphy_adopted" and metric == "Mass_Acc" and rid == repeat_id:
                cache.put(cached)
        for condition_id in conditions:
            adapter = ConditionAdapter(condition_id, provider, shared_cache=cache)
            for item in inputs:
                obs = item["observation"]
                key_prefix = {"condition": condition_id, "repeat_id": repeat_id, "input_id": obs.input_id,
                              "manifest_sha256": config_snapshot["manifest_sha256"], "model_version": cfg["model"]}
                existing = {metric: row for (cid, rid, iid, metric), row in reuse_index.items()
                            if cid == condition_id and rid == repeat_id and iid == obs.input_id}
                new_rows: list[dict[str, Any]] = []
                try:
                    if condition_id == "siphy_adopted":
                        rows = []
                        if "Mass_Acc" not in existing or "Crit" not in existing:
                            from tuj.m1_scene.siphy_backend import SiPhyBackend
                            backend = SiPhyBackend(model=cfg["model"], temperature=cfg["temperature"])
                            rows.extend(_siphy_rows(obs, repeat_id, cfg, config_snapshot["manifest_sha256"], backend))
                            new_rows.extend(rows)
                        else:
                            rows.extend(existing[m] for m in ("Mass_Acc", "Crit"))
                        cache.put(next(r for r in rows if r.get("metric") == "Mass_Acc"))
                        extra_metrics = [metric for metric in METRICS[condition_id]
                                         if metric not in {"Mass_Acc", "Crit"} and metric not in existing]
                        if extra_metrics:
                            # SiPhy mass remains the repository backend output;
                            # downstream decisions are requested separately and
                            # never silently fabricated from the mass estimate.
                            fresh = ConditionAdapter(condition_id, provider,
                                                     shared_cache=cache).predict_bundle(obs, extra_metrics)
                            rows.extend(fresh)
                            new_rows.extend(fresh)
                        rows.extend(existing[m] for m in METRICS[condition_id]
                                     if m not in {"Mass_Acc", "Crit"} and m in existing)
                    else:
                        missing_metrics = [metric for metric in METRICS[condition_id] if metric not in existing]
                        rows = [existing[metric] for metric in METRICS[condition_id] if metric in existing]
                        if missing_metrics:
                            fresh = adapter.predict_bundle(obs, missing_metrics)
                            rows.extend(fresh)
                            new_rows.extend(fresh)
                    # one provider request is represented by every parsed metric,
                    # but counted once through request_id/prompt/input.
                    unique_requests = {(r.get("request_id"), r.get("prompt_sha256"), r.get("input_id")) for r in new_rows if r.get("independent_model_call")}
                    counts["provider_calls"] += len(unique_requests)
                    for row in rows:
                        row.update(key_prefix, repeat_id=repeat_id, requested_seed=repeat_id,
                                  provider_seed_supported=False, provider_seed_sent=False)
                        _append_jsonl(raw_path, row)
                        _append_jsonl(parsed_path, row)
                        _append_jsonl(checkpoint_path, row)
                        counts["prediction_rows"] += 1
                except Exception as error:  # keep independent inputs running
                    counts["failures"] += 1
                    _append_jsonl(run_dir / "failures.jsonl", {**key_prefix, "failure_reason": _redact(str(error))})
                    detail = str(error).lower()
                    if any(token in detail for token in ("authentication", "api key", "401", "permission", "quota exceeded")):
                        log_lines.append("fatal_provider_error=1")
                        (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
                        return {"status": "BLOCKED_PROVIDER_FAILURE", **counts,
                                "model_calls": counts["provider_calls"], "reason": _redact(str(error))}
    (run_dir / "run.log").write_text("\n".join(log_lines + ["status=COMPLETE", *(f"{k}={v}" for k, v in counts.items()), "sim_gt_read=0"]) + "\n", encoding="utf-8")
    # Keep the same replay/metadata artifacts for successful and blocked runs.
    if not metadata_path.exists():
        metadata_path.write_text("", encoding="utf-8")
    failures_path = run_dir / "failures.csv"
    if not failures_path.exists():
        failure_rows = []
        failure_jsonl = run_dir / "failures.jsonl"
        if failure_jsonl.exists():
            failure_rows = [json.loads(line) for line in failure_jsonl.read_text(encoding="utf-8").splitlines() if line]
        fields = ["condition", "repeat_id", "input_id", "manifest_sha256", "model_version", "failure_reason"]
        with failures_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field) for field in fields} for row in failure_rows)
    (run_dir / "reproduce_inference.bat").write_text(
            f"@echo off\nsetlocal\nset PREP={prep}\n.venv\\Scripts\\python.exe experiments\\c4_table3\\sim_d\\run_inference.py --prep %PREP% --output {run_dir} --repeats {repeats} --conditions {' '.join(conditions)}" + (f" --reuse-predictions {reuse_predictions}" if reuse_predictions else "") + "\n",
        encoding="utf-8")
    return {"status": "COMPLETE", **counts, "model_calls": counts["provider_calls"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--conditions", nargs="+", choices=CONDITION_ORDER, default=list(CONDITION_ORDER))
    parser.add_argument("--shared-predictions", type=Path,
                        help="existing parsed JSONL containing repeat-matched SiPhy Mass_Acc rows")
    parser.add_argument("--reuse-predictions", type=Path,
                        help="existing parsed JSONL; compatible successful rows are reused without provider calls")
    args = parser.parse_args()
    result = run(args.prep, args.output, args.repeats, tuple(args.conditions), args.shared_predictions, args.reuse_predictions)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status", "").startswith("BLOCKED"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
