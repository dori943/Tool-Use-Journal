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
    "name_only": ["Mass_Acc", "Crit"],
    "affordance_labels": ["Mass_Acc", "Crit"],
    "siphy_adopted": ["Mass_Acc", "Crit"],
    "geometric_grounding": ["Mass_Acc", "Clearance_RelErr", "Feasibility_Acc", "DA", "Crit"],
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


def run(prep: Path, run_dir: Path, repeats: int = 3) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest, inputs = _load_manifest(prep)
    _write_snapshot(run_dir, prep, manifest)
    cfg = provider_config()
    config_snapshot = {"panel": "D_SIM", "prep_dir": str(prep), "manifest_sha256": sha256_file(prep / "sim_manifest.yaml"),
                       "conditions": list(CONDITION_ORDER), "metrics": METRICS, "repeats": repeats,
                       "provider": {k: v for k, v in cfg.items() if k != "configured"},
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
            f"@echo off\nsetlocal\nset PREP={prep}\n.venv\\Scripts\\python.exe experiments\\c4_table3\\sim_d\\run_inference.py --prep %PREP% --output {run_dir} --repeats {repeats}\n",
            encoding="utf-8")
        return failure
    provider = OpenAIJsonProvider(cfg)
    cache = SharedPredictionCache()
    counts = {"provider_calls": 0, "prediction_rows": 0, "failures": 0}
    raw_path, parsed_path, metadata_path, checkpoint_path = [run_dir / name for name in ("raw_predictions.jsonl", "parsed_predictions.jsonl", "request_metadata.jsonl", "checkpoint.jsonl")]
    for repeat_id in range(repeats):
        for condition_id in CONDITION_ORDER:
            adapter = ConditionAdapter(condition_id, provider, shared_cache=cache)
            for item in inputs:
                obs = item["observation"]
                key_prefix = {"condition": condition_id, "repeat_id": repeat_id, "input_id": obs.input_id,
                              "manifest_sha256": config_snapshot["manifest_sha256"], "model_version": cfg["model"]}
                try:
                    rows = adapter.predict_bundle(obs, METRICS[condition_id])
                    # one provider request is represented by every parsed metric,
                    # but counted once through request_id/prompt/input.
                    unique_requests = {(r.get("request_id"), r.get("prompt_sha256"), r.get("input_id")) for r in rows if r.get("independent_model_call")}
                    counts["provider_calls"] += len(unique_requests)
                    for row in rows:
                        row.update(key_prefix, repeat_id=repeat_id, requested_seed=100 + repeat_id,
                                  provider_seed_supported=False, provider_seed_sent=False)
                        _append_jsonl(raw_path, row)
                        _append_jsonl(parsed_path, row)
                        _append_jsonl(checkpoint_path, row)
                        counts["prediction_rows"] += 1
                except Exception as error:  # keep independent inputs running
                    counts["failures"] += 1
                    _append_jsonl(run_dir / "failures.jsonl", {**key_prefix, "failure_reason": _redact(str(error))})
    (run_dir / "run.log").write_text("\n".join(log_lines + ["status=COMPLETE", *(f"{k}={v}" for k, v in counts.items()), "sim_gt_read=0"]) + "\n", encoding="utf-8")
    return {"status": "COMPLETE", **counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = run(args.prep, args.output, args.repeats)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status", "").startswith("BLOCKED"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
