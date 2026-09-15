from __future__ import annotations

import json
from pathlib import Path

from .generate_sim_gt import generate
from .run_inference import run


def test_provider_missing_blocks_without_gt_or_model_call(tmp_path: Path, monkeypatch):
    prep = tmp_path / "prep"
    out = tmp_path / "run"
    generate(prep)
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "TUJ_LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    result = run(prep, out, repeats=1)
    assert result["status"] == "BLOCKED_PROVIDER_NOT_CONFIGURED"
    assert result["model_calls"] == 0
    assert not (out / "sim_gt.yaml").exists()
    assert (out / "raw_predictions.jsonl").read_text(encoding="utf-8") == ""
    failure = json.loads((out / "failures.json").read_text(encoding="utf-8"))
    assert failure["model_calls"] == 0

