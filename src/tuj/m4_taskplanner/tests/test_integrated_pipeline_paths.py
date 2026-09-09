"""Regression tests for isolated integrated-pipeline artifact paths."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[4]


def _load_script(name: str):
    path = REPOSITORY / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_inputs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "m1.json").write_text(
        json.dumps({"nodes": [{"id": "object_1"}], "edges": []}),
        encoding="utf-8",
    )
    (out / "m2.json").write_text(
        json.dumps(
            {
                "task": "test task",
                "m2_subgoals": [
                    {
                        "subgoal_id": "SG1",
                        "goal": "move object",
                        "object_ids": ["object_1"],
                        "target_ids": [],
                        "tool_candidate_ids": [],
                        "details": [
                            {
                                "detail_id": "SG1_d1",
                                "action_type": "PICK",
                                "pre": [],
                                "establish": [],
                                "destroy": [],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_assemble_gk_writes_to_explicit_output_dir(tmp_path):
    module = _load_script("assemble_gk")
    custom = tmp_path / "isolated-run"
    _write_inputs(custom)

    assert module.main(["c1_1", "--output-dir", str(custom)]) == 0

    assert (custom / "gk_SG1.json").is_file()


def test_assemble_gk_preserves_default_output_layout(monkeypatch, tmp_path):
    module = _load_script("assemble_gk")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    default = tmp_path / "output" / "c1_1"
    _write_inputs(default)

    assert module.main(["c1_1"]) == 0

    assert (default / "gk_SG1.json").is_file()


def test_integrated_stage_forwards_isolated_output_dir(monkeypatch, tmp_path):
    module = _load_script("run")
    captured = {}
    monkeypatch.setattr(module, "load_script", lambda name: object())
    monkeypatch.setattr(
        module,
        "call_main",
        lambda script, argv, label: captured.update(argv=argv, label=label),
    )

    assert module.stage_gk("c1_1", tmp_path) == []

    assert captured["label"] == "assemble_gk"
    assert captured["argv"] == ["c1_1", "--output-dir", str(tmp_path)]
