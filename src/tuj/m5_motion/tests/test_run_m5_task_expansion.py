from __future__ import annotations

import runpy
from pathlib import Path


def _runner(monkeypatch):
    repository = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repository))
    return runpy.run_path(str(repository / "scripts" / "run_m5.py"))


def test_generic_task_expansion_never_injects_scenario_grasp_profile(monkeypatch):
    runner = _runner(monkeypatch)

    for task, environment in runner["TASK_ENVS"].items():
        expanded = runner["_expand_task"]([task])
        assert "--grasp-profile" not in expanded
        assert "--motion-profile" not in expanded
        assert expanded[expanded.index("--environment") + 1] == environment


def test_generic_task_expansion_preserves_explicit_grasp_profile(monkeypatch):
    runner = _runner(monkeypatch)

    for task in runner["TASK_ENVS"]:
        expanded = runner["_expand_task"]([
            task, "--grasp-profile", "explicit-gripper-profile.json",
        ])
        assert expanded.count("--grasp-profile") == 1
        assert expanded[expanded.index("--grasp-profile") + 1] == (
            "explicit-gripper-profile.json"
        )


def test_task_expansion_does_not_inject_task_specific_camera(monkeypatch):
    runner = _runner(monkeypatch)

    for task in runner["TASK_ENVS"]:
        expanded = runner["_expand_task"]([task])
        assert "--camera" not in expanded
        assert "--width" not in expanded
        assert "--height" not in expanded
