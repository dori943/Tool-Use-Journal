from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tuj.m5_motion import object_function_grasp as module


@dataclass
class Attachment:
    object_id: str
    mode: str = "KINEMATIC"


@pytest.fixture
def fake_grasp_lab(monkeypatch):
    package = ModuleType("grasp_lab")
    package.__path__ = []
    catalog_runtime = ModuleType("grasp_lab.catalog_runtime")
    runtime_module = ModuleType("grasp_lab.runtime")

    class CatalogContext:
        from_runtime = None

    catalog_runtime.CatalogContext = CatalogContext

    def save_json(path, payload):
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    runtime_module.save_json = save_json
    monkeypatch.setitem(sys.modules, "grasp_lab", package)
    monkeypatch.setitem(sys.modules, "grasp_lab.catalog_runtime", catalog_runtime)
    monkeypatch.setitem(sys.modules, "grasp_lab.runtime", runtime_module)
    return CatalogContext


@pytest.fixture
def rig(monkeypatch, tmp_path, fake_grasp_lab):
    calls = []
    runtime = SimpleNamespace(
        active_ee="2F",
        attached_object_id=None,
        held_tool_id=None,
        attachment=None,
        env=object(),
    )

    def attach(target, **kwargs):
        calls.append("attach")
        runtime.attached_object_id = target
        runtime.attachment = Attachment(target)

    runtime.attach_object = attach
    runtime.command_gripper = lambda **kwargs: calls.append("command")
    runtime.capture_gripper_hold = lambda: calls.append("hold")
    runtime.mark_attached_object_as_tool = lambda target: setattr(
        runtime, "held_tool_id", target
    )
    context = SimpleNamespace(
        runtime=runtime,
        env=runtime.env,
        data=SimpleNamespace(time=12.0),
        close=lambda: calls.append("close"),
    )

    def factory(*args, **kwargs):
        kwargs["output"].mkdir()
        return context

    fake_grasp_lab.from_runtime = factory
    world = SimpleNamespace(model_dump_json=lambda **kwargs: "{}")
    monkeypatch.setattr(module, "snapshot", lambda runtime: world)

    def function(context, object_id):
        calls.append("function_finished_lift")
        return {"status": "SUCCESS"}

    catalog = SimpleNamespace(
        get_recipe=lambda _: SimpleNamespace(ee_id="2F"),
        get_grasp_function=lambda _: function,
    )
    monkeypatch.setattr(module, "catalog_library", lambda _: catalog)
    return runtime, context, catalog, calls, tmp_path / "run"


def run(rig, **kwargs):
    runtime, _, _, _, output = rig
    return module.execute_object_function_grasp(
        runtime,
        object_id="whisk",
        repository=Path("."),
        output_dir=output,
        **kwargs,
    )


def test_attach_only_after_existing_function_success(rig):
    result = run(rig, resource_kind="tool")
    assert rig[3] == ["function_finished_lift", "command", "attach", "hold", "close"]
    assert rig[0].held_tool_id == "whisk"
    assert result.attachment_reused is False
    record = json.loads((rig[4] / "post_grasp_attachment.json").read_text())
    assert record["runtime_shared"] and record["environment_shared"]


def test_failed_result_does_not_attach(rig):
    rig[2].get_grasp_function = lambda _: lambda *args, **kwargs: {
        "status": "FAILED",
        "failure_reason": "slip",
    }
    with pytest.raises(module.ObjectFunctionGraspError, match="FUNCTION_GRASP_FAILED"):
        run(rig)
    assert rig[0].attachment is None
    assert rig[3] == ["close"]


def test_existing_vacuum_attachment_not_duplicated(rig):
    rig[0].active_ee = "vac"
    rig[2].get_recipe = lambda _: SimpleNamespace(ee_id="vac")

    def vacuum(context, object_id):
        context.runtime.attach_object(object_id)
        return {"status": "SUCCESS"}

    rig[2].get_grasp_function = lambda _: vacuum
    assert run(rig).attachment_reused
    assert rig[3].count("attach") == 1


def test_wrong_ee_rejected_before_function(rig):
    rig[0].active_ee = "3F"
    with pytest.raises(module.ObjectFunctionGraspError, match="requires 2F"):
        run(rig)
    assert rig[3] == []


def test_release_orders_detach_before_open():
    calls = []
    runtime = SimpleNamespace(
        active_ee="2F",
        detach_object=lambda obj: calls.append(("detach", obj)),
        command_gripper=lambda **kwargs: calls.append(("open", kwargs)),
    )
    module.release_object(runtime, "whisk")
    assert calls == [
        ("detach", "whisk"),
        ("open", {"engaged": False, "suction": False}),
    ]


def test_catalog_library_accepts_configured_package_parent(monkeypatch, tmp_path):
    package = tmp_path / "catalog_checkout" / "grasp_lab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "catalog.py").write_text("SOURCE = 'configured'\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv(module.SCRIPTED_GRASP_LIBRARY_ENV, raising=False)
    monkeypatch.delitem(sys.modules, "grasp_lab", raising=False)
    monkeypatch.delitem(sys.modules, "grasp_lab.catalog", raising=False)

    catalog = module.catalog_library(Path("."), library_root=package.parent)

    assert catalog.SOURCE == "configured"
    assert Path(catalog.__file__).resolve().parent == package.resolve()


def test_configured_catalog_path_must_contain_grasp_lab(monkeypatch, tmp_path):
    monkeypatch.setenv(module.SCRIPTED_GRASP_LIBRARY_ENV, str(tmp_path))
    with pytest.raises(
        module.ObjectFunctionGraspError,
        match="must name a grasp_lab package",
    ):
        module.catalog_library(Path("."))
