from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from tuj.m4_motion.ee_exchange import EEExchangeTemplateGenerator
from tuj.m4_motion.workcell_models import EEWorkcellCollisionModelCompiler


@pytest.fixture(scope="module")
def rack_env_and_compiler():
    pytest.importorskip("robosuite")
    from experiment.ee_rack_layout_demo import EERackLayoutEnv

    env = EERackLayoutEnv(has_renderer=False, has_offscreen_renderer=False)
    env.reset()
    compiler = EEWorkcellCollisionModelCompiler.from_ee_rack_env(env)
    try:
        yield env, compiler
    finally:
        env.close()


def _is_descendant(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
    current = body_id
    for _ in range(model.nbody + 1):
        if current == root_id:
            return True
        if current == 0:
            return False
        current = int(model.body_parentid[current])
    return False


def test_compiler_builds_bare_and_all_hard_attached_variants(
    rack_env_and_compiler,
) -> None:
    _, compiler = rack_env_and_compiler
    bare = compiler.compile(None)

    assert bare.model.nq == 6
    assert bare.active_ee is None
    for ee in compiler.ee_names:
        attached = compiler.compile(ee)
        robot_root_id = attached.model.body(attached.robot_root_body_name).id
        qc_id = attached.model.body("gripper0_right_qc_master_base").id
        active_id = attached.model.body(dict(attached.entity_body_names)[ee]).id

        assert attached.model.nq == 6
        assert attached.active_ee == ee
        assert int(attached.model.body_parentid[active_id]) == qc_id
        assert _is_descendant(attached.model, active_id, robot_root_id)
        for inactive_ee in set(compiler.ee_names) - {ee}:
            inactive_id = attached.model.body(
                dict(attached.entity_body_names)[inactive_ee]
            ).id
            assert int(attached.model.body_parentid[inactive_id]) == 0


def test_attached_root_uses_qc_mjcf_coupling_position(
    rack_env_and_compiler,
) -> None:
    _, compiler = rack_env_and_compiler
    attached = compiler.compile("vacuum")
    model = attached.model
    data = mujoco.MjData(model)
    data.qpos[:] = attached.baseline_qpos
    mujoco.mj_forward(model, data)
    qc_id = model.body("gripper0_right_qc_master_base").id
    active_id = model.body(dict(attached.entity_body_names)["vacuum"]).id
    qc_rotation = np.asarray(data.xmat[qc_id]).reshape(3, 3)
    qc_asset = ET.parse("experiment/assets/quick_changer_master.xml").getroot()
    coupling_body = qc_asset.find(".//body[@name='eef']")
    assert coupling_body is not None
    coupling_position = np.fromstring(coupling_body.get("pos", ""), sep=" ")
    expected = np.asarray(data.xpos[qc_id]) + qc_rotation @ np.array(
        coupling_position
    )

    assert data.xpos[active_id] == pytest.approx(expected, abs=1e-9)
    assert np.asarray(data.xmat[active_id]).reshape(3, 3) == pytest.approx(
        qc_rotation,
        abs=1e-9,
    )


def test_compilation_is_deterministic(rack_env_and_compiler) -> None:
    _, compiler = rack_env_and_compiler

    first = compiler.compile("2f")
    second = compiler.compile("2f")

    assert first.mjcf_sha256 == second.mjcf_sha256
    assert first.baseline_qpos == second.baseline_qpos


def test_registry_routes_every_ee_exchange_collision_context(
    rack_env_and_compiler,
) -> None:
    env, compiler = rack_env_and_compiler
    contexts = EEExchangeTemplateGenerator().build_collision_contexts(
        from_ee="2f",
        to_ee="vacuum",
    )
    registry = compiler.build_collision_registry(
        contexts,
        collision_margin_m=0.005,
    )
    initial_q = tuple(float(value) for value in env.sim.data._data.qpos[:6])

    reports = {
        context_id: registry.check(initial_q, context=context)
        for context_id, context in contexts.items()
    }

    assert all(report.valid for report in reports.values())
    assert registry.joint_names == tuple(env.robots[0].robot_model.joints)


def test_attached_vacuum_validator_includes_stalk_collision(
    rack_env_and_compiler,
) -> None:
    _, compiler = rack_env_and_compiler
    attached = compiler.compile("vacuum")
    validator = attached.make_validator(collision_margin_m=0.005)

    assert any(
        name.endswith("vac_mount_col") for name in validator.robot_geom_names
    )


def test_vacuum_mjcf_has_separate_visual_and_collision_stalk() -> None:
    root = ET.parse("scripts/assets/vacuum_gripper.xml").getroot()
    geoms = {geom.get("name"): geom for geom in root.findall(".//geom")}

    assert geoms["vac_mount_col"].get("contype") == "1"
    assert geoms["vac_mount_col"].get("group") == "0"
    assert geoms["vac_mount"].get("contype") == "0"
    assert geoms["vac_mount"].get("group") == "1"
    assert geoms["vac_mount_col"].get("size") == geoms["vac_mount"].get("size")
    assert geoms["vac_mount_col"].get("pos") == geoms["vac_mount"].get("pos")
