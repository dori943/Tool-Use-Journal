import mujoco
import numpy as np
import pytest

from tuj.m5_motion.adhesion_load import payload_adhesion_load


XML = '''<mujoco><worldbody>
<body name="ee"><geom size=".02"/></body>
<body name="payload"><freejoint/><geom size=".02" mass=".3"/>
<body name="payload_child" pos="0 0 .1"><geom size=".02" mass=".1"/></body></body>
</worldbody><actuator><adhesion body="ee" gain="80" ctrlrange="0 1"/></actuator></mujoco>'''


def load(model):
    return payload_adhesion_load(model, model.body("payload").id, model.body("ee").id)


def test_base_force_uses_live_subtree_mass_and_preserves_capacity():
    model = mujoco.MjModel.from_xml_string(XML)
    gains = model.actuator_gainprm.copy()
    first = load(model)
    assert first.mass_kg == pytest.approx(.4)
    assert first.base_force_n == pytest.approx(.4 * 9.81)
    assert (first.command + 1) / 2 * 80 == pytest.approx(first.base_force_n)
    model.body_mass[model.body("payload_child").id] = .2
    second = load(model)
    assert second.mass_kg == pytest.approx(.5)
    np.testing.assert_array_equal(model.actuator_gainprm, gains)


def test_explicit_force_limit_is_respected_without_changing_command_scale():
    model = mujoco.MjModel.from_xml_string(XML.replace('gain="80"', 'gain="80" forcelimited="true" forcerange="0 5"'))
    result = load(model)
    assert result.capacity_n == 5
    assert (result.command + 1) / 2 * 80 == pytest.approx(.4 * 9.81)
    model.body_mass[model.body("payload").id] = 1
    with pytest.raises(ValueError, match="exceeds"):
        load(model)


@pytest.mark.parametrize("change", ["no_actuator", "zero_gravity", "bad_binding"])
def test_unsupported_load_or_binding_fails_closed(change):
    model = mujoco.MjModel.from_xml_string(XML if change != "no_actuator" else XML.replace('<adhesion body="ee" gain="80" ctrlrange="0 1"/>', ''))
    if change == "zero_gravity":
        model.opt.gravity[:] = 0
    with pytest.raises(ValueError):
        if change == "bad_binding":
            payload_adhesion_load(model, model.body("payload").id, -1)
        else:
            load(model)
