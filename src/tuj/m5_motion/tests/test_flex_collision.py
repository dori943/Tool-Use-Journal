import xml.etree.ElementTree as ET
import mujoco

from environments.deformable_material import MaterialParameters
from environments.objects.deformable_ellipsoid import DeformableEllipsoidObject
from tuj.m5_motion.mujoco_collision import MuJoCoCollisionValidator
from tuj.m5_motion.schema import RelativeKeyframeSpec, KeyframeType, CollisionContext


def test_native_flex_contact_is_rejected_unless_exact_pair_is_allowed():
    obj = DeformableEllipsoidObject('material', [.06, .06, .06],
                                   MaterialParameters(500, 20000, 1000, 1000))
    root = ET.fromstring('''<mujoco><worldbody><body name="robot">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="tool" type="sphere" size=".01"/>
    </body></worldbody></mujoco>''')
    root.find('worldbody').append(obj.get_obj())
    ET.SubElement(root, 'deformable').append(obj.flex)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    context = CollisionContext(context_id='contact', collision_model_version='native-flex',
                               allowed_collision_pairs=[('tool', 'target')])
    validator = MuJoCoCollisionValidator(model, joint_names=('slide',),
        robot_root_body_name='robot', collision_margin_m=.002,
        collision_model_version='native-flex', entity_geoms={'target': (obj.root_body,)},
        collision_contexts={'contact': context})
    key = RelativeKeyframeSpec(keyframe_id='check', keyframe_type=KeyframeType.CUSTOM,
                              frame_ref='world', anchor='origin',
                              approach_axis_xyz=(0, 0, 1), planner='JOINT')
    assert validator.check((.1,), key).valid
    assert not validator.check((.042,), key).valid  # positive gap below 2 mm clearance
    collision = validator.check((.03,), key)
    assert not collision.valid
    assert any(obj.flex.get('name') in (c.geom_a, c.geom_b) for c in collision.contacts)
    key.collision_context_id = 'contact'
    assert validator.check((.03,), key).valid
    # The same allowance must not approve a different tool entity.
    validator._collision_contexts['contact'] = CollisionContext(
        context_id='contact', collision_model_version='native-flex',
        allowed_collision_pairs=[('different_tool', 'target')])
    assert not validator.check((.03,), key).valid
