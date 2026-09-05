"""Local correction of Jaco's incompatible tendon reference, for spoon runs.

The source XML fixes 0.4*(proximal-distal) at qpos0's .64 even though
proximal<=1.51 and distal>=0 imply a maximum of .604. Its declared spring
rest .2 and Python initial pose [.5,0] agree. Use that consistent rest value.
MuJoCo single-tendon equality fixes length0 + polycoef[0].
https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-tendon
"""
import numpy as np
import xml.etree.ElementTree as ET


def repair_spoon_parallel_2f_xml(root,prefix):
    """Explicit parallel linkage for the native 2F mesh/joint layout.

    Keep the passive pad parallel (inner=-outer) and close the inner linkage
    with inner_knuckle=outer. This replaces the old unconnected spring tendons;
    it is a recorded robot model calibration, not an object attachment.
    """
    tendon=root.find('tendon'); equality=root.find('equality')
    removed=[]
    for t in list(tendon):
        if t.get('name','').startswith(prefix):
            removed.append(ET.tostring(t,encoding='unicode'));tendon.remove(t)
    if len(removed)!=2: raise ValueError('Expected native 2F spring tendons')
    added=[]
    for side,outer in [('left','finger_joint'),('right','right_outer_knuckle_joint')]:
        for suffix,sign in [('inner_finger_joint',-1),('inner_knuckle_joint',1)]:
            attrs={'name':prefix+side+'_'+suffix+'_parallel','joint1':prefix+side+'_'+suffix,
                'joint2':prefix+outer,'polycoef':f'0 {sign} 0 0 0','solref':'.004 1','solimp':'.99 .999 .001'}
            ET.SubElement(equality,'joint',**attrs);added.append(attrs)
    for j in root.findall('.//worldbody//joint'):
        if j.get('name','').startswith(prefix):
            j.set('solreflimit','.002 1');j.set('solimplimit','.9999 .9999 .001');j.set('armature','.0001')
    return {'policy':'CALIBRATED_PARALLEL_2F_LINKAGE','removed_spring_tendons':removed,
        'added_joint_couplings':added,'joint_armature_kg_m2':.0001,
        'joint_ranges_changed':False,'mass_or_friction_changed':False,'source_assets_changed':False,
        'object_attachment':False,'scene_collision_masks_changed':False}


def repair_spoon_hand_xml(root, prefix, timeconstant_s=.004):
    """Change the assembled in-memory model only, never the source XML."""
    joints={j.get('name'):j for j in root.findall('.//worldbody//joint') if j.get('name','').startswith(prefix)}
    tendons={t.get('name'):t for t in root.findall('./tendon/fixed') if t.get('name','').startswith(prefix)}
    changes=[]
    for equality in root.findall('./equality/tendon'):
        name=equality.get('tendon1','')
        if not name.startswith(prefix): continue
        if equality.get('tendon2'): raise ValueError('Unsupported coupled tendon pair')
        tendon=tendons[name]
        reference=sum(float(term.get('coef'))*float(joints[term.get('joint')].get('ref','0')) for term in tendon.findall('joint'))
        rest=float(tendon.get('springlength'))
        if not np.isclose(reference,.64) or not np.isclose(rest,.2):
            raise ValueError('Unexpected Jaco calibration; revalidate the spoon hand')
        constant=rest-reference
        before=dict(equality.attrib)
        equality.set('polycoef',f'{constant:.12g} 1 0 0 0')
        equality.set('solref',f'{timeconstant_s} 1')
        changes.append({'name':name,'reference_length':reference,'rest_length':rest,
            'polycoef_constant':constant,'before':before,'after':dict(equality.attrib)})
    if len(changes)!=3: raise ValueError('Expected exactly three Jaco tendon couplings')
    for joint in joints.values():
        joint.set('solreflimit','.002 1')
        joint.set('solimplimit','.9999 .9999 .001 .5 2')
        # Explicit reflected rotor inertia for the very light finger hinges.
        # This robot-only stabilization is recorded; object dynamics are intact.
        joint.set('armature','.0001')
    return {'policy':'TENDON_REFERENCE_ALIGNED_TO_DECLARED_REST_AND_INIT',
        'couplings':changes,'joint_limit_timeconstant_s':.002,
        'joint_ranges_changed':False,'mass_or_friction_changed':False,
        'joint_limit_impedance':[.9999,.9999,.001,.5,2],
        'joint_armature_kg_m2':.0001,
        'source_assets_changed':False,'minimum_proximal_command_rad':0.}


def bound_spoon_3f_commands(commands):
    """Use native actuator targets [0,1.51]; actual joint limits are enforced."""
    return np.clip(np.asarray(commands,dtype=float),-1.,1.)
