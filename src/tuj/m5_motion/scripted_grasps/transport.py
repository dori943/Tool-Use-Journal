"""Ground M4's implicit region transport without changing the physical grasp."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from tuj.m5_motion.geometry import tool_rotation_from_axis
from .frames import inverse, transform


def ground_held_transport(request, retention):
    task=request.task
    if task.action_type.upper() not in {'TRANSPORT','MOVE'}:
        return
    if not task.metadata.get('scripted_m4_implicit_object_pose', False):
        return
    if not task.goal.target_region_id or task.target_ids != [retention.entry.object_id]:
        return
    # Explicit task poses remain authoritative. Only replace M4's implicit
    # fallback to the carried object's current pose, which is not a destination.
    if task.metadata.get('action_parameters',{}).get('target_pose') is not None or task.grasp is not None:
        return
    record=request.world.objects.get(task.goal.target_region_id)
    if not record or 'pose' not in record or 'dimensions_m' not in record:
        raise ValueError('TRANSPORT_REGION_GEOMETRY_REQUIRED')
    pose=record['pose']
    if pose.get('frame_id')!='world':
        raise ValueError('TRANSPORT_REGION_WORLD_FRAME_REQUIRED')
    T_WR=transform(pose['position_m'],rotation=Rotation.from_quat(pose['orientation_xyzw']).as_matrix())
    region_center=np.asarray(record.get('anchors',{}).get('center',[0.,0.,0.]))
    region_world=(T_WR@np.r_[region_center,1.])[:3]
    region_half_height=float(np.abs(T_WR[2,:3])@np.asarray(record['dimensions_m'])/2.)
    c=retention.context
    T_WB,T_WG=c.body_pose(),c.grip_pose()
    center=(T_WB@np.r_[c.center_in_body,1.])[:3]
    half_height=float(np.abs(T_WB[2,:3])@np.asarray(c.local_size)/2.)
    desired_center=region_world.copy()
    desired_center[2]=max(center[2],region_world[2]+region_half_height+half_height+.05)
    destination=T_WG.copy()
    destination[:3,3]+=desired_center-center
    anchor='held_transport_goal'
    record.setdefault('anchors',{})[anchor]=(inverse(T_WR)@destination)[:3,3].tolist()
    z,x=destination[:3,2],destination[:3,0]
    base=tool_rotation_from_axis(z,0.)
    task.goal.target_pose=None
    task.metadata['held_transport_goal']={
        'frame_ref':'object:'+task.goal.target_region_id,'anchor':anchor,
        'approach_axis_xyz':(T_WR[:3,:3].T@(-z)).tolist(),
        'tool_axis_to_align':'-z','roll_rad':math.atan2(float(x@base[:,1]),float(x@base[:,0])),
        'offset_along_approach_m':0.,'preserve_grasp_orientation':True,
        'object_id':retention.entry.object_id,'source':'LIVE_GRASP_AND_DESTINATION_BBOX'}
