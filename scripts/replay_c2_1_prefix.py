"""Replay the saved, validated EE-attach prefix with a wide camera."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from tuj.m5_motion import generic_runner as runner
from tuj.m5_motion.orchestration import SelectedPlanPlanningResult, _predicted_world
from tuj.m5_motion.schema import MotionPlan, MotionPlanRequest, WorldSnapshot

OUT = ROOT / 'output/c2_1'
load = lambda p: json.loads(p.read_text(encoding='utf-8'))
initial = WorldSnapshot.model_validate(load(OUT / 'm5/initial_world.json'))
plans = tuple(MotionPlan.model_validate(load(p)) for p in sorted((OUT / 'm5/plans').glob('*.json')))
requests = tuple(MotionPlanRequest.model_validate(load(p)) for p in sorted((OUT / 'm5/requests').glob('*.json')))
final = initial.model_copy(deep=True)
for request, plan in zip(requests, plans, strict=True):
    assert request.request_id == plan.request_id
    final = _predicted_world(final, request, plan, completed_subgoal=None)

original = runner.GenericSimulationVideoRecorder._write_frame
def wide_frame(self, env):
    cid = env.sim.model.camera_name2id(self.camera)
    eye = np.array([1.35, -1.65, 2.05])
    target = np.array([-.25, -.1, 1.05])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    quat = Rotation.from_matrix(np.column_stack((right, up, -forward))).as_quat()
    env.sim.model.cam_pos[cid] = eye
    env.sim.model.cam_quat[cid] = quat[[3, 0, 1, 2]]
    env.sim.model.cam_fovy[cid] = 48
    env.sim.forward()
    original(self, env)
runner.GenericSimulationVideoRecorder._write_frame = wide_frame
result = runner.execute_planning_result(SelectedPlanPlanningResult(requests, plans, final),
    repository=ROOT, initial_world=initial, output_dir=OUT / 'm5/wide_replay',
    mode='controller', seed=0, show_viewer=False, realtime_factor=0,
    hold_seconds=0, video=OUT / 'c2_1_wide_raw.mp4', camera='agentview',
    width=1280, height=720, video_fps=24, video_hold_seconds=3)
print(result.status.value)
