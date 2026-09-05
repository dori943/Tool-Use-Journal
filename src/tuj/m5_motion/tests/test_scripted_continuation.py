import numpy as np
import pytest
from types import SimpleNamespace
from tuj.m5_motion import UR5eKinematics
from tuj.m5_motion.scripted_grasps.ik_continuity import ContinuousIK
from tuj.m5_motion.scripted_grasps.objects.spatula import SpatulaRecipe
from tuj.m5_motion.scripted_grasps.spatula_runtime import update_three_finger_commands


def test_held_pose_ik_preserves_negative_wrist_turn_and_residual_budget():
    inner=UR5eKinematics()
    seed=(.3,-1.8,1.8,-1.7,-4.6,.4)
    goal_q=np.asarray(seed)+[.02,0.,.01,0.,0.,0.]
    target,rotation=inner.forward_pose_world(goal_q)
    choices=ContinuousIK(inner,seed).solve_all_ik(target,rotation)
    nearest=min(choices.solutions,key=lambda s:np.linalg.norm(np.asarray(s.qpos)-seed))
    assert nearest.position_error_m<=1e-5
    assert nearest.orientation_error_rad<=5e-5
    assert abs(nearest.qpos[4]-seed[4])<.2
    assert max(abs(np.asarray(nearest.qpos)-seed))<.3


def test_small_balanced_contact_errors_do_not_walk_fingers_open():
    recipe=SpatulaRecipe(three_finger_force_deadband_n=.25)
    commands=np.array([-.70,-.52,-.63])
    original=commands.copy()
    for _ in range(500):
        commands=update_three_finger_commands(commands,[2.91,1.72,1.47],recipe)
    np.testing.assert_allclose(commands,original)
    # A real loss of force must still close the fingers, within one-tick bounds.
    next_commands=update_three_finger_commands(commands,[0.,0.,0.],recipe)
    assert np.all(next_commands<commands)
    assert np.max(commands-next_commands)<=.01


def test_reused_planner_reads_current_joint_reference_before_each_generation():
    from tuj.m5_motion.pipeline import MotionPlanningPipeline
    kinematics=ContinuousIK(UR5eKinematics(), [0.]*6)
    class Provider:
        def generate(self, request):
            assert kinematics.initial_q == tuple(request.world.robot_state.joint_positions_rad)
            raise RuntimeError('generation reached')
    pipeline=MotionPlanningPipeline(Provider(), kinematics)
    for q in ([.2]*6, [-.4]*6):
        request=SimpleNamespace(world=SimpleNamespace(robot_state=SimpleNamespace(joint_positions_rad=q)))
        with pytest.raises(RuntimeError, match='generation reached'):
            pipeline.plan(request)


def test_post_grasp_arm_gain_is_restored_on_release(monkeypatch):
    from tuj.m5_motion.scripted_grasps.retention import GraspRetention
    from tuj.m5_motion.tests.test_scripted_grasps import runtime_stub
    controller=SimpleNamespace(kp=np.full(6,150.),kd=np.full(6,2*np.sqrt(150.)))
    original=(controller.kp.copy(),controller.kd.copy())
    context=SimpleNamespace(grip_pose=lambda:np.eye(4),body_pose=lambda:np.eye(4),
        gripper=SimpleNamespace(current_action=[0.]), recipe=SimpleNamespace(post_grasp_arm_kp=300.),
        robot=SimpleNamespace(part_controllers={'right':controller}))
    monkeypatch.setattr(GraspRetention,'_forces',lambda self:{})
    retention=GraspRetention(context,SimpleNamespace(driver='catalog'))
    np.testing.assert_allclose(controller.kp,300.)
    np.testing.assert_allclose(controller.kd,2*np.sqrt(300.))
    runtime=runtime_stub()
    runtime.scripted_grasp_retention=retention
    runtime.command_gripper(engaged=False,suction=False)
    np.testing.assert_allclose(controller.kp,original[0])
    np.testing.assert_allclose(controller.kd,original[1])


def test_live_capture_uses_actual_runtime_but_file_capture_still_validates(monkeypatch):
    from tuj.m5_motion.scripted_grasps import cli
    from tuj.m5_motion import generic_runner
    observed=object()
    monkeypatch.setattr(cli,'snapshot',lambda runtime,preview:observed)
    def mismatch(*args):
        raise generic_runner.GenericMotionRunnerError('snapshot mismatch')
    monkeypatch.setattr(generic_runner,'_validate_runtime_start',mismatch)
    assert cli.execution_initial_world(object(),object(),externally_supplied=False) is observed
    with pytest.raises(generic_runner.GenericMotionRunnerError,match='snapshot mismatch'):
        cli.execution_initial_world(object(),object(),externally_supplied=True)


def test_collision_variant_factory_releases_each_environment_before_the_next():
    from tuj.m5_motion.tool_use_journal import ToolUseJournalCollisionModelCompiler
    from tuj.m5_motion.tests.test_tool_use_journal import _fake_env
    reference=_fake_env('2F')
    alive=[]
    created=[]
    def factory(ee):
        assert not alive
        assert ee!='2F'  # The actual mounted environment is already available.
        env=_fake_env(ee)
        env.reset=lambda:None
        env.close=lambda:alive.remove(ee)
        alive.append(ee)
        created.append(ee)
        return env
    compiler=ToolUseJournalCollisionModelCompiler.from_environment_factory(reference,factory)
    assert not alive
    assert created==[None,'3F','vac']
    for ee in (None,'2F','3F','vac'):
        assert compiler.compile(ee).active_ee==ee


def test_held_motion_limit_never_loosens_a_stricter_task_constraint(monkeypatch,tmp_path):
    from tuj.m5_motion.scripted_grasps import live
    from tuj.m5_motion.tests.test_scripted_grasps import request_for,ENTRIES
    request=request_for(ENTRIES[4],action='MOVE')
    request.constraints.velocity_scaling=.05
    request.constraints.acceleration_scaling=.5
    monkeypatch.setattr(live,'snapshot',lambda runtime,previous=None:request.world.model_copy(deep=True))
    recipe=SimpleNamespace(post_grasp_velocity_scaling=.1,post_grasp_acceleration_scaling=.1)
    runtime=SimpleNamespace(env=object(),scripted_grasp_retention=SimpleNamespace(
        context=SimpleNamespace(recipe=recipe),samples=[]))
    def factory(*args,**kwargs):
        def planner(actual):
            assert actual.constraints.velocity_scaling==.05
            assert actual.constraints.acceleration_scaling==.1
            raise RuntimeError('limits checked')
        return planner
    session=live.ScriptedGraspSession(runtime,tmp_path,tmp_path,planner_factory=factory)
    with pytest.raises(RuntimeError,match='limits checked'):
        session.execute_request(request)
    assert request.constraints.acceleration_scaling==.5
