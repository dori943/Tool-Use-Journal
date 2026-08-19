"""설치 확인용 비대화형 robosuite 스모크 테스트."""

import sys
import time

import numpy as np


def main() -> int:
    import robosuite as suite
    from robosuite.environments import ALL_ENVIRONMENTS
    from robosuite.robots import ALL_ROBOTS, ROBOT_CLASS_MAPPING

    print(f"robosuite version: {suite.__version__}")

    import mujoco

    print(f"mujoco version: {mujoco.__version__}")

    robots = sorted(ALL_ROBOTS)
    print(f"Registered robots ({len(robots)}): {robots}")

    ur5e_registered = "UR5e" in ALL_ROBOTS
    ur5e_mapped = "UR5e" in ROBOT_CLASS_MAPPING
    print(f"UR5e in ALL_ROBOTS: {ur5e_registered}")
    print(f"UR5e in ROBOT_CLASS_MAPPING: {ur5e_mapped}")

    envs = sorted(ALL_ENVIRONMENTS)
    print(f"Registered environments ({len(envs)}): {envs}")

    env_name = "Lift"
    if env_name not in ALL_ENVIRONMENTS:
        print(f"ERROR: {env_name} not found in registered environments", file=sys.stderr)
        return 1

    print(f"\nCreating {env_name} environment with Panda robot...")
    env = suite.make(
        env_name=env_name,
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    print("Calling env.reset()...")
    obs = env.reset()
    print(f"reset() OK - observation keys: {list(obs.keys())}")

    env.viewer.set_camera(camera_id=0)
    print("Rendering 50 random-action steps (viewer window should open)...")
    for step in range(50):
        action = np.random.randn(*env.action_spec[0].shape)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.04)

    print("Smoke test passed.")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
