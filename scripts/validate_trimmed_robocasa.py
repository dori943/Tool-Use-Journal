"""Validate C1-T2/C2-T2 while requiring a selected trimmed RoboCasa import."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime))
    sys.path.insert(1, str(PROJECT_ROOT))

    import robocasa
    import robosuite as suite

    imported_from = Path(robocasa.__file__).resolve()
    try:
        imported_from.relative_to(runtime)
    except ValueError as exc:
        raise RuntimeError(f"RoboCasa was not imported from trimmed runtime: {imported_from}") from exc

    import environments  # noqa: F401

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    for seed in seeds:
        for env_name in ("C1_2_DoughFlatten", "C2_2_SandwichAssembly"):
            env = suite.make(
                env_name=env_name,
                robots="UR5e",
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                render_camera=None,
                seed=seed,
            )
            try:
                observation = env.reset()
                action = env.action_spec[0] * 0.0
                env.step(action)
                print(f"PASS {env_name} seed={seed} observations={len(observation)}")
            finally:
                env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
