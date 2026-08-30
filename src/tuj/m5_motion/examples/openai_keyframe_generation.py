"""Generate a frozen keyframe candidate artifact from a MotionPlanRequest JSON.

PowerShell:
    $env:OPENAI_API_KEY = "<new project key>"
    python src/tuj/m5_motion/examples/openai_keyframe_generation.py request.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
REPOSITORY = SRC.parent
for package_root in (REPOSITORY, SRC):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from tuj.m5_motion.schema import MotionPlanRequest  # noqa: E402
from tuj.m5_motion.vlm_provider import (  # noqa: E402
    OpenAIKeyframeProvider,
    OpenAIKeyframeProviderConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path, help="MotionPlanRequest JSON")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    request = MotionPlanRequest.model_validate_json(
        args.request.read_text(encoding="utf-8")
    )
    overrides = {
        "candidate_count": args.candidates,
        "cache_dir": args.cache_dir,
    }
    if args.model:
        overrides["model"] = args.model
    config = OpenAIKeyframeProviderConfig.from_environment(**overrides)
    artifact = OpenAIKeyframeProvider(config).generate(request)
    rendered = artifact.model_dump_json(indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
