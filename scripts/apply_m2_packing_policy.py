"""Apply a task-owned packing sequence to an already grounded M2 artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY / "src"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from tuj.m2_subgoal.core import (
        add_container_packing_sequence_pres,
        partial_order,
    )

    payload = json.loads(args.m2.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    logs = add_container_packing_sequence_pres(
        payload["m2_subgoals"], policy["target_order"]
    )
    if len(logs) != max(0, len(policy["target_order"]) - 1):
        raise ValueError(
            "packing policy did not resolve every adjacent target pair: "
            f"expected {len(policy['target_order']) - 1}, got {len(logs)}"
        )

    details = [
        detail
        for subgoal in payload["m2_subgoals"]
        for detail in subgoal["details"]
    ]
    edges, mutex = partial_order(details)
    payload["m2_partial_order"] = edges
    payload["m2_mutex"] = mutex
    payload["m2_packing_policy"] = policy
    payload.setdefault("m2_stats", {})["n_edges"] = len(edges)
    payload["m2_stats"]["n_mutex"] = len(mutex)

    output = args.output or args.m2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for line in logs:
        print(line)
    print(f"[M2] packing policy applied: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
