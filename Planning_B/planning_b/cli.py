"""Command-line interface.

    python -m planning_b.cli plan \
        --planner-a examples/bottle_plate_planner_a.json \
        --resources examples/bottle_plate_resources.json \
        --candidates examples/bottle_plate_candidates.json \
        --output result.json

    python -m planning_b.cli replan \
        --request request.json \
        --execution-state execution_state.json \
        --failure failure.json \
        --output replanned_result.json
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from pydantic import ValidationError

from planning_b.diagnostics import PlanStatus
from planning_b.models import (
    CandidateProposal,
    ExecutionState,
    FailureFeedback,
    PlannerAOutput,
    PlanningBRequest,
    PlanningPolicy,
    ResourceCatalog,
)
from planning_b.planner_a_adapter import build_request_from_current_planner_a
from planning_b.gk_adapter import build_request_from_gk
from planning_b.planner import plan
from planning_b.replanning import replan
from planning_b.serialization import PlanningResult, dump_result, load_json

_EXIT_CODES: dict[PlanStatus, int] = {
    PlanStatus.SUCCESS: 0,
    PlanStatus.INVALID_INPUT: 2,
    PlanStatus.INFEASIBLE_REDECOMPOSE: 3,
    PlanStatus.INFEASIBLE_NO_CANDIDATE: 3,
    PlanStatus.INFEASIBLE_NO_PLAN: 3,
    PlanStatus.SEARCH_LIMIT_REACHED: 4,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="planning_b", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="run the planner")
    source = p_plan.add_mutually_exclusive_group(required=True)
    source.add_argument("--planner-a")
    source.add_argument(
        "--gk", help="GK JSON (single record or gk_by_subgoal bundle)"
    )
    p_plan.add_argument(
        "--m1", help="M1 action/DAG JSON required by --gk"
    )
    p_plan.add_argument("--m0", help="optional M0 scene graph for --gk")
    p_plan.add_argument(
        "--robot-spec", help="optional robot/EE specification for --gk"
    )
    p_plan.add_argument(
        "--id-aliases",
        help="optional JSON mapping upstream object IDs to world IDs",
    )
    p_plan.add_argument(
        "--scenario",
        default=None,
        help="companion Planner-A scenario for detailed_subgoals outputs",
    )
    p_plan.add_argument("--resources", default=None)
    p_plan.add_argument("--candidates", default=None)
    p_plan.add_argument("--policy", default=None)
    p_plan.add_argument("--output", required=True)

    p_replan = sub.add_parser("replan", help="replan after an execution failure")
    p_replan.add_argument("--request", required=True)
    p_replan.add_argument("--execution-state", required=True)
    p_replan.add_argument("--failure", required=True)
    p_replan.add_argument("--output", required=True)
    return parser


def _load_proposals(path: str) -> dict[str, list[CandidateProposal]]:
    raw = load_json(path)
    return {
        sg_id: [CandidateProposal.model_validate(p) for p in proposals]
        for sg_id, proposals in raw.items()
    }


def _finish(result: PlanningResult, output: str) -> int:
    dump_result(result, output)
    if result.status is not PlanStatus.SUCCESS:
        reasons = "; ".join(
            f"{r.reason_code.value}: {r.message}" for r in result.rejections[:5]
        )
        print(
            f"planning_b: {result.status.value}"
            + (f" ({reasons})" if reasons else ""),
            file=sys.stderr,
        )
    return _EXIT_CODES[result.status]


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            policy = (
                PlanningPolicy.model_validate(load_json(args.policy))
                if args.policy is not None
                else None
            )
            proposals = (
                _load_proposals(args.candidates)
                if args.candidates is not None
                else None
            )
            resources = (
                ResourceCatalog.model_validate(load_json(args.resources))
                if args.resources is not None
                else None
            )
            if args.gk is not None:
                if args.m1 is None:
                    raise ValueError(
                        "--gk requires --m1 because GK has no executable "
                        "action details or partial-order DAG"
                    )
                aliases = (
                    load_json(args.id_aliases)
                    if args.id_aliases is not None
                    else None
                )
                if aliases is not None and not isinstance(aliases, dict):
                    raise ValueError("--id-aliases must contain a JSON object")
                request = build_request_from_gk(
                    load_json(args.gk),
                    load_json(args.m1),
                    m0_payload=(
                        load_json(args.m0) if args.m0 is not None else None
                    ),
                    robot_spec_payload=(
                        load_json(args.robot_spec)
                        if args.robot_spec is not None
                        else None
                    ),
                    resource_catalog=resources,
                    candidate_proposals=proposals,
                    planning_policy=policy,
                    id_aliases=aliases,
                )
            else:
                planner_raw = load_json(args.planner_a)
                if "detailed_subgoals" in planner_raw:
                    if args.scenario is None:
                        raise ValueError(
                            "current Planner-A outputs require --scenario because "
                            "feasible EE sets and robot state are not in the DAG JSON"
                        )
                    request = build_request_from_current_planner_a(
                        planner_raw,
                        load_json(args.scenario),
                        resource_catalog=resources,
                        candidate_proposals=proposals,
                        planning_policy=policy,
                    )
                else:
                    if resources is None:
                        raise ValueError(
                            "normalized Planner-A inputs require --resources"
                        )
                    request = PlanningBRequest(
                        planner_a=PlannerAOutput.model_validate(planner_raw),
                        resource_catalog=resources,
                        candidate_proposals=proposals,
                        planning_policy=policy or PlanningPolicy(),
                    )
            return _finish(plan(request), args.output)
        request = PlanningBRequest.model_validate(load_json(args.request))
        execution_state = ExecutionState.model_validate(
            load_json(args.execution_state)
        )
        failure = FailureFeedback.model_validate(load_json(args.failure))
        return _finish(
            replan(request, execution_state, failure), args.output
        )
    except (ValidationError, OSError, ValueError) as exc:
        print(f"planning_b: invalid input: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
