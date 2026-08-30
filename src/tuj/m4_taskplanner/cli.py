"""Command-line interface.

    python -m tuj.m4_taskplanner.cli plan \
        --gk output/c1_1/gk_bundle.json \
        --m2 output/c1_1/m2.json \
        --output result.json

    python -m tuj.m4_taskplanner.cli replan \
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

from tuj.m4_taskplanner.diagnostics import PlanStatus
from tuj.m4_taskplanner.models import (
    CandidateProposal,
    ExecutionState,
    FailureFeedback,
    InitialState,
    TaskPlannerRequest,
    PlanningPolicy,
    ResourceCatalog,
)
from tuj.m4_taskplanner.gk_adapter import build_request_from_gk
from tuj.m4_taskplanner.planner import plan
from tuj.m4_taskplanner.replanning import replan
from tuj.m4_taskplanner.serialization import PlanningResult, dump_result, load_json

_EXIT_CODES: dict[PlanStatus, int] = {
    PlanStatus.SUCCESS: 0,
    PlanStatus.INVALID_INPUT: 2,
    PlanStatus.INFEASIBLE_REDECOMPOSE: 3,
    PlanStatus.INFEASIBLE_NO_CANDIDATE: 3,
    PlanStatus.INFEASIBLE_NO_PLAN: 3,
    PlanStatus.SEARCH_LIMIT_REACHED: 4,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="task_planner", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="run the planner")
    p_plan.add_argument(
        "--gk",
        required=True,
        help="GK JSON (single record or gk_by_subgoal bundle)",
    )
    p_plan.add_argument(
        "--m2",
        required=True,
        help="M2 scene graph JSON or legacy executable action/DAG JSON",
    )
    p_plan.add_argument("--m1", help="optional M1 scene graph for --gk")
    p_plan.add_argument(
        "--robot-spec", help="optional robot/EE specification for --gk"
    )
    p_plan.add_argument(
        "--id-aliases",
        help="optional JSON mapping upstream object IDs to world IDs",
    )
    p_plan.add_argument(
        "--initial-state",
        help=(
            "optional normalized InitialState JSON; overrides robot_spec "
            "current_ee, held_tool, rack occupancy, and facts"
        ),
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
            f"task_planner: {result.status.value}"
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
            aliases = (
                load_json(args.id_aliases)
                if args.id_aliases is not None
                else None
            )
            if aliases is not None and not isinstance(aliases, dict):
                raise ValueError("--id-aliases must contain a JSON object")
            initial_state = (
                InitialState.model_validate(load_json(args.initial_state))
                if args.initial_state is not None
                else None
            )
            request = build_request_from_gk(
                load_json(args.gk),
                load_json(args.m2),
                m1_payload=(
                    load_json(args.m1) if args.m1 is not None else None
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
                initial_state=initial_state,
            )
            return _finish(plan(request), args.output)
        request = TaskPlannerRequest.model_validate(load_json(args.request))
        execution_state = ExecutionState.model_validate(
            load_json(args.execution_state)
        )
        failure = FailureFeedback.model_validate(load_json(args.failure))
        return _finish(
            replan(request, execution_state, failure), args.output
        )
    except (ValidationError, OSError, ValueError) as exc:
        print(f"task_planner: invalid input: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
