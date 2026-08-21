from __future__ import annotations

from conftest import make_request, prop, sg

from planning_b.models import GraspSpec
from planning_b.planner import plan


def test_selected_assignment_preserves_motion_grounding() -> None:
    grasp = GraspSpec(
        grasp_id="grasp-1",
        owner_kind="object",
        owner_id="part",
        pose={
            "frame_id": "world",
            "position_m": [0.4, 0.1, 0.2],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )
    request = make_request(
        [
            sg(
                "pick-part",
                targets=["part"],
                action="acquire",
                feasible=["A"],
            )
        ],
        proposals={
            "pick-part": [
                prop(
                    "candidate-1",
                    "pick-part",
                    "A",
                    grasp=grasp,
                    metadata={"approach_distance_m": 0.05},
                )
            ]
        },
    )

    result = plan(request)

    assert result.selected_plan is not None
    assignment = result.selected_plan.candidate_assignments[0]
    assert assignment.action_type == "acquire"
    assert assignment.target_ids == ["part"]
    assert assignment.grasp_id == "grasp-1"
    assert assignment.grasp == grasp
    assert assignment.action_parameters["approach_distance_m"] == 0.05
