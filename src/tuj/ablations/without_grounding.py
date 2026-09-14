"""Upper-level grounding ablation. Execution geometry belongs to M5 only."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from tuj.m2_subgoal.core import TOOL_KINDS
from tuj.m2_subgoal.pipeline import run_m2
from tuj.m2_subgoal.rough import (
    _image_part,
    make_llm_client,
    track_usage,
    validate_subgoals,
)
from tuj.m4_taskplanner.gk_adapter import build_request_from_gk

MODE = "without_grounding"
POLICY_VERSION = "upper-level-visual-semantic-v1"
NODE_KEYS = ("id", "canonical_id", "class", "source_name")
ROBOT_KEYS = (
    "robot_id",
    "model",
    "reach_mm",
    "current_ee",
    "held_tool",
    "hand_empty",
    "ee_swap_cost_s",
)
EE_KEYS = (
    "ee_id",
    "type",
    "capabilities",
    "stroke_mm",
    "aperture_mm",
    "payload_kg",
    "grip_force_n",
    "fingerpad_friction",
    "home_slot",
    "pass_height_mm",
    "seal_diameter_mm",
    "flatness_tol_rms_mm",
)


def decision_scene(raw: dict) -> dict:
    """An allowlist, not recursive field deletion: unknown fields cannot leak."""
    nodes = []
    ids = set()
    for node in raw["nodes"]:
        item = {k: node[k] for k in NODE_KEYS if k in node}
        if not isinstance(item.get("id"), str) or item["id"] in ids:
            raise ValueError("scene object IDs must be unique strings")
        if not all(isinstance(v, str) for v in item.values()):
            raise ValueError("identity fields must be strings")
        ids.add(item["id"])
        nodes.append(item)
    if not nodes:
        raise ValueError("scene must contain objects")
    return {"nodes": nodes, "edges": []}


def decision_robot(raw: dict) -> dict:
    """Keep intrinsic hardware specs; never per-object compatibility or facts."""
    result = {k: copy.deepcopy(raw[k]) for k in ROBOT_KEYS if k in raw}
    if any(isinstance(v, (dict, list)) for v in result.values()):
        raise ValueError("robot identity/resource state must contain scalars only")
    result["ee_pool"] = [
        {k: copy.deepcopy(ee[k]) for k in EE_KEYS if k in ee} for ee in raw["ee_pool"]
    ]
    # Robot files are configuration, but reject arbitrary nested scene fields.
    for ee in result["ee_pool"]:
        if not isinstance(ee.get("ee_id"), str):
            raise TypeError("each EE needs an ee_id")
        for key, value in ee.items():
            if isinstance(value, dict) or (
                isinstance(value, list)
                and any(isinstance(v, (dict, list)) for v in value)
            ):
                raise ValueError(f"unexpected nested robot specification: {key}")
    return result


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RoughSubgoal(_Record):
    subgoal_id: str
    goal: str
    kind: Literal[
        "relocate", "sweep_collect", "flatten", "stack", "scoop_transfer", "extract"
    ]
    target_ids: list[str] = Field(min_length=1)
    container_id: str | None
    ordered: bool
    confidence: float = Field(ge=0, le=1)


class VisualDecision(_Record):
    subgoal_id: str
    selected_tool_id: str | None
    # One hypothesis per grasped object. M4 still chooses among these EE IDs.
    ee_candidates_by_object: dict[str, list[str]]
    reason: str
    confidence: float = Field(ge=0, le=1)


def validate_rough(value, scene):
    records = TypeAdapter(list[RoughSubgoal]).validate_python(value)
    if not records:
        raise ValueError("at least one subgoal is required")
    return validate_subgoals(
        [r.model_dump() for r in records], [n["id"] for n in scene["nodes"]]
    )


def validate_decisions(value, subgoals, scene, robot):
    records = TypeAdapter(list[VisualDecision]).validate_python(value)
    by_id = {r.subgoal_id: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {s["subgoal_id"] for s in subgoals}:
        raise ValueError("return exactly one decision for each subgoal")
    known = {n["id"] for n in scene["nodes"]}
    ees = {e["ee_id"] for e in robot["ee_pool"]}
    for s in subgoals:
        r = by_id[s["subgoal_id"]]
        if s["kind"] in TOOL_KINDS:
            if r.selected_tool_id not in known - set(s["target_ids"]) - {
                s["container_id"]
            }:
                raise ValueError(
                    f"{r.subgoal_id}: select a scene tool distinct from targets/destination"
                )
            owners = {r.selected_tool_id}
        else:
            if r.selected_tool_id is not None:
                raise ValueError("relocate/stack use direct object manipulation")
            owners = set(s["target_ids"])
        if set(r.ee_candidates_by_object) != owners:
            raise ValueError(
                f"{r.subgoal_id}: EE hypotheses must cover exactly {sorted(owners)}"
            )
        for candidates in r.ee_candidates_by_object.values():
            if len(candidates) != len(set(candidates)) or not set(candidates) <= ees:
                raise ValueError(
                    "EE hypotheses must be unique IDs from the hardware inventory"
                )
        # Empty hypotheses are valid negative decisions, not silently filled in.
    return {sid: r.model_dump() for sid, r in by_id.items()}


class VisualSemanticRough:
    """Two vision calls: semantic decomposition, then tool/EE hypotheses.

    No measured data, memory, scripted grasp registry, or template answer is read.
    API errors propagate; malformed outputs receive one retry, as in full M2.
    """

    def __init__(self, model: str, frame: Path, robot: dict, client=None):
        if not frame.is_file():
            raise ValueError("without_grounding requires the matching scene frame")
        self.model, self.frame = model, frame
        self.robot = decision_robot(robot)
        self.client = client
        self.usage = {}

    def _call(self, prompt, validate, stage):
        if self.client is None:
            self.client = make_llm_client()
        error = None
        for attempt in range(2):
            message = prompt + (
                f"\nFix the previous response: {error}" if attempt else ""
            )
            t0 = time.monotonic()
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            _image_part(str(self.frame)),
                        ],
                    }
                ],
            )
            track_usage(self.usage, stage, response, time.monotonic() - t0)
            try:
                content = response.choices[0].message.content or ""
                value = json.loads(content[content.find("[") : content.rfind("]") + 1])
                return validate(value)
            except ValueError as exc:
                error = str(exc)
        raise ValueError(f"{stage}: invalid model response after 2 attempts: {error}")

    def generate(self, task, scene):
        scene = decision_scene(scene)
        context = json.dumps(
            {"task": task, "objects": scene["nodes"]}, ensure_ascii=False
        )
        subgoals = self._call(
            "Decompose the requested robot task into semantic subgoals using the image and "
            "object identities only. Do not invent extra cleanup goals. Choose only IDs in "
            "the inventory. A means/tool mentioned in the task is not a separate goal. "
            "Use relocate for moving to a destination, stack for ordered layers, "
            "sweep_collect, flatten, scoop_transfer or extract for tool actions. Group targets "
            "by destination. If order matters (always for stack), order target_ids by execution "
            "order using the image and task. No measured geometry, material or feasibility "
            "is available. Return only a JSON array with this record schema: "
            + json.dumps(RoughSubgoal.model_json_schema())
            + "\nINPUT:\n"
            + context,
            lambda value: validate_rough(value, scene),
            "visual_decomposition",
        )
        decisions = self._call(
            "For each subgoal choose the tool and plausible EE candidates using the image, "
            "object names, task and intrinsic hardware specifications only. These are uncertain "
            "visual/semantic hypotheses, NOT measured feasibility. For flatten, sweep_collect, "
            "scoop_transfer and extract select one available object as tool; consider unusual "
            "uses visible in the image. For relocate/stack selected_tool_id is null. "
            "ee_candidates_by_object must cover only the selected tool for tool tasks, otherwise "
            "every target object. Use EE IDs from the hardware inventory. An empty EE list "
            "means you predict no suitable EE; do not assume all EEs work. Do not output "
            "measurements, coordinates, ground-truth checks or extra fields. Return a JSON array "
            "of records with schema: "
            + json.dumps(VisualDecision.model_json_schema())
            + "\nINPUT:\n"
            + context
            + "\nSUBGOALS:\n"
            + json.dumps(subgoals, ensure_ascii=False)
            + "\nHARDWARE:\n"
            + json.dumps(self.robot, ensure_ascii=False),
            lambda value: validate_decisions(value, subgoals, scene, self.robot),
            "visual_selection",
        )
        for s in subgoals:
            decision = decisions[s["subgoal_id"]]
            tool = decision["selected_tool_id"]
            s["selected_tool_id"] = tool
            s["tool_candidate_ids"] = [tool] if tool else []
            s["object_ids"] = sorted(
                set(
                    s["target_ids"]
                    + ([tool] if tool else [])
                    + ([s["container_id"]] if s["container_id"] else [])
                )
            )
            s["selection_by"] = "visual_semantic"
            s["visual_decision"] = decision
        return subgoals


def build_m2(task: str, scene: dict, rough) -> dict:
    scene = decision_scene(scene)
    out = run_m2(task, scene, rough=rough)
    out["ablation"] = {"mode": MODE, "policy_version": POLICY_VERSION}
    for s in out["m2_subgoals"]:
        # Do not call apply_grounding/update_confidence/regroup on this path.
        for d in s["details"]:
            for p in d["pre"]:
                if p["eval_by"] == "m3":
                    p.update(status="unknown", evidence=[])
    return out


def build_m4_request(scene: dict, m2: dict, robot: dict, assemble):
    """Reuse the real adapter/search, with explicit hypotheses and no ground truth."""
    if m2.get("ablation") != {"mode": MODE, "policy_version": POLICY_VERSION}:
        raise ValueError("M4 requires an M2 artifact from this ablation policy")
    scene, robot = decision_scene(scene), decision_robot(robot)
    bundle = {"gk_by_subgoal": assemble(scene, m2)}
    request = build_request_from_gk(
        bundle, m2, m1_payload=scene, robot_spec_payload=robot
    )
    by_id = {s["subgoal_id"]: s for s in m2["m2_subgoals"]}
    decisions = validate_decisions(
        [s["visual_decision"] for s in by_id.values()],
        list(by_id.values()),
        scene,
        robot,
    )
    aliases = request.task_graph.log["id_aliases"]
    for sg in request.task_graph.subgoals:
        parent = by_id[sg.source_kg_subgoal_id]
        hypotheses = {
            aliases.get(k, k): set(v)
            for k, v in decisions[parent["subgoal_id"]][
                "ee_candidates_by_object"
            ].items()
        }
        owners = [sg.tool_id] if sg.tool_id else sg.target_ids
        sets = [hypotheses[o] for o in owners]
        sg.feasible_ee = sorted(set.intersection(*sets)) if sets else []
        sg.feasible_ee_source = "request"
        sg.condition_source = POLICY_VERSION
        sg.model_extra["ee_hypothesis_source"] = "visual_semantic_unverified"
    request.task_graph.log["ablation"] = m2["ablation"]
    # build_request_from_gk already defers unknown measurement checks; preserve
    # symbolic ordering/resource checks and the normal planner cost/search.
    assert request.planning_policy.unknown_feasibility_policy == "defer"
    return request, bundle
