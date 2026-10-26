# -*- coding: utf-8 -*-
"""w/o Reasoner ablation: Geometry & Physical Scene Reasoner 의 측정값만 VLM 추정으로 대체한다.

논문 Table III "w/o Reasoner" — "the Reasoner is removed and the same attributes are
inferred directly by the base VLM". 바뀌는 것은 *측정값의 출처* 하나뿐이다.

  유지 (시뮬레이터/검출기 관측값) : 물체 identity, center_mm, bbox_mm, edges 의 위상(near/on)
  대체 (VLM 이 이미지+이름만 보고 추정) : extents/footprint/height, cylinder_like, surface_rms,
      seal_patch_rms, material, density, mass, youngs, mu, 그리고 edge 의 gap_mm,
      컨테이너의 inner_bbox/wall/floor (원본 노드에 그 키가 있을 때만)
  동일 (Full 과 같은 코드) : EE 판정 규칙 `m1_scene.ee_rules.evaluate_ee`, 단항 술어,
      reachability, 그리고 M2 → G_k → M4 → M5 전부

추정값 기준 규칙에서 어떤 EE 도 통과하지 못하면 feasible_ee 는 빈 목록이다.
Full 과 같은 판정을 보존하며, 실행을 강제하기 위해 실패한 EE 를 통과시키지 않는다.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from tuj.m1_scene.ee_rules import evaluate_ee, reach_check
from tuj.m1_scene.relations import flat_face, region_clear, top_exposed
from tuj.m2_subgoal.rough import _image_part, make_llm_client, track_usage

MODE = "without_reasoner"
POLICY_VERSION = "vlm-prior-attributes-v3"
PositiveDimension = Annotated[float, Field(gt=0, allow_inf_nan=False)]

# 노드에서 VLM 추정으로 갈아끼우는 키 (geometry 는 통째로 재구성)
_REPLACED_NODE_KEYS = (
    "material", "density_kgm3", "mass_kg", "youngs_gpa", "mu", "confidence",
    "mass_range_kg", "materials_topk", "caption", "mass_source",
    "geometry_source", "intrinsic_source", "grounding_source", "ee", "predicates",
    "ee_fallback",
)
_CONTAINER_KEYS = ("inner_bbox_mm", "wall_mm", "floor_mm")


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectEstimate(_Record):
    id: str
    extents_mm: list[PositiveDimension] = Field(min_length=3, max_length=3)  # principal dimensions
    height_z_mm: PositiveDimension  # vertical height in the observed pose
    footprint_mm: list[PositiveDimension] = Field(min_length=2, max_length=2)  # tabletop footprint
    cylinder_like: bool
    surface_rms_mm: float = Field(ge=0)        # 상면 거칠기 RMS
    seal_patch_rms_mm: float = Field(ge=0)     # 흡착컵 크기 평면 패치 RMS (없으면 큰 값)
    material: str
    density_kgm3: float = Field(gt=0)
    mass_kg: float = Field(gt=0)
    youngs_gpa: float = Field(gt=0)
    mu: float = Field(gt=0, le=2.0)
    confidence: float = Field(ge=0, le=1)
    inner_bbox_mm: list[float] | None = None   # 컨테이너일 때만 [x, y, z]
    wall_mm: float | None = None
    floor_mm: float | None = None


class EdgeEstimate(_Record):
    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    gap_mm: float = Field(ge=0)


class SceneEstimate(_Record):
    objects: list[ObjectEstimate]
    edges: list[EdgeEstimate]


def normalize_ee_pool(robot_spec: dict) -> list[dict]:
    """run_m1 / gt.labels 와 동일: vac 의 flatness_tol_rms_mm → seal_rms_tol_mm."""
    pool = []
    for entry in robot_spec["ee_pool"]:
        e = copy.deepcopy(entry)
        if "seal_rms_tol_mm" not in e and "flatness_tol_rms_mm" in e:
            e["seal_rms_tol_mm"] = e["flatness_tol_rms_mm"]
        pool.append(e)
    return pool


def identity_view(m1: dict) -> dict:
    """VLM 에 보여줄 것: identity + 컨테이너 여부. 치수/질량/위치는 보여주지 않는다."""
    nodes = []
    for n in m1["nodes"]:
        item = {k: n[k] for k in ("id", "class", "source_name") if k in n}
        if any(k in n for k in _CONTAINER_KEYS):
            item["is_container"] = True
        nodes.append(item)
    edges = [
        {"from": e["from"], "to": e["to"], "type": e.get("type")}
        for e in m1.get("edges") or []
        if "gap_mm" in e
    ]
    return {"objects": nodes, "edges": edges}


class VLMPriorEstimator:
    """이미지 + 물체 이름 + 하드웨어 스펙만으로 물리/기하 속성을 추정한다."""

    def __init__(self, model: str, frame: Path, client=None, attempts: int = 3):
        if not Path(frame).is_file():
            raise ValueError("without_reasoner requires the matching scene frame.png")
        self.model, self.frame, self.client, self.attempts = model, Path(frame), client, attempts
        self.usage: dict = {}

    def estimate(self, view: dict, task_instruction: str) -> SceneEstimate:
        if self.client is None:
            self.client = make_llm_client()
        ids = [o["id"] for o in view["objects"]]
        containers = {o["id"] for o in view["objects"] if o.get("is_container")}
        edge_keys = {(e["from"], e["to"]) for e in view["edges"]}
        prompt = (
            "You are replacing a metric 3D perception module. From the image and object "
            "names ONLY, estimate the physical and geometric attributes of every listed "
            "object. No depth, point cloud, mass sensor or ground truth is available; use "
            "visual appearance and common knowledge of such objects. Units: millimetres, "
            "kilograms, kg/m^3, GPa. extents_mm contains the object's three principal "
            "dimensions in descending order; these are NOT world x/y/z axes. Separately "
            "estimate height_z_mm (vertical height above the support surface in the "
            "observed pose) and footprint_mm (the two horizontal dimensions on that "
            "surface). Never assume the shortest principal dimension is the height: an "
            "upright mug can be tallest vertically. surface_rms_mm is the roughness of "
            "the top surface; seal_patch_rms_mm is the plane-fit residual of the flattest "
            "patch of a suction-cup-sized region on top (use >= 5.0 if the top is rounded, "
            "concave or has no such flat patch). mu is the friction coefficient against a "
            "rubber gripper pad. For objects marked is_container also estimate inner_bbox_mm "
            "[x, y, z] interior size, wall_mm wall thickness and floor_mm floor thickness. "
            "For each listed edge estimate gap_mm, the free horizontal clearance between the "
            "two objects' nearest surfaces (0 if touching). Return ONLY a JSON object "
            '{"objects": [...], "edges": [...]} with exactly one record per listed object id '
            "and one per listed edge, following this schema: "
            + json.dumps(SceneEstimate.model_json_schema())
            + "\nTASK: " + task_instruction
            + "\nOBJECTS AND EDGES:\n" + json.dumps(view, ensure_ascii=False)
        )
        error = None
        for attempt in range(self.attempts):
            message = prompt + (f"\nFix the previous response: {error}" if error else "")
            t0 = time.monotonic()
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": message}, _image_part(str(self.frame))]}],
            )
            track_usage(self.usage, "vlm_prior_attributes", response, time.monotonic() - t0)
            try:
                content = response.choices[0].message.content or ""
                payload = json.loads(content[content.find("{"): content.rfind("}") + 1])
                est = TypeAdapter(SceneEstimate).validate_python(payload)
                got = [o.id for o in est.objects]
                if sorted(got) != sorted(ids) or len(got) != len(set(got)):
                    raise ValueError(f"objects must cover exactly {ids}")
                got_e = {(e.from_id, e.to_id) for e in est.edges}
                if got_e != edge_keys:
                    raise ValueError(f"edges must cover exactly {sorted(edge_keys)}")
                for o in est.objects:
                    if o.id in containers and (o.inner_bbox_mm is None or o.wall_mm is None
                                               or o.floor_mm is None):
                        raise ValueError(f"{o.id} is a container: inner_bbox_mm/wall_mm/floor_mm required")
                return est
            except (ValueError, TypeError) as exc:
                error = str(exc)[:600]
        raise ValueError(f"vlm_prior_attributes: invalid model response after {self.attempts} attempts: {error}")


def _geometry_from_estimate(o: ObjectEstimate, node: dict) -> dict:
    ext = sorted(float(v) for v in o.extents_mm)[::-1]
    length, width, _ = ext
    height = float(o.height_z_mm)
    geom = {
        "length_mm": round(length, 1),
        "diameter_mm": round(width, 1) if o.cylinder_like else None,
        "extents_mm": [round(v, 1) for v in ext],
        "height_z_mm": round(height, 1),
        "cylinder_like": bool(o.cylinder_like),
        "surface_rms_mm": round(float(o.surface_rms_mm), 3),
        "seal_patch_rms_mm": round(float(o.seal_patch_rms_mm), 3),
        "footprint_mm": [round(float(v), 1) for v in o.footprint_mm],
        "height_mm": round(height, 1),
    }
    # 관측값(시뮬레이터)은 Full 과 동일하게 유지
    if "center_mm" in node:
        geom["center"] = list(node["center_mm"])
    if "bbox_mm" in node:
        geom["aabb_size"] = list(node["bbox_mm"])
    return geom


def build_prior_scene(full_m1: dict, est: SceneEstimate, robot_spec: dict) -> dict:
    """Full M1 payload 에서 Reasoner 측정값만 VLM 추정으로 바꾼 새 m1 payload."""
    ee_pool = normalize_ee_pool(robot_spec)
    reach_mm = robot_spec.get("reach_mm")
    by_id = {o.id: o for o in est.objects}
    gaps = {(e.from_id, e.to_id): float(e.gap_mm) for e in est.edges}

    out = copy.deepcopy(full_m1)
    edges = []
    for e in out.get("edges") or []:
        e = dict(e)
        if "gap_mm" in e:
            e["gap_mm"] = round(gaps.get((e["from"], e["to"]), e["gap_mm"]), 1)
            e["gap_source"] = "vlm_prior"
        edges.append(e)
    out["edges"] = edges

    for node in out["nodes"]:
        o = by_id[node["id"]]
        for k in _REPLACED_NODE_KEYS:
            node.pop(k, None)
        geom = _geometry_from_estimate(o, node)
        node["geometry"] = geom
        node["material"] = o.material
        node["density_kgm3"] = round(float(o.density_kgm3), 1)
        node["mass_kg"] = round(float(o.mass_kg), 4)
        node["youngs_gpa"] = round(float(o.youngs_gpa), 3)
        node["mu"] = {"mu": round(float(o.mu), 3), "stage": 0, "material": o.material,
                      "rms_mm": geom["surface_rms_mm"]}
        node["confidence"] = round(float(o.confidence), 3)
        node["geometry_source"] = node["intrinsic_source"] = node["grounding_source"] = "vlm_prior"
        for k in _CONTAINER_KEYS:
            if k in node:
                v = getattr(o, k)
                node[k] = [round(float(x), 1) for x in v] if isinstance(v, list) else round(float(v), 1)
        intrinsic = {"geometry": geom, "mass_kg": node["mass_kg"], "mu": node["mu"]}
        verdicts = {e["ee_id"]: evaluate_ee(e, intrinsic) for e in ee_pool}
        # Preserve the common EE rules, including an empty feasible set.
        # A negative margin is diagnostic evidence, not permission to grasp.
        node["ee"] = verdicts
        if reach_mm is not None and "center_mm" in node:
            node["reachability"] = reach_check(reach_mm, node["center_mm"])
    for node in out["nodes"]:
        nid = node["id"]
        node["predicates"] = {
            "top_exposed": top_exposed(nid, out["edges"]),
            "clear": region_clear(nid, out["edges"]),
            "flat_face": flat_face(node["geometry"]),
        }
    out["ablation"] = {"mode": MODE, "policy_version": POLICY_VERSION,
                       "ee_fallback": {},
                       "ee_feasibility_policy": "shared_ee_rules_no_override",
                       "replaced": "geometry/material/density/mass/youngs/mu/ee/predicates/edge.gap_mm",
                       "kept_from_observation": "id/class/center_mm/bbox_mm/edge topology"}
    return out


def feasible_ee_table(m1: dict) -> dict[str, list[str]]:
    return {n["id"]: sorted(k for k, v in (n.get("ee") or {}).items() if v.get("feasible"))
            for n in m1["nodes"]}
