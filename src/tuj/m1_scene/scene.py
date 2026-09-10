"""M1·M3 통합 — 씬 전체 접지 패스 (에피소드당 1회).

build_m1(bbox 노드 + coarse 관계) 위에 씬의 **모든 객체**에 대해
  ① 기하: 점군에서 1회 (extents / footprint / 상면 RMS / 흡착 패치) — M3 중복 계산 제거
  ② 물성: M0 Object Knowledge (exact / BBox→C3 density) HIT 이면 재사용, miss 면 Full SiPhy
  ③ EE 판정: robot_spec ee_pool 각 EE 의 {feasible, margin, reason, checks}
  ④ 리치, 단항 술어: reachability / top_exposed / clear / flat_face
를 채우고, 새로 접지된 물성은 memory.json 에 적재해 다음 에피소드부터 재사용한다.

종전 M3 는 M2 가 고른 객체만 lazy 접지했다. 통합 후에는 씬의 모든 객체를 선접지하므로
첫 씬의 VLM 콜은 늘지만 memory 가 쌓일수록 0 으로 수렴하고, M2 는 질의 왕복 없이
m1.json 만 읽으면 된다. 집합·쌍 술어(batch/swept_space/fits/gap_accessible)는
서브골이 있어야 계산할 수 있으므로 relations 모듈 함수로 M2 가 직접 호출한다.
"""

from __future__ import annotations

from pathlib import Path

from tuj.m0_memory.object_knowledge import _as_size3

from .ee_rules import evaluate_ee, reach_check
from .grounding import (
    FrictionHead,
    MockBackend,
    apply_memory_hit_to_observation,
    ground_intrinsic,
)
from .relations import flat_face, region_clear, top_exposed


def ground_scene(
    m1: dict,
    *,
    backend=None,
    memory=None,
    ee_pool: list[dict] = (),
    reach_mm: float | None = None,
    crops_dir=None,
    friction=None,
    logger=None,
    source: str = "m1",
    density_infer=None,
    retrieval_debug: dict | None = None,
) -> dict:
    """m1 = build_m1() 결과 (nodes 에 _points 필요). 노드를 제자리에서 채운다.

    density_infer:
        M0 cross-task C3 callback (crop → DensityOnlyResult).

    retrieval_debug:
        optional dict filled per node_id with lookup_or_retrieve debug.

    Returns:
        {
            "memory_hits",
            "geom_refreshed",
            "grounded",
            "n_nodes",
            ...
        }
    """

    backend = backend or MockBackend()
    friction = friction or FrictionHead()
    log = logger or (lambda **kw: None)
    crops_dir = Path(crops_dir) if crops_dir else None

    edges = m1.get("edges", [])

    stats = {
        "memory_hits": 0,
        "geom_refreshed": 0,
        "grounded": 0,
        "n_nodes": len(m1["nodes"]),
    }

    cache: dict[str, dict] = {}

    if retrieval_debug is None:
        retrieval_debug = {}

    task_id = getattr(memory, "task_id", None) if memory is not None else None
    use_m0 = memory is not None and task_id is not None

    for node in m1["nodes"]:
        nid = node["id"]

        # 최신 main의 M1 → M5 geometry contract.
        # 현재 episode에서 관측된 center / bbox는 항상 최신값을 사용한다.
        observation_geometry = {
            "center": list(node["center_mm"]),
            "aabb_size": list(node["bbox_mm"]),
        }

        # cross-task C3 및 Full grounding 모두 동일 crop을 사용한다.
        crop = crops_dir / f"{nid}.png" if crops_dir else None
        crop = crop if (crop is not None and crop.exists()) else None

        intr = None
        how = None

        # -------------------------------------------------------------
        # M0 Object Knowledge retrieval
        #
        # 1) same-task exact HIT
        # 2) cross-task BBox candidate
        # 3) candidate 존재 시 C3 density-only
        # 4) density threshold HIT → Object Knowledge 재사용
        # -------------------------------------------------------------
        if use_m0:
            bbox = _as_size3(node.get("bbox_mm"))
            infer = density_infer or (lambda _crop: None)

            reused, debug = memory.lookup_or_retrieve(
                task_id,
                nid,
                bbox,
                crop,
                infer,
            )

            retrieval_debug[nid] = debug

            if reused is not None:
                # Intrinsic/material만 memory에서 재사용하고,
                # geometry·mass는 항상 현재 observation으로 재결합한다.
                # (과거 footprint/mass가 EE feasibility에 들어가면 안 됨)
                intr = apply_memory_hit_to_observation(node, reused)
                stats["memory_hits"] += 1
                stats["geom_refreshed"] += 1

                debug = retrieval_debug[nid]
                debug["geometry_source"] = "current_observation"
                debug["intrinsic_source"] = "memory"

                log(
                    module="m1",
                    event="memory_hit",
                    node=nid,
                    lookup_type=debug.get("lookup_type"),
                    geometry_source="current_observation",
                    intrinsic_source="memory",
                )

                how = "memory"

        # -------------------------------------------------------------
        # Memory MISS 또는 M0 미사용
        # → 현재 observation에서 Full physical grounding
        # -------------------------------------------------------------
        if intr is None:
            intr = ground_intrinsic(
                node,
                crop,
                backend,
                friction,
            )

            stats["grounded"] += 1

            if use_m0:
                debug = retrieval_debug.setdefault(nid, {})
                debug.update(
                    full_m3_called=True,
                    full_m3_skipped=False,
                )

            log(
                module="m1",
                event="grounded",
                node=nid,
                mu_stage=intr["mu"]["stage"],
            )

            how = "backend"

        # -------------------------------------------------------------
        # Memory update용 intrinsic cache
        # -------------------------------------------------------------
        cache[nid] = intr

        # material / density / mass / mu / geometry 등을 node에 반영
        node.update(intr)

        # intr["geometry"]는 reusable shape / surface 정보를 담고,
        # center / aabb_size는 현재 episode의 관측값을 사용한다.
        #
        # M5가 현재 관측 geometry를 소비하므로 observation 값이 우선한다.
        node["geometry"] = {
            **intr.get("geometry", {}),
            **observation_geometry,
        }

        node["grounding_source"] = how

        # -------------------------------------------------------------
        # End-effector evaluation
        # -------------------------------------------------------------
        node["ee"] = {
            e["ee_id"]: evaluate_ee(e, intr)
            for e in ee_pool
        }

        # -------------------------------------------------------------
        # Reachability
        # -------------------------------------------------------------
        if reach_mm is not None:
            node["reachability"] = reach_check(
                reach_mm,
                node["center_mm"],
            )

        # -------------------------------------------------------------
        # Unary predicates
        # -------------------------------------------------------------
        node["predicates"] = {
            "top_exposed": top_exposed(nid, edges),
            "clear": region_clear(nid, edges),
            "flat_face": flat_face(intr["geometry"]),
        }

    # -------------------------------------------------------------
    # 새로 grounding한 값 및 현재 cache를 Object Knowledge에 반영
    # MemoryStore가 기존 Failure-Recovery Experience는 보존한다.
    # -------------------------------------------------------------
    if memory is not None:
        stats["memory_update"] = memory.update(
            cache,
            source=source,
        )
        memory.save()

    stats["retrieval_debug"] = retrieval_debug

    return stats