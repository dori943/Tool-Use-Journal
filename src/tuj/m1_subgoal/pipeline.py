# -*- coding: utf-8 -*-
"""M1 — Subgoal Decomposition & Constraint Extraction (진입점).

입력  자연어 task + M0 그래프 (m0_scene.serialize() 형태. Materializer 연동 시 원본 m0)
출력  M1Output — 필드는 전부 m1_ 접두 (모듈 간 필드 충돌 방지 합의):
      m1_subgoals        planning-level 서브골 + 상세 detail + 술어(expr, eval_by)
      m1_partial_order   DAG 하드 제약 (완전순서 아님 — 위상정렬·EE 최적화는 M3)
      m1_mutex           자원 배타 (M3가 순서를 짤 때 지켜야 하는 인터리브 규칙)
      m1_queries         M2 질의 사양 (queried_by = 술어 id → M5 역추적)
      m1_invariants      태스크 수준 유지 조건 (경계 검문·성공 판정)
정합  G_k에 실리는 M1 몫(subgoal_id, goal, role, predicates.expr)은
      run_m1_with_m2()가 Materializer를 호출하며 채운다. G_k 조립 주체는 M2.
"""
from __future__ import annotations

from .core import build_queries, decompose, invariants_for, partial_order
from .rough import TemplateRough


def run_m1(task: str, m0_serialized: dict, rough=None) -> dict:
    rough = rough or TemplateRough()
    subgoals = rough.generate(task, m0_serialized)

    all_details, all_edges, all_mutex, all_queries = [], [], [], []
    for s in subgoals:
        details = decompose(s)
        s["details"] = details
        all_details += details
        all_queries += build_queries(s, details)
    edges, mutex = partial_order(all_details)

    # 서브골별 객체 선택(object_ids)은 LLMRough 2차 호출이 담당한다 (0821 결정).
    # 선택 결과가 없는 경로(TemplateRough 테스트)만 질의 기반으로 파생해 채운다.
    for s in subgoals:
        if "object_ids" in s:
            continue
        picked = []
        for q in all_queries:
            if q["subgoal_id"] != s["subgoal_id"]:
                continue
            c = q["m2_call"]
            for n in (c.get("node_id"), c.get("a"), c.get("b")):
                if n and n not in picked:
                    picked.append(n)
        s["object_ids"] = picked

    return {
        "task": task,
        "m1_subgoals": subgoals,
        "m1_partial_order": edges,
        "m1_mutex": mutex,
        "m1_queries": all_queries,
        "m1_invariants": invariants_for(subgoals),
        "m1_stats": {
            "n_subgoals": len(subgoals),
            "n_details": len(all_details),
            "n_edges": len(edges),
            "n_mutex": len(mutex),
            "n_m2_queries": len(all_queries),
        },
    }


def run_m1_with_m2(task: str, m0: dict, ee_pool: list[dict], reach_mm: float,
                  rough=None, backend=None) -> tuple[dict, list[dict]]:
    """M1 실행 + M2(Materializer) 질의까지 수행해 서브골별 G_k를 채워 온다.

    m0는 build_m0() 원본(_points 포함)이어야 한다. 반환: (M1Output, [G_k ...])
    """
    from tuj.m2_grounding.materialize import Materializer, new_gk

    out = run_m1(task, {"nodes": [{k: v for k, v in n.items() if k != "_points"}
                                  for n in m0["nodes"]],
                        "edges": m0["edges"]}, rough=rough)
    mat = Materializer(m0, backend=backend)
    gks = []
    for s in out["m1_subgoals"]:
        gk = new_gk(s["subgoal_id"])
        gk["goal"] = s["goal"]                                 # ← M1
        gk["roles"] = {"target": s["target_ids"],              # ← M1
                       "container": s["container_id"],
                       "tool_candidates": s["tool_candidate_ids"]}
        gk["predicates"] = [                                   # ← M1 (status는 M2/M3 몫)
            {"id": p["id"], "expr": p["expr"], "eval_by": p["eval_by"]}
            for d in s["details"] for p in d["pre"]]
        for q in (x for x in out["m1_queries"] if x["subgoal_id"] == s["subgoal_id"]):
            c = q["m2_call"]
            if c["kind"] == "intrinsic":
                mat.query_intrinsic(gk, c["node_id"], q["queried_by"])
            elif c["kind"] == "relational":
                mat.query_relational(gk, c["a"], c["b"], c["relation"], q["queried_by"])
            elif c["kind"] == "ee":
                mat.query_ee(gk, c["node_id"], ee_pool, q["queried_by"], reach_mm=reach_mm)
        gks.append(gk)
    return out, gks
