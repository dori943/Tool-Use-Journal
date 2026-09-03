# -*- coding: utf-8 -*-
"""M2 — Subgoal Decomposition & Constraint Extraction (진입점).

입력  자연어 task + M1 그래프 (m1_scene.serialize() 형태. Materializer 연동 시 원본 m1)
출력  M1Output — 필드는 전부 m2_ 접두 (모듈 간 필드 충돌 방지 합의):
      m2_subgoals        planning-level 서브골 + 상세 detail + 술어(expr, eval_by)
      m2_partial_order   DAG 하드 제약 (완전순서 아님 — 위상정렬·EE 최적화는 M4)
      m2_mutex           자원 배타 (M4가 순서를 짤 때 지켜야 하는 인터리브 규칙)
      m2_queries         M3 질의 사양 (queried_by = 술어 id → M6 역추적)
      m2_invariants      태스크 수준 유지 조건 (경계 검문·성공 판정)
정합  G_k에 실리는 M2 몫(subgoal_id, goal, role, predicates.expr)은
      run_m2_with_m3()가 Materializer를 호출하며 채운다. G_k 조립 주체는 M3.
"""
from __future__ import annotations

from .core import add_container_seal_pres, build_queries, decompose, invariants_for, partial_order
from .rough import TemplateRough


def run_m2(task: str, m1_serialized: dict, rough=None) -> dict:
    rough = rough or TemplateRough()
    subgoals = rough.generate(task, m1_serialized)

    all_details, all_edges, all_mutex, all_queries = [], [], [], []
    for s in subgoals:
        s["details"] = decompose(s)
    add_container_seal_pres(subgoals)          # 0903: 담기 ≺ 덮기 (서브골 간)
    for s in subgoals:
        all_details += s["details"]
        all_queries += build_queries(s, s["details"])
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
            c = q["m3_call"]
            for n in (c.get("node_id"), c.get("a"), c.get("b")):
                if n and n not in picked:
                    picked.append(n)
        s["object_ids"] = picked

    return {
        "task": task,
        "m2_subgoals": subgoals,
        "m2_partial_order": edges,
        "m2_mutex": mutex,
        "m2_queries": all_queries,
        "m2_invariants": invariants_for(subgoals),
        "m2_stats": {
            "n_subgoals": len(subgoals),
            "n_details": len(all_details),
            "n_edges": len(edges),
            "n_mutex": len(mutex),
            "n_m3_queries": len(all_queries),
            # 단계별 LLM 토큰 (0828 — Fig1 token 그래프, 호출 통합 A/B 근거)
            "llm_usage": getattr(rough, "usage", {}),
        },
    }


def run_m2_with_m3(task: str, m1: dict, ee_pool: list[dict], reach_mm: float,
                  rough=None, backend=None) -> tuple[dict, list[dict]]:
    """M2 실행 + M3(Materializer) 질의까지 수행해 서브골별 G_k를 채워 온다.

    m1는 build_m1() 원본(_points 포함)이어야 한다. 반환: (M1Output, [G_k ...])
    """
    from tuj.m3_grounding.materialize import Materializer, new_gk

    out = run_m2(task, {"nodes": [{k: v for k, v in n.items() if k != "_points"}
                                  for n in m1["nodes"]],
                        "edges": m1["edges"]}, rough=rough)
    mat = Materializer(m1, backend=backend)
    gks = []
    for s in out["m2_subgoals"]:
        gk = new_gk(s["subgoal_id"])
        gk["goal"] = s["goal"]                                 # ← M2
        gk["roles"] = {"target": s["target_ids"],              # ← M2
                       "container": s["container_id"],
                       "tool_candidates": s["tool_candidate_ids"]}
        gk["predicates"] = [                                   # ← M2 (status는 M3/M4 몫)
            {"id": p["id"], "expr": p["expr"], "eval_by": p["eval_by"]}
            for d in s["details"] for p in d["pre"]]
        for q in (x for x in out["m2_queries"] if x["subgoal_id"] == s["subgoal_id"]):
            c = q["m3_call"]
            if c["kind"] == "intrinsic":
                mat.query_intrinsic(gk, c["node_id"], q["queried_by"])
            elif c["kind"] == "relational":
                mat.query_relational(gk, c["a"], c["b"], c["relation"], q["queried_by"])
            elif c["kind"] == "ee":
                mat.query_ee(gk, c["node_id"], ee_pool, q["queried_by"], reach_mm=reach_mm)
            elif c["kind"] in ("batch", "swept_space"):
                # 0828 신규 질의 — 측정 모듈(m3_grounding)에 구현되기 전까지는 건너뛴다
                # (해당 술어는 unanswered로 남고, 분할 없이 기존 동작으로 폴백)
                fn = getattr(mat, "query_batch" if c["kind"] == "batch"
                             else "query_swept_space", None)
                if fn is not None:
                    fn(gk, c, q["queried_by"])
        gks.append(gk)
    return out, gks
