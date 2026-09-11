# -*- coding: utf-8 -*-
"""M2 — Subgoal Decomposition & Constraint Extraction (진입점).

입력  자연어 task + M1 그래프 (0908 통합 뒤에는 접지값이 노드에 실려 온다)
출력  M1Output — 필드는 전부 m2_ 접두 (모듈 간 필드 충돌 방지 합의):
      m2_subgoals        planning-level 서브골 + 상세 detail + 술어(expr, eval_by)
      m2_partial_order   DAG 하드 제약 (완전순서 아님 — 위상정렬·EE 최적화는 M4)
      m2_mutex           자원 배타 (M4가 순서를 짤 때 지켜야 하는 인터리브 규칙)
      m2_invariants      태스크 수준 유지 조건 (경계 검문·성공 판정)

0908 파이프라인 재구조: [M1 & M3] → M2 → M4 → M5 한 방향.
M3로 나가던 질의 목록(m2_queries)은 없어졌다. 술어 판정은 ingest.apply_grounding이
M1 접지값과 관계 함수(ground.py)로 직접 수행하고, G_k 조립도 M2(gk.py)가 한다.
"""
from __future__ import annotations

from .core import (add_container_seal_pres, add_uncover_effects, decompose,
                   prune_tool_candidates,
                   invariants_for, partial_order, plan_evaluations)
from .rough import TemplateRough


def object_ids_of(subgoal: dict, m1_serialized: dict | None = None) -> list[str]:
    """서브골이 건드리는 노드 목록 — 판정 사양이 지목한 노드에서 뽑는다."""
    picked = []
    for q in plan_evaluations(subgoal, subgoal.get("details", []), m1=m1_serialized):
        c = q["call"]
        for n in (c.get("node_id"), c.get("a"), c.get("b"),
                  c.get("tool_id"), c.get("target_id")):
            if n and n not in picked:
                picked.append(n)
        for n in c.get("member_ids", []):
            if n not in picked:
                picked.append(n)
    return picked


def run_m2(task: str, m1_serialized: dict, rough=None) -> dict:
    rough = rough or TemplateRough()
    subgoals = rough.generate(task, m1_serialized)

    # 0911: 접지 질의(plan_evaluations)가 후보를 batch actor 로 쓰므로, 후보
    # 정리는 반드시 decompose 이전에 끝나야 한다.
    for _line in prune_tool_candidates(subgoals):
        print(_line)

    all_details = []
    for s in subgoals:
        s["details"] = decompose(s)
    add_container_seal_pres(subgoals)          # 0903: 담기 ≺ 덮기 (서브골 간)
    add_uncover_effects(subgoals, m1_serialized)   # 0908: 치우면 드러남을 효과로
    for s in subgoals:
        all_details += s["details"]
    edges, mutex = partial_order(all_details)

    # 서브골별 객체 선택(object_ids)은 LLMRough 2차 호출이 담당한다 (0821 결정).
    # 선택 결과가 없는 경로(TemplateRough 테스트)만 판정 사양에서 파생해 채운다.
    for s in subgoals:
        if "object_ids" not in s:
            s["object_ids"] = object_ids_of(s, m1_serialized)

    return {
        "task": task,
        "m2_subgoals": subgoals,
        "m2_partial_order": edges,
        "m2_mutex": mutex,
        "m2_invariants": invariants_for(subgoals),
        "m2_stats": {
            "n_subgoals": len(subgoals),
            "n_details": len(all_details),
            "n_edges": len(edges),
            "n_mutex": len(mutex),
            # 단계별 LLM 토큰 (0828 — Fig1 token 그래프, 호출 통합 A/B 근거)
            "llm_usage": getattr(rough, "usage", {}),
        },
    }
