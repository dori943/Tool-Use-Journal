# -*- coding: utf-8 -*-
"""M1-2~4 — 상세 분해 · 조건(술어) 부착 · 부분순서 DAG.

원칙 (0817 확정):
  · EE-agnostic — 술어는 ?EE 를 자유변수로 갖는다. EE/도구의 '선택'은 하지 않는다.
  · 값을 내지 않는다 — "무엇이 중요한가"(술어)만 정하고, 값은 M2 질의로 얻는다.
  · 순서는 DAG(부분순서 하드 제약)까지만 — 완전순서는 M3의 위상정렬이 만든다.
    (M1이 완전순서를 내면 M3의 교체비용 최적화 자유도가 0이 된다)

상세 분해 (kind별 고정 구조 — 구조는 결정론, LLM 0회):
  relocate       d1 acquire(target) → d2 transport(target, container) → d3 place
  sweep_collect  d1 acquire(?tool) → d2 tool_act:sweep(targets→container)
                 → d2b transport(?tool, rest) → d3 place(?tool, rest)

술어 eval_by (누가 판정하나 — M5 역추적의 근거):
  m1     계획 상태만으로 판정 (hand_empty, holding, above, in)
  m2     측정 필요 → m1_queries로 나감 (reachable, fits, clear, ee_usable, tool_sweepable)
  motion 실행해봐야 앎 — 계획 시점 미결정으로 유보 (path_clear)
"""
from __future__ import annotations

# action_type 템플릿: (사전조건, establish, destroy). ?o=대상, ?r=목적지, ?t=도구
TEMPLATES = {
    "acquire": (["reachable(?o)", "top_exposed(?o)", "ee_usable(?EE, ?o)", "hand_empty"],
                ["holding(?o)"], ["hand_empty"]),
    "transport": (["holding(?o)", "path_clear(?o, ?r)"],
                  ["above(?o, ?r)"], []),
    "place": (["holding(?o)", "above(?o, ?r)", "fits(?o, ?r)", "clear(?r)"],
              ["in(?o, ?r)", "hand_empty"], ["holding(?o)", "above(?o, ?r)"]),
    "tool_act:sweep": (["holding(?t)", "tool_sweepable(?t, ?targets)", "path_clear(?t, ?r)"],
                       ["in(?targets, ?r)"], []),
}

EVAL_BY = {"hand_empty": "m1", "holding": "m1", "above": "m1", "in": "m1",
           "reachable": "m2", "top_exposed": "m2", "fits": "m2", "clear": "m2",
           "ee_usable": "m2", "tool_sweepable": "m2",
           "path_clear": "motion"}


def _cond(sid, i, kind, template, binding):
    expr = template
    # 긴 변수명부터 치환한다. (?t를 먼저 바꾸면 ?targets의 앞부분이 먹혀
    #  "?toolargets" 같은 깨진 문자열이 생긴다)
    for var, val in sorted(binding.items(), key=lambda kv: -len(kv[0])):
        expr = expr.replace(var, str(val))
    head = template.split("(")[0]
    return {"id": f"{sid}_{kind}{i}", "expr": expr,
            "head": head, "eval_by": EVAL_BY[head]}


def _detail(sid, action, binding, group, note=""):
    pre_t, est_t, des_t = TEMPLATES[action]
    return {
        "detail_id": sid,
        "action_type": action,
        "binding": binding,
        "group_id": group,
        "note": note,
        "pre": [_cond(sid, i, "p", t, binding) for i, t in enumerate(pre_t)],
        "establish": [_cond(sid, i, "e", t, binding) for i, t in enumerate(est_t)],
        "destroy": [_cond(sid, i, "d", t, binding) for i, t in enumerate(des_t)],
    }


def decompose(subgoal: dict) -> list[dict]:
    """planning-level 서브골 1개 → 상세 서브골 목록."""
    sid, kind = subgoal["subgoal_id"], subgoal["kind"]
    if kind == "relocate":
        o, r = subgoal["target_ids"][0], subgoal["container_id"]
        g = f"G_{o}"
        return [
            _detail(f"{sid}_d1", "acquire", {"?o": o}, g, f"{o} 확보"),
            _detail(f"{sid}_d2", "transport", {"?o": o, "?r": r}, g, f"{o}를 {r}로 운반"),
            _detail(f"{sid}_d3", "place", {"?o": o, "?r": r}, g, f"{o}를 {r}에 배치"),
        ]
    if kind == "sweep_collect":
        r = subgoal["container_id"]
        targets = "{" + ",".join(subgoal["target_ids"]) + "}"
        g = "G_tool"
        d2b = _detail(f"{sid}_d2b", "transport", {"?o": "?tool", "?r": "tool_rest"}, g, "도구를 거치 위치로 운반")
        # 도구 반환은 sweep 완료 후여야 한다. 이 상식을 조건으로 명시해야
        # partial_order가 causal link(d2→d2b)를 유도한다. 안 걸면 DAG가
        # "반환 후 sweep" 같은 순서도 허용해 버린다.
        d2b["pre"].append({"id": f"{sid}_d2b_p{len(d2b['pre'])}",
                           "expr": f"in({targets}, {r})", "head": "in", "eval_by": "m1"})
        return [
            _detail(f"{sid}_d1", "acquire", {"?o": "?tool"}, g, "sweep 도구 확보 (?tool: 후보 중 M3 선택)"),
            _detail(f"{sid}_d2", "tool_act:sweep", {"?t": "?tool", "?targets": targets, "?r": r},
                    g, f"블록들을 {r}로 sweep"),
            d2b,
            _detail(f"{sid}_d3", "place", {"?o": "?tool", "?r": "tool_rest"}, g, "도구 반환"),
        ]
    raise ValueError(f"모르는 kind: {kind}")


def partial_order(details: list[dict]) -> tuple[list[dict], list[dict]]:
    """causal link / threat → DAG 엣지 + mutex.

    엣지 근거를 why에 남긴다 (제약의 출처 설명 — 논문·M5용).
      causal_link  A가 만들어야(establish) B의 사전조건이 참이 된다 → A 먼저
      threat       B에 필요한 조건을 C가 깬다(destroy) → B 먼저
    같은 자원(hand_empty)을 두 그룹이 다투면 순서를 강제하지 않고 mutex로 넘긴다.
    """
    INITIALLY_TRUE = {"hand_empty"}    # 시작 상태에서 참 — 생산자가 필요 없다.
                                       # (그룹 간 재확립 요구는 아래 threat/mutex가 담당)
    edges, threat_edges, mutex = [], [], []

    for b in details:
        for p in b["pre"]:
            if p["eval_by"] == "motion":          # 계획 시점 미결정 — 순서 근거로 쓰지 않음
                continue
            if p["expr"] in INITIALLY_TRUE:
                continue
            producers = [a for a in details if a is not b and
                         any(e["expr"] == p["expr"] for e in a["establish"])]
            # 같은 그룹의 생산자만 하드 순서로 못 박는다.
            for a in producers:
                if a["group_id"] == b["group_id"]:
                    edges.append({"from": a["detail_id"], "to": b["detail_id"],
                                  "why": f"causal_link: {p['expr']}"})
            # 그룹 밖 생산자만 있으면(예: hand_empty) 배타 자원 — mutex
            if producers and not any(a["group_id"] == b["group_id"] for a in producers):
                pass  # 초기 참 조건(hand_empty)은 아래 threat 처리에서 다룬다

        # threat: b의 사전조건 p를 c가 깬다(destroy).
        #   같은 그룹  → b가 c보다 먼저 와야 한다는 하드 순서 엣지 (b→c)
        #              (예: sweep은 도구 반환보다 먼저 — 반환이 holding을 파괴)
        #   다른 그룹  → 순서를 강제하지 않고 mutex로 M3에 넘긴다
        for p in b["pre"]:
            if p["eval_by"] != "m1":
                continue
            for c in details:
                if c is b or not any(d["expr"] == p["expr"] for d in c["destroy"]):
                    continue
                if c["group_id"] == b["group_id"]:
                    threat_edges.append({"from": b["detail_id"], "to": c["detail_id"],
                                         "why": f"threat: {c['detail_id']}가 {p['expr']}를 파괴"})
                    continue
                key = tuple(sorted((b["group_id"], c["group_id"])))
                if not any(m["groups"] == list(key) and m["resource"] == p["expr"] for m in mutex):
                    mutex.append({
                        "resource": p["expr"],
                        "groups": list(key),
                        "rule": "두 그룹의 작업을 끼워 넣으려면 이 조건을 재확립하는 "
                                "detail(place의 establish: hand_empty)이 사이에 있어야 한다",
                    })

    # 중복 엣지 제거 (같은 엣지가 causal과 threat 양쪽에서 나오면 causal 근거를 남긴다)
    seen, uniq = set(), []
    for e in edges + threat_edges:
        k = (e["from"], e["to"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq, mutex


def build_queries(subgoal: dict, details: list[dict]) -> list[dict]:
    """eval_by == m2 인 술어 인스턴스 → Materializer 호출 사양.

    m2_call.kind ∈ intrinsic | relational | ee  (m2_grounding.Materializer의 질의 3종)
    queried_by에 술어 id를 실어 보낸다 — M5 역추적 계약.
    """
    q = []
    tool_ids = subgoal.get("tool_candidate_ids", [])
    for d in details:
        for p in d["pre"]:
            if p["eval_by"] != "m2":
                continue
            b = d["binding"]
            head = p["head"]
            if head in ("reachable", "top_exposed"):
                oid = b.get("?o")
                targets = tool_ids if oid == "?tool" else [oid]
                for t in targets:
                    q.append({"subgoal_id": subgoal["subgoal_id"], "queried_by": p["id"],
                              "m2_call": {"kind": "intrinsic", "node_id": t}})
            elif head == "ee_usable":
                oid = b.get("?o")
                targets = tool_ids if oid == "?tool" else [oid]
                for t in targets:
                    q.append({"subgoal_id": subgoal["subgoal_id"], "queried_by": p["id"],
                              "m2_call": {"kind": "ee", "node_id": t}})
            elif head in ("fits", "clear"):
                oid, rid = b.get("?o"), b.get("?r")
                if oid in (None, "?tool") or rid in (None, "tool_rest"):
                    continue                       # 도구 거치 위치는 측정 대상 아님
                rel = "fits_inside" if head == "fits" else "clearance"
                q.append({"subgoal_id": subgoal["subgoal_id"], "queried_by": p["id"],
                          "m2_call": {"kind": "relational", "a": oid, "b": rid,
                                      "relation": rel}})
            elif head == "tool_sweepable":
                for t in tool_ids:
                    q.append({"subgoal_id": subgoal["subgoal_id"], "queried_by": p["id"],
                              "m2_call": {"kind": "ee", "node_id": t}})
    # 동일 호출 중복 제거 (같은 노드 intrinsic 2회 등 — M2 캐시가 있지만 명세도 깨끗하게)
    seen, uniq = set(), []
    for x in q:
        k = (x["queried_by"], tuple(sorted(x["m2_call"].items())))
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def invariants_for(subgoals: list[dict]) -> list[dict]:
    """실행 내내 유지·최종 만족해야 하는 태스크 수준 조건 (경계 검문 + 성공 판정)."""
    inv = []
    for s in subgoals:
        if s["kind"] == "relocate":
            inv.append({"id": f"INV_{s['subgoal_id']}",
                        "expr": f"in({s['target_ids'][0]}, {s['container_id']})",
                        "check_at": "final"})
        elif s["kind"] == "sweep_collect":
            inv.append({"id": f"INV_{s['subgoal_id']}",
                        "expr": f"all_in({{{','.join(s['target_ids'])}}}, {s['container_id']})",
                        "check_at": "final"})
    return inv
