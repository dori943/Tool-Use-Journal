# -*- coding: utf-8 -*-
"""M2-2~4 — 상세 분해 · 조건(술어) 부착 · 부분순서 DAG.

원칙 (0817 확정, 0821 개정):
  · EE-agnostic — 술어는 ?EE 를 자유변수로 갖는다. EE의 최종 '선택'은 하지 않는다.
    (객체 선택은 M2-1 2차 호출이 한다 — 도구 후보 선별까지. 어느 후보를 쓸지는 M4)
  · 값을 내지 않는다 — "무엇이 중요한가"(술어)만 정하고, 값은 M3 질의로 얻는다.
  · 순서는 DAG(부분순서 하드 제약)까지만 — 완전순서는 M4의 위상정렬이 만든다.
    (M2이 완전순서를 내면 M4의 교체비용 최적화 자유도가 0이 된다)

상세 분해 (kind별 고정 구조 — 구조는 결정론, LLM 0회):
  relocate       d1 acquire(target) → d2 transport(target, container) → d3 place
  sweep_collect  d1 acquire(?tool) → d2 tool_act:sweep(targets→container)
                 → d2b transport(?tool, rest) → d3 place(?tool, rest)
  flatten        sweep_collect과 같은 도구 골격, d2가 tool_act:flatten (C1_2)
  stack          층마다 acquire → transport → place_on, 아래층 완성 후 다음 층 (C2_2, C4_1)
  scoop_transfer sweep_collect과 같은 도구 골격, d2가 tool_act:scoop (C3_2, 양 개념 협의중)
  extract        도구 골격, d2가 tool_act:extract — 틈에서 빼내기까지만 (C4_2, 협의중)

술어 eval_by (누가 판정하나 — M6 역추적의 근거):
  m2     계획 상태만으로 판정 (hand_empty, holding, above, in)
  m3     측정 필요 → m2_queries로 나감 (reachable, fits, clear, ee_usable,
         batch_feasible, act_space_clear — 뒤 2종은 0828 신규: 그룹 동시 처리·실행 공간)
  motion 실행해봐야 앎 — 계획 시점 미결정으로 유보 (path_clear)
"""
from __future__ import annotations

import json

# action_type 템플릿: (사전조건, establish, destroy). ?o=대상, ?r=목적지, ?t=도구
TEMPLATES = {
    "acquire": (["reachable(?o)", "top_exposed(?o)", "ee_usable(?EE, ?o)",
                 "batch_feasible(?EE, ?o)", "hand_empty"],
                ["holding(?o)"], ["hand_empty"]),
    "transport": (["holding(?o)", "act_space_clear(?EE, ?o, ?r)", "path_clear(?o, ?r)"],
                  ["above(?o, ?r)"], []),
    "place": (["holding(?o)", "above(?o, ?r)", "fits(?o, ?r)", "clear(?r)"],
              ["in(?o, ?r)", "hand_empty"], ["holding(?o)", "above(?o, ?r)"]),
    "tool_act:sweep": (["holding(?t)", "batch_feasible(?t, ?targets)",
                        "act_space_clear(?t, ?targets, ?r)", "path_clear(?t, ?r)"],
                       ["in(?targets, ?r)"], []),
    # --- 0901 확장: 태스크 8종 커버 (C1_2, C2_2, C3_2, C4_1, C4_2) ---
    "tool_act:flatten": (["holding(?t)", "flat_face(?t)",
                          "act_space_clear(?t, ?targets, ?work)", "path_clear(?t, ?work)"],
                         ["flattened(?targets)"], []),
    "place_on": (["holding(?o)", "above(?o, ?base)", "top_exposed(?base)", "fits(?o, ?base)"],
                 ["on(?o, ?base)", "hand_empty"], ["holding(?o)", "above(?o, ?base)"]),
    "tool_act:scoop": (["holding(?t)", "batch_feasible(?t, ?targets)",
                        "act_space_clear(?t, ?targets, ?r)", "path_clear(?t, ?r)"],
                       ["in(?targets, ?r)"], []),
    "tool_act:extract": (["holding(?t)", "gap_accessible(?t, ?o)", "path_clear(?t, ?o)"],
                         ["extracted(?o)"], []),
}

EVAL_BY = {"hand_empty": "m2", "holding": "m2", "above": "m2", "in": "m2",
           "reachable": "m3", "top_exposed": "m3", "fits": "m3", "clear": "m3",
           "ee_usable": "m3", "batch_feasible": "m3", "act_space_clear": "m3",
           "path_clear": "motion",
           # 0901 확장분. flat_face(도구에 평평한 작업면이 있는가)와
           # gap_accessible(도구가 틈에 진입 가능한가)은 M3 신규 질의 — 도희 협의 필요.
           "on": "m2", "flattened": "m2", "extracted": "m2",
           "flat_face": "m3", "gap_accessible": "m3"}

# 도구를 전제로 하는 kind — 분해가 ?tool 자유변수를 쓰고, 검문이 도구 후보를 요구한다
TOOL_KINDS = {"sweep_collect", "flatten", "scoop_transfer", "extract"}


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
        ts = subgoal["target_ids"]
        # 0828: 다중 target 허용 — 몇 그룹으로 나눌지는 측정(batch 질의) 후 regroup이 정한다
        o = ts[0] if len(ts) == 1 else "{" + ",".join(ts) + "}"
        r = subgoal["container_id"]
        g = f"G_{sid}"
        return [
            _detail(f"{sid}_d1", "acquire", {"?o": o}, g, f"{o} 확보"),
            _detail(f"{sid}_d2", "transport", {"?o": o, "?r": r}, g, f"{o}를 {r}로 운반"),
            _detail(f"{sid}_d3", "place", {"?o": o, "?r": r}, g, f"{o}를 {r}에 배치"),
        ]
    if kind == "sweep_collect":
        r = subgoal["container_id"]
        targets = "{" + ",".join(subgoal["target_ids"]) + "}"
        g = f"G_{sid}_tool"      # 서브골별 그룹. 전역 "G_tool"로 두면 sweep 서브골이
                                 # 여럿일 때 서로 같은 그룹으로 묶여 threat가 양방향
                                 # 하드 엣지가 되고 DAG에 사이클이 생긴다.
        d2b = _detail(f"{sid}_d2b", "transport", {"?o": "?tool", "?r": "tool_rest"}, g, "도구를 거치 위치로 운반")
        # 도구 반환은 sweep 완료 후여야 한다. 이 상식을 조건으로 명시해야
        # partial_order가 causal link(d2→d2b)를 유도한다. 안 걸면 DAG가
        # "반환 후 sweep" 같은 순서도 허용해 버린다.
        d2b["pre"].append({"id": f"{sid}_d2b_p{len(d2b['pre'])}",
                           "expr": f"in({targets}, {r})", "head": "in", "eval_by": "m2"})
        return [
            _detail(f"{sid}_d1", "acquire", {"?o": "?tool"}, g, "sweep 도구 확보 (?tool: 후보 중 M4 선택)"),
            _detail(f"{sid}_d2", "tool_act:sweep", {"?t": "?tool", "?targets": targets, "?r": r},
                    g, f"블록들을 {r}로 sweep"),
            d2b,
            _detail(f"{sid}_d3", "place", {"?o": "?tool", "?r": "tool_rest"}, g, "도구 반환"),
        ]
    if kind == "flatten":
        # C1_2 (반죽 펴기). sweep_collect와 같은 도구 골격: 확보 → 도구 행위 → 거치 → 반납.
        targets = "{" + ",".join(subgoal["target_ids"]) + "}"
        work = subgoal.get("container_id") or targets   # 별도 작업 영역이 없으면 대상 자리에서 수행
        g = f"G_{sid}_tool"
        d2b = _detail(f"{sid}_d2b", "transport", {"?o": "?tool", "?r": "tool_rest"}, g, "도구를 거치 위치로 운반")
        d2b["pre"].append({"id": f"{sid}_d2b_p{len(d2b['pre'])}",
                           "expr": f"flattened({targets})", "head": "flattened", "eval_by": "m2"})
        return [
            _detail(f"{sid}_d1", "acquire", {"?o": "?tool"}, g, "flatten 도구 확보 (?tool: 후보 중 M4 선택)"),
            _detail(f"{sid}_d2", "tool_act:flatten", {"?t": "?tool", "?targets": targets, "?work": work},
                    g, f"{targets}를 평평하게 편다"),
            d2b,
            _detail(f"{sid}_d3", "place", {"?o": "?tool", "?r": "tool_rest"}, g, "도구 반환"),
        ]
    if kind == "stack":
        # C2_2 (샌드위치) / C4_1 (뚜껑 덮기). target_ids 순서 = 쌓는 순서 (아래 → 위).
        # container_id가 있으면 첫 받침(접시 등), 없으면 첫 대상이 바닥이 된다.
        ts = list(subgoal["target_ids"])
        base = subgoal.get("container_id")
        if base is None:
            base, ts = ts[0], ts[1:]
        g = f"G_{sid}"
        out, prev = [], None                        # prev = (직전에 올린 물체, 그 받침)
        for i, o in enumerate(ts, 1):
            d1 = _detail(f"{sid}_s{i}a", "acquire", {"?o": o}, g, f"{o} 확보")
            if prev:
                # 아래층이 완성된 뒤에만 다음 층 확보 — sweep의 d2b와 같은 수법.
                # 이 조건이 직전 place_on의 establish(on)와 causal link를 만들어
                # 층 순서(아래→위)를 하드 엣지로 못 박는다.
                d1["pre"].append({"id": f"{sid}_s{i}a_p{len(d1['pre'])}",
                                  "expr": f"on({prev[0]}, {prev[1]})", "head": "on", "eval_by": "m2"})
            out += [
                d1,
                _detail(f"{sid}_s{i}b", "transport", {"?o": o, "?r": base}, g, f"{o}를 {base} 위로 운반"),
                _detail(f"{sid}_s{i}c", "place_on", {"?o": o, "?base": base}, g, f"{o}를 {base} 위에 놓기"),
            ]
            prev, base = (o, base), o               # 다음 층의 받침은 방금 올린 물체
        return out
    if kind == "scoop_transfer":
        # C3_2 (파스타 2인분). sweep_collect와 같은 도구 골격, d2만 tool_act:scoop.
        # [협의] '2인분' 같은 양 개념의 측정·판정 방식은 미정 — 현재는 대상 집합
        # 전체 이동으로 근사한다 (양 판정이 정해지면 establish 술어를 교체).
        r = subgoal["container_id"]
        targets = "{" + ",".join(subgoal["target_ids"]) + "}"
        g = f"G_{sid}_tool"
        d2b = _detail(f"{sid}_d2b", "transport", {"?o": "?tool", "?r": "tool_rest"}, g, "도구를 거치 위치로 운반")
        d2b["pre"].append({"id": f"{sid}_d2b_p{len(d2b['pre'])}",
                           "expr": f"in({targets}, {r})", "head": "in", "eval_by": "m2"})
        return [
            _detail(f"{sid}_d1", "acquire", {"?o": "?tool"}, g, "scoop 도구 확보 (?tool: 후보 중 M4 선택)"),
            _detail(f"{sid}_d2", "tool_act:scoop", {"?t": "?tool", "?targets": targets, "?r": r},
                    g, f"{targets}를 {r}로 떠서 옮긴다"),
            d2b,
            _detail(f"{sid}_d3", "place", {"?o": "?tool", "?r": "tool_rest"}, g, "도구 반환"),
        ]
    if kind == "extract":
        # C4_2 (틈새 카드). [협의] gap_accessible(도구가 틈에 진입 가능한가) 판정은
        # 틈새 기하 질의가 필요 — M3(도희) 협의 후 확정. 빼낸 뒤의 목적지 배치는
        # 별도 relocate 서브골이 담당한다 (extract는 '꺼내기'까지만).
        ts = subgoal["target_ids"]
        o = ts[0] if len(ts) == 1 else "{" + ",".join(ts) + "}"
        g = f"G_{sid}_tool"
        d2b = _detail(f"{sid}_d2b", "transport", {"?o": "?tool", "?r": "tool_rest"}, g, "도구를 거치 위치로 운반")
        d2b["pre"].append({"id": f"{sid}_d2b_p{len(d2b['pre'])}",
                           "expr": f"extracted({o})", "head": "extracted", "eval_by": "m2"})
        return [
            _detail(f"{sid}_d1", "acquire", {"?o": "?tool"}, g, "extract 도구 확보 (?tool: 후보 중 M4 선택)"),
            _detail(f"{sid}_d2", "tool_act:extract", {"?t": "?tool", "?o": o}, g, f"{o}를 틈에서 빼낸다"),
            d2b,
            _detail(f"{sid}_d3", "place", {"?o": "?tool", "?r": "tool_rest"}, g, "도구 반환"),
        ]
    raise ValueError(f"모르는 kind: {kind}")


# 0903: 컨테이너에 "담는" 서브골과 같은 컨테이너 "위에 놓는"(덮는) 서브골 사이의 순서.
# c4_2에서 M2가 "긴 물건 담기"(relocate → 상자)와 "뚜껑 덮기"(stack, container_id=상자)를
# 별개 서브골로 내자 서브골 간 제약이 없어 M4가 뚜껑을 먼저 덮었다.
# 물리 규칙: 컨테이너를 덮고 나면 안에 넣을 수 없다 → 담기 ≺ 덮기.
# 구현은 sweep의 d2b, stack의 층 순서와 같은 수법: 덮는 서브골의 첫 상세(acquire) pre에
# 담는 서브골이 establish하는 in(x, C)를 그대로 넣어 partial_order가 causal link로 잡게 한다.
# 분할 뒤에는 자식들이 in(원소, C)를 각각 establish하므로 다시 호출한다(auto 태그로 교체).
INSIDE_KINDS = {"relocate", "sweep_collect", "scoop_transfer"}


def add_container_seal_pres(subgoals: list[dict]) -> list[str]:
    logs = []
    for b in subgoals:
        if b.get("kind") != "stack" or not b.get("details"):
            continue
        # 받침: container_id가 있으면 그것, 없으면 첫 target (decompose의 stack 규칙과 동일)
        C = b.get("container_id") or (b.get("target_ids") or [None])[0]
        if not C:
            continue
        first = b["details"][0]
        first["pre"] = [q for q in first["pre"] if not q.get("auto")]
        added = []
        for a in subgoals:
            if a is b or a.get("kind") not in INSIDE_KINDS or a.get("container_id") != C:
                continue
            for d in a.get("details", []):
                for e in d.get("establish", []):
                    if e.get("head") == "in":
                        first["pre"].append({"id": f"{first['detail_id']}_p{len(first['pre'])}",
                                             "expr": e["expr"], "head": "in", "eval_by": "m2",
                                             "auto": "container_seal"})
                        added.append(a["subgoal_id"])
        if added:
            logs.append(f"  [순서] {b['subgoal_id']}({C} 덮기)는 "
                        f"{sorted(set(added))}(담기) 뒤에 — 사전조건 {len(added)}건 부착")
    return logs


def partial_order(details: list[dict]) -> tuple[list[dict], list[dict]]:
    """causal link / threat → DAG 엣지 + mutex.

    엣지 근거를 why에 남긴다 (제약의 출처 설명 — 논문·M6용).
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
            # 예외(0903): 컨테이너 담기 ≺ 덮기 사전조건(auto=container_seal)은 서브골(그룹)을
            # 가로지르는 물리 제약이라 그룹이 달라도 하드 엣지로 둔다.
            for a in producers:
                if a["group_id"] == b["group_id"] or p.get("auto") == "container_seal":
                    edges.append({"from": a["detail_id"], "to": b["detail_id"],
                                  "why": f"causal_link: {p['expr']}"})
            # 그룹 밖 생산자만 있으면(예: hand_empty) 배타 자원 — mutex
            if producers and not any(a["group_id"] == b["group_id"] for a in producers):
                pass  # 초기 참 조건(hand_empty)은 아래 threat 처리에서 다룬다

        # threat: b의 사전조건 p를 c가 깬다(destroy).
        #   같은 그룹  → b가 c보다 먼저 와야 한다는 하드 순서 엣지 (b→c)
        #              (예: sweep은 도구 반환보다 먼저 — 반환이 holding을 파괴)
        #   다른 그룹  → 순서를 강제하지 않고 mutex로 M4에 넘긴다
        for p in b["pre"]:
            if p["eval_by"] != "m2":
                continue
            for c in details:
                if c is b or not any(d["expr"] == p["expr"] for d in c["destroy"]):
                    continue
                if c["group_id"] == b["group_id"]:
                    # 0901: b와 c가 같은 자원을 둘 다 소비·파괴하면(예: stack 한 그룹의
                    # acquire 여럿이 hand_empty를 다툼) 방향 정보가 없다 — 양방향 엣지는
                    # 사이클이 되므로 하드 엣지를 만들지 않는다. 순서는 causal link가
                    # 정한다 (stack은 다음 층 acquire의 on(...) 사전조건이 그 역할).
                    symmetric = (any(d["expr"] == p["expr"] for d in b["destroy"]) and
                                 any(q["expr"] == p["expr"] for q in c["pre"]))
                    if not symmetric:
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


def _set_members(v) -> list[str]:
    """"{a,b,c}" 집합 표기 → 원소 리스트. 단일 id는 1원소 리스트."""
    if v is None:
        return []
    v = str(v)
    if v.startswith("{") and v.endswith("}"):
        return [x for x in v[1:-1].split(",") if x]
    return [v]


def build_queries(subgoal: dict, details: list[dict]) -> list[dict]:
    """eval_by == m3 인 술어 인스턴스 → Materializer 호출 사양.

    m3_call.kind ∈ intrinsic | relational | ee  (m3_grounding.Materializer의 질의 3종)
    queried_by에 술어 id를 실어 보낸다 — M6 역추적 계약.
    """
    q = []
    tool_ids = subgoal.get("tool_candidate_ids", [])
    for d in details:
        for p in d["pre"]:
            if p["eval_by"] != "m3":
                continue
            b = d["binding"]
            head = p["head"]
            if head in ("reachable", "top_exposed"):
                oid = b.get("?o")
                # 집합 표기 "{a,b}"는 원소별로 펼쳐 질의 (0828 — C2_1에서 발견)
                targets = tool_ids if oid == "?tool" else _set_members(oid)
                for t in targets:
                    q.append({"subgoal_id": subgoal["subgoal_id"], "queried_by": p["id"],
                              "m3_call": {"kind": "intrinsic", "node_id": t}})
            elif head == "ee_usable":
                oid = b.get("?o")
                targets = tool_ids if oid == "?tool" else _set_members(oid)
                for t in targets:
                    q.append({"subgoal_id": subgoal["subgoal_id"], "queried_by": p["id"],
                              "m3_call": {"kind": "ee", "node_id": t}})
            elif head in ("fits", "clear"):
                oid, rid = b.get("?o"), b.get("?r")
                if oid in (None, "?tool") or rid in (None, "tool_rest"):
                    continue                       # 도구 거치 위치는 측정 대상 아님
                rel = "fits_inside" if head == "fits" else "clearance"
                for t in _set_members(oid):          # 집합이면 원소별 관계 질의
                    q.append({"subgoal_id": subgoal["subgoal_id"], "queried_by": p["id"],
                              "m3_call": {"kind": "relational", "a": t, "b": rid,
                                          "relation": rel}})
            elif head in ("batch_feasible", "act_space_clear"):
                # 0828 신규 질의. 물체 1개짜리는 묶기 판정이 필요 없어 질의를 내지 않는다
                # (not_queried — tool_rest와 같은 취급). 액션 주체(actor):
                # 객체를 쓰는 서브골이면 후보 객체마다 1회, 아니면 EE 풀 기준 1회.
                members = _set_members(b.get("?targets") or b.get("?o"))
                if "?tool" in members:
                    continue
                # 물체 1개면 묶기 판정 불필요 → 질의 생략 (not_queried).
                # 단 분할된 자식 서브골은 1개여도 발행한다 — 측정 모듈이 서브골별
                # 서브그래프를 질의 기준으로 조립하므로, 재검증 겸 자기 몫을 남긴다.
                if len(members) < 2 and "split_from" not in subgoal:
                    continue
                actors = ([{"type": "object", "id": t} for t in tool_ids]
                          if tool_ids else [{"type": "ee_pool"}])
                call = {"kind": "batch" if head == "batch_feasible" else "swept_space",
                        "action_type": subgoal["kind"], "member_ids": members}
                if head == "act_space_clear":
                    rid = b.get("?r")
                    if rid in (None, "tool_rest"):
                        continue
                    call["to"] = rid
                for a in actors:
                    q.append({"subgoal_id": subgoal["subgoal_id"], "queried_by": p["id"],
                              "m3_call": {**call, "actor": a}})
    # 동일 호출 중복 제거 (같은 노드 intrinsic 2회 등 — M3 캐시가 있지만 명세도 깨끗하게)
    seen, uniq = set(), []
    for x in q:
        k = (x["queried_by"], json.dumps(x["m3_call"], sort_keys=True, ensure_ascii=False))
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def invariants_for(subgoals: list[dict]) -> list[dict]:
    """실행 내내 유지·최종 만족해야 하는 태스크 수준 조건 (경계 검문 + 성공 판정)."""
    inv = []
    for s in subgoals:
        if s["kind"] == "relocate":
            ts = s["target_ids"]
            expr = (f"in({ts[0]}, {s['container_id']})" if len(ts) == 1
                    else f"all_in({{{','.join(ts)}}}, {s['container_id']})")
            inv.append({"id": f"INV_{s['subgoal_id']}", "expr": expr,
                        "check_at": "final"})
        elif s["kind"] in ("sweep_collect", "scoop_transfer"):
            inv.append({"id": f"INV_{s['subgoal_id']}",
                        "expr": f"all_in({{{','.join(s['target_ids'])}}}, {s['container_id']})",
                        "check_at": "final"})
        elif s["kind"] == "flatten":
            inv.append({"id": f"INV_{s['subgoal_id']}",
                        "expr": f"flattened({{{','.join(s['target_ids'])}}})",
                        "check_at": "final"})
        elif s["kind"] == "stack":
            ts, base = list(s["target_ids"]), s.get("container_id")
            if base is None:
                base, ts = ts[0], ts[1:]
            for j, o in enumerate(ts, 1):           # 층마다 제 받침 위에 있는가
                inv.append({"id": f"INV_{s['subgoal_id']}_{j}",
                            "expr": f"on({o}, {base})", "check_at": "final"})
                base = o
        elif s["kind"] == "extract":
            ts = s["target_ids"]
            o = ts[0] if len(ts) == 1 else "{" + ",".join(ts) + "}"
            inv.append({"id": f"INV_{s['subgoal_id']}",
                        "expr": f"extracted({o})", "check_at": "final"})
    return inv
