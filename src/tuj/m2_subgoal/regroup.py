# -*- coding: utf-8 -*-
"""M2-6 — 측정 반영 후 서브골 분할 (0828 결정).

batch 질의 응답의 partition(그룹 구성)대로, 서브골 하나를 그룹별 서브골 여러 개로
나눈다. apply_m3(판정·객체 확정)가 partition_plan을 채운 뒤 호출한다.

분할 시 순서 규칙 (0828 문서 "서브골 분할 시 순서 규칙"):
  · 반복 부분(액션)만 그룹 수만큼 복제한다.
  · 객체 확보(acquire)는 첫 서브골에만, 반환(transport/place)은 마지막 서브골에만.
    전부 복제하면 같은 객체를 집었다 놨다 그룹 수만큼 반복하는 계획이 된다.
  · 분할된 서브골들은 group_id를 공유한다 — 같은 객체를 계속 쓰는 하나의 흐름.
    (0821에 sweep 다수가 전역 그룹을 공유해 사이클이 난 것과 다르다. 그때는 서로 다른
     서브골이 하나의 그룹으로 묶인 것이고, 여기는 한 서브골이 나뉜 조각들이다.)
  · 분할된 서브골 사이 순서는 강제하지 않는다 — 순서 최적화는 태스크 플래너 몫.
  · 반환(d2b)의 사전조건은 그룹별로 하나씩 붙인다. partial_order가 조건식 문자열의
    완전 일치로 생산자를 찾으므로, 전체 집합 하나로 걸면 어느 액션의 establish와도
    매칭되지 않아 "다 처리하기 전에 반환" 순서가 허용되어 버린다.
  · 분할 후 DAG 사이클을 검증한다.
"""
from __future__ import annotations

from .core import _detail, build_queries, decompose, invariants_for, partial_order


def _cycle_check(details: list[dict], edges: list[dict]) -> list[str]:
    """Kahn 위상정렬로 사이클 검출. 사이클에 걸린 detail_id 목록 반환 (없으면 빈 리스트)."""
    ids = [d["detail_id"] for d in details]
    indeg = {i: 0 for i in ids}
    adj: dict[str, list[str]] = {i: [] for i in ids}
    for e in edges:
        if e["from"] in adj and e["to"] in indeg:
            adj[e["from"]].append(e["to"])
            indeg[e["to"]] += 1
    queue = [i for i in ids if indeg[i] == 0]
    seen = 0
    while queue:
        n = queue.pop()
        seen += 1
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return [] if seen == len(ids) else [i for i in ids if indeg[i] > 0]


def _child(s: dict, sid: str, k: int, total: int, members: list[str], verb: str) -> dict:
    """분할 서브골 뼈대: 원 서브골 필드 승계 + 분할 추적 필드(split_from, split_index)."""
    child = {kk: vv for kk, vv in s.items() if kk not in ("details", "partition_plan")}
    child.update({
        "subgoal_id": f"{sid}_s{k}",
        "target_ids": list(members),
        "goal": f"대상 {len(members)}개를 {s['container_id']}로 {verb} ({k}/{total})",
        "split_from": sid, "split_index": k,
    })
    # 객체는 이미 확정됨 — 자식의 후보 목록은 확정된 것 하나로 축소
    # (재생성되는 질의가 탈락 후보까지 다시 묻지 않게)
    if s.get("selected_tool_id"):
        child["tool_candidate_ids"] = [s["selected_tool_id"]]
    # 신뢰도 갱신은 분할 전에 돌므로(LLM 호출 추가 없음 — 0825 랩미팅의 호출 수 지적),
    # 그 시점의 "batch_feasible 불충족"은 분할로 해소된 상태다. 자식 기록에서 정리해 둔다.
    conf = s.get("confidence")
    if isinstance(conf, dict) and isinstance(conf.get("after"), dict):
        after = dict(conf["after"])
        after["uncertain_about"] = [u for u in after.get("uncertain_about", [])
                                    if "batch_feasible" not in u]
        after["post_split_note"] = "batch_feasible 불충족은 그룹 분할로 해소됨"
        child["confidence"] = {**conf, "after": after}
    return child


def _split_sweep(s: dict, part: list[list[str]]) -> list[dict]:
    """sweep_collect 분할 — 확보는 첫 서브골, 반환은 마지막 서브골에만."""
    sid, r = s["subgoal_id"], s["container_id"]
    g = f"G_{sid}_tool"                  # 분할 서브골 전원이 공유하는 객체 점유 흐름
    tstrs = ["{" + ",".join(m) + "}" for m in part]
    children = []
    for k, members in enumerate(part, 1):
        child = _child(s, sid, k, len(part), members, "쓸어 담는다")
        csid = child["subgoal_id"]
        details = []
        if k == 1:
            details.append(_detail(f"{csid}_d1", "acquire", {"?o": "?tool"}, g,
                                   "sweep 객체 확보 (분할 서브골 공용 — 첫 서브골에만)"))
        details.append(_detail(f"{csid}_d2", "tool_act:sweep",
                               {"?t": "?tool", "?targets": tstrs[k - 1], "?r": r}, g,
                               f"그룹 {k}/{len(part)}을 {r}로 sweep"))
        if k == len(part):
            d2b = _detail(f"{csid}_d2b", "transport", {"?o": "?tool", "?r": "tool_rest"}, g,
                          "객체를 거치 위치로 운반 (마지막 서브골에만)")
            for tj in tstrs:             # 그룹별 사전조건 — 각 sweep의 establish와 문자열 일치
                d2b["pre"].append({"id": f"{csid}_d2b_p{len(d2b['pre'])}",
                                   "expr": f"in({tj}, {r})", "head": "in", "eval_by": "m2"})
            details.append(d2b)
            details.append(_detail(f"{csid}_d3", "place", {"?o": "?tool", "?r": "tool_rest"},
                                   g, "객체 반환 (마지막 서브골에만)"))
        child["details"] = details
        children.append(child)
    return children


def _split_relocate(s: dict, part: list[list[str]]) -> list[dict]:
    """relocate 분할 — 공용 확보/반환 단계가 없어 그룹마다 독립 체인. decompose를 그대로 쓴다."""
    sid = s["subgoal_id"]
    children = []
    for k, members in enumerate(part, 1):
        child = _child(s, sid, k, len(part), members, "옮긴다")
        child["details"] = decompose(child)
        children.append(child)
    return children


def split_after_m3(m2_out: dict) -> list[str]:
    """partition_plan이 있는 서브골을 그룹별 서브골로 나누고 순서·불변식·통계를 재계산한다.

    partition_plan이 하나도 없으면 아무것도 바꾸지 않는다 (측정 모듈 미구현 시 폴백).
    반환: 사람이 읽을 로그 라인.
    """
    logs, new_subs, changed = [], [], False
    for s in m2_out["m2_subgoals"]:
        part = s.get("partition_plan")
        if not part or len(part) < 2:
            new_subs.append(s)
            continue
        kids = (_split_sweep if s["kind"] == "sweep_collect" else _split_relocate)(s, part)
        new_subs += kids
        changed = True
        logs.append(f"  [분할] {s['subgoal_id']}: 그룹 {len(part)}개 → "
                    + ", ".join(f"{k['subgoal_id']}({len(k['target_ids'])}개)" for k in kids))
    if not changed:
        return logs

    m2_out["m2_subgoals"] = new_subs
    all_details = [d for s2 in new_subs for d in s2["details"]]
    edges, mutex = partial_order(all_details)
    stuck = _cycle_check(all_details, edges)
    if stuck:
        raise ValueError(f"분할 후 DAG에 사이클 발생: {stuck}")
    m2_out["m2_partial_order"], m2_out["m2_mutex"] = edges, mutex
    m2_out["m2_invariants"] = invariants_for(new_subs)
    # 질의 목록 재생성: 분할 안 된 서브골 것은 유지, 자식은 자기 detail 기준으로 새로 발행
    # (측정 모듈이 이 목록으로 서브골별 서브그래프를 다시 조립한다. 접지는 캐시라 저렴)
    kept_ids = {s2["subgoal_id"] for s2 in new_subs if "split_from" not in s2}
    queries = [q for q in m2_out.get("m2_queries", []) if q["subgoal_id"] in kept_ids]
    for s2 in new_subs:
        if "split_from" in s2:
            queries += build_queries(s2, s2["details"])
    m2_out["m2_queries"] = queries
    st = m2_out.setdefault("m2_stats", {})
    st.update({"n_subgoals": len(new_subs), "n_details": len(all_details),
               "n_edges": len(edges), "n_mutex": len(mutex),
               "n_m3_queries": len(queries),
               "n_split_subgoals": sum(1 for s2 in new_subs if "split_from" in s2)})
    logs.append(f"  [분할] 재계산: 서브골 {len(new_subs)} / 상세 {len(all_details)} / "
                f"엣지 {len(edges)} / mutex {len(mutex)} / 사이클 없음")
    return logs
