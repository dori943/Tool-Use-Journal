# -*- coding: utf-8 -*-
"""M1-5 — M2 응답 반영: 질의로 나갔던 술어에 판정 결과를 되채운다.

입력  M1Output + M2 응답 리스트 (합의 형식: 항목마다 queried_by, node_id 에코)
출력  eval_by==m2 술어에 status(sat|unsat|unknown|unanswered)와 evidence 부착,
      사람이 읽을 로그 라인 목록 반환 (도희 요청: 질의가 반영되는지 확인할 로그)

판정 규칙 (EE-agnostic 유지 — 후보 중 하나라도 되면 sat, 최종 선택은 M3 몫):
  reachable       reachability.reachable
  top_exposed     응답에 판정 필드 없음 → unknown (M2에 필드 추가 논의 항목)
  ee_usable       ee 판정 중 feasible=true 존재 여부
  tool_sweepable  ee 대체 판정(잠정) — sweep 적합성 질의 종류는 M2와 협의 중
  fits / clear    relational 응답의 pass
"""
from __future__ import annotations

import json


def _node_view(responses: list[dict]) -> dict:
    """같은 노드에 대한 응답 여러 건(intrinsic, ee 등)을 노드별 한 뷰로 합친다."""
    view: dict[str, dict] = {}
    for r in responses:
        n = view.setdefault(r.get("node_id", "?"), {})
        for k, v in r.items():
            if k not in ("subgoal_id", "queried_by", "node_id"):
                n[k] = v
    return view


def _judge(head: str, nodes: list[str], view: dict, rels: list[dict]):
    """술어 1건 판정 → (status, evidence). evidence는 후보 노드별 근거."""
    ev = []
    if head in ("fits", "clear"):
        if not rels:
            return "unanswered", []
        ok = [bool(r.get("pass")) for r in rels]
        ev = [{"check": r.get("check"), "value_mm": r.get("value_mm"),
               "pass": bool(r.get("pass"))} for r in rels]
        return ("sat" if all(ok) else "unsat"), ev

    verdicts = []
    for n in nodes:
        info = view.get(n, {})
        if head == "reachable":
            r = info.get("reachability", {})
            v = r.get("reachable")
            ev.append({"node": n, "reachable": v, "margin_mm": r.get("margin_mm")})
        elif head == "top_exposed":
            v = info.get("top_exposed")          # 현재 M2 응답에 없음 → None
            ev.append({"node": n, "top_exposed": v})
        elif head in ("ee_usable", "tool_sweepable"):
            ee = info.get("ee")
            v = any(x.get("feasible") for x in ee.values()) if ee else None
            ev.append({"node": n,
                       "feasible_ees": [k for k, x in (ee or {}).items() if x.get("feasible")],
                       "proxy": head == "tool_sweepable"})
        else:
            v = None
            ev.append({"node": n})
        verdicts.append(v)

    if all(v is None for v in verdicts):
        return "unknown", ev
    if any(v for v in verdicts):
        return "sat", ev                        # 후보 중 하나라도 가능하면 계획은 유효
    return "unsat", ev


PROMPT_UPDATE = """너는 이 로봇 계획을 만든 계획자다. 계획을 세울 때는 측정값이 없어서
아래와 같이 스스로 신뢰도를 매겼다.

태스크: {task}
서브골: {goal}
- 서브골 분해 신뢰도: {c_dec}
- 객체 선택 신뢰도: {c_sel}  (선택한 객체: {objects}, 도구 후보: {tools})
- 그때 불확실하다고 본 것: {unc}

이제 M2 측정 모듈이 조건들을 실측한 결과가 도착했다:
{summary}

이 측정 결과를 반영해 두 신뢰도를 갱신하라. 측정으로 해소된 불확실성은 반영하되,
근거 없이 후하게 주지 말 것.
(0.9 이상 = 거의 확실 / 0.7 = 대체로 확신 / 0.5 = 반반 / 0.3 = 실패 가능성 높음)
uncertain_about에는 아직 남아 있는 불확실 요소만 적어라. 없으면 빈 배열.
reason은 한국어 한 문장.

JSON 객체만 출력:
{{"decomposition": 0.0, "object_selection": 0.0, "uncertain_about": ["..."], "reason": "..."}}"""


def _summarize_for_subgoal(s: dict) -> str:
    """한 서브골의 M2 판정 결과 요약. 상태 성격을 구분해 전달한다
    (not_queried는 불확실성이 아니라 의도적 비측정)."""
    buckets = {"sat": [], "unsat": [], "unknown": [], "not_queried": []}
    for d in s["details"]:
        for p in d["pre"]:
            if p.get("eval_by") != "m2" or "status" not in p:
                continue
            ev = "; ".join(str(e) for e in p.get("evidence", [])[:2])[:120]
            proxy = any(e.get("proxy") for e in p.get("evidence", []) if isinstance(e, dict))
            tag = " [대체 판정]" if proxy else ""
            buckets.get(p["status"], buckets["unknown"]).append(
                f"  - {p['expr'][:60]}: ({ev}){tag}")
    parts = [f"판정 요약: 충족 {len(buckets['sat'])} / 불충족 {len(buckets['unsat'])} / "
             f"근거 미제공 {len(buckets['unknown'])} / 의도적 비측정 {len(buckets['not_queried'])}"]
    labels = {"sat": "충족된 조건:", "unsat": "불충족 조건:",
              "unknown": "측정을 요청했으나 판정 근거가 없던 조건:",
              "not_queried": "의도적으로 측정하지 않은 조건 (측정 대상 아님, 불확실성으로 취급하지 말 것):"}
    for k, label in labels.items():
        if buckets[k]:
            parts.append(label)
            parts += buckets[k]
    if s.get("selected_tool_id"):
        parts.append(f"도구 확정: {s['selected_tool_id']} "
                     f"(근거 {json.dumps(s.get('selection_evidence', []), ensure_ascii=False)[:150]})")
    return "\n".join(parts)


def update_confidence(m1_out: dict, client, model: str = "gpt-4o") -> list[str]:
    """M2 반영 후 자기보고 신뢰도 갱신 (LLM 3차 호출).

    apply_m2()로 판정·도구 확정이 끝난 뒤 호출한다. 서브골마다
    confidence["after"]를 채우고 로그 라인을 돌려준다.
    """
    logs = []
    for s in m1_out["m1_subgoals"]:
        conf = s.setdefault("confidence", {})
        before = conf.get("before", {})
        prompt = PROMPT_UPDATE.format(
            task=m1_out.get("task", ""), goal=s.get("goal", ""),
            c_dec=before.get("decomposition"), c_sel=before.get("object_selection"),
            objects=s.get("object_ids", []), tools=s.get("tool_candidate_ids", []),
            unc=before.get("uncertain_about", []), summary=_summarize_for_subgoal(s))
        err = None
        for attempt in (1, 2):
            msg = prompt if attempt == 1 else prompt + f"\n\n이전 출력 문제: {err}. JSON 객체만."
            r = client.chat.completions.create(
                model=model, temperature=0, messages=[{"role": "user", "content": msg}])
            text = r.choices[0].message.content.strip()
            try:
                obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
                for k in ("decomposition", "object_selection"):
                    v = obj.get(k)
                    if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                        raise ValueError(f"{k}가 0~1 숫자가 아님: {v!r}")
                conf["after"] = obj
                conf["delta"] = {k: round(obj[k] - (before.get(k) or 0.0), 3)
                                 for k in ("decomposition", "object_selection")}
                logs.append(
                    f"  [신뢰도 갱신] {s['subgoal_id']}: "
                    f"분해 {before.get('decomposition')} → {obj['decomposition']} "
                    f"({conf['delta']['decomposition']:+.2f}), "
                    f"객체 선택 {before.get('object_selection')} → {obj['object_selection']} "
                    f"({conf['delta']['object_selection']:+.2f})")
                if obj.get("uncertain_about"):
                    logs.append(f"      남은 불확실 요소: {obj['uncertain_about']}")
                break
            except (ValueError, json.JSONDecodeError) as e:
                err = str(e)
        else:
            logs.append(f"  [신뢰도 갱신] {s['subgoal_id']}: 실패 ({err})")
    return logs


def apply_m2(m1_out: dict, responses: list[dict]) -> list[str]:
    """M2 응답을 M1 출력의 m2 술어에 반영하고 로그 라인을 돌려준다. (m1_out은 제자리 수정)"""
    by_q: dict[str, list[dict]] = {}
    for r in responses:
        by_q.setdefault(r.get("queried_by"), []).append(r)
    view = _node_view(responses)

    # 질의 사양에서 술어 id → 대상 노드 목록 (질의가 실제로 향했던 노드들)
    q_nodes: dict[str, list[str]] = {}
    for q in m1_out.get("m1_queries", []):
        c = q["m2_call"]
        n = c.get("node_id") or c.get("a")
        if n:
            q_nodes.setdefault(q["queried_by"], [])
            if n not in q_nodes[q["queried_by"]]:
                q_nodes[q["queried_by"]].append(n)

    logs, n_sat = [], {"sat": 0, "unsat": 0, "unknown": 0, "unanswered": 0, "not_queried": 0}
    for s in m1_out["m1_subgoals"]:
        for d in s["details"]:
            for p in d["pre"]:
                if p["eval_by"] != "m2":
                    continue
                rs = by_q.get(p["id"], [])
                rels = [r for r in rs if "pass" in r or "check" in r]
                answered_nodes = [r["node_id"] for r in rs if "node_id" in r]
                nodes = q_nodes.get(p["id"], []) or answered_nodes
                if p["id"] not in q_nodes:      # 질의 자체를 안 낸 술어 (예: tool_rest는 측정 대상 아님)
                    status, ev = "not_queried", []
                elif not rs:
                    status, ev = "unanswered", []
                else:
                    status, ev = _judge(p["head"], nodes, view, rels)
                p["status"], p["evidence"] = status, ev
                n_sat[status] += 1
                missing = [n for n in nodes if n not in answered_nodes]
                line = f"  {p['id']:14s} {p['expr'][:52]:52s} -> {status}"
                if status == "unknown":
                    line += "  (응답에 판정 근거 필드 없음)"
                if status == "not_queried":
                    line += "  (질의 대상 아님: 도구 거치 위치 등)"
                if missing:
                    line += f"  [미회신 노드: {', '.join(missing)}]"
                logs.append(line)

    # 도구 확정 (0821 확정: 객체 선택은 M1이 완결한다 — M2 측정값을 근거로
    # 어느 물체를 도구로 쓸지까지 M1이 정한다. M3는 EE 선택·순서 최적화만)
    for s in m1_out["m1_subgoals"]:
        cands = s.get("tool_candidate_ids", [])
        if not cands:
            continue
        scored = []
        for c in cands:
            info = view.get(c, {})
            ee = info.get("ee") or {}
            feasible = [k for k, x in ee.items() if x.get("feasible")]
            reach = info.get("reachability") or {}
            if ee and not feasible:
                continue                      # 어떤 EE로도 못 잡는 후보는 탈락
            if reach.get("reachable") is False:
                continue                      # 팔이 안 닿는 후보 탈락
            scored.append((len(feasible), float(reach.get("margin_mm") or 0.0), c, feasible))
        if not scored:
            logs.append(f"  [도구 확정] {s['subgoal_id']}: 통과한 후보 없음")
            continue
        scored.sort(reverse=True)             # 사용 가능 EE 수 최대 → 리치 여유 최대
        n_ee, margin, chosen, _ = scored[0]
        s["selected_tool_id"] = chosen
        s["selection_evidence"] = [{"node": c2, "feasible_ees": f2, "reach_margin_mm": m2}
                                   for (n2, m2, c2, f2) in scored]
        logs.append(f"  [도구 확정] {s['subgoal_id']}: {chosen} 선택 "
                    f"(사용 가능 EE {n_ee}종, 리치 여유 {margin}mm)")

    m1_out["m1_stats"]["m2_predicates"] = n_sat
    total = sum(n_sat.values())
    logs.insert(0, f"[M2 반영] 응답 {len(responses)}건 수신, m2 술어 {total}건 판정: "
                   f"sat {n_sat['sat']} / unsat {n_sat['unsat']} / "
                   f"unknown {n_sat['unknown']} / 미회신 {n_sat['unanswered']} / "
                   f"질의대상아님 {n_sat['not_queried']}")
    return logs
