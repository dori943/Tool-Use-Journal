# -*- coding: utf-8 -*-
"""M2-5 — M3 응답 반영: 질의로 나갔던 술어에 판정 결과를 되채운다.

입력  M1Output + M3 응답 리스트 (합의 형식: 항목마다 queried_by, node_id 에코)
출력  eval_by==m3 술어에 status(sat|unsat|unknown|unanswered)와 evidence 부착,
      사람이 읽을 로그 라인 목록 반환 (도희 요청: 질의가 반영되는지 확인할 로그)

판정 규칙 (EE-agnostic 유지 — 후보 중 하나라도 되면 sat, 최종 선택은 M4 몫):
  reachable       reachability.reachable
  top_exposed     응답에 판정 필드 없음 → unknown (M3에 필드 추가 논의 항목)
  ee_usable       ee 판정 중 feasible=true 존재 여부
  batch_feasible  batch 응답(0828 신규)의 feasible — 그룹 동시 처리 가능 여부
  act_space_clear swept_space 응답(0828 신규)의 clear — 실행 공간 확보 여부
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


def _judge(head: str, nodes: list[str], view: dict, rels: list[dict],
           require_all: bool = False):
    """술어 1건 판정 → (status, evidence). evidence는 노드별 근거.

    require_all=False: tool 후보 의미론 — 후보 중 하나라도 되면 sat (기존).
    require_all=True:  그룹 원소 의미론 — 전원이어야 sat, 하나라도 안 되면 unsat
                       (0828 — relocate 다중 target의 원소들은 전부 옮겨야 하므로).
    """
    ev = []
    if head in ("fits", "clear"):
        if not rels:
            return "unanswered", []
        ok = [bool(r.get("pass")) for r in rels]
        ev = [{"node": r.get("from") or r.get("node_id"), "check": r.get("check"),
               "value_mm": r.get("value_mm"), "pass": bool(r.get("pass"))} for r in rels]
        # 0908: 도구 후보 서브골(require_all=False)은 후보 중 하나라도 되면 sat.
        # 전원 통과를 요구하면 fits(?tool, tool_rest)가 후보 하나 때문에 unsat이 되어
        # 재분해(sweep_collect → relocate)를 잘못 트리거한다 (c1_1 0908 실행).
        return ("sat" if (all(ok) if require_all else any(ok)) else "unsat"), ev

    verdicts = []
    for n in nodes:
        info = view.get(n, {})
        if head == "reachable":
            r = info.get("reachability", {})
            v = r.get("reachable")
            ev.append({"node": n, "reachable": v, "margin_mm": r.get("margin_mm")})
        elif head == "top_exposed":
            # 0905: M3 query_top_exposed 응답 {type: top_exposed, node_id, value, blockers}
            r = next((x for x in rels
                      if x.get("type") == "top_exposed" and x.get("node_id") == n), None)
            v = r.get("value") if r else info.get("top_exposed")
            ev.append({"node": n, "top_exposed": v,
                       "blockers": (r or {}).get("blockers", [])})
        elif head == "ee_usable":
            ee = info.get("ee")
            v = any(x.get("feasible") for x in ee.values()) if ee else None
            ev.append({"node": n,
                       "feasible_ees": [k for k, x in (ee or {}).items() if x.get("feasible")]})
        elif head == "flat_face":
            # M3 query_flat_face 응답 {type: flat_face, node_id, value, pass, check}
            r = next((x for x in rels
                      if x.get("type") == "flat_face" and x.get("node_id") == n), None)
            v = r.get("value") if r else None
            ev.append({"node": n, "flat_face": v, "check": (r or {}).get("check")})
        elif head == "gap_accessible":
            # M3 query_gap_accessible 응답 {type: gap_accessible, from: tool, to: target, pass, value_mm}
            r = next((x for x in rels
                      if x.get("type") == "gap_accessible" and x.get("from") == n), None)
            v = r.get("pass") if r else None
            ev.append({"node": n, "gap_accessible": v, "margin_mm": (r or {}).get("value_mm")})
        else:
            v = None
            ev.append({"node": n})
        verdicts.append(v)

    if all(v is None for v in verdicts):
        return "unknown", ev
    if require_all:                             # 그룹 원소: 전원 충족이어야 sat
        if any(v is False for v in verdicts):
            return "unsat", ev
        if all(v for v in verdicts):
            return "sat", ev
        return "unknown", ev                    # 일부 미판정
    if any(v for v in verdicts):
        return "sat", ev                        # 후보 중 하나라도 가능하면 계획은 유효
    return "unsat", ev


def _judge_group(head: str, rs: list[dict]):
    """batch / swept_space 응답(0828 신규) 판정. 액션 주체(actor) 후보별 응답 리스트를 받는다.

    후보 중 하나라도 가능하면 sat (EE-agnostic 원칙과 동일 — 최종 선택은 도구 확정이 한다).
    batch가 unsat이어도 응답의 partition이 있으면 regroup이 그 구성대로 서브골을 나눈다.
    """
    ev, ok = [], []
    for r in rs:
        if head == "batch_feasible":
            v = r.get("feasible")
            ev.append({"actor": r.get("actor"), "feasible": v,
                       "binding_check": r.get("binding_check"),
                       "partition": r.get("partition")})
        else:
            v = r.get("clear")
            ev.append({"actor": r.get("actor"), "clear": v,
                       "margin_mm": r.get("margin_mm"), "blockers": r.get("blockers")})
        ok.append(v)
    if all(v is None for v in ok):
        return "unknown", ev
    return ("sat" if any(ok) else "unsat"), ev


PROMPT_UPDATE = """너는 이 로봇 계획을 만든 계획자다. 계획을 세울 때는 측정값이 없어서
아래와 같이 스스로 신뢰도를 매겼다.

태스크: {task}
서브골: {goal}
- 서브골 분해 신뢰도: {c_dec}
- 객체 선택 신뢰도: {c_sel}  (선택한 객체: {objects}, 도구 후보: {tools})
- 그때 불확실하다고 본 것: {unc}

이제 M3 측정 모듈이 조건들을 실측한 결과가 도착했다:
{summary}

이 측정 결과를 반영해 두 신뢰도를 갱신하라. 측정으로 해소된 불확실성은 반영하되,
근거 없이 후하게 주지 말 것.
(0.9 이상 = 거의 확실 / 0.7 = 대체로 확신 / 0.5 = 반반 / 0.3 = 실패 가능성 높음)
uncertain_about에는 아직 남아 있는 불확실 요소만 적어라. 없으면 빈 배열.
reason은 한국어 한 문장.

JSON 객체만 출력:
{{"decomposition": 0.0, "object_selection": 0.0, "uncertain_about": ["..."], "reason": "..."}}"""


# 0908: 도구 후보가 있는 서브골에는 신뢰도 갱신 호출에 도구 판정을 합친다 (호출 수 동일).
PROMPT_TOOL_BLOCK = """
이 서브골은 도구가 필요하다 (kind: {kind}). 도구 후보별 측정값:
{table}

후보 중 이 작업에 가장 적합한 도구 하나를 골라 selected_tool_id에 쓰고, tool_reason에
이유를 한국어 한 문장으로 쓰라. 후보 나열 순서에는 의미가 없다.
작업별 적합 기준:
- flatten(펴기): 반죽을 굴리거나 눌러 펴야 한다. 원통형 또는 두께가 두꺼운 단단한 몸체
  (병, 밀대, 컵)가 적합하고, 두께가 몇 mm인 얇은 판(주걱, 뒤집개)은 힘을 주면 휘어 부적합하다.
- sweep_collect(쓸어 모으기): 넓고 평평한 면. 필요 액션 횟수가 적은 것.
- extract(틈에서 꺼내기): 틈보다 얇고 대상까지 닿을 만큼 긴 것.
- scoop_transfer(떠 옮기기): 오목한 면이 있는 것.
공통 기준:
- 핵심 술어({core})가 false인 후보는 고르지 말 것. 잡을 수 있는 EE가 없는 후보도 제외.
- 리치 여유는 양수면 충분하며 우열 기준이 아니다. 잡을 수 있는 EE 종류 수도 우열 기준이 아니다.
- 질량과 재질은 VLM 추정치라 오차가 크다. 크기, 두께, 형상을 우선하라.
JSON 객체에 "selected_tool_id"와 "tool_reason" 필드를 추가하라."""


def _tool_table(measured: list[dict]) -> str:
    rows = []
    for m in measured:
        ext = m.get("extents_mm")
        size = "x".join(str(round(x)) for x in ext) + "mm" if ext else "크기 미상"
        mass = f"{m['mass_kg']}kg(추정)" if m.get("mass_kg") is not None else "질량 미상"
        parts = [size]
        if ext:
            parts.append(f"두께(최소 치수) {round(min(ext), 1)}mm")
        parts.append(mass)
        if m.get("material") and m["material"] != "unknown":
            parts.append(f"재질 {m['material']}")
        if m.get("cylinder_like"):
            parts.append("원통형")
        parts.append(f"잡을 수 있는 EE {m.get('feasible_ees') or []}")
        if m.get("reach_margin_mm") is not None:
            parts.append(f"리치 여유 {m['reach_margin_mm']}mm")
        if m.get("batch_groups"):
            parts.append(f"필요 액션 {m['batch_groups']}회")
        if m.get("core_pred"):
            v = m.get("core_value")
            parts.append(f"{m['core_pred']}={'미측정' if v is None else v}")
        rows.append(f"  - {m['node']}: " + ", ".join(parts))
    return "\n".join(rows)


def _apply_llm_tool_choice(s: dict, obj: dict, logs: list[str]) -> None:
    """LLM이 고른 도구를 검증해 채택한다. 부적합하면 규칙 선택을 유지한다."""
    pick = obj.get("selected_tool_id")
    reason = obj.get("tool_reason", "")
    measured = {m["node"]: m for m in s.get("tool_candidates_measured", [])}
    rule = s.get("selected_tool_id")
    if pick not in measured:
        logs.append(f"  [도구 판정] {s['subgoal_id']}: LLM 선택 {pick!r}이 측정된 후보에 없음 — "
                    f"규칙 선택 {rule} 유지")
        return
    m = measured[pick]
    if not m.get("feasible_ees"):
        logs.append(f"  [도구 판정] {s['subgoal_id']}: LLM 선택 {pick}은 잡을 수 있는 EE 없음 — "
                    f"규칙 선택 {rule} 유지")
        return
    if m.get("core_value") is False:
        logs.append(f"  [도구 판정] {s['subgoal_id']}: LLM 선택 {pick}은 {m.get('core_pred')}=false — "
                    f"규칙 선택 {rule} 유지")
        return
    s["selection_by"] = "llm"
    s["selection_reason_tool"] = reason
    if pick == rule:
        logs.append(f"  [도구 판정] {s['subgoal_id']}: LLM이 규칙 선택 {rule} 동의 ({reason})")
        return
    s["selected_tool_id_rule"] = rule
    s["selected_tool_id"] = pick
    part = m.get("partition")
    if part and len(part) > 1:
        s["partition_plan"] = part
    else:
        s.pop("partition_plan", None)
    logs.append(f"  [도구 판정] {s['subgoal_id']}: {rule} → {pick} 로 변경 ({reason})")


def _summarize_for_subgoal(s: dict) -> str:
    """한 서브골의 M3 판정 결과 요약. 상태 성격을 구분해 전달한다
    (not_queried는 불확실성이 아니라 의도적 비측정)."""
    buckets = {"sat": [], "unsat": [], "unknown": [], "not_queried": []}
    for d in s["details"]:
        for p in d["pre"]:
            if p.get("eval_by") != "m3" or "status" not in p:
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


def update_confidence(m2_out: dict, client, model: str = "gpt-4o",
                      usage_acc: dict | None = None) -> list[str]:
    """M3 반영 후 자기보고 신뢰도 갱신 (LLM 3차 호출).

    apply_m3()로 판정·도구 확정이 끝난 뒤 호출한다. 서브골마다
    confidence["after"]를 채우고 로그 라인을 돌려준다.
    """
    logs = []
    for s in m2_out["m2_subgoals"]:
        conf = s.setdefault("confidence", {})
        before = conf.get("before", {})
        prompt = PROMPT_UPDATE.format(
            task=m2_out.get("task", ""), goal=s.get("goal", ""),
            c_dec=before.get("decomposition"), c_sel=before.get("object_selection"),
            objects=s.get("object_ids", []), tools=s.get("tool_candidate_ids", []),
            unc=before.get("uncertain_about", []), summary=_summarize_for_subgoal(s))
        # 0908: 도구 후보 측정표가 있으면 같은 호출에서 도구 판정까지 받는다
        measured = s.get("tool_candidates_measured") or []
        want_tool = bool(measured) and s.get("selected_tool_id") in {m["node"] for m in measured}
        if want_tool:
            # 규칙 선택은 LLM에 알리지 않는다 (앵커링 방지: c1_2에서 "1차 선택 spatula"를
            # 보여주니 2mm 두께가 휜다고 스스로 적으면서도 동의했음). 후보는 id순으로 중립 나열.
            prompt += "\n" + PROMPT_TOOL_BLOCK.format(
                kind=s.get("kind"),
                table=_tool_table(sorted(measured, key=lambda m: m["node"])),
                core=(measured[0].get("core_pred") or "없음"))
        err = None
        for attempt in (1, 2):
            msg = prompt if attempt == 1 else prompt + f"\n\n이전 출력 문제: {err}. JSON 객체만."
            import time as _time
            t0 = _time.monotonic()
            r = client.chat.completions.create(
                model=model, temperature=0, messages=[{"role": "user", "content": msg}])
            if usage_acc is not None:
                from .rough import track_usage
                track_usage(usage_acc, "confidence_update", r,
                            seconds=_time.monotonic() - t0)
            text = r.choices[0].message.content.strip()
            try:
                obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
                for k in ("decomposition", "object_selection"):
                    v = obj.get(k)
                    if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                        raise ValueError(f"{k}가 0~1 숫자가 아님: {v!r}")
                if want_tool and not obj.get("selected_tool_id"):
                    raise ValueError("selected_tool_id가 없음 (도구 후보 중 하나를 골라라)")
                if want_tool:
                    _apply_llm_tool_choice(s, obj, logs)
                conf["after"] = {k: v for k, v in obj.items()
                                 if k not in ("selected_tool_id", "tool_reason")}
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


def measurement_feedback(m2_out: dict) -> str | None:
    """직전 왕복에서 unsat 난 술어를 사실 요약 텍스트로 만든다 (재분해 프롬프트 주입용).

    0831 설계: 해법("도구를 써라")은 절대 넣지 않는다 — 측정 사실만 전달하고,
    대안 도출은 분해 LLM의 몫으로 남긴다. 그래야 도구 사용이 창발로 성립한다.
    """
    # 0908: unsat을 두 등급으로 나눈다. 이전엔 unsat 하나만 있어도 "다른 kind를 내라"고
    # 강제해 c2_2(stack → relocate → extract), c4_1(extract → 맨손 relocate),
    # c4_2(relocate → stack → ...)가 전부 엉뚱한 kind로 튀었다.
    #   kind 등급  수행 방식 자체가 불가: reachable, ee_usable(도구 없이 못 잡음),
    #              flat_face / gap_accessible(도구 후보 전원 부적합)
    #   조정 등급  같은 kind 안에서 순서/짝만 고치면 되는 것: top_exposed(위에 다른 물체가
    #              있음 → 그 물체를 먼저 치우거나 쌓기 순서 변경), fits(이 쌍만 재검토)
    # 0908 2차: "수행 방식 불가"를 다시 둘로 나눈다. 도구 후보 전원이 핵심 술어에서
    # 떨어진 것(flat_face/gap_accessible 전부 false)은 kind가 아니라 후보가 문제다.
    # c1_2에서 M0 캐시 버그로 flat_face가 전부 false가 되자 LLM이 flatten 대신
    # sweep_collect("병으로 반죽을 도마에 쓸어 담기")를 내놓았다. 펴기를 쓸기로 바꾸는 건
    # 어떤 상황에서도 답이 아니므로 kind는 유지하고 도구 후보 확장만 요구한다.
    kind_lines, tool_lines, adjust_lines = [], [], []
    for s in m2_out.get("m2_subgoals", []):
        for d in s.get("details", []):
            for p in d.get("pre", []):
                if p.get("status") != "unsat":
                    continue
                head = p.get("head")
                # batch/act_space는 unsat이어도 분할(partition)과 제외 목록으로
                # 해소되는 신호라 재분해 사유가 아니다 (0831 — kind 진동 방지)
                if head in ("batch_feasible", "act_space_clear"):
                    continue
                # 0903: 목적지 clear 불충족은 수행 방식(kind)을 바꿔도 해소되지 않는다
                # (트레이가 막힌 건 쓸어 담아도 마찬가지). 목적지 정리는 별도 서브골 문제.
                if head == "clear":
                    continue
                ev = p.get("evidence") or []
                if head == "top_exposed":
                    blk = {n for e in ev if e.get("top_exposed") is False
                           for n in (e.get("blockers") or [])}
                    who = [e.get("node") for e in ev if e.get("top_exposed") is False]
                    adjust_lines.append(
                        f"- {p['expr']} -> 불충족: {', '.join(who)} 위에 다른 물체가 있음"
                        + (f" ({', '.join(sorted(blk))})" if blk else "")
                        + ". 위에 있는 물체를 먼저 다루거나 쌓는 순서를 바꿀 것")
                    continue
                if head == "fits":
                    bad = [e.get("node") for e in ev if "pass" in e and not e.get("pass")]
                    adjust_lines.append(
                        f"- {p['expr']} -> 불충족 (검사 {len(ev)}건 중 {len(bad)}건: "
                        f"{', '.join(str(x) for x in bad if x)}). 이 대상만 다른 목적지/받침으로 "
                        "바꾸거나 별도 서브골로 분리할 것")
                    continue
                if head in ("flat_face", "gap_accessible"):
                    bad = [e.get("node") for e in ev if e.get(head) is False]
                    tool_lines.append(
                        f"- {p['expr']} -> 불충족: 도구 후보 {', '.join(str(x) for x in bad if x)} 전부 "
                        f"{head} 조건을 만족하지 못함")
                    continue
                detail = ""
                falsy = [e for e in ev if e.get("reachable") is False]
                if falsy:
                    margins = [e.get("margin_mm") for e in falsy
                               if e.get("margin_mm") is not None]
                    detail = f" (대상 {len(ev)}개 중 {len(falsy)}개는 로봇 팔이 닿지 않음"
                    if margins:
                        detail += f", 최대 {abs(min(margins)):.1f}mm 부족"
                    detail += ")"
                elif ev and all("pass" in e for e in ev):
                    n_f = sum(1 for e in ev if not e.get("pass"))
                    detail = f" (검사 {len(ev)}건 중 {n_f}건 불충족)"
                kind_lines.append(f"- {p['expr']} -> 불충족{detail}")
    if not kind_lines and not tool_lines and not adjust_lines:
        return None
    out = ["직전 계획 평가: 아래 조건이 실측에서 충족되지 않았다.",
           "공통: 재분해하더라도 지시문이 요구하는 최종 상태(펴기, 담기, 쌓기, 꺼내기 등)는 "
           "바뀌면 안 된다. 바꿀 수 있는 것은 수행 방식, 도구 후보, 순서, 짝이다."]
    if kind_lines:
        out.append("[맨손 수행 불가] 아래 제약은 로봇과 장면의 물리적 사실이므로, 같은 kind의 "
                   "분해를 반복하면 동일하게 실행 불가로 판정된다. 같은 목표를 달성하는 다른 "
                   "수행 방식(kind, 예: 도구를 쓰는 방식)의 분해를 출력하라.")
        out += kind_lines
    if tool_lines:
        out.append("[도구 후보 부적합] kind는 그대로 유지하라. 대신 장면의 다른 물체 중 이 작업에 "
                   "쓸 수 있는 것을 도구 후보(tool_candidate_ids)에 추가하거나 교체하라. "
                   "추가할 물체가 없으면 기존 분해를 그대로 출력하라.")
        out += tool_lines
    if adjust_lines:
        out.append("[순서/짝 조정] 아래 제약은 수행 방식(kind)의 문제가 아니다. kind는 그대로 "
                   "유지하고, 지목된 물체의 순서, 받침, 목적지, 그룹 구성만 바꿔서 다시 분해하라.")
        out += adjust_lines
    return "\n".join(out)


# kind별 도구 적합성의 핵심 m3 술어 (측정 없이는 후보를 가를 근거가 없는 것)
_CORE_PRED = {"sweep_collect": "batch_feasible", "scoop_transfer": "batch_feasible",
              "flatten": "flat_face", "extract": "gap_accessible"}


def apply_m3(m2_out: dict, responses: list[dict]) -> list[str]:
    """M3 응답을 M2 출력의 m3 술어에 반영하고 로그 라인을 돌려준다. (m2_out은 제자리 수정)"""
    by_q: dict[str, list[dict]] = {}
    for r in responses:
        by_q.setdefault(r.get("queried_by"), []).append(r)
    view = _node_view(responses)

    # 질의 사양에서 술어 id → 대상 노드 목록 (질의가 실제로 향했던 노드들)
    q_nodes: dict[str, list[str]] = {}
    # 0903: 관계 질의는 (a, b) 쌍까지 기억한다. from만 보면 접지 컴파일 보충이 만든
    # 엉뚱한 쌍(c2_1: fits(빵, 머그) — 빵을 머그에 넣는지 검사)이 from=빵으로 통과해
    # unsat을 만들고, 그게 재분해 피드백으로 들어가 kind가 stack으로 튀었다.
    q_pairs: dict[str, set] = {}
    for q in m2_out.get("m2_queries", []):
        c = q["m3_call"]
        if c.get("kind") in ("batch", "swept_space"):   # 0828 신규 — 노드 대신 그룹 대상
            q_nodes.setdefault(q["queried_by"], [])
            continue
        n = c.get("node_id") or c.get("a") or c.get("tool_id")   # 0908: gap_accessible은 tool_id
        if n:
            q_nodes.setdefault(q["queried_by"], [])
            if n not in q_nodes[q["queried_by"]]:
                q_nodes[q["queried_by"]].append(n)
        if c.get("kind") == "relational" and c.get("a") and c.get("b"):
            q_pairs.setdefault(q["queried_by"], set()).add((c["a"], c["b"]))

    logs, n_sat = [], {"sat": 0, "unsat": 0, "unknown": 0, "unanswered": 0, "not_queried": 0}
    for s in m2_out["m2_subgoals"]:
        for d in s["details"]:
            for p in d["pre"]:
                if p["eval_by"] != "m3":
                    continue
                rs = by_q.get(p["id"], [])
                expected = set(q_nodes.get(p["id"], []))
                # 관계 응답은 from/to 키를 쓴다. 접지 쪽 컴파일 보충이 만든 예상 밖 쌍
                # (예: fits({집합}, 영역)을 (원소, 원소)로 오파싱)이 판정을 오염시키지
                # 않도록, 우리가 질의를 발행한 노드(from)의 응답만 판정에 쓴다 (0831).
                rels = [r for r in rs if ("pass" in r or "check" in r)
                        and (not expected or r.get("from") in expected
                             or r.get("node_id") in expected)]
                pairs = q_pairs.get(p["id"])
                if pairs:                        # 관계 술어: 발행한 (from, to) 쌍만 인정
                    dropped = [r for r in rels if r.get("from") and r.get("to")
                               and (r["from"], r["to"]) not in pairs]
                    if dropped:
                        logs.append(f"  [무시] {p['id']}: 발행하지 않은 쌍 "
                                    + ", ".join(f"({r['from']}, {r['to']})" for r in dropped)
                                    + " (접지 컴파일 보충)")
                    rels = [r for r in rels if r not in dropped]
                answered_nodes = [r.get("node_id") or r.get("from")
                                  for r in rs
                                  if (r.get("node_id") or r.get("from"))]
                nodes = q_nodes.get(p["id"], []) or answered_nodes
                # 0908: M2가 직접 발행하지 않은 술어(flat_face, gap_accessible)도 run_m3 컴파일
                # 보충이 응답을 주면 판정에 쓴다. 응답 노드가 하나도 없을 때만 not_queried
                # (tool_rest처럼 error 응답만 오는 경우). 도희 0907 리포트: c1_2 flat_face가
                # 4건 응답됐는데 q_nodes 게이팅에 걸려 미측정 처리 → 리치 여유로 spatula 확정.
                if p["id"] not in q_nodes and not answered_nodes:
                    status, ev = "not_queried", []
                elif not rs:
                    status, ev = "unanswered", []
                elif p["head"] in ("batch_feasible", "act_space_clear"):
                    status, ev = _judge_group(p["head"], rs)
                else:
                    # tool 후보가 있는 서브골(sweep류)은 any, 없는 서브골(relocate
                    # 그룹)은 원소 전원 충족 의미론 (0828)
                    status, ev = _judge(p["head"], nodes, view, rels,
                                        require_all=not s.get("tool_candidate_ids"))
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

    # 도구 확정 (0821 확정: 객체 선택은 M2이 완결한다 — M3 측정값을 근거로
    # 어느 물체를 도구로 쓸지까지 M2이 정한다. M4는 EE 선택·순서 최적화만)
    # 0828: batch 응답(그룹 동시 처리)을 후보 객체별로 모은다 — 확정 기준에 동시처리 용량 추가
    batch_by: dict[str, dict] = {}
    for r in responses:
        a = r.get("actor")
        if r.get("kind") == "batch" and isinstance(a, dict) and a.get("type") == "object":
            batch_by.setdefault(r.get("subgoal_id"), {})[a.get("id")] = r

    for s in m2_out["m2_subgoals"]:
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
            b = batch_by.get(s["subgoal_id"], {}).get(c)
            part = b.get("partition") if b else None
            # 측정값이 하나도 없는 후보는 채점 대상이 아니다 (0831 — 무데이터 동점이
            # id 정렬로 갈리는 오확정 방지. 전략 전환 직후 라운드가 이 경우다)
            if not ee and not reach and b is None:
                continue
            # 필요한 액션 횟수(그룹 수)가 적을수록 우선. batch 응답이 없으면 최하위
            # (batch 질의가 아예 없던 기존 실행에서는 전원이 같아 순위 변화 없음)
            batch_rank = -len(part) if part else float("-inf")
            # 0905: 확정 기준을 액션 횟수 최소 → 리치 여유 최대로 변경. 사용 가능 EE 수는
            # 위에서 "하나라도 있는가"로만 거른다 (도구의 태스크 적합성과 무관한 값이라
            # 점수에 넣으면 c1_1에서 ladle(EE 3종, 12회)이 plate(vac 1종, 2회)를 이겼음)
            scored.append((batch_rank, float(reach.get("margin_mm") or 0.0),
                           len(feasible), c, feasible, part))
        if not scored:
            s.pop("selected_tool_id", None)      # 이전 라운드의 무근거 확정 잔재 제거
            s.pop("selection_evidence", None)
            logs.append(f"  [도구 확정] {s['subgoal_id']}: 측정된 후보 없음 — 확정 보류"
                        " (다음 접지 응답 후 확정)")
            continue
        scored.sort(reverse=True)   # 액션 횟수 최소 → 리치 여유 최대 → EE 수 (0905 개정)
        batch_rank, margin, n_ee, chosen, _, part = scored[0]
        s["selected_tool_id"] = chosen
        # 0903: kind의 핵심 술어가 미측정이면 확정은 하되 근거에 남기고 경고한다.
        # (c4_1 gap_accessible, c1_2 flat_face가 M3에 아직 없어 리치 여유 몇 mm로 갈렸음.
        #  보류로 바꾸면 M4까지 못 가므로 일단 확정 + 표시. M3 구현 시 자동 해소)
        core = _CORE_PRED.get(s.get("kind"))
        if core:
            st = {p.get("status") for d in s.get("details", []) for p in d.get("pre", [])
                  if p.get("head") == core}
            if not (st & {"sat", "unsat"}):
                s["selection_note"] = f"{core} 미측정 상태에서 확정 (EE 수/리치 여유 기준)"
                logs.append(f"  [경고] {s['subgoal_id']}: 핵심 술어 {core}가 미측정 — "
                            f"도구 확정이 측정 근거 없이 EE 수/리치 여유로 결정됨")
        if part and len(part) > 1:
            s["partition_plan"] = part        # regroup이 이 구성대로 서브골을 나눈다
        s["selection_evidence"] = [
            {"node": c2, "feasible_ees": f2, "reach_margin_mm": m3,
             "batch_groups": (len(p2) if p2 else None)}
            for (br2, m3, n2, c2, f2, p2) in scored]
        s["selection_by"] = "rule"
        # 0908: 후보별 측정표. update_confidence의 LLM 도구 판정 입력이 된다.
        # 규칙 스코어러(액션 횟수 → 리치 여유)는 태스크 적합성(펴기엔 굴릴 몸체, 꺼내기엔
        # 얇고 긴 것)을 못 보고 c1_1 ladle, c1_2 spatula처럼 무관한 값으로 갈렸음.
        core_val: dict[str, object] = {}
        if core:
            for d in s.get("details", []):
                for p in d.get("pre", []):
                    if p.get("head") != core:
                        continue
                    for e in p.get("evidence") or []:
                        n = e.get("node")
                        if n is None:
                            continue
                        v = e.get(core, e.get("pass"))
                        if v is not None:
                            core_val[n] = v
        measured = []
        for (br2, m3, n2, c2, f2, p2) in scored:
            info = view.get(c2, {})
            g = info.get("geometry") or {}
            measured.append({
                "node": c2,
                "extents_mm": g.get("extents_mm"),
                "length_mm": g.get("length_mm"),
                "cylinder_like": g.get("cylinder_like"),
                "mass_kg": info.get("mass_kg"),
                "material": info.get("material"),
                "feasible_ees": f2,
                "reach_margin_mm": m3,
                "batch_groups": (len(p2) if p2 else None),
                "partition": p2,
                "core_pred": core,
                "core_value": core_val.get(c2),
            })
        s["tool_candidates_measured"] = measured
        logs.append(f"  [도구 확정] {s['subgoal_id']}: {chosen} 선택 "
                    f"(사용 가능 EE {n_ee}종, 리치 여유 {margin}mm"
                    + (f", 필요 액션 {len(part)}회" if part else "") + ")")

    # 0903: 도구 없는 서브골(relocate 다중 target)의 그룹 분할.
    # 위 블록은 도구 후보가 있는 서브골만 partition_plan을 채우므로, 물체 5개짜리
    # relocate는 batch 응답과 무관하게 한 덩어리(binding 통짜)로 M4에 갔다(0903 1차
    # 실행 c2_1/c4_2). 접지의 batch 응답(actor=ee_pool)이 partition을 주면 그대로
    # 쓰고, 미측정(feasible None — M3 쪽 ee_pool 동시 파지 계산이 아직 없음)이면
    # 물체별 1개씩 나눈다. 맨손 이동은 한 번에 하나씩 옮기는 것이 안전한 기본값이다.
    pool_batch: dict[str, dict] = {}
    for r in responses:
        a = r.get("actor")
        if r.get("kind") == "batch" and isinstance(a, dict) and a.get("type") == "ee_pool":
            pool_batch[r.get("subgoal_id")] = r
    for s in m2_out["m2_subgoals"]:
        if s.get("tool_candidate_ids") or s.get("partition_plan"):
            continue                          # 도구 서브골은 위에서 처리, 이미 분할 계획 있음
        targets = s.get("target_ids") or []
        if len(targets) < 2:
            continue
        b = pool_batch.get(s["subgoal_id"])
        if b is None:
            continue                          # batch 질의 자체가 없던 서브골
        part = b.get("partition")
        if part and len(part) > 1:
            s["partition_plan"] = [list(g) for g in part]
            why = "접지 partition"
        elif b.get("feasible") is True:
            continue                          # 동시 처리 가능 판정 — 분할 불필요
        else:
            s["partition_plan"] = [[t] for t in targets]
            why = ("batch 미측정(ee_pool) 폴백: 물체별 1개" if b.get("feasible") is None
                   else "batch unsat인데 partition 없음: 물체별 1개")
        logs.append(f"  [분할 계획] {s['subgoal_id']}: 그룹 {len(s['partition_plan'])}개 ({why})")

    m2_out["m2_stats"]["m3_predicates"] = n_sat
    total = sum(n_sat.values())
    logs.insert(0, f"[M3 반영] 응답 {len(responses)}건 수신, m3 술어 {total}건 판정: "
                   f"sat {n_sat['sat']} / unsat {n_sat['unsat']} / "
                   f"unknown {n_sat['unknown']} / 미회신 {n_sat['unanswered']} / "
                   f"질의대상아님 {n_sat['not_queried']}")
    return logs
