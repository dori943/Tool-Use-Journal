# -*- coding: utf-8 -*-
"""M2-5 — 접지값으로 술어 판정 + 도구 확정.

0908 파이프라인 재구조 (M1&M3 통합): M3 질의/응답 왕복이 없어졌다.
입력  M1Output + m1 (노드에 ee/reachability/predicates/물성이 실려 온다)
출력  eval_by==m3 술어에 status(sat|unsat|unknown)와 evidence 부착, 서브골에
      measurements(원본 판정 결과 — assemble_gk가 gk에 실어 M4/M6가 본다) 부착,
      사람이 읽을 로그 라인 목록 반환

판정 소스 (EE-agnostic 유지 — 후보 중 하나라도 되면 sat, 최종 선택은 M4 몫):
  reachable        node.reachability.reachable
  ee_usable        node.ee 중 feasible 하나 이상
  top_exposed      node.predicates.top_exposed.value
  clear(영역)      node.predicates.clear.value
  clear(담기)      + 원소별 relations.depth_clearance(member, container).pass 전부
  fits             relations.fits_inside, 집합이면 원소별 all()
  flat_face        node.predicates.flat_face.value
  gap_accessible   relations.gap_access(tool, target)
  batch_feasible   relations.batch_partition(members, tool) — partition을 분할에 사용
(act_space_clear는 0908에 motion 유보로 옮겼다 — 통로 점유는 궤적이 정해져야 안다)
"""
from __future__ import annotations

import json

from .core import TOOL_KINDS, binds_tool

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
            return "unknown", []
        ok = [bool(r.get("pass")) for r in rels]
        ev = [{"node": r.get("from") or r.get("node_id"), "check": r.get("check"),
               "value_mm": r.get("value_mm"), "pass": bool(r.get("pass"))}
              | ({"unfit_members": r["unfit_members"]} if r.get("unfit_members") else {})
              for r in rels]
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
        elif head == "graspable_on_support":
            # 지지면 위 접근 가능성 {type, node_id, value, graspable_ees, per_ee}
            r = next((x for x in rels
                      if x.get("type") == "graspable_on_support"
                      and x.get("node_id") == n), None)
            v = r.get("value") if r else None
            ev.append({"node": n, "graspable_on_support": v,
                       "graspable_ees": (r or {}).get("graspable_ees", []),
                       "supports": sorted({s for e in ((r or {}).get("per_ee") or {}).values()
                                           for s in (e.get("supports") or [])}),
                       "height_mm": next((e.get("height_mm")
                                          for e in ((r or {}).get("per_ee") or {}).values()
                                          if e.get("height_mm") is not None), None)})
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
- stack(쌓기) / relocate(옮기기): 대상 하나를 옮기는 데는 도구가 필요 없다. 도구를
  쓸 이유는 옮기는 횟수가 줄어드는 경우뿐이므로, 대상이 여럿일 때만 한 번에 여러
  개를 받쳐 옮길 수 있는 넓고 평평한 면을 고른다. 대상이 하나면 null 로 둔다.
공통 기준:
- 핵심 술어({core})가 false인 후보는 고르지 말 것. 잡을 수 있는 EE가 없는 후보도 제외.
- 리치 여유는 양수면 충분하며 우열 기준이 아니다. 잡을 수 있는 EE 종류 수도 우열 기준이 아니다.
- 질량과 재질은 VLM 추정치라 오차가 크다. 크기, 두께, 형상을 우선하라.{optional}
JSON 객체에 "selected_tool_id"와 "tool_reason" 필드를 추가하라."""

# 0911: 도구 없이도 되는 서브골에서 억지로 하나를 고르지 않게 하는 선택지.
# 액션 스키마에 ?tool 이 박혀 있는 TOOL_KINDS 에는 붙이지 않는다.
TOOL_OPTIONAL_LINE = """
- 후보 전부가 이 작업에 부적합하거나, 대상을 EE 로 직접 집을 수 있어 도구가
  불필요하면 selected_tool_id 를 null 로 두고 tool_reason 에 그 이유를 적어라.
  도구를 쓰는 것 자체는 이득이 아니다. 도구가 있어야 되는 일에만 쓴다."""


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


def _apply_llm_tool_choice(s: dict, obj: dict, logs: list[str],
                           optional: bool = False) -> None:
    """LLM이 고른 도구를 검증해 채택한다. 부적합하면 규칙 선택을 유지한다.

    optional 이면 null(도구 불필요)도 유효한 답이다. 이때는 후보 목록까지 비워야
    한다. 남겨 두면 뒤 단계가 다시 ?tool 을 바인딩할 대상으로 읽는다.
    """
    pick = obj.get("selected_tool_id")
    reason = obj.get("tool_reason", "")
    measured = {m["node"]: m for m in s.get("tool_candidates_measured", [])}
    rule = s.get("selected_tool_id")
    if optional and pick is not None and pick in measured:
        # 액션 스키마에 ?tool 이 없으므로 여기서 확정하면 뒤 단계가 바인딩할 곳이 없다.
        # 이 서브골은 도구 사용 형태로 다시 분해되어야 한다는 신호로만 남긴다.
        s["tool_needed_hint"] = {"tool_id": pick, "reason": reason}
        s["tool_candidate_ids"] = []
        s.pop("selected_tool_id", None)
        # 0911: partition_plan 은 지우지 않는다. 도구가 없는 서브골의 분할 계획은
        # ee_pool_one_per_grasp 가 만든 "물체당 한 그룹"이고, 이것이 없으면 여러
        # 물체가 한 서브골에 남아 공통 EE 교집합이 비어 버린다 (c3_1 에서 사과와
        # 접시가 한 그룹에 묶여 EMPTY_FEASIBLE_EE 가 났다). 도구 판정 결과와
        # 무관하게 맨손 분할은 그대로 살려 둔다.
        logs.append(f"  [도구 판정] {s['subgoal_id']}: 도구 필요 신호 {pick} ({reason}) — "
                    f"이 kind({s.get('kind')})에는 도구 액션이 없어 확정하지 않는다")
        return
    if optional and pick is None:
        s["selection_by"] = "llm"
        s["selection_reason_tool"] = reason
        s["selected_tool_id_rule"] = rule
        s["selected_tool_id"] = None
        s["tool_candidate_ids"] = []
        # partition_plan 은 위와 같은 이유로 유지한다 (맨손 분할).
        logs.append(f"  [도구 판정] {s['subgoal_id']}: 도구 불필요로 판정 "
                    f"(규칙 선택 {rule} 기각) ({reason})")
        return
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
    buckets = {"sat": [], "unsat": [], "split": [], "unknown": [], "not_queried": []}
    for d in s["details"]:
        for p in d["pre"]:
            if p.get("eval_by") != "m3" or "status" not in p:
                continue
            ev = "; ".join(str(e) for e in p.get("evidence", [])[:2])[:120]
            proxy = any(e.get("proxy") for e in p.get("evidence", []) if isinstance(e, dict))
            tag = " [대체 판정]" if proxy else ""
            # 0908: batch_feasible 불충족은 실패가 아니라 그룹 분할의 트리거다. M2는 일부러
            # 넓게 분해한 뒤(예: relocate({5개} → 트레이)) 이 측정으로 몇 그룹인지 정한다.
            # 종전에는 이게 "불충족 조건"으로 들어가 신뢰도 갱신 LLM이 계획이 나빠진 신호로
            # 읽었고, c2_1/c3_1/c3_2/c4_2의 분해 신뢰도가 전부 이 사유로 깎였다
            # ("일괄 조작 불가로 서브골 분해 수정 필요" — 애초에 일괄 조작이 목표가 아니었다).
            bucket = ("split" if (p["status"] == "unsat"
                                  and p.get("head") == "batch_feasible")
                      else p["status"])
            buckets.get(bucket, buckets["unknown"]).append(
                f"  - {p['expr'][:60]}: ({ev}){tag}")
    parts = [f"판정 요약: 충족 {len(buckets['sat'])} / 불충족 {len(buckets['unsat'])} / "
             f"분할로 해소 {len(buckets['split'])} / "
             f"근거 미제공 {len(buckets['unknown'])} / 의도적 비측정 {len(buckets['not_queried'])}"]
    labels = {"sat": "충족된 조건:", "unsat": "불충족 조건:",
              "split": ("한 번에 처리하기엔 대상이 많아 그룹 분할로 진행하는 조건 "
                        "(계획 실패가 아니라 예정된 절차다. 이 때문에 신뢰도를 낮추지 말 것):"),
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
        tool_optional = s.get("kind") not in TOOL_KINDS
        if want_tool:
            # 규칙 선택은 LLM에 알리지 않는다 (앵커링 방지: c1_2에서 "1차 선택 spatula"를
            # 보여주니 2mm 두께가 휜다고 스스로 적으면서도 동의했음). 후보는 id순으로 중립 나열.
            prompt += "\n" + PROMPT_TOOL_BLOCK.format(
                kind=s.get("kind"),
                table=_tool_table(sorted(measured, key=lambda m: m["node"])),
                core=(measured[0].get("core_pred") or "없음"),
                optional=TOOL_OPTIONAL_LINE if tool_optional else "")
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
                if want_tool and not tool_optional and not obj.get("selected_tool_id"):
                    raise ValueError("selected_tool_id가 없음 (도구 후보 중 하나를 골라라)")
                if want_tool and tool_optional and "selected_tool_id" not in obj:
                    raise ValueError("selected_tool_id가 없음 (도구를 고르거나 null 로 두라)")
                if want_tool:
                    _apply_llm_tool_choice(s, obj, logs, optional=tool_optional)
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
    # 0908: 계획 자신이 옮길 물체는 blocker로 세지 않는다. 초기 상태에서 A 위에 B가
    # 있어도 B가 이 태스크의 대상이면 실행 순서가 그것을 해소한다 (c2_2: 빵 두 개가
    # 겹쳐 있는데 아래 빵이 5층, 위 빵이 1층이라 1층을 놓는 순간 노출된다).
    # 이런 위반을 재분해 사유로 넘기면 초기 상태는 그대로라 순서를 바꿔도 풀리지 않고,
    # LLM이 상한까지 같은 분해를 반복한다. 판정 결과(evidence)에는 남겨 M4가 본다.
    planned = {t for s in m2_out.get("m2_subgoals", [])
               for t in (s.get("target_ids") or [])}
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
                ev = p.get("evidence") or []
                if head == "clear":
                    # 0903: 목적지가 다른 물체로 막힌 것은 kind를 바꿔도 해소되지 않는다
                    # (트레이가 막힌 건 쓸어 담아도 마찬가지). 목적지 정리는 별도 문제.
                    # 0908: 다만 "이 물체가 목적지에 안 들어간다"(depth_clearance 불충족)는
                    # 성격이 fits와 같다 — 그 대상만 빼거나 다른 목적지로 보내면 풀린다.
                    unfit = sorted({m for e in ev for m in (e.get("unfit_members") or [])})
                    if not unfit:
                        continue
                    adjust_lines.append(
                        f"- {p['expr']} -> 불충족: {', '.join(unfit)}는 이 목적지에 "
                        "안정적으로 담기지 않음 (바닥 면적 부족 또는 넘어짐). 해당 대상만 "
                        "다른 목적지로 보내거나 대상에서 제외할 것")
                    continue
                if head == "top_exposed":
                    blk = {n for e in ev if e.get("top_exposed") is False
                           for n in (e.get("blockers") or [])}
                    who = [e.get("node") for e in ev if e.get("top_exposed") is False]
                    if blk and blk <= planned:
                        continue            # 계획이 먼저 치울 물체 — 재분해로는 못 푼다
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
                if head == "graspable_on_support":
                    # 0909: 어떤 EE 로도 지지면 위에서 접근할 수 없다는 뜻이다. EE 하나라도
                    # 가능하면 sat 이 되고 그 선택은 M4 몫이므로, 여기 오는 것은 맨손 자체가
                    # 불가한 경우뿐이다. 해법은 넣지 않고 관측 사실만 적는다.
                    bad = [e for e in ev if e.get("graspable_on_support") is False]
                    for e in bad:
                        where = ", ".join(e.get("supports") or []) or "지지면"
                        kind_lines.append(
                            f"- {p['expr']} -> 불충족: {e.get('node')}"
                            + (f"(높이 {e['height_mm']}mm)"
                               if e.get("height_mm") is not None else "")
                            + f"가 {where} 위에 놓여 있어 어떤 EE 로도 직접 접근할 수 없음")
                    if bad:
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


def evaluate(m2_out: dict, m1: dict) -> list[tuple[dict, dict]]:
    """0908: 판정 사양을 M1 접지값 + 관계 함수로 직접 계산한다 (M3 왕복 없음).

    반환: [(plan_item, result), ...]. result는 종전 M3 응답과 같은 스키마라
    아래 판정 로직(_judge 계열)을 그대로 쓴다.
    """
    from . import ground
    from .core import plan_evaluations

    nodes = {n["id"]: n for n in m1["nodes"]}
    out = []
    for s in m2_out["m2_subgoals"]:
        ignore = set(s.get("ignore_ids") or [])
        for q in plan_evaluations(s, s.get("details", []), m1=m1):
            c, qid = q["call"], q["queried_by"]
            kind = c["kind"]
            r: dict = {"queried_by": qid}
            try:
                if kind in ("intrinsic", "ee"):
                    n = nodes[c["node_id"]]
                    r["node_id"] = n["id"]
                    r |= {k: v for k, v in n.items()
                          if k in ("geometry", "material", "density_kgm3", "mass_kg",
                                   "youngs_gpa", "mu", "confidence", "caption")}
                    # reachable/ee_usable은 같은 노드 뷰에서 읽히므로 둘 다 실어 둔다
                    if "ee" in n:
                        r |= {"ee": n["ee"], "reachability": n.get("reachability")}
                    elif kind == "ee":
                        r["error"] = f"접지값 없음(ee): {n['id']}"
                elif kind == "relational":
                    r |= {"from": c["a"], "to": c["b"]}
                    r |= ground.fits_inside(nodes[c["a"]], nodes[c["b"]])
                elif kind in ("top_exposed", "clear"):
                    n = nodes[c["node_id"]]
                    pred = (n.get("predicates") or {}).get(kind) or {}
                    r |= {"node_id": n["id"], "type": kind,
                          "value": pred.get("value"), "pass": pred.get("value")}
                    if kind == "top_exposed":
                        r["blockers"] = pred.get("blockers", [])
                    else:
                        r["occupants"] = pred.get("occupants", [])
                        # 0908 팀 스펙: 담는 목적지는 원소별 수용 여부까지 본다.
                        # 영역이 비어 있어도 물체가 안 들어가면 계획이 성립하지 않는다.
                        mem = [m for m in c.get("members", []) if m in nodes]
                        holds = [ground.depth_clearance(nodes[m], n) for m in mem]
                        if holds:
                            unfit = [m for m, h in zip(mem, holds) if not h.get("pass")]
                            ok = bool(r.get("value")) and not unfit
                            r |= {"value": ok, "pass": ok, "depth_clearance": holds,
                                  "unfit_members": unfit}
                elif kind == "graspable_on_support":
                    n = nodes[c["node_id"]]
                    r |= {"node_id": n["id"]}
                    r |= ground.graspable_on_support(n, m1.get("edges", []))
                elif kind == "flat_face":
                    n = nodes[c["node_id"]]
                    pred = (n.get("predicates") or {}).get("flat_face") or {}
                    r |= {"node_id": n["id"], "type": "flat_face",
                          "value": pred.get("value"), "pass": pred.get("value"),
                          "check": pred.get("check")}
                elif kind == "gap_accessible":
                    r |= {"from": c["tool_id"], "to": c["target_id"],
                          "type": "gap_accessible"}
                    r |= ground.gap_access(nodes[c["tool_id"]], nodes[c["target_id"]],
                                           gap_width_mm=c.get("gap_width_mm"))
                elif kind in ("batch", "swept_space"):
                    actor = c.get("actor") or {}
                    tool = nodes.get(actor["id"]) if actor.get("type") == "object" else None
                    members = [nodes[i] for i in c.get("member_ids", []) if i in nodes]
                    r |= {"subgoal_id": s["subgoal_id"], "kind": kind, "actor": actor}
                    if not members:
                        r |= {"feasible": None, "clear": None, "partition": None}
                    elif kind == "batch":
                        r |= ground.batch_partition(members, tool=tool)
                    else:
                        to = nodes.get(c.get("to"))
                        exclude = ({m["id"] for m in members} | ignore | {c.get("to")}
                                   | ({tool["id"]} if tool else set()))
                        others = [n for n in nodes.values() if n["id"] not in exclude]
                        r |= (ground.swept_space(members, to, others, tool=tool) if to
                              else {"clear": None, "margin_mm": None, "blockers": []})
                else:
                    r["error"] = f"unsupported kind: {kind}"
            except KeyError as e:
                r["error"] = f"node not in m1: {e}"
            except Exception as e:                   # 판정 실패는 미판정으로 남긴다
                r["error"] = f"{type(e).__name__}: {e}"
            out.append((q, {"subgoal_id": s["subgoal_id"]} | r))
    return out


def assign_container_slots(m2_out: dict, m1: dict) -> list[str]:
    """같은 컨테이너로 가는 분할 형제들에게 서로 다른 목표 자리를 준다 (0909).

    분할까지는 "각자 하나씩 트레이로 옮긴다"만 정해질 뿐 어디에 놓을지는 아무도
    정하지 않는다. 실행계는 매번 영역 중심을 고르므로 먼저 놓인 것 위로 내려온다
    (c3_1: 접시 위 머그 관통, 놓기 전략 8개 전부 COLLISION_FILTERED_ALL).
    어느 물체를 어디에 놓을지는 계획이 아는 사실이니 여기서 배치를 잡는다.

    자리는 컨테이너 내부 반치수 기준 정규화 좌표로 싣는다 — M1 점군 치수와 실행계
    치수가 어긋나도 비율은 옮겨 가고, 실행계가 실제 점유 상황으로 다시 검증한다.
    """
    from . import ground

    nodes = {n["id"]: n for n in m1["nodes"]}
    groups: dict[tuple[str, str], list[dict]] = {}
    for s in m2_out["m2_subgoals"]:
        parent, container = s.get("split_from"), s.get("container_id")
        if not parent or not container or s.get("kind") != "relocate":
            continue
        groups.setdefault((parent, container), []).append(s)

    logs: list[str] = []
    for (parent, container), sibs in groups.items():
        if len(sibs) < 2 or container not in nodes:
            continue
        members = [nodes[t] for s in sibs for t in s.get("target_ids", [])
                   if t in nodes and "bbox_mm" in nodes[t]]
        if len(members) < 2:
            continue
        layout = ground.container_layout(members, nodes[container])
        if not layout or not layout.get("slots"):
            continue
        slots = layout["slots"]
        placed = 0
        for s in sibs:
            mine = [t for t in s.get("target_ids", []) if t in slots]
            if not mine:
                continue
            slot = slots[mine[0]]
            for d in s.get("details", []):
                if (d.get("binding") or {}).get("?r") != container:
                    continue
                d.setdefault("action_parameters", {})["placement_slot"] = {
                    "region": container, "target": mine[0], "uv": list(slot["uv"]),
                    "offset_mm": list(slot["offset_mm"]), "row": slot["row"],
                    "source": "m2_container_layout"}
                placed += 1
            s["placement_slot"] = {"region": container, "uv": list(slot["uv"]),
                                   "row": slot["row"]}
        note = "" if layout.get("pass") else f", 한 층 초과 {layout.get('overflow')}"
        logs.append(f"  [배치] {parent} -> {container}: {len(slots)}자리 / "
                    f"{layout.get('rows')}줄, 상세 {placed}건에 부여{note}")
    return logs


def apply_grounding(m2_out: dict, m1: dict) -> list[str]:
    """M1 접지값으로 m3 술어를 판정하고 도구를 확정한다. (m2_out은 제자리 수정)

    0908 재구조: 종전 apply_m3(m2_out, m3_responses)를 대체한다. 측정값은 m1에 실려
    오고, 쌍/집합 술어는 evaluate()가 관계 함수로 직접 계산한다. 원본 판정 결과는
    서브골 measurements에 남겨 gk로 실린다 (M4/M6가 본다).
    """
    pairs = evaluate(m2_out, m1)
    plan = [q for q, _ in pairs]
    responses = [r for _, r in pairs]
    m2_out.setdefault("m2_stats", {})["n_evaluations"] = len(responses)
    for s in m2_out["m2_subgoals"]:
        s["measurements"] = [r for r in responses
                             if r.get("subgoal_id") == s["subgoal_id"]]
    return _apply_responses(m2_out, plan, responses)


def _apply_responses(m2_out: dict, plan: list[dict], responses: list[dict]) -> list[str]:
    """판정 결과를 술어 status/evidence에 반영하고 도구를 확정한다."""
    by_q: dict[str, list[dict]] = {}
    for r in responses:
        by_q.setdefault(r.get("queried_by"), []).append(r)
    view = _node_view(responses)

    # 판정 사양에서 술어 id → 대상 노드 목록 (그 술어가 실제로 본 노드들)
    q_nodes: dict[str, list[str]] = {}
    for q in plan:
        c = q["call"]
        if c.get("kind") in ("batch", "swept_space"):   # 0828 신규 — 노드 대신 그룹 대상
            q_nodes.setdefault(q["queried_by"], [])
            continue
        n = c.get("node_id") or c.get("a") or c.get("tool_id")   # gap_accessible은 tool_id
        if n:
            q_nodes.setdefault(q["queried_by"], [])
            if n not in q_nodes[q["queried_by"]]:
                q_nodes[q["queried_by"]].append(n)

    logs, n_sat = [], {"sat": 0, "unsat": 0, "unknown": 0, "not_queried": 0}
    for s in m2_out["m2_subgoals"]:
        for d in s["details"]:
            for p in d["pre"]:
                if p["eval_by"] != "m3":
                    continue
                rs = by_q.get(p["id"], [])
                rels = [r for r in rs if ("pass" in r or "check" in r)]
                answered_nodes = [r.get("node_id") or r.get("from")
                                  for r in rs
                                  if (r.get("node_id") or r.get("from"))]
                nodes = q_nodes.get(p["id"], []) or answered_nodes
                # 판정 사양이 대상을 못 잡은 술어(도구 거치 위치 tool_rest 등)는 not_queried.
                # 나머지는 전부 이번 실행에서 계산된 결과가 있다 (왕복이 없어 미회신 없음).
                if p["id"] not in q_nodes and not answered_nodes:
                    status, ev = "not_queried", []
                elif p["head"] in ("batch_feasible", "act_space_clear"):
                    status, ev = _judge_group(p["head"], rs)
                else:
                    # tool 액션이 있는 서브골(sweep류)은 any, 없는 서브골(relocate
                    # 그룹)은 원소 전원 충족 의미론 (0828). 0911: 기준을 후보 유무에서
                    # ?tool 바인딩 유무로 바꾼다 — 이제 relocate/stack 도 후보를 받으므로
                    # 후보 유무로 가르면 쌓기의 대상 판정이 any 로 느슨해진다.
                    status, ev = _judge(p["head"], nodes, view, rels,
                                        require_all=not binds_tool(s.get("details", [])))
                p["status"], p["evidence"] = status, ev
                n_sat[status] += 1
                line = f"  {p['id']:14s} {p['expr'][:52]:52s} -> {status}"
                if status == "unknown":
                    line += "  (접지값에 판정 근거 필드 없음)"
                if status == "not_queried":
                    line += "  (판정 대상 아님: 도구 거치 위치 등)"
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
        # 0908: 분할 뒤 재판정에서 이 블록이 다시 돈다. LLM이 이미 고른 서브골은
        # 후보가 그 하나로 좁혀져 있어 규칙이 같은 답을 내지만, 그대로 두면
        # selection_by가 "rule"로 덮여 어블레이션 로그에서 LLM 판정분이 사라진다.
        llm_kept = s.get("selection_by") == "llm" and s.get("selected_tool_id") == chosen
        s["selected_tool_id"] = chosen
        # 0908 통합: 접지값이 m1에 전부 실려 오므로 "핵심 술어 미측정" 경고는 없앴다.
        core = _CORE_PRED.get(s.get("kind"))
        if part and len(part) > 1:
            s["partition_plan"] = part        # regroup이 이 구성대로 서브골을 나눈다
        s["selection_evidence"] = [
            {"node": c2, "feasible_ees": f2, "reach_margin_mm": m3,
             "batch_groups": (len(p2) if p2 else None)}
            for (br2, m3, n2, c2, f2, p2) in scored]
        s["selection_by"] = "llm" if llm_kept else "rule"
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
        logs.append(f"  [도구 확정] {s['subgoal_id']}: {chosen} "
                    + ("유지 (LLM 판정)" if llm_kept else "선택 (규칙)")
                    + f" — 사용 가능 EE {n_ee}종, 리치 여유 {margin}mm"
                    + (f", 필요 액션 {len(part)}회" if part else ""))

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
    logs.insert(0, f"[판정] 접지값 {len(responses)}건으로 m3 술어 {total}건 판정: "
                   f"sat {n_sat['sat']} / unsat {n_sat['unsat']} / "
                   f"unknown {n_sat['unknown']} / 판정대상아님 {n_sat['not_queried']}")
    return logs
