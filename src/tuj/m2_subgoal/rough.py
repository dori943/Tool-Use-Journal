# -*- coding: utf-8 -*-
"""M2-1 — 서브골 분해 + 서브골별 객체 선택 (LLM 2회 호출, 0821 결정).

  1차 호출  자연어 task + M1 노드 → planning-level 서브골 분해
  2차 호출  서브골별로 장면 노드 중 관련 객체 선택 (object_ids)
            도구가 필요한 서브골이면 도구로 쓸 물체를 tool_candidate_ids로 선별
            선택 이유를 selection_reason에 남긴다

모든 id는 M1 노드 id 중에서만 고른다 (새 이름 생성 금지). 호출마다 검문 + 1회 재시도.

구현체
  LLMRough       기본 경로 (OPENAI_API_KEY 필수)
  TemplateRough  결정론 폴백 — 태스크 키워드 + class→역할 규칙표. 회귀·테스트용.

출력 스키마 (두 구현체 공통, 최소):
  [{"subgoal_id", "goal",                       # goal: G_k에 실리는 한 줄 (← M2)
    "kind": "relocate" | "sweep_collect",
    "target_ids": [...], "container_id": str|None,
    "tool_candidate_ids": [...]}]               # 도구 후보만 나열 — 선택은 M4
"""
from __future__ import annotations

import json
import os
import time


class TemplateRough:
    """결정론 서브골 생성. category_map: class → container id (분류형 태스크용)."""

    name = "template"

    def __init__(self, category_map: dict | None = None):
        self.category_map = category_map or {}

    def generate(self, task: str, m1: dict) -> list[dict]:
        nodes = m1["nodes"]
        ids = [n["id"] for n in nodes]

        def by_class(*keys):
            return [n["id"] for n in nodes if any(k in n["class"] for k in keys)]

        # ── 수거/쓸기형: "쓸어" / "sweep" ──
        if ("쓸어" in task) or ("sweep" in task.lower()):
            blocks = by_class("block", "lego")
            zone = (by_class("collection", "zone", "tray") or [None])[0]
            tools = by_class("plate", "board", "bottle")   # 후보만 — 적합성은 M3, 선택은 M4
            return [{
                "subgoal_id": "SG1",
                "goal": f"흩어진 블록 {len(blocks)}개를 {zone}로 쓸어 담는다",
                "kind": "sweep_collect",
                "target_ids": blocks,
                "container_id": zone,
                "tool_candidate_ids": tools,
            }]

        # ── 분류/운반형: category_map이 주어진 경우 ──
        if self.category_map:
            out, i = [], 0
            for n in nodes:
                dest = self.category_map.get(n["class"])
                if dest is None:
                    continue
                i += 1
                out.append({
                    "subgoal_id": f"SG{i}",
                    "goal": f"{n['id']}를 {dest}로 옮긴다",
                    "kind": "relocate",
                    "target_ids": [n["id"]],
                    "container_id": dest,
                    "tool_candidate_ids": [],
                })
            if out:
                return out

        raise ValueError(f"TemplateRough가 처리 못 하는 task: {task!r} — LLMRough 사용 또는 규칙 추가")


PROMPT = """로봇 매니퓰레이션 태스크를 planning-level 서브골로 분해하라.

태스크: {task}
장면 노드 id: {ids}

규칙:
- 서브골은 로봇 세부 동작이 아니라 의미 단위 작업이다.
- 태스크에 명시된 목표만 서브골로 만든다. 태스크에 없는 목표(예: 장애물 치우기,
  정리하기)를 발명하지 않는다.
- target_ids / container_id / tool_candidate_ids 에는 위 노드 id만 쓴다.
- 도구를 '선택'하지 말라. 후보만 나열한다 (선택은 뒤 모듈이 한다).
- kind는 relocate(옮기기) 또는 sweep_collect(쓸어 담기) 중 하나.
- kind와 무관하게, 같은 목적지로 가는 대상 전체를 서브골 하나의 target_ids에 담는다.
  대상별로 쪼개지 않는다. 몇 그룹으로 나눠 처리할지는 뒤 모듈이 측정값을 보고 정한다.
- 도구는 여기서 다루지 않는다. 도구 선별은 다음 단계(객체 선택)가 한다.
- goal은 한국어 한 문장으로 쓴다.
- confidence: 이 서브골 분해가 옳다는 확신을 0.0~1.0으로 매긴다. 지금은 물체의
  질량·재질·파지 가능 여부 등 측정값이 없는 상태다. 근거 없이 후하게 주지 말 것.
  (0.9 이상 = 거의 확실 / 0.7 = 대체로 확신 / 0.5 = 반반 / 0.3 = 실패 가능성 높음)
- JSON 배열만 출력한다.

출력 형식:
[{{"subgoal_id":"SG1","goal":"...","kind":"relocate","target_ids":["..."],"container_id":"...","confidence":0.0}}]"""


PROMPT_SELECT = """너는 로봇 계획의 객체 선택 단계다. 각 서브골에 대해, 장면 노드 중
그 서브골 수행에 관련된 객체를 선택하라.

태스크: {task}
장면 노드 (id: class):
{nodes}

서브골:
{subgoals}

규칙:
- object_ids에는 이 서브골에서 물리·기하 추론이 필요한 객체만 담는다.
  장면 노드를 전부 나열하는 것은 선택이 아니다.
- kind가 sweep_collect면 도구로 쓸 수 있는 물체를 장면에서 골라
  tool_candidate_ids에 1개 이상 담는다 (object_ids에도 포함).
  후보는 복수 가능하며 어느 것을 쓸지 최종 선택은 뒤 모듈이 측정값을 보고 한다.
- kind가 relocate면 대상과 목적지를 object_ids에 담는다 (들어가는지 판정에 필요).
- 위 장면 노드 id만 쓴다. reason에 선택 이유를 한국어 한 문장으로 쓴다.
- confidence: 이 객체 선택이 옳다는 확신을 0.0~1.0으로 매긴다. 지금은 물체의
  질량·재질·파지 가능 여부 등 측정값이 없는 상태다. 근거 없이 후하게 주지 말 것.
  (0.9 이상 = 거의 확실 / 0.7 = 대체로 확신 / 0.5 = 반반 / 0.3 = 실패 가능성 높음)
- uncertain_about: 지금 가장 불확실한 요소 1~2개를 구체적으로 적는다.
- JSON 배열만 출력한다.

출력 형식:
[{{"subgoal_id":"SG1","object_ids":["..."],"tool_candidate_ids":[],"reason":"...",
  "confidence":0.0,"uncertain_about":["..."]}}]"""


PROMPT_COMBINED = """로봇 매니퓰레이션 태스크를 planning-level 서브골로 분해하고,
각 서브골에서 다룰 객체 선택까지 한 번에 수행하라.

태스크: {task}
장면 노드 (id: class):
{nodes}

분해 규칙:
- 서브골은 로봇 세부 동작이 아니라 의미 단위 작업이다.
- 태스크에 명시된 목표만 서브골로 만든다. 태스크에 없는 목표(예: 장애물 치우기,
  정리하기)를 발명하지 않는다.
- target_ids / container_id / tool_candidate_ids / object_ids 에는 위 노드 id만 쓴다.
- kind는 relocate(옮기기) 또는 sweep_collect(쓸어 담기) 중 하나.
- kind와 무관하게, 같은 목적지로 가는 대상 전체를 서브골 하나의 target_ids에 담는다.
  대상별로 쪼개지 않는다. 몇 그룹으로 나눠 처리할지는 뒤 모듈이 측정값을 보고 정한다.
- goal은 한국어 한 문장으로 쓴다.

선택 규칙:
- object_ids에는 이 서브골에서 물리·기하 추론이 필요한 객체만 담는다.
  장면 노드를 전부 나열하는 것은 선택이 아니다.
- kind가 sweep_collect면 도구로 쓸 수 있는 물체를 장면에서 골라
  tool_candidate_ids에 1개 이상 담는다 (object_ids에도 포함).
  후보는 복수 가능하며 어느 것을 쓸지 최종 선택은 뒤 모듈이 측정값을 보고 한다.
- kind가 relocate면 대상과 목적지를 object_ids에 담는다 (들어가는지 판정에 필요).
- reason에 선택 이유를 한국어 한 문장으로 쓴다.
- confidence: 분해와 객체 선택 각각의 확신을 0.0~1.0으로 매긴다. 지금은 물체의
  질량·재질·파지 가능 여부 등 측정값이 없는 상태다. 근거 없이 후하게 주지 말 것.
  (0.9 이상 = 거의 확실 / 0.7 = 대체로 확신 / 0.5 = 반반 / 0.3 = 실패 가능성 높음)
- uncertain_about: 지금 가장 불확실한 요소 1~2개를 구체적으로 적는다.
- JSON 배열만 출력한다.

출력 형식:
[{{"subgoal_id":"SG1","goal":"...","kind":"relocate","target_ids":["..."],"container_id":"...",
  "object_ids":["..."],"tool_candidate_ids":[],"reason":"...",
  "confidence":{{"decomposition":0.0,"object_selection":0.0}},"uncertain_about":["..."]}}]"""


def track_usage(acc: dict, stage: str, response, seconds: float = 0.0) -> None:
    """OpenAI 응답의 usage와 소요 시간을 단계별로 누적한다. usage가 없으면 무시."""
    u = getattr(response, "usage", None)
    if u is None:
        return
    e = acc.setdefault(stage, {"calls": 0, "prompt_tokens": 0,
                               "completion_tokens": 0, "tokens": 0, "seconds": 0.0})
    e["calls"] += 1
    e["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
    e["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
    e["tokens"] += getattr(u, "total_tokens", 0) or 0
    e["seconds"] = round(e.get("seconds", 0.0) + seconds, 2)


def _check_conf(v, where: str):
    """자기보고 신뢰도 검문 — 0.0~1.0 숫자여야 한다."""
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.0 <= v <= 1.0:
        raise ValueError(f"{where}: confidence가 0~1 숫자가 아님 ({v!r})")


def validate_subgoals(subs: list[dict], ids: list[str]) -> list[dict]:
    """LLM 출력 검문 + 정규화. 프롬프트는 지시, 여기는 검문 — 둘 다 있어야 안전하다.

    검사: 장면에 없는 id / sweep_collect의 빈 도구 후보 → ValueError (호출부가 재시도)
    정규화: 같은 (kind, 목적지) 서브골 병합, subgoal_id 재부여
    (0828: relocate 물체당 강제 분리 제거 — 분할은 측정 후 regroup이 한다)
    """
    known = set(ids)
    for s in subs:
        bad = [x for x in s.get("target_ids", []) if x not in known]
        cid = s.get("container_id")
        if cid is not None and cid not in known:
            bad.append(cid)
        bad += [x for x in s.get("tool_candidate_ids", []) if x not in known]
        if bad:
            raise ValueError(f"장면에 없는 id: {bad}")
        _check_conf(s.get("confidence"), f"{s.get('subgoal_id')} 분해")

    out = []
    for s in subs:
        s.setdefault("container_id", None)
        s.setdefault("tool_candidate_ids", [])
        out.append(s)   # 0828: relocate 물체당 강제 분리 제거 — 분할은 측정 후 regroup이 한다

    # 같은 (kind, 목적지) 서브골은 하나로 합친다. 분해 시점에는 목적지 단위로만 묶고,
    # 몇 그룹으로 나눠 처리할지는 측정(batch 질의) 후 regroup이 정한다 (0828 결정).
    merged, by_dst = [], {}
    for s in out:
        key = (s.get("kind"), s.get("container_id"))
        if key in by_dst:
            t = by_dst[key]
            t["target_ids"] += [x for x in s["target_ids"] if x not in t["target_ids"]]
            t["tool_candidate_ids"] += [x for x in s["tool_candidate_ids"]
                                        if x not in t["tool_candidate_ids"]]
            verb = "쓸어 담는다" if s.get("kind") == "sweep_collect" else "옮긴다"
            t["goal"] = f"대상 {len(t['target_ids'])}개를 {t['container_id']}로 {verb}"
            continue
        by_dst[key] = s
        merged.append(s)
    out = merged

    for i, s in enumerate(out, 1):                  # 분리·병합 후 id 재부여 (중복 방지)
        s["subgoal_id"] = f"SG{i}"
    return out


def validate_selection(sel: list[dict], subgoals: list[dict], ids: list[str]) -> dict:
    """2차(객체 선택) 출력 검문. 통과 시 {subgoal_id: 선택 항목} 반환, 위반 시 ValueError."""
    known = set(ids)
    by_id = {}
    for e in sel:
        sid = e.get("subgoal_id")
        bad = [x for x in e.get("object_ids", []) + e.get("tool_candidate_ids", [])
               if x not in known]
        if bad:
            raise ValueError(f"{sid}: 장면에 없는 id {bad}")
        missing_tools = [t for t in e.get("tool_candidate_ids", [])
                         if t not in e.get("object_ids", [])]
        if missing_tools:
            raise ValueError(f"{sid}: 도구 후보 {missing_tools}는 object_ids에도 포함하라")
        _check_conf(e.get("confidence"), f"{sid} 객체 선택")
        by_id[sid] = e
    for s in subgoals:
        e = by_id.get(s["subgoal_id"])
        if e is None:
            raise ValueError(f"{s['subgoal_id']}의 선택 항목이 없다. 서브골마다 하나씩 출력하라")
        if s["kind"] == "sweep_collect" and not e.get("tool_candidate_ids"):
            raise ValueError(f"{s['subgoal_id']}(sweep_collect): 도구로 쓸 물체를 "
                             "tool_candidate_ids에 1개 이상 골라라")
    return by_id


class LLMRough:
    """LLM 2회 호출: 1차 서브골 분해 → 2차 서브골별 객체 선택 (0821 결정)."""

    name = "llm"

    def __init__(self, model: str | None = None):
        self._client = None                     # 지연 생성 (테스트에서 키 없이 로드 가능)
        self.model = model or os.environ.get("TUJ_M2_MODEL", "gpt-4o")
        # 단계별 토큰 집계 (0828 — 랩미팅 지적: 호출 1회 통합 A/B의 비교 근거)
        self.usage: dict[str, dict] = {}

    def _json_call(self, prompt: str, validate, stage: str):
        """호출 → JSON 파싱 → 검문. 실패 시 오류를 붙여 1회 재시도."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        err = None
        for attempt in (1, 2):
            msg = prompt if attempt == 1 else prompt + f"\n\n이전 출력의 문제: {err}. 고쳐서 JSON 배열만 다시 출력하라."
            t0 = time.monotonic()
            r = self._client.chat.completions.create(
                model=self.model, temperature=0,
                messages=[{"role": "user", "content": msg}])
            track_usage(self.usage, stage, r, seconds=time.monotonic() - t0)
            text = r.choices[0].message.content.strip()
            try:
                parsed = json.loads(text[text.find("["): text.rfind("]") + 1])
                return validate(parsed)
            except (ValueError, json.JSONDecodeError) as e:
                err = str(e)
        raise ValueError(f"LLM {stage} 2회 실패: {err}")

    def generate(self, task, m1):
        ids = [n["id"] for n in m1["nodes"]]
        node_lines = "\n".join(f"  {n['id']}: {n.get('class', '?')}" for n in m1["nodes"])

        # ── 실험 모드 (0828, 랩미팅 지적): TUJ_M2_SINGLE_CALL=1 이면
        #    분해+선택을 1회 호출로 통합. 검문은 기존 2단 검문을 그대로 재사용.
        if os.environ.get("TUJ_M2_SINGLE_CALL") == "1":
            return self._generate_single(task, ids, node_lines)

        # 1차 — 서브골 분해
        subgoals = self._json_call(
            PROMPT.format(task=task, ids=", ".join(ids)),
            lambda subs: validate_subgoals(subs, ids), "서브골 분해")

        # 2차 — 서브골별 객체 선택 (도구로 쓸 물체 선별 포함)
        brief = [{"subgoal_id": s["subgoal_id"], "goal": s["goal"], "kind": s["kind"],
                  "target_ids": s["target_ids"], "container_id": s["container_id"]}
                 for s in subgoals]
        sel = self._json_call(
            PROMPT_SELECT.format(task=task, nodes=node_lines,
                                 subgoals=json.dumps(brief, ensure_ascii=False, indent=2)),
            lambda x: validate_selection(x, subgoals, ids), "객체 선택")

        for s in subgoals:
            e = sel[s["subgoal_id"]]
            s["object_ids"] = e["object_ids"]
            s["tool_candidate_ids"] = e.get("tool_candidate_ids", [])
            s["selection_reason"] = e.get("reason", "")
            # 생성 시점 자기보고 신뢰도 (측정값 없는 상태). M3 반영 후 갱신은 ingest가 채운다.
            s["confidence"] = {
                "before": {"decomposition": s.pop("confidence", None),
                           "object_selection": e.get("confidence"),
                           "uncertain_about": e.get("uncertain_about", [])},
            }
        return subgoals

    def _generate_single(self, task, ids, node_lines):
        """실험: 분해+선택 통합 1회 호출. 출력 구조는 2회 호출과 동일하게 맞춘다."""
        def validate(parsed):
            for e in parsed:
                conf = e.get("confidence") or {}
                if not isinstance(conf, dict):
                    raise ValueError(f"{e.get('subgoal_id')}: confidence는 "
                                     "{decomposition, object_selection} 객체여야 한다")
                _check_conf(conf.get("decomposition"), f"{e.get('subgoal_id')} 분해")
                _check_conf(conf.get("object_selection"), f"{e.get('subgoal_id')} 객체 선택")
            # 기존 1차 검문 재사용 (confidence는 분해값을 숫자로 실어 통과시킨다)
            subs = [dict(e, _conf=e.get("confidence") or {},
                         confidence=(e.get("confidence") or {}).get("decomposition"))
                    for e in parsed]
            subs = validate_subgoals(subs, ids)
            # 기존 2차 검문 재사용
            sel = [{"subgoal_id": s["subgoal_id"],
                    "object_ids": s.get("object_ids", []),
                    "tool_candidate_ids": s.get("tool_candidate_ids", []),
                    "reason": s.get("reason", ""),
                    "confidence": s.get("_conf", {}).get("object_selection"),
                    "uncertain_about": s.get("uncertain_about", [])}
                   for s in subs]
            validate_selection(sel, subs, ids)
            return subs

        subgoals = self._json_call(
            PROMPT_COMBINED.format(task=task, nodes=node_lines),
            validate, "통합(분해+선택)")
        for s in subgoals:
            conf = s.pop("_conf", {})
            s["object_ids"] = s.get("object_ids", [])
            s["selection_reason"] = s.pop("reason", "")
            s.pop("confidence", None)
            s["confidence"] = {
                "before": {"decomposition": conf.get("decomposition"),
                           "object_selection": conf.get("object_selection"),
                           "uncertain_about": s.pop("uncertain_about", [])},
            }
        return subgoals
