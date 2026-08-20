# -*- coding: utf-8 -*-
"""M1-1 — planning-level 서브골 생성 (M1의 유일한 LLM 호출 지점).

자연어 task + M0 그래프(노드 어휘)에서 의미 단위 서브골을 생성한다.
target/container/tool 후보는 반드시 M0 노드 id 중에서 고른다 (새 이름 생성 금지).

구현체
  LLMRough       OPENAI_API_KEY 있으면 사용. 최소 프롬프트, 파싱 실패 1회 재시도.
  TemplateRough  결정론 폴백 — 태스크 키워드 + class→역할 규칙표. 회귀·오프라인 산출용.
                 (LLM 경로와 출력 스키마 동일. 오늘의 C1-T1/C2-T1 JSON 산출은 이 경로)

출력 스키마 (두 구현체 공통, 최소):
  [{"subgoal_id", "goal",                       # goal: G_k에 실리는 한 줄 (← M1)
    "kind": "relocate" | "sweep_collect",
    "target_ids": [...], "container_id": str|None,
    "tool_candidate_ids": [...]}]               # 도구 후보만 나열 — 선택은 M3
"""
from __future__ import annotations

import json
import os


class TemplateRough:
    """결정론 서브골 생성. category_map: class → container id (분류형 태스크용)."""

    name = "template"

    def __init__(self, category_map: dict | None = None):
        self.category_map = category_map or {}

    def generate(self, task: str, m0: dict) -> list[dict]:
        nodes = m0["nodes"]
        ids = [n["id"] for n in nodes]

        def by_class(*keys):
            return [n["id"] for n in nodes if any(k in n["class"] for k in keys)]

        # ── 수거/쓸기형: "쓸어" / "sweep" ──
        if ("쓸어" in task) or ("sweep" in task.lower()):
            blocks = by_class("block", "lego")
            zone = (by_class("collection", "zone", "tray") or [None])[0]
            tools = by_class("plate", "board", "bottle")   # 후보만 — 적합성은 M2, 선택은 M3
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
- relocate의 target_ids는 1개다. 옮길 물체가 여러 개면 물체당 서브골을 하나씩 만든다.
- sweep_collect는 반대다. 같은 목적지로 쓸어 담는 대상 전체를 서브골 하나의
  target_ids에 담는다. 대상별로 쪼개지 않는다 (한 번의 sweep으로 여러 개를 담는 동작이다).
- sweep_collect에는 도구가 될 만한 노드를 tool_candidate_ids에 반드시 1개 이상 나열한다.
- goal은 한국어 한 문장으로 쓴다.
- JSON 배열만 출력한다.

출력 형식:
[{{"subgoal_id":"SG1","goal":"...","kind":"relocate","target_ids":["..."],"container_id":"...","tool_candidate_ids":[]}}]"""


def validate_subgoals(subs: list[dict], ids: list[str]) -> list[dict]:
    """LLM 출력 검문 + 정규화. 프롬프트는 지시, 여기는 검문 — 둘 다 있어야 안전하다.

    검사: 장면에 없는 id / sweep_collect의 빈 도구 후보 → ValueError (호출부가 재시도)
    정규화: 다중 target relocate를 물체당 서브골로 분리, subgoal_id 재부여
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
        if s.get("kind") == "sweep_collect" and not s.get("tool_candidate_ids"):
            raise ValueError("sweep_collect에 tool_candidate_ids가 비었다. 도구 후보를 나열하라")

    out = []
    for s in subs:
        s.setdefault("container_id", None)
        s.setdefault("tool_candidate_ids", [])
        if s.get("kind") == "relocate" and len(s.get("target_ids", [])) > 1:
            for t in s["target_ids"]:               # 물체당 서브골로 분리
                out.append({**s, "target_ids": [t],
                            "goal": f"{t}를 {s['container_id']}로 옮긴다"})
        else:
            out.append(s)

    # sweep_collect는 반대로 합친다. 같은 목적지를 대상별로 쪼개면
    # 도구를 대상 수만큼 잡았다 놓는 계획이 된다.
    merged, sweep_by_dst = [], {}
    for s in out:
        if s.get("kind") == "sweep_collect" and s.get("container_id") in sweep_by_dst:
            t = sweep_by_dst[s["container_id"]]
            t["target_ids"] += [x for x in s["target_ids"] if x not in t["target_ids"]]
            t["tool_candidate_ids"] += [x for x in s["tool_candidate_ids"]
                                        if x not in t["tool_candidate_ids"]]
            t["goal"] = f"대상 {len(t['target_ids'])}개를 {t['container_id']}로 쓸어 담는다"
            continue
        if s.get("kind") == "sweep_collect":
            sweep_by_dst[s["container_id"]] = s
        merged.append(s)
    out = merged

    for i, s in enumerate(out, 1):                  # 분리·병합 후 id 재부여 (중복 방지)
        s["subgoal_id"] = f"SG{i}"
    return out


class LLMRough:
    name = "llm"

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.environ.get("TUJ_M1_MODEL", "gpt-4o")

    def generate(self, task, m0):
        ids = [n["id"] for n in m0["nodes"]]
        prompt = PROMPT.format(task=task, ids=", ".join(ids))
        err = None
        for attempt in (1, 2):
            msg = prompt if attempt == 1 else prompt + f"\n\n이전 출력 파싱 실패: {err}. JSON 배열만."
            r = self.client.chat.completions.create(
                model=self.model, temperature=0,
                messages=[{"role": "user", "content": msg}])
            text = r.choices[0].message.content.strip()
            try:
                t = text[text.find("["): text.rfind("]") + 1]
                subs = json.loads(t)
                return validate_subgoals(subs, ids)
            except (ValueError, json.JSONDecodeError) as e:
                err = str(e)
        raise ValueError(f"LLM 러프 분해 2회 실패: {err}")
