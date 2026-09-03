# -*- coding: utf-8 -*-
"""confidence 실험 — M3 접지 전후 LLM 자기보고 신뢰도 변화 측정.

목적: "측정값 없이 내린 계획"과 "M3 측정값을 반영한 계획"에 대해 LLM이 스스로
매기는 신뢰도가 어떻게 변하는지 실측 → 교수님 상의 자료.

사용법:
  python scripts/exp_confidence.py c1_1              # 1회 측정
  python scripts/exp_confidence.py c1_1 --runs 3     # 3회 반복 (평균·편차 확인용)

전제: output/<task>/m1.json 과 m3.json 이 있어야 한다. OPENAI_API_KEY 필수.
출력: experiments/confidence/<task>.json + 콘솔 표 (사전 / 사후 / 변화량)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tuj.m2_subgoal.ingest import apply_m3
from tuj.m2_subgoal.pipeline import run_m2
from tuj.m2_subgoal.rough import LLMRough

TASKS = {
    "c1_1": "모든 레고 블록을 초록색 영역 내부로 이동시켜라",
    "c2_1": "테이블 위 물건을 전부 수거함에 담아라",
}

ASPECTS = ["decomposition", "object_selection", "overall"]

PROMPT_CONF = """너는 방금 아래 로봇 계획을 만든 계획자다. 스스로의 판단에 대한
신뢰도를 평가하라.

태스크: {task}

네가 만든 계획 요약:
{plan}

{grounding}

각 항목에 0.0~1.0 신뢰도를 매겨라. 근거 없이 후하게 주지 말 것.
척도 기준: 0.9 이상 = 거의 확실 / 0.7 = 대체로 확신 / 0.5 = 반반 /
0.3 = 실패 가능성 높음 / 0.1 이하 = 거의 확실히 실패
- decomposition: 서브골 분해가 태스크를 정확히 담았는가
- object_selection: 객체 선택(도구 포함)이 옳았는가
- overall: 이 계획대로 실행하면 태스크가 성공할 것인가

uncertain_about에는 지금 가장 불확실한 요소를 1~2개만 구체적으로 적어라.

JSON 객체만 출력:
{{"decomposition": 0.0, "object_selection": 0.0, "overall": 0.0,
  "uncertain_about": ["..."], "reason": "한 문장"}}"""

NO_GROUNDING = ("지금은 측정값이 전혀 없다. 장면에는 물체의 대략적 위치와 종류만 있고,"
                " 물체의 질량·재질·잡을 수 있는지 여부는 모르는 상태다.")
WITH_GROUNDING = """M3 측정 모듈이 계획의 조건들을 실측한 결과가 도착했다:
{summary}
이 측정 결과까지 반영한 상태의 신뢰도를 평가하라."""


def plan_summary(out: dict) -> str:
    lines = []
    for s in out["m2_subgoals"]:
        lines.append(f"- {s['subgoal_id']} ({s['kind']}): {s['goal']}")
        lines.append(f"  대상 {len(s['target_ids'])}개, 목적지 {s['container_id']}")
        lines.append(f"  선택 객체 {len(s.get('object_ids', []))}개, "
                     f"도구 후보 {s.get('tool_candidate_ids', [])}")
    lines.append(f"- 실행 순서 제약 {len(out['m2_partial_order'])}개, "
                 f"M3 측정 요청 {len(out['m2_queries'])}건")
    return "\n".join(lines)


def m3_summary(out: dict) -> str:
    """측정 결과 요약. 상태의 성격을 구분해서 전달한다 —
    not_queried는 불확실성이 아니라 의도적 비측정이므로 리스크로 읽히지 않게 분리."""
    sat, unsat, unknown, skipped = [], [], [], []
    for s in out["m2_subgoals"]:
        for d in s["details"]:
            for p in d["pre"]:
                if p.get("eval_by") != "m3" or "status" not in p:
                    continue
                ev = "; ".join(str(e) for e in p.get("evidence", [])[:2])[:120]
                proxy = any(e.get("proxy") for e in p.get("evidence", []) if isinstance(e, dict))
                tag = " [대체 판정: 도구 파지 가능성 기준]" if proxy else ""
                line = f"  - {p['expr'][:60]}: ({ev}){tag}"
                {"sat": sat, "unsat": unsat, "unknown": unknown,
                 "not_queried": skipped}.get(p["status"], unknown).append(line)
    parts = [f"판정 요약: 충족 {len(sat)} / 불충족 {len(unsat)} / "
             f"근거 미제공 {len(unknown)} / 의도적 비측정 {len(skipped)}"]
    if sat:
        parts.append("충족된 조건 (측정 근거 포함):")
        parts += sat
    if unsat:
        parts.append("불충족 조건:")
        parts += unsat
    if unknown:
        parts.append("측정을 요청했으나 응답에 판정 근거가 없던 조건:")
        parts += unknown
    if skipped:
        parts.append("의도적으로 측정하지 않은 조건 (도구 거치 위치 등 측정 대상 아님 — 불확실성으로 취급하지 말 것):")
        parts += skipped
    for s in out["m2_subgoals"]:
        if s.get("selected_tool_id"):
            parts.append(f"도구 확정: {s['selected_tool_id']} "
                         f"(근거 {json.dumps(s.get('selection_evidence', []), ensure_ascii=False)[:150]})")
    return "\n".join(parts)


def ask_confidence(client, model, task: str, plan: str, grounding: str) -> dict:
    prompt = PROMPT_CONF.format(task=task, plan=plan, grounding=grounding)
    err = None
    for attempt in (1, 2):
        msg = prompt if attempt == 1 else prompt + f"\n\n이전 출력 문제: {err}. JSON 객체만."
        r = client.chat.completions.create(model=model, temperature=0,
                                           messages=[{"role": "user", "content": msg}])
        text = r.choices[0].message.content.strip()
        try:
            obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
            for a in ASPECTS:
                v = obj.get(a)
                if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                    raise ValueError(f"{a} 값이 0~1 숫자가 아님: {v!r}")
            if not isinstance(obj.get("uncertain_about", []), list):
                raise ValueError("uncertain_about는 문자열 배열이어야 한다")
            return obj
        except (ValueError, json.JSONDecodeError) as e:
            err = str(e)
    raise ValueError(f"confidence 응답 2회 실패: {err}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "c1_1"
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 1
    task = TASKS[name]
    tdir = os.path.join("output", name)

    with open(os.path.join(tdir, "m1.json"), encoding="utf-8") as f:
        m1 = json.load(f)
    with open(os.path.join(tdir, "m3.json"), encoding="utf-8") as f:
        raw = json.load(f)
    responses = raw["responses"] if isinstance(raw, dict) else raw

    rough = LLMRough()
    results = []
    for i in range(runs):
        out = run_m2(task, m1, rough=rough)            # 분해 + 객체 선택 (LLM 2회)
        if rough._client is None:                       # generate가 만든 클라이언트 재사용
            raise RuntimeError("LLM 클라이언트가 없음")
        client, model = rough._client, rough.model

        before = ask_confidence(client, model, task, plan_summary(out), NO_GROUNDING)
        apply_m3(out, responses)                        # M3 응답 반영 + 도구 확정
        after = ask_confidence(client, model, task, plan_summary(out),
                               WITH_GROUNDING.format(summary=m3_summary(out)))

        row = {"run": i + 1, "before": before, "after": after,
               "delta": {a: round(after[a] - before[a], 3) for a in ASPECTS}}
        results.append(row)
        print(f"\n[run {i+1}] {'항목':16s} {'사전':>6s} {'사후':>6s} {'변화':>7s}")
        for a in ASPECTS:
            print(f"        {a:16s} {before[a]:6.2f} {after[a]:6.2f} {row['delta'][a]:+7.2f}")
        print(f"        사전 이유: {before.get('reason', '')}")
        print(f"        사전 불확실 요소: {before.get('uncertain_about', [])}")
        print(f"        사후 이유: {after.get('reason', '')}")
        print(f"        사후 불확실 요소: {after.get('uncertain_about', [])}")

    os.makedirs("experiments/confidence", exist_ok=True)
    path = f"experiments/confidence/{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"task": task, "runs": results}, f, ensure_ascii=False, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
