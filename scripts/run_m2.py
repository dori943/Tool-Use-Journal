# -*- coding: utf-8 -*-
"""M2 실행기 — task_registry 에 등록된 태스크에 M2을 돌려 서브골 JSON을 산출.

사용법:
  python scripts/run_m2.py c1_1              # output/c1_1/m1.json (접지 포함) 사용
  python scripts/run_m2.py c1_2              # 지시문은 task_registry 가 단일 출처
  python scripts/run_m2.py c1_1 --m1-json path.json   # M1 JSON 경로 직접 지정

0908 파이프라인 재구조 (M1&M3 통합):
  입력은 m1.json 하나다. 접지값(ee/reachability/predicates/물성)이 노드에 실려 온다.
  M3 왕복(m2_queries → m3.json → 재분해)은 없어졌고, 한 번의 실행 안에서
    분해(LLM) → 판정(apply_grounding) → [피드백 재분해(LLM)] → 도구 확정(LLM) → 분할 → 재판정
  까지 끝낸다. 쌍/집합 술어는 M2가 관계 함수를 import해 직접 계산한다 (VLM 0회).

서브골 생성은 항상 LLM이다. 기본 Gemini(GEMINI_API_KEY), TUJ_LLM_PROVIDER=openai 면 OpenAI.

출력: output/<task>/m2.json — 서브골·부분순서·mutex·invariant·서브골별 measurements
      G_k 조립은 scripts/assemble_gk.py (m1.json + m2.json).
"""
from __future__ import annotations

import copy
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)                      # task_registry (단일 출처)

from tuj.m2_subgoal.ground import load_scene
from tuj.m2_subgoal.ingest import (apply_grounding, assign_container_slots,
                                   measurement_feedback, update_confidence)
from tuj.m2_subgoal.pipeline import run_m2
from tuj.m2_subgoal.regroup import split_by_partition
from tuj.m2_subgoal.rough import LLMRough
from task_registry import instruction as task_instruction

MAX_REDECOMPOSE = 2                     # 피드백 재분해 상한 (왕복이 없어 M2 안에서 끝낸다)


def apply_task_packing_policy(out, policy_path):
    """Apply every requested adjacent packing precedence or fail atomically."""
    from tuj.m2_subgoal.core import (
        add_container_packing_sequence_pres,
        partial_order,
    )

    with open(policy_path, encoding="utf-8") as f:
        packing_policy = json.load(f)
    target_order = packing_policy["target_order"]
    logs = add_container_packing_sequence_pres(
        out["m2_subgoals"], target_order
    )
    expected = max(0, len(target_order) - 1)
    if len(logs) != expected:
        raise ValueError(
            "packing policy did not resolve every adjacent target pair: "
            f"expected {expected}, got {len(logs)}"
        )
    all_details = [
        detail
        for subgoal in out["m2_subgoals"]
        for detail in subgoal["details"]
    ]
    out["m2_partial_order"], out["m2_mutex"] = partial_order(all_details)
    out["m2_packing_policy"] = packing_policy
    out["m2_stats"]["n_edges"] = len(out["m2_partial_order"])
    out["m2_stats"]["n_mutex"] = len(out["m2_mutex"])
    return logs


def _serialized(m1: dict) -> dict:
    """분해 LLM에 넘길 M1 뷰 — 점군만 뺀다 (접지값은 프롬프트 컨텍스트로 유용)."""
    return {"nodes": [{k: v for k, v in n.items() if k != "_points"} for n in m1["nodes"]],
            "edges": m1["edges"]}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "c1_1"
    m1_json = None
    if "--m1-json" in sys.argv:
        m1_json = sys.argv[sys.argv.index("--m1-json") + 1]
    tdir = (os.path.abspath(sys.argv[sys.argv.index("--output-dir") + 1])
            if "--output-dir" in sys.argv else os.path.join("output", name))
    os.makedirs(tdir, exist_ok=True)
    if not m1_json and os.path.exists(os.path.join(tdir, "m1.json")):
        m1_json = os.path.join(tdir, "m1.json")             # M1 모듈 출력 자동 사용

    # 지시문은 task_registry 가 단일 출처다. 예전에는 c1_1 이 아니면 전부
    # c2_1 문장으로 떨어져, 장면에 없는 물체(수거함·트레이)를 분해에 요구하다
    # 검문에서 죽었다. 미등록이면 조용히 다른 문장을 쓰지 않고 즉시 멈춘다.
    try:
        task = task_instruction(name)
    except KeyError as e:
        sys.exit(f"[중단] {e}.\n"
                 "  task_registry.py 의 TASKS 에 instruction 을 등록하십시오.")
    print(f"[M2] 지시문: {task}")

    # M1&M3 통합 후 접지값(ee/reachability/predicates/물성)은 m1.json에 실려 온다.
    # mock 씬 폴백은 없앴다 — 접지값이 없으면 술어가 전부 unknown이라 판정이 무의미하다.
    if not m1_json:
        sys.exit(f"[중단] {tdir}/m1.json 이 없습니다.\n"
                 f"  먼저 python scripts/run_m1.py {name} 을 돌리십시오.")
    m1 = load_scene(m1_json)
    print(f"[M1] {m1_json} 사용")

    m1s = _serialized(m1)

    rough = LLMRough()                          # 서브골 생성은 항상 LLM
    # 0908: 순서 요구 서브골(ordered)의 VLM 순서 판정용 장면 이미지 (M1이 저장한 frame.png)
    _frame = os.path.join(tdir, "frame.png")
    rough.frame_path = _frame if os.path.exists(_frame) else None

    # ── 분해 → 판정 → (기각되면) 재분해. 왕복이 없으므로 여기서 수렴시킨다 ──
    out = run_m2(task, m1s, rough=rough)
    for line in apply_grounding(out, m1):
        print(line)
    for attempt in range(1, MAX_REDECOMPOSE + 1):
        fb = measurement_feedback(out)
        if not fb:
            break
        # 0831: 해법은 넣지 않는다 — 측정 사실만 전달하고 대안 도출은 분해 LLM 몫.
        # (그래야 도구 사용이 창발로 성립한다)
        print(f"[M2] 판정에서 기각된 조건이 있어 재분해 ({attempt}/{MAX_REDECOMPOSE}):")
        for ln in fb.splitlines():
            print("    " + ln)
        rough.feedback = fb
        prev = copy.deepcopy(out)
        out = run_m2(task, m1s, rough=rough)
        for line in apply_grounding(out, m1):
            print(line)
        if measurement_feedback(out) == fb:      # 같은 사유 반복 — 더 돌려도 같다
            print("[M2] 재분해해도 같은 조건이 기각됨 — 직전 분해로 진행")
            out = prev
            break
    else:
        print(f"[M2] 재분해 상한({MAX_REDECOMPOSE}회) 도달 — 현재 분해로 진행")

    # 측정 반영 후 자기보고 신뢰도 갱신 + 도구 확정 (LLM, 서브골당 1회)
    if rough._client is None:
        from tuj.m2_subgoal.rough import make_llm_client
        rough._client = make_llm_client()
    for line in update_confidence(out, rough._client, rough.model, usage_acc=rough.usage):
        print(line)

    # partition대로 서브골 분할 → 자식 기준으로 술어 재판정 (VLM 0회라 저렴)
    split_logs = split_by_partition(out)
    for line in split_logs:
        print(line)
    if split_logs:
        for line in apply_grounding(out, m1):
            print(line)
        # 같은 컨테이너로 가는 형제들에게 서로 다른 자리를 준다 (0909) — 목적지가
        # 하나뿐이면 실행계가 매번 영역 중심을 골라 먼저 놓인 것 위로 내려온다.
        for line in assign_container_slots(out, m1):
            print(line)

    unresolved = [s["subgoal_id"] for s in out["m2_subgoals"]
                  if s.get("tool_candidate_ids") and not s.get("selected_tool_id")]
    if unresolved:
        print(f"[M2] 경고: 도구 미확정 서브골 {unresolved} — assemble_gk에서 멈춥니다")
    out["m2_stats"]["llm_usage"] = rough.usage

    # A task-owned packing order is a hard execution constraint. If M1/M2
    # identifiers drift and even one adjacent pair cannot be linked, do not
    # emit an apparently valid DAG without the policy.
    policy_path = os.path.join(_ROOT, "configs", f"{name}_packing_policy.json")
    if os.path.exists(policy_path):
        for line in apply_task_packing_policy(out, policy_path):
            print(line)

    with open(os.path.join(tdir, "m2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    s = out["m2_stats"]
    usage = s.get("llm_usage") or {}
    if usage:
        total = sum(e["tokens"] for e in usage.values())
        tsec = sum(e.get("seconds", 0.0) for e in usage.values())
        parts = " / ".join(
            f"{k} {e['tokens']}tok({e['calls']}회, {e.get('seconds', 0):.1f}s)"
            for k, e in usage.items())
        print(f"  [M2 tokens] {parts} | 합계 {total}tok, {tsec:.1f}s")
    print(f"[{name}] 서브골 {s['n_subgoals']} → 상세 {s['n_details']} | "
          f"DAG 엣지 {s['n_edges']} | mutex {s['n_mutex']} | 술어 판정 {s.get('n_evaluations', 0)}")
    for e in out["m2_partial_order"]:
        print(f"  {e['from']} -> {e['to']}   ({e['why']})")
    print(f"-> {tdir}/m2.json")


if __name__ == "__main__":
    main()
