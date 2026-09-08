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

import numpy as np

from tuj.m1_scene.abstraction import build_m1, serialize
from tuj.m2_subgoal.ground import ensure_measurements, load_ee_pool, load_scene
from tuj.m2_subgoal.ingest import apply_grounding, measurement_feedback, update_confidence
from tuj.m2_subgoal.pipeline import run_m2
from tuj.m2_subgoal.regroup import split_by_partition
from tuj.m2_subgoal.rough import LLMRough
from task_registry import instruction as task_instruction

RNG = np.random.default_rng(0)          # mock 점군 결정론
MAX_REDECOMPOSE = 2                     # 피드백 재분해 상한 (왕복이 없어 M2 안에서 끝낸다)


def box_points(center_mm, size_mm, n=400):
    c, s = np.asarray(center_mm, float), np.asarray(size_mm, float)
    return c + (RNG.random((n, 3)) - 0.5) * s


def spec_to_objects(spec):
    return [{"name": name, "cls": cls, "points": box_points(c, s)}
            for name, cls, c, s in spec]


# ── 씬 스펙: (name, class, center_mm, bbox_mm) — 예빈 env 근사 ──────────────

def scene_c1_1():
    spec = []
    for i in range(12):                  # 레고 12개, 테이블 위 분산
        spec.append((f"block_{i}", "block",
                     [420 + 30 * (i % 4), -90 + 30 * (i // 4), 810], [20, 20, 12]))
    spec += [
        ("light_plate", "plate", [250, 250, 810], [200, 220, 10]),
        ("heavy_plate", "plate", [250, -250, 810], [200, 220, 10]),
        ("bottle_distractor", "bottle", [150, 150, 860], [60, 60, 120]),
        ("collection_zone_visual", "collection_zone", [650, 0, 802], [250, 180, 4]),
    ]
    task = task_instruction("c1_1")     # 지시문 출처: task_registry
    return task, spec


def scene_c2_1():
    spec = [
        ("apple", "apple", [420, -150, 840], [75, 75, 75]),
        ("bread", "bread", [470, -60, 830], [100, 60, 50]),
        ("mug", "mug", [430, 40, 845], [90, 80, 95]),
        ("plate", "plate", [500, 140, 808], [200, 200, 15]),
        ("spoon", "spoon", [380, 210, 805], [150, 30, 10]),
        ("green_tray", "tray", [680, -200, 818], [250, 180, 35]),
        ("blue_tray", "tray", [700, 0, 818], [250, 180, 35]),
        ("red_tray", "tray", [680, 200, 818], [250, 180, 35]),
    ]
    task = task_instruction("c2_1")     # 지시문 출처: task_registry
    return task, spec


# mock M1 을 만들 수 있는 태스크만 등록한다 (나머지는 실제 m1.json 필요).
MOCK_SCENES = {"c1_1": scene_c1_1, "c2_1": scene_c2_1}


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
    robot_spec = (sys.argv[sys.argv.index("--robot-spec") + 1]
                  if "--robot-spec" in sys.argv
                  else os.path.join(_ROOT, "configs", "robot_spec.json"))
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

    if m1_json:                          # 실제 M1 출력 (접지값 포함)
        m1 = load_scene(m1_json)
        print(f"[M1] {m1_json} 사용")
    else:                                # mock M1 (씬 근사 수치)
        make_scene = MOCK_SCENES.get(name)
        if make_scene is None:
            sys.exit(f"[중단] '{name}' 은 mock 씬이 없습니다.\n"
                     f"  먼저 run_m1 으로 output/{name}/m1.json 을 만드십시오.")
        _, spec = make_scene()
        m1 = serialize(build_m1(spec_to_objects(spec)))
        print("[M1] mock 수치 사용 (실제 M1 JSON 없음)")

    ee_pool, reach_mm = load_ee_pool(robot_spec)
    for line in ensure_measurements(m1, ee_pool, reach_mm, out_dir=tdir):
        print(line)
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

    unresolved = [s["subgoal_id"] for s in out["m2_subgoals"]
                  if s.get("tool_candidate_ids") and not s.get("selected_tool_id")]
    if unresolved:
        print(f"[M2] 경고: 도구 미확정 서브골 {unresolved} — assemble_gk에서 멈춥니다")
    out["m2_stats"]["llm_usage"] = rough.usage

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
