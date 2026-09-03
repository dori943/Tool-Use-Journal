# -*- coding: utf-8 -*-
"""M2 실행기 — task_registry 에 등록된 태스크에 M2을 돌려 서브골 JSON을 산출.

사용법:
  python scripts/run_m2.py c1_1              # output/c1_1/m1.json 있으면 그걸로, 없으면 mock
  python scripts/run_m2.py c1_2              # 지시문은 task_registry 가 단일 출처
  python scripts/run_m2.py c1_1 --m1-json path.json   # M1 JSON 경로 직접 지정

서브골 생성은 항상 LLM(gpt-4o)이다 → OPENAI_API_KEY 필수.

출력: output/<task>/m2.json (팀 구조 — main의 M1·M3 산출물 폴더 명명과 동일: output/c1_1/)
      내용: 서브골·부분순서·mutex·M3 질의 사양·invariant

G_k는 여기서 만들지 않는다. G_k 조립은 M3 몫 — M2은 질의 목록(m2_queries)만
넘기고, M3가 답을 채워 G_k를 조립한다. (0820 합의: M2 아웃풋에서 G_k 제외)

mock M1 수치는 예빈 씬 노션 표 기준의 근사값 — 실제 실행 시 M1가 대체한다.
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)                      # task_registry (단일 출처)

import numpy as np

from tuj.m1_scene.abstraction import build_m1, serialize
from tuj.m2_subgoal.pipeline import run_m2
from tuj.m2_subgoal.rough import LLMRough
from task_registry import instruction as task_instruction

RNG = np.random.default_rng(0)          # mock 점군 결정론


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


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "c1_1"
    m1_json = None
    if "--m1-json" in sys.argv:
        m1_json = sys.argv[sys.argv.index("--m1-json") + 1]
    tdir = os.path.join("output", name)      # output/c1_1/ (main의 M1·M3 폴더와 동일 명명)
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

    if m1_json:                          # 실제 M1 serialize() 출력으로
        with open(m1_json, encoding="utf-8") as f:
            m1s = json.load(f)
        print(f"[M1] {m1_json} 사용")
    else:                                # mock M1 (씬 근사 수치)
        make_scene = MOCK_SCENES.get(name)
        if make_scene is None:
            sys.exit(f"[중단] '{name}' 은 mock 씬이 없습니다.\n"
                     f"  먼저 run_m1 으로 output/{name}/m1.json 을 만드십시오.")
        _, spec = make_scene()
        m1s = serialize(build_m1(spec_to_objects(spec)))
        print("[M1] mock 수치 사용 (실제 M1 JSON 없음)")

    # ── M3 응답 경로 결정: output/<task-id>/m3.json 자동 또는 --m3-json 경로 ──
    m3_json = None
    if "--m3-json" in sys.argv:
        m3_json = sys.argv[sys.argv.index("--m3-json") + 1]
    elif os.path.exists(os.path.join(tdir, "m3.json")):
        m3_json = os.path.join(tdir, "m3.json")

    # ── 0828 안전장치(사전): 직전 왕복이 이미 완료된 세트면 덮어쓰지 않는다 ──
    # 분할된 m2.json과 그 자식 질의에 대한 m3.json이 짝으로 남아 있을 때 run_m2을
    # 또 돌리면, 새 분해(부모 id)와 자식 응답이 매칭되지 않아 분할이 풀린 계획으로
    # 덮어써진다. 왕복의 마지막은 항상 run_m3다.
    prev_path = os.path.join(tdir, "m2.json")
    if m3_json and os.path.exists(prev_path):
        with open(prev_path, encoding="utf-8") as f:
            prev = json.load(f)
        with open(m3_json, encoding="utf-8") as f:
            raw0 = json.load(f)
        prev_q = {q["queried_by"] for q in prev.get("m2_queries", [])}
        resp_q = {r.get("queried_by")
                  for r in (raw0["responses"] if isinstance(raw0, dict) else raw0)
                  if r.get("queried_by")}
        if prev.get("m2_stats", {}).get("n_split_subgoals") and resp_q and resp_q <= prev_q:
            sys.exit("[중단] 분할 완료된 m2.json과 m3.json이 이미 짝이 맞는 최종 세트입니다.\n"
                     "  여기서 run_m2을 다시 돌리면 분할이 풀린 계획으로 덮어써집니다.\n"
                     "  다음 단계로 진행하거나, 처음부터 다시 돌리려면 m2.json을 지운 뒤 실행하십시오.")

    rough = LLMRough()                          # 서브골 생성은 항상 LLM

    # ── 0831 측정 피드백: 직전 왕복(m2.json+m3.json)이 있으면 unsat 판정 요약을
    #    재분해 프롬프트에 사실로 첨부한다 (호출 추가 없음). 해법은 넣지 않는다.
    if m3_json and os.path.exists(prev_path):
        import copy as _copy
        from tuj.m2_subgoal.ingest import apply_m3 as _apply_prev, measurement_feedback
        judged = _copy.deepcopy(prev)
        _apply_prev(judged, raw0["responses"] if isinstance(raw0, dict) else raw0)
        fb = measurement_feedback(judged)
        if fb:
            rough.feedback = fb
            print("[M2] 직전 측정 피드백을 재분해에 반영:")
            for ln in fb.splitlines():
                print("    " + ln)
        else:
            # 0831: 기각 사유가 없으면 직전 분해를 유지한다 — 분해 LLM 생략.
            # (재분해는 측정이 계획을 기각했을 때만. 불필요한 재분해는 결과가
            #  되돌아갈 수 있고(kind 진동) 호출 낭비다.)
            keep = _copy.deepcopy(prev)
            print("[M2] 직전 측정에서 재분해 사유 없음 — 기존 분해 유지 (분해 LLM 생략)")

    out = locals().get("keep")
    if out is None:
        out = run_m2(task, m1s, rough=rough)

    if m3_json:
        from tuj.m2_subgoal.ingest import apply_m3, update_confidence
        with open(m3_json, encoding="utf-8") as f:
            raw = json.load(f)
        responses = raw["responses"] if isinstance(raw, dict) else raw
        # ── 0828 안전장치(사후): 응답이 이번 분해의 질의와 하나도 안 맞으면 중단 ──
        qids = {q["queried_by"] for q in out["m2_queries"]}
        rids = {r.get("queried_by") for r in responses if r.get("queried_by")}
        if rids and not (qids & rids):
            sys.exit("[중단] m3.json 응답이 이번 분해의 질의와 하나도 매칭되지 않습니다.\n"
                     "  run_m3 -> run_m2 -> run_m3 왕복 순서를 확인하십시오. (m2.json 미변경)")
        for line in apply_m3(out, responses):
            print(line)
        # 측정 반영 후 자기보고 신뢰도 갱신 (LLM 3차 호출)
        if rough._client is None:               # 분해 생략 경로에서도 클라이언트 보장
            from openai import OpenAI
            rough._client = OpenAI()
        for line in update_confidence(out, rough._client, rough.model,
                                      usage_acc=rough.usage):
            print(line)
        out["m2_stats"]["llm_usage"] = rough.usage
        # 0828: batch 응답의 partition대로 서브골 분할 (확보/반환은 한 번만 생성)
        from tuj.m2_subgoal.regroup import split_after_m3
        for line in split_after_m3(out):
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
          f"DAG 엣지 {s['n_edges']} | mutex {s['n_mutex']} | M3 질의 {s['n_m3_queries']}")
    for e in out["m2_partial_order"]:
        print(f"  {e['from']} -> {e['to']}   ({e['why']})")
    print(f"-> {tdir}/m2.json")


if __name__ == "__main__":
    main()
