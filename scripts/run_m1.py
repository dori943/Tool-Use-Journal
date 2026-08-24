# -*- coding: utf-8 -*-
"""M1 실행기 — 예빈 태스크 2종(C1-T1, C2-T1)에 M1을 돌려 서브골 JSON을 산출.

사용법:
  python scripts/run_m1.py c1_1              # output/c1_1/m0.json 있으면 그걸로, 없으면 mock
  python scripts/run_m1.py c2_1
  python scripts/run_m1.py c1_1 --m0-json path.json   # M0 JSON 경로 직접 지정

서브골 생성은 항상 LLM(gpt-4o)이다 → OPENAI_API_KEY 필수.

출력: output/<task>/m1.json (팀 구조 — main의 M0·M2 산출물 폴더 명명과 동일: output/c1_1/)
      내용: 서브골·부분순서·mutex·M2 질의 사양·invariant

G_k는 여기서 만들지 않는다. G_k 조립은 M2 몫 — M1은 질의 목록(m1_queries)만
넘기고, M2가 답을 채워 G_k를 조립한다. (0820 합의: M1 아웃풋에서 G_k 제외)

mock M0 수치는 예빈 씬 노션 표 기준의 근사값 — 실제 실행 시 M0가 대체한다.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tuj.m0_scene.abstraction import build_m0, serialize
from tuj.m1_subgoal.pipeline import run_m1
from tuj.m1_subgoal.rough import LLMRough

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
    task = "테이블에 흩어진 레고 블록을 수거 영역으로 쓸어 담아라"
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
    task = "사과와 빵은 초록 트레이, 머그는 파랑 트레이, 접시와 숟가락은 빨강 트레이로 옮겨라"
    return task, spec


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "c1_1"
    m0_json = None
    if "--m0-json" in sys.argv:
        m0_json = sys.argv[sys.argv.index("--m0-json") + 1]
    tdir = os.path.join("output", name)      # output/c1_1/ (main의 M0·M2 폴더와 동일 명명)
    os.makedirs(tdir, exist_ok=True)
    if not m0_json and os.path.exists(os.path.join(tdir, "m0.json")):
        m0_json = os.path.join(tdir, "m0.json")             # M0 모듈 출력 자동 사용

    if m0_json:                          # 실제 M0 serialize() 출력으로
        with open(m0_json, encoding="utf-8") as f:
            m0s = json.load(f)
        task, _ = scene_c1_1() if name == "c1_1" else scene_c2_1()
        print(f"[M0] {m0_json} 사용")
    else:                                # mock M0 (씬 근사 수치)
        task, spec = scene_c1_1() if name == "c1_1" else scene_c2_1()
        m0s = serialize(build_m0(spec_to_objects(spec)))
        print("[M0] mock 수치 사용 (실제 M0 JSON 없음)")

    out = run_m1(task, m0s, rough=LLMRough())   # 서브골 생성은 항상 LLM

    # ── M2 응답 반영 (있으면): output/<task-id>/m2.json 자동 또는 --m2-json 경로 ──
    m2_json = None
    if "--m2-json" in sys.argv:
        m2_json = sys.argv[sys.argv.index("--m2-json") + 1]
    elif os.path.exists(os.path.join(tdir, "m2.json")):
        m2_json = os.path.join(tdir, "m2.json")
    if m2_json:
        from tuj.m1_subgoal.ingest import apply_m2
        with open(m2_json, encoding="utf-8") as f:
            raw = json.load(f)
        responses = raw["responses"] if isinstance(raw, dict) else raw
        for line in apply_m2(out, responses):
            print(line)

    with open(os.path.join(tdir, "m1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    s = out["m1_stats"]
    print(f"[{name}] 서브골 {s['n_subgoals']} → 상세 {s['n_details']} | "
          f"DAG 엣지 {s['n_edges']} | mutex {s['n_mutex']} | M2 질의 {s['n_m2_queries']}")
    for e in out["m1_partial_order"]:
        print(f"  {e['from']} -> {e['to']}   ({e['why']})")
    print(f"-> {tdir}/m1.json")


if __name__ == "__main__":
    main()
