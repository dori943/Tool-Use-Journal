# -*- coding: utf-8 -*-
"""M1 실행기 — 예빈 태스크 2종(C1-T1, C2-T1)에 M1을 돌려 서브골 JSON을 산출.

사용법:
  python scripts/run_m1.py c1_1              # mock M0 (아래 표 수치) + M2 mock 접지
  python scripts/run_m1.py c2_1
  python scripts/run_m1.py c1_1 --m0-json path.json   # 실제 M0 serialize() 출력으로 실행
                                                       # (점군이 없으므로 M2 접지는 생략)

출력: outputs/m1/<task>.m1.json (M1 출력) / <task>.gk.json (서브골별 G_k, M2 mock 값 포함)

mock M0 수치는 예빈 씬 노션 표 기준의 근사값 — 실제 실행 시 M0가 대체한다.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tuj.m0_scene.abstraction import build_m0
from tuj.m1_subgoal.pipeline import run_m1, run_m1_with_m2
from tuj.m1_subgoal.rough import TemplateRough

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
    return task, spec, TemplateRough()


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
    rough = TemplateRough(category_map={
        "apple": "obj_tray_green_tray", "bread": "obj_tray_green_tray",
        "mug": "obj_tray_blue_tray",
        "plate": "obj_tray_red_tray", "spoon": "obj_tray_red_tray",
    })
    return task, spec, rough


def load_ee_pool():
    p = os.path.join(os.path.dirname(__file__), "..", "configs", "robot_spec.json")
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    pool = cfg["ee_pool"]
    for e in pool:                       # 키 정합: ee_conditioned는 seal_rms_tol_mm을 기대
        if e.get("type") == "vacuum" and "seal_rms_tol_mm" not in e:
            e["seal_rms_tol_mm"] = e.get("flatness_tol_rms_mm", 1.5)
    return pool, cfg["reach_mm"]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "c1_1"
    m0_json = None
    if "--m0-json" in sys.argv:
        m0_json = sys.argv[sys.argv.index("--m0-json") + 1]
    use_llm = "--llm" in sys.argv          # OPENAI_API_KEY 필요. 없으면 Template 경로

    os.makedirs("outputs/m1", exist_ok=True)

    if m0_json:                          # 실제 M0 출력으로 (점군 없음 → M2 생략)
        with open(m0_json, encoding="utf-8") as f:
            m0s = json.load(f)
        task, _, rough = scene_c1_1() if name == "c1_1" else scene_c2_1()
        if use_llm:
            from tuj.m1_subgoal.rough import LLMRough
            rough = LLMRough()
        out = run_m1(task, m0s, rough=rough)
        gks = []
    else:
        task, spec, rough = scene_c1_1() if name == "c1_1" else scene_c2_1()
        if use_llm:
            from tuj.m1_subgoal.rough import LLMRough
            rough = LLMRough()
        m0 = build_m0(spec_to_objects(spec))
        ee_pool, reach = load_ee_pool()
        out, gks = run_m1_with_m2(task, m0, ee_pool, reach, rough=rough)

    with open(f"outputs/m1/{name}.m1.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(f"outputs/m1/{name}.gk.json", "w", encoding="utf-8") as f:
        json.dump(gks, f, ensure_ascii=False, indent=2)

    s = out["m1_stats"]
    print(f"[{name}] 서브골 {s['n_subgoals']} → 상세 {s['n_details']} | "
          f"DAG 엣지 {s['n_edges']} | mutex {s['n_mutex']} | M2 질의 {s['n_m2_queries']}")
    for e in out["m1_partial_order"]:
        print(f"  {e['from']} -> {e['to']}   ({e['why']})")
    print(f"-> outputs/m1/{name}.m1.json, {name}.gk.json")


if __name__ == "__main__":
    main()
