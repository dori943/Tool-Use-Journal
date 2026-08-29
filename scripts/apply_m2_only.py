# -*- coding: utf-8 -*-
"""m2.json 응답을 기존 m1.json에 반영만 한다 (재분해 없음, LLM 0회). 0828.

용도: 분할 뒤 마지막 run_m2가 만든 자식 질의 응답(m2.json)을, 분할된 m1.json의
사전조건 status/evidence로 되채울 때. run_m1은 분해부터 다시 하므로 이 용도로
못 쓴다(안전장치가 막기도 함).

사용: python scripts/apply_m2_only.py c1_1
이후 run_m2_c*_1.py를 다시 돌리면 gk의 pre에 status가 실린다.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tuj.m1_subgoal.ingest import apply_m2


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "c1_1"
    tdir = os.path.join("output", name)
    with open(os.path.join(tdir, "m1.json"), encoding="utf-8") as f:
        m1 = json.load(f)
    with open(os.path.join(tdir, "m2.json"), encoding="utf-8") as f:
        raw = json.load(f)
    responses = raw["responses"] if isinstance(raw, dict) else raw
    qids = {q["queried_by"] for q in m1.get("m1_queries", [])}
    rids = {r.get("queried_by") for r in responses if r.get("queried_by")}
    if rids and not (qids & rids):
        sys.exit("[중단] m2.json 응답이 m1.json 질의와 매칭되지 않습니다.")
    for line in apply_m2(m1, responses):
        print(line)
    with open(os.path.join(tdir, "m1.json"), "w", encoding="utf-8") as f:
        json.dump(m1, f, ensure_ascii=False, indent=2)
    print(f"-> {tdir}/m1.json (판정 반영, 재분해 없음)")


if __name__ == "__main__":
    main()
