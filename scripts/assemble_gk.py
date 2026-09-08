# -*- coding: utf-8 -*-
"""gk 조립기 — m1.json(통합 접지) + m2.json → gk_<SG>.json (M4 입력).

M1·M3 통합 후 질의층(run_m3)이 없어졌으므로, M4 가 읽던 서브골별 서브그래프는
질의 없이 두 산출물만으로 조립한다.
  · nodes : 서브골이 다루는 객체(object_ids ∪ target ∪ container ∪ tool 후보 ∪ selected_tool)의
            m1.json 노드 전체 (기하·물성·ee·reachability·predicates 포함)
  · edges : 그 노드들 사이의 m1 coarse 관계
  · roles / details / partial_order / mutex / invariants / split_from : m2.json 그대로
  · measurements : M2 가 서브골에 남긴 술어 판정(relations 결과)이 있으면 그대로 실음
M4 는 nodes[].ee / mass_kg / geometry, roles.selected_tool, details, mutex, partial_order 를 읽는다.

사용법: python scripts/assemble_gk.py c1_1
이전 실행(다른 분할 구성)의 stale gk_*.json 은 지운다 (gk_bundle.json 은 M4 산출물이라 제외).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def assemble(m1: dict, m2: dict) -> list[dict]:
    nodes = {n["id"]: n for n in m1["nodes"]}
    edges = m1.get("edges", [])
    gks = []
    for s in m2.get("m2_subgoals", []):
        sid = s["subgoal_id"]
        want = set(s.get("object_ids", [])) | set(s.get("target_ids", [])) \
            | set(s.get("tool_candidate_ids", []))
        for k in ("container_id", "selected_tool_id"):
            if s.get(k):
                want.add(s[k])
        gk = {
            "subgoal_id": sid,
            "task": m2.get("task"),
            "goal": s.get("goal"),
            "subgoal_kind": s.get("kind"),
            "nodes": {i: dict(nodes[i]) for i in sorted(want) if i in nodes},
            "edges": [e for e in edges if e.get("from") in want and e.get("to") in want],
            "roles": {"target": s.get("target_ids", []),
                      "container": s.get("container_id"),
                      "tool_candidates": s.get("tool_candidate_ids", []),
                      "selected_tool": s.get("selected_tool_id"),
                      "selection_evidence": s.get("selection_evidence")},
            "details": [
                {"detail_id": d["detail_id"], "action_type": d["action_type"],
                 "binding": d.get("binding"),
                 "pre": [{k: p[k] for k in ("id", "expr", "eval_by", "status", "evidence")
                          if k in p} for p in d.get("pre", [])],
                 "establish": [p["expr"] for p in d.get("establish", [])],
                 "destroy": [p["expr"] for p in d.get("destroy", [])]}
                for d in s.get("details", [])],
            "partial_order": [e for e in m2.get("m2_partial_order", [])
                              if e["from"].startswith(sid + "_") or e["to"].startswith(sid + "_")],
            "mutex": [x for x in m2.get("m2_mutex", [])
                      if any(any(did.startswith(sid + "_") for did in g)
                             for g in x.get("groups", []))],
            "invariants": [i for i in m2.get("m2_invariants", []) if i.get("id") == f"INV_{sid}"],
        }
        if s.get("measurements") is not None:     # M2 의 relations 판정 결과 (있으면)
            gk["measurements"] = s["measurements"]
        if s.get("partition_plan"):
            gk["partition_plan"] = s["partition_plan"]
        if s.get("split_from"):
            gk["split_from"] = s["split_from"]
            gk["split_index"] = s.get("split_index")
        missing = sorted(want - set(nodes))
        if missing:
            print(f"  [gk] {sid}: m1 에 없는 객체 {missing} (M2 가 참조한 id 확인)")
        gks.append(gk)
    return gks


def write_gks(out: Path, gks: list[dict]) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for gk in gks:
        p = out / f"gk_{gk['subgoal_id']}.json"
        p.write_text(json.dumps(gk, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(p)
    keep = {p.name for p in paths}
    stale = [q for q in out.glob("gk_*.json") if q.name not in keep and q.name != "gk_bundle.json"]
    for q in stale:
        q.unlink()
    if stale:
        print(f"[gk] stale gk 정리 {len(stale)}건: {sorted(q.name for q in stale)}")
    return paths


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    name = argv[0] if argv and not argv[0].startswith("-") else "c1_1"
    out = ROOT / "output" / name
    for f in ("m1.json", "m2.json"):
        if not (out / f).exists():
            sys.exit(f"[err] {out / f} 없음")
    gks = assemble(_read(out / "m1.json"), _read(out / "m2.json"))
    paths = write_gks(out, gks)
    print(f"[gk] {name}: 서브골 {len(paths)}개 -> " + ", ".join(p.name for p in paths))
    for gk in gks:
        tool = gk["roles"].get("selected_tool")
        feas = ([k for k, v in gk["nodes"][tool]["ee"].items() if v["feasible"]]
                if tool in gk["nodes"] and gk["nodes"][tool].get("ee") else None)
        print(f"     {gk['subgoal_id']}: nodes={len(gk['nodes'])} edges={len(gk['edges'])} "
              f"selected_tool={tool} feasible_ee={feas}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
