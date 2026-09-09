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

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _graspable_ees(subgoal: dict) -> dict[str, list[str]]:
    """서브골의 graspable_on_support 판정에서 노드별 접근 가능 EE 목록을 모은다."""
    found: dict[str, list[str]] = {}
    for d in subgoal.get("details", []):
        for cond in d.get("pre", []):
            if cond.get("head") != "graspable_on_support":
                continue
            for e in cond.get("evidence") or []:
                node_id = e.get("node")
                if node_id is None or e.get("graspable_on_support") is None:
                    continue
                found[node_id] = list(e.get("graspable_ees") or [])
    return found


def _narrow_ee(node: dict, usable: list[str]) -> dict:
    """지지면 위에서 접근할 수 없는 EE 를 노드에서 내린다.

    m1.json 의 ee 는 물체를 고립시켜 본 판정이라, 지지면에 붙어 있어 손가락이 들어갈
    자리가 없는 물체도 feasible 로 남는다. M4 는 이 필드로 EE 를 고르므로, 여기서
    좁히지 않으면 M2 가 판정한 결과가 하류에 전달되지 않는다 (c2_2 에서 두께 1.6mm
    치즈에 2F 가 배정되어 M5 파지가 전부 막혔음).
    """
    ee = node.get("ee")
    if not ee:
        return node
    narrowed = {}
    for ee_id, spec in ee.items():
        if ee_id in usable or not spec.get("feasible"):
            narrowed[ee_id] = spec
            continue
        narrowed[ee_id] = dict(spec, feasible=False,
                               reason="지지면 위에서 이 EE 로 접근할 수 없음"
                                      " (graspable_on_support)")
    return dict(node, ee=narrowed)


def assemble(m1: dict, m2: dict) -> list[dict]:
    nodes = {n["id"]: n for n in m1["nodes"]}
    edges = m1.get("edges", [])
    gks = []
    for s in m2.get("m2_subgoals", []):
        sid = s["subgoal_id"]
        graspable = _graspable_ees(s)
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
            "nodes": {i: (_narrow_ee(nodes[i], graspable[i])
                          if i in graspable else dict(nodes[i]))
                      for i in sorted(want) if i in nodes},
            "edges": [e for e in edges if e.get("from") in want and e.get("to") in want],
            "roles": {"target": s.get("target_ids", []),
                      "container": s.get("container_id"),
                      "tool_candidates": s.get("tool_candidate_ids", []),
                      "selected_tool": s.get("selected_tool_id"),
                      "selection_evidence": s.get("selection_evidence")},
            "details": [
                {"detail_id": d["detail_id"], "action_type": d["action_type"],
                 "binding": d.get("binding"), "group_id": d.get("group_id"),
                 # 0909: M2 가 정한 실행 매개변수(예: 컨테이너 안 배치 자리).
                 # 없으면 키 자체를 빼 gk 를 부풀리지 않는다.
                 **({"action_parameters": d["action_parameters"]}
                    if d.get("action_parameters") else {}),
                 "pre": [{k: p[k] for k in ("id", "expr", "head", "eval_by",
                                            "status", "evidence")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="c1_1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory containing m1.json/m2.json and receiving gk_*.json",
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    name = args.task
    out = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT / "output" / name
    )
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
