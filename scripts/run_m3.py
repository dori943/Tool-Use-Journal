# -*- coding: utf-8 -*-
"""M3 실행기 — M1 산출물 + m2.json으로 M3 접지 → gk_<SG>.json, m3.json 산출.

사용법:
  python scripts/run_m3.py c1_1                       # 기본: SiPhy 백엔드 (VLM 1콜/객체)
  python scripts/run_m3.py c1_1 --backend mock        # 배관 검증용 mock (물성값은 MockBackend 기본표)
  python scripts/run_m3.py c2_1
선행: python scripts/run_m1.py <task>                 (m1 산출물 생성)

입력: output/<task>/m2.json — M2 출력. "m2_queries" 리스트를 실행.
      m3_call.kind ∈ intrinsic | ee | relational | top_exposed | clear | batch | swept_space
      queried_by가 술어 id → 응답과 G_k에 각인. m2.json 없으면 내장 데모 폴백.
출력: output/<task>/m3.json          — M2 응답 {"responses": [{subgoal_id, queried_by, ...}]}
      output/<task>/gk_<SG>.json     — 서브골별 G_k (M4 입력)
      output/<task>/m3_intrinsic.json — Tier-2 접지 캐시

siphy 백엔드: OPENAI_API_KEY(우선) 또는 GEMINI_API_KEY/GOOGLE_API_KEY 환경변수, 혹은 레포 루트 my_api_key.py.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tuj.m3_grounding import Materializer, MockBackend, PropertyMemory, SiPhyBackend, new_gk


# ── 술어 → m3_call 컴파일 (m2_queries가 안 실은 eval_by:m3 술어 보충) ──
HEAD_MAP = {
    "reachable":      ("ee", None),
    "ee_usable":      ("ee", None),
    "tool_sweepable": ("ee", None),
    "top_exposed":    ("top_exposed", None),
    "clear":          ("clear", None),
    "fits":           ("relational", "fits_inside"),
    "gap":            ("relational", "gap"),
    "distance":       ("relational", "distance"),
    "clearance":      ("relational", "clearance"),
    # 0901 어휘 확장(도희): flat_face(도구 평평면), gap_accessible(도구가 틈 진입 가능)
    "flat_face":      ("flat_face", None),
    "gap_accessible": ("gap_accessible", None),
    # 집합형(batch/swept_space): m2_queries가 직접 실어야 함, 컴파일러는 스킵
    "batch_feasible":  ("__skip__", None),
    "act_space_clear": ("__skip__", None),
}
NEEDS_COMPILE = (None, "not_queried", "unknown", "unanswered")


def load_m1(out):
    m1 = json.loads((out / "m1.json").read_text(encoding="utf-8"))
    pts = np.load(out / "m1_points.npz")
    for n in m1["nodes"]:
        n["_points"] = pts[n["id"]]
    return m1


def crop_of(out, node_id):
    p = out / "crops" / f"{node_id}.png"
    return p if p.exists() else None


def _split_args(expr):
    """expr의 최상위 인자만 분리 — 중괄호 집합 {a,b,c}는 하나의 토큰으로 유지.
    (종전: 단순 콤마 분리가 집합을 쪼개 원소가 개별 인자로 새어 나갔다 —
     c2_1 fits(집합,트레이) → (빵,머그) 같은 엉뚱한 쌍의 원인)."""
    inner = expr.split("(", 1)[-1].rsplit(")", 1)[0]
    args, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "{":
            depth += 1; cur += ch
        elif ch == "}":
            depth -= 1; cur += ch
        elif ch == "," and depth == 0:
            args.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur)
    return [a for a in (x.strip() for x in args) if a]


def compile_predicates(m2, ids, aliases):
    """details의 eval_by:m3 술어 중 미질의/미해결 → 질의 목록으로 컴파일.
    집합 {a,b,c} 인자는 원소별로 펼쳐 질의한다 (0828 C2_1 계약)."""
    def resolve(tok, sg):
        tok = tok.strip()
        if tok.startswith("{") and tok.endswith("}"):     # 집합 → 원소별 해석 후 합침
            acc = []
            for m in tok[1:-1].split(","):
                acc += resolve(m, sg)
            return acc
        if tok.startswith("obj_"):
            return [tok] if tok in ids else []
        if tok.startswith("?"):
            return [t for t in sg.get("tool_candidate_ids", []) if t in ids]
        return aliases.get(tok, [])

    out = []
    for sg in m2.get("m2_subgoals", []):
        for d in sg.get("details", []):
            for p in d.get("pre", []) + d.get("establish", []):
                if p.get("eval_by") != "m3" or p.get("status", None) not in NEEDS_COMPILE:
                    continue
                head = p.get("head")
                if head not in HEAD_MAP:
                    out.append({"subgoal_id": sg["subgoal_id"], "queried_by": p["id"],
                                "kind": "error", "error": f"no mapping for head: {head}"})
                    continue
                kind, relation = HEAD_MAP[head]
                if kind == "__skip__":
                    continue
                targets = [resolve(a, sg) for a in _split_args(p["expr"])] or [[]]
                base = {"subgoal_id": sg["subgoal_id"], "queried_by": p["id"], "kind": kind}
                if kind == "relational":
                    a_list = targets[0]
                    b_list = targets[1] if len(targets) > 1 else []
                    for a in a_list:                        # 집합 target → (원소, 컨테이너)
                        for b in b_list:
                            out.append(base | {"a": a, "b": b, "relation": relation})
                    if not (a_list and b_list):
                        out.append(base | {"kind": "error",
                                           "error": f"unresolved args in {p['expr']!r} "
                                                    f"(별칭/노드 미해결 — m1 추적 대상 확인)"})
                elif kind == "gap_accessible":              # (?tool, ?target)
                    tools = targets[0]
                    tgts = targets[1] if len(targets) > 1 else []
                    for ttool in tools:
                        for tgt in tgts:
                            out.append(base | {"tool_id": ttool, "target_id": tgt})
                    if not (tools and tgts):
                        out.append(base | {"kind": "error",
                                           "error": f"unresolved args in {p['expr']!r}"})
                else:                                       # ee/top_exposed/clear/flat_face
                    nodes = targets[-1] if head == "ee_usable" else targets[0]
                    for nd in nodes:
                        out.append(base | {"node_id": nd})
                    if not nodes:
                        out.append(base | {"kind": "error",
                                           "error": f"unresolved args in {p['expr']!r}"})
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "c1_1"
    args = sys.argv[2:]
    backend_name = args[args.index("--backend") + 1] if "--backend" in args else "siphy"
    model = args[args.index("--model") + 1] if "--model" in args else "gpt-4o-mini"
    memory_path = args[args.index("--memory") + 1] if "--memory" in args else str(ROOT / "output" / "memory.json")

    OUT = ROOT / "output" / name
    if not (OUT / "m1.json").exists():
        sys.exit(f"[err] {OUT}/m1.json 없음 — 먼저 python scripts/run_m1.py {name}")

    m1 = load_m1(OUT)
    print(f"[M3] task={name} m1 로드: nodes={len(m1['nodes'])} edges={len(m1['edges'])} "
          f"backend={backend_name}")

    if backend_name == "siphy":
        backend = SiPhyBackend(model=model, repo_root=ROOT, verbose=True)
    else:
        backend = MockBackend()

    spec = json.loads((ROOT / "configs" / "robot_spec.json").read_text(encoding="utf-8"))
    ee_pool = []
    for e in spec["ee_pool"]:
        e = dict(e)
        if "flatness_tol_rms_mm" in e:
            e["seal_rms_tol_mm"] = e["flatness_tol_rms_mm"]
        ee_pool.append(e)

    memory = None if memory_path == "none" else PropertyMemory(memory_path)
    mat = Materializer(m1, backend=backend, memory=memory,
                       logger=lambda **kw: print("  [m3]", kw))
    preloaded = len(mat._cache)
    if preloaded:
        print(f"[M3] memory preload: {preloaded}개 객체 hit (VLM 스킵)")
    ids = {n["id"]: n for n in m1["nodes"]}

    m2_path = OUT / "m2.json"
    m2 = {}
    if m2_path.exists():
        m2 = json.loads(m2_path.read_text(encoding="utf-8"))
        queries = [{"subgoal_id": q.get("subgoal_id", "SG"),
                    "queried_by": q["queried_by"], **q["m3_call"]}
                   for q in m2["m2_queries"]]
        # 영역 별칭: tool_rest → 랙 몸체(ee_rack)만. 랙에 꽂힌 EE 부품(gripperrack_*)은 제외.
        aliases = {"tool_rest": [i for i in ids
                                 if "rack" in i.lower() and "gripperrack" not in i.lower()]}
        compiled = compile_predicates(m2, ids, aliases)
        key = lambda q: (q["queried_by"], q["kind"],
                         q.get("node_id") or (q.get("a"), q.get("b"))
                         or (q.get("tool_id"), q.get("target_id")))
        already = {key(q) for q in queries}
        compiled = [q for q in compiled if key(q) not in already]
        for q in compiled:
            tgt = (q.get("node_id") or (q.get("a"), q.get("b"))
                   or (q.get("tool_id"), q.get("target_id")) or q.get("error"))
            print(f"  [compile] {q['queried_by']} -> {q['kind']}({tgt})")
        queries += compiled
        for sg in m2.get("m2_subgoals", []):
            sel = set(sg.get("object_ids", []))
            asked = {q.get("node_id") for q in queries if q["subgoal_id"] == sg["subgoal_id"]} \
                  | {x for q in queries if q["subgoal_id"] == sg["subgoal_id"]
                     for x in (q.get("a"), q.get("b")) if x} \
                  | {x for q in queries if q["subgoal_id"] == sg["subgoal_id"]
                     for x in q.get("member_ids", [])}
            if asked - sel - {None}:
                print(f"  [lint] {sg['subgoal_id']}: object_ids 밖 질의 {sorted(asked - sel - {None})}")
            never = sel - asked - set(sg.get("target_ids", []))
            if never:
                print(f"  [lint] {sg['subgoal_id']}: 선택됐지만 질의 0건 {sorted(never)}")
        print(f"[M3] m2.json 로드: task={m2.get('task', '?')!r} "
              f"queries={len(queries)} (컴파일 보충 {len(compiled)}건)")
    else:
        def find(sub):
            return next((i for i in ids if sub in i), None)
        light = find("light_plate") or next(iter(ids))
        queries = [{"subgoal_id": "SG1", "queried_by": "demo_ee", "kind": "ee", "node_id": light}]
        print("[M3] m2.json 없음 → 내장 데모 질의 사용")

    responses = []
    gk_of = {}
    for q in queries:
        gk = gk_of.setdefault(q["subgoal_id"], new_gk(q["subgoal_id"]))
        qid, kind = q["queried_by"], q["kind"]
        try:
            if kind == "ee":
                r = mat.query_ee(gk, q["node_id"], ee_pool, queried_by=qid,
                                 reach_mm=spec["reach_mm"], crop_rgb=crop_of(OUT, q["node_id"]))
            elif kind == "intrinsic":
                r = mat.query_intrinsic(gk, q["node_id"], queried_by=qid,
                                        crop_rgb=crop_of(OUT, q["node_id"]))
            elif kind == "relational":
                a = q.get("a") or q.get("a_id") or q.get("from")
                b = q.get("b") or q.get("b_id") or q.get("to")
                r = mat.query_relational(gk, a, b, q.get("relation", "distance"),
                                         queried_by=qid, **q.get("kw", {}))
            elif kind == "top_exposed":
                r = mat.query_top_exposed(gk, q["node_id"], queried_by=qid)
            elif kind == "clear":
                r = mat.query_clear(gk, q["node_id"], queried_by=qid)
            elif kind == "flat_face":
                r = mat.query_flat_face(gk, q["node_id"], queried_by=qid)
            elif kind == "gap_accessible":
                r = mat.query_gap_accessible(gk, q["tool_id"], q["target_id"],
                                             queried_by=qid)
            elif kind in ("batch", "swept_space"):
                call = {"kind": kind, "action_type": q.get("action_type"),
                        "actor": q.get("actor"), "member_ids": q.get("member_ids", []),
                        "to": q.get("to"),
                        "ignore_ids": q.get("ignore_ids", [])}
                fn = mat.query_batch if kind == "batch" else mat.query_swept_space
                r = fn(gk, call, queried_by=qid)
            elif kind == "error":
                r = {"queried_by": qid, "error": q["error"]}
                print(f"  [warn] {qid}: {q['error']}")
            else:
                r = {"queried_by": qid, "error": f"unsupported kind: {kind}"}
                print(f"  [warn] {r['error']}")
        except KeyError as e:
            r = {"queried_by": qid, "node_id": q.get("node_id"),
                 "error": f"node not in m1: {e}"}
            print(f"  [warn] {qid}: {r['error']}")
        responses.append({"subgoal_id": q["subgoal_id"]} | r)

    # 서브골 자기완결 gk 조립
    subs = {s["subgoal_id"]: s for s in m2.get("m2_subgoals", [])}
    for sid in subs:
        gk_of.setdefault(sid, new_gk(sid))
    extra = defaultdict(lambda: {"batch": [], "swept_space": []})
    for r0 in responses:
        if r0.get("kind") in ("batch", "swept_space"):
            extra[r0.get("subgoal_id")][r0["kind"]].append(r0)
    gks = list(gk_of.values())
    for gk in gks:
        sid = gk["subgoal_id"]
        s = subs.get(sid)
        if not s:
            continue
        gk["task"] = m2.get("task")
        gk["goal"] = s.get("goal")
        gk["subgoal_kind"] = s.get("kind")
        gk["roles"] = {"target": s.get("target_ids", []),
                       "container": s.get("container_id"),
                       "tool_candidates": s.get("tool_candidate_ids", []),
                       "selected_tool": s.get("selected_tool_id"),
                       "selection_evidence": s.get("selection_evidence")}
        gk["details"] = [
            {"detail_id": d["detail_id"], "action_type": d["action_type"],
             "binding": d["binding"],
             "pre": [{k: p[k] for k in ("id", "expr", "eval_by", "status", "evidence")
                      if k in p} for p in d.get("pre", [])],
             "establish": [p["expr"] for p in d.get("establish", [])],
             "destroy": [p["expr"] for p in d.get("destroy", [])]}
            for d in s.get("details", [])]
        gk["partial_order"] = [e for e in m2.get("m2_partial_order", [])
                               if e["from"].startswith(sid + "_")
                               or e["to"].startswith(sid + "_")]
        gk["mutex"] = [x for x in m2.get("m2_mutex", [])
                       if any(any(did.startswith(sid + "_") for did in g)
                              for g in x.get("groups", []))]
        gk["invariants"] = [i for i in m2.get("m2_invariants", [])
                            if i.get("id") == f"INV_{sid}"]
        gk["batch"] = extra[sid]["batch"]
        gk["swept_space"] = extra[sid]["swept_space"]
        if s.get("split_from"):
            gk["split_from"] = s["split_from"]
            gk["split_index"] = s.get("split_index")
    # selected_tool 노드 복제 (얕은 복사 + queried_by 초기화 + replicated_from 태그)
    for gk in gks:
        tool = (gk.get("roles") or {}).get("selected_tool")
        sid = gk["subgoal_id"]
        if tool and tool not in gk.get("nodes", {}):
            for other in gks:
                src = other.get("nodes", {}).get(tool)
                if src is None or other["subgoal_id"] == sid:
                    continue
                clone = dict(src)
                clone["queried_by"] = []
                clone["replicated_from"] = other["subgoal_id"]
                gk["nodes"][tool] = clone
                break

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, list):
            return [strip(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    for gk in gks:
        (OUT / f"gk_{gk['subgoal_id']}.json").write_text(
            json.dumps(strip(gk), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "m3.json").write_text(
        json.dumps({"responses": strip(responses)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT / "m3_intrinsic.json").write_text(
        json.dumps(strip(dict(mat._cache)), ensure_ascii=False, indent=2), encoding="utf-8")

    n_t2 = len(mat._cache)
    if memory is not None:
        stats = memory.update(mat._cache, source=backend_name)
        memory.save()
        print(f"[M3] memory update: new={stats['new']} upgraded={stats['upgraded']} "
              f"kept={stats['kept']} -> {memory_path}")
    print(f"\n[{name}] tier-2 접지 {n_t2}/{len(m1['nodes'])} 객체 (lazy, preload {preloaded}건 포함),"
          f" 응답 {len(responses)}건 -> m3.json, " + ", ".join(f"gk_{g['subgoal_id']}.json" for g in gks))
    for gk in gks:
        for nid, entry in gk["nodes"].items():
            if entry.get("ee"):
                feas = [k for k, v in entry["ee"].items() if v["feasible"]]
                print(f"     {nid}: mass_est={entry.get('mass_kg')}kg feasible_ee={feas}"
                      + (f" [replicated_from={entry['replicated_from']}]" if 'replicated_from' in entry else ""))
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
