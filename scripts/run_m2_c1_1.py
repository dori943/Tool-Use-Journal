"""M0 산출물 + m1.json(M1 질의 요청)으로 M2 실행 → gk_<SG>.json, m2.json.

배치: Tool-Use-Journal/scripts/run_m2_c1_1.py
선행: python scripts/run_m0_c1_1.py             (m0 산출물 생성)
실행: python scripts/run_m2_c1_1.py                      # mock 백엔드 (오프라인)
      python scripts/run_m2_c1_1.py --backend siphy      # SiPhy 이식 백엔드 (VLM 1콜/객체)

입력: output/<task>/m1.json — M1 출력. "m1_queries" 리스트를 실행:
  {"m1_queries": [{"subgoal_id": "SG1", "queried_by": "SG1_d1_p2",
                   "m2_call": {"kind": "intrinsic|ee|relational", "node_id": "<id>",
                               ("relational"이면) "a": "<id>", "b": "<id>",
                               "relation": "distance|fits_inside|clearance"}}]}
  · queried_by가 술어 id → 응답과 G_k에 각인. 같은 술어가 여러 후보에 걸릴 수 있어
    응답은 리스트 (queried_by, node_id 조합으로 구별).
  · node id는 m0.json의 nodes[].id 그대로. (m1.json 없으면 내장 SG1 데모로 폴백)

출력: output/<task>/m2.json          — M1 응답 {"responses": [{subgoal_id, queried_by, node_id, ...}]}
      output/<task>/gk_<SG>.json     — 서브골별 G_k (M3 입력)
      output/<task>/m2_intrinsic.json — Tier-2 접지 캐시 (lazy: 질의된 객체만)

siphy 백엔드: OPENAI_API_KEY 환경변수 또는 레포 루트 my_api_key.py 필요 (SiPhy 관례).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tuj.m2_grounding import Materializer, MockBackend, SiPhyBackend, new_gk

TASK = "c1_1"
OUT = ROOT / "output" / TASK   # 모듈 공용 출력 (m0.json, gk_*.json 등 전부 여기)


class C1Backend(MockBackend):
    """C1-1 클래스용 mock 물성 (SiPhy 연결 전 배관 검증용)."""
    TABLE = dict(MockBackend.TABLE)
    TABLE.update({
        "plate":  dict(material="plastic", density_kgm3=950, youngs_gpa=2.0),
        "block":  dict(material="plastic", density_kgm3=500, youngs_gpa=2.0),
        "bottle": dict(material="glass", density_kgm3=2500, youngs_gpa=70),
        "zone":   dict(material="unknown", density_kgm3=0, youngs_gpa=None),
    })


def load_m0() -> dict:
    """m0.json(직렬화본) + m0_points.npz(점군) → build_m0 결과 형태로 복원."""
    m0 = json.loads((OUT / "m0.json").read_text(encoding="utf-8"))
    pts = np.load(OUT / "m0_points.npz")
    for n in m0["nodes"]:
        n["_points"] = pts[n["id"]]
    return m0


def crop_of(node_id):
    """M0가 저장한 마스크 크롭 경로 (siphy 백엔드 VLM 입력, 없으면 None)."""
    p = OUT / "crops" / f"{node_id}.png"
    return p if p.exists() else None


# ── 술어 → m2_call 컴파일 (m1_queries가 안 실은 eval_by:m2 술어 보충) ──
# head별 질의 종류. relational은 (relation, 이항) / 나머지는 단항.
HEAD_MAP = {
    "reachable":    ("ee", None),          # 도달성은 ee 응답의 reachability
    "ee_usable":    ("ee", None),
    "tool_sweepable": ("ee", None),
    "top_exposed":  ("top_exposed", None),
    "clear":        ("clear", None),
    "fits":         ("relational", "fits_inside"),
    "gap":          ("relational", "gap"),
    "distance":     ("relational", "distance"),
    "clearance":    ("relational", "clearance"),
}
NEEDS_COMPILE = (None, "not_queried", "unknown", "unanswered")


def compile_predicates(m1, ids, aliases):
    """details의 eval_by:m2 술어 중 미질의/미해결 → 질의 목록으로 컴파일."""
    def resolve(tok, sg):
        tok = tok.strip()
        if tok.startswith("obj_"):
            return [tok] if tok in ids else []
        if tok.startswith("?"):                        # ?tool 등 도구 변수 → 후보 전개
            return [t for t in sg.get("tool_candidate_ids", []) if t in ids]
        return aliases.get(tok, [])                    # tool_rest 등 영역 별칭

    out = []
    for sg in m1.get("m1_subgoals", []):
        for d in sg.get("details", []):
            for p in d.get("pre", []) + d.get("establish", []):
                if p.get("eval_by") != "m2" or p.get("status", None) not in NEEDS_COMPILE:
                    continue
                head = p.get("head")
                if head not in HEAD_MAP:
                    out.append({"subgoal_id": sg["subgoal_id"], "queried_by": p["id"],
                                "kind": "error", "error": f"no mapping for head: {head}"})
                    continue
                kind, relation = HEAD_MAP[head]
                args = [a for a in p["expr"].split("(", 1)[-1].rstrip(")").split(",")
                        if not a.strip().startswith("{")]       # 집합 인자는 제외
                targets = [resolve(a, sg) for a in args] or [[]]
                base = {"subgoal_id": sg["subgoal_id"], "queried_by": p["id"], "kind": kind}
                if kind == "relational":
                    a_list, b_list = (targets + [[]])[:2]
                    for a in a_list:
                        for b in b_list:
                            out.append(base | {"a": a, "b": b, "relation": relation})
                    if not (a_list and b_list):
                        out.append(base | {"kind": "error",
                                           "error": f"unresolved args in {p['expr']!r} "
                                                    f"(별칭/노드 미해결 — m0 추적 대상 확인)"})
                else:
                    nodes = targets[-1] if head == "ee_usable" else targets[0]  # ee_usable(?EE,?tool)
                    for n in nodes:
                        out.append(base | {"node_id": n})
                    if not nodes:
                        out.append(base | {"kind": "error",
                                           "error": f"unresolved args in {p['expr']!r}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "siphy"], default="mock")
    ap.add_argument("--model", default="gpt-4o-mini", help="siphy 백엔드 VLM")
    args = ap.parse_args()

    if not (OUT / "m0.json").exists():
        sys.exit("[err] m0.json 없음 — 먼저 scripts/run_m0_c1_1.py 실행")
    m0 = load_m0()
    print(f"[M2] m0 로드: nodes={len(m0['nodes'])} edges={len(m0['edges'])} "
          f"backend={args.backend}")

    if args.backend == "siphy":
        backend = SiPhyBackend(model=args.model, repo_root=ROOT, verbose=True)
    else:
        backend = C1Backend()

    # ── robot_spec 로드 + 필드명 정규화 (vac seal 허용치) ──
    spec = json.loads((ROOT / "configs" / "robot_spec.json").read_text(encoding="utf-8"))
    ee_pool = []
    for e in spec["ee_pool"]:
        e = dict(e)
        if "flatness_tol_rms_mm" in e:
            e["seal_rms_tol_mm"] = e["flatness_tol_rms_mm"]
        ee_pool.append(e)

    mat = Materializer(m0, backend=backend,
                       logger=lambda **kw: print("  [m2]", kw))
    ids = {n["id"]: n for n in m0["nodes"]}

    # ── M1 질의 로드 (output/<task>/m1.json의 m1_queries) — 없으면 내장 SG1 데모 ──
    m1_path = OUT / "m1.json"
    if m1_path.exists():
        m1 = json.loads(m1_path.read_text(encoding="utf-8"))
        queries = [{"subgoal_id": q.get("subgoal_id", "SG"),
                    "queried_by": q["queried_by"], **q["m2_call"]}
                   for q in m1["m1_queries"]]
        # 영역 별칭: tool_rest → m0의 rack 노드 (없으면 미해결 error 응답)
        aliases = {"tool_rest": [i for i in ids if "rack" in i.lower()]}
        compiled = compile_predicates(m1, ids, aliases)
        key = lambda q: (q["queried_by"], q["kind"],
                         q.get("node_id") or (q.get("a"), q.get("b")))
        already = {key(q) for q in queries}
        compiled = [q for q in compiled if key(q) not in already]
        for q in compiled:
            print(f"  [compile] {q['queried_by']} -> {q['kind']}"
                  f"({q.get('node_id') or (q.get('a'), q.get('b')) or q.get('error')})")
        queries += compiled
        # lint: object_ids ↔ 질의 일관성
        for sg in m1.get("m1_subgoals", []):
            sel = set(sg.get("object_ids", []))
            asked = {q.get("node_id") for q in queries if q["subgoal_id"] == sg["subgoal_id"]} \
                  | {x for q in queries if q["subgoal_id"] == sg["subgoal_id"]
                     for x in (q.get("a"), q.get("b")) if x}
            if asked - sel - {None}:
                print(f"  [lint] {sg['subgoal_id']}: object_ids 밖 질의 {sorted(asked - sel - {None})}")
            never = sel - asked - set(sg.get("target_ids", []))
            if never:
                print(f"  [lint] {sg['subgoal_id']}: 선택됐지만 질의 0건 {sorted(never)}")
        print(f"[M2] m1.json 로드: task={m1.get('task', '?')!r} "
              f"queries={len(queries)} (컴파일 보충 {len(compiled)}건)")
    else:
        def find(sub):
            return next((i for i in ids if sub in i), None)
        light, heavy = find("light_plate"), find("heavy_plate")
        queries = [{"subgoal_id": "SG1", "queried_by": "demo_ee", "kind": "ee", "node_id": t}
                   for t in (light, heavy) if t]
        print("[M2] m1.json 없음 → 내장 데모 질의 사용 (스키마는 파일 상단 docstring 참고)")

    # ── 질의 실행: 서브골별 G_k + M1 응답(m2.json 리스트) ──
    responses: list[dict] = []
    gk_of: dict[str, dict] = {}
    for q in queries:
        gk = gk_of.setdefault(q["subgoal_id"], new_gk(q["subgoal_id"]))
        qid, kind = q["queried_by"], q["kind"]
        try:
            if kind == "ee":
                r = mat.query_ee(gk, q["node_id"], ee_pool, queried_by=qid,
                                 reach_mm=spec["reach_mm"], crop_rgb=crop_of(q["node_id"]))
            elif kind == "intrinsic":
                r = mat.query_intrinsic(gk, q["node_id"], queried_by=qid,
                                        crop_rgb=crop_of(q["node_id"]))
            elif kind == "relational":
                a = q.get("a") or q.get("a_id") or q.get("from")
                b = q.get("b") or q.get("b_id") or q.get("to")
                r = mat.query_relational(gk, a, b, q.get("relation", "distance"),
                                         queried_by=qid, **q.get("kw", {}))
            elif kind == "top_exposed":
                r = mat.query_top_exposed(gk, q["node_id"], queried_by=qid)
            elif kind == "clear":
                r = mat.query_clear(gk, q["node_id"], queried_by=qid)
            elif kind == "error":                      # 컴파일 미해결 → 응답으로 전달
                r = {"queried_by": qid, "error": q["error"]}
                print(f"  [warn] {qid}: {q['error']}")
            else:
                r = {"queried_by": qid, "error": f"unsupported kind: {kind}"}
                print(f"  [warn] {r['error']}")
        except KeyError as e:                          # M0에 없는 노드 등 — 실패도 응답으로
            r = {"queried_by": qid, "node_id": q.get("node_id"),
                 "error": f"node not in m0: {e}"}
            print(f"  [warn] {qid}: {r['error']}")
        responses.append({"subgoal_id": q["subgoal_id"]} | r)
    gks = list(gk_of.values())

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
    (OUT / "m2.json").write_text(
        json.dumps({"responses": strip(responses)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT / "m2_intrinsic.json").write_text(
        json.dumps(strip(dict(mat._cache)), ensure_ascii=False, indent=2), encoding="utf-8")

    n_t2 = len(mat._cache)
    print(f"\n[M2] tier-2 접지 {n_t2}/{len(m0['nodes'])} 객체 (lazy), 응답 {len(responses)}건"
          f" -> m2.json, " + ", ".join(f"gk_{g['subgoal_id']}.json" for g in gks))
    for gk in gks:
        for nid, entry in gk["nodes"].items():
            if entry.get("ee"):
                feas = [k for k, v in entry["ee"].items() if v["feasible"]]
                print(f"     {nid}: mass_est={entry['mass_kg']}kg feasible_ee={feas}")
    print(f"[DONE] -> {OUT}")


if __name__ == "__main__":
    main()
