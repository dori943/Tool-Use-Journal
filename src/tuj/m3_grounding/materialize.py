"""M3 — materialize: M2 질의 → 접지 실행 → G_k에 심기 + margin 게이트.

핵심 계약:
  · pull 방식 — query된 객체만 접지 (lazy). 캐시로 중복 방지.
  · 모든 결과에 queried_by(질의한 술어 id) 각인 → M6의 역추적 라우팅 근거.
  · margin 게이트: |margin|<ε 항목을 모아 에스컬레이션 훅(재관측/프로브) 실행.
  · logger: common/logging.py 의 로거 주입 (없으면 no-op) — 모듈별 비용 분리 집계.

G_k 형태 (common/schemas.py 검증 대상):
  {"subgoal_id", "nodes": {id: {intrinsic..., ee: {ee_id: verdict}, queried_by: [...]}},
   "edges": [{from,to,type,value_mm,pass,queried_by}], "flags": {near_threshold: [...]}}
"""
from __future__ import annotations

from . import relational
from .ee_conditioned import evaluate_ee, grip_slip_margin_fn, reach_check
from .intrinsic import FrictionHead, MockBackend, ground_intrinsic


class Materializer:
    def __init__(self, m1: dict, backend=None, friction: FrictionHead | None = None,
                 logger=None, eps_margin: float = 0.1,
                 remeasure_fn=None, probe_fn=None, memory=None):
        """m1: m1_scene.build_m1() 결과. memory: PropertyMemory (선택) —
        지속 메모리 hit 시 VLM 콜 스킵, 신규만 접지. remeasure_fn/probe_fn: 에스컬레이션 훅."""
        self.nodes = {n["id"]: n for n in m1["nodes"]}
        self.edges = list(m1.get("edges", []))         # coarse 관계 (top_exposed/clear 판정 근거)
        self.backend = backend or MockBackend()
        self.friction = friction or FrictionHead()
        self.log = logger or (lambda **kw: None)
        self.eps = eps_margin
        self.remeasure_fn = remeasure_fn
        self.probe_fn = probe_fn
        self.memory = memory
        self._cache: dict[str, dict] = {}
        if memory is not None:                         # 씬 노드에 해당하는 엔트리 preload
            for nid in self.nodes:
                hit = memory.lookup(nid)
                if hit is not None:
                    self._cache[nid] = hit
                    self.log(module="m3", event="memory_hit", node=nid,
                             stage=max(int(hit.get("mass_stage", 0)),
                                       int(hit.get("mu", {}).get("stage", 0))))

    # ── 질의 3종 (M2의 술어가 부름) ─────────────────────

    def query_intrinsic(self, gk: dict, node_id: str, queried_by: str,
                        crop_rgb=None, margin_fn=None) -> dict:
        """→ {queried_by, node_id, geometry..., material, mass_kg, mu, ...} (M2 응답 스키마)"""
        if node_id not in self._cache:
            hooks = {}
            if margin_fn is not None:                     # 마찰 에스컬레이션 활성화
                hooks = dict(
                    margin_fn=margin_fn, eps=self.eps,
                    remeasure_fn=(lambda: self.remeasure_fn(node_id)) if self.remeasure_fn else None,
                    # TODO(M5): probe_push 프리미티브 완성 시 주석 해제 (2단 마찰 프로브)
                    # probe_fn=(lambda: self.probe_fn(node_id)) if self.probe_fn else None,
                    probe_fn=None)
            self._cache[node_id] = ground_intrinsic(
                self.nodes[node_id], crop_rgb, self.backend, self.friction, **hooks)
            self.log(module="m3", event="intrinsic", node=node_id,
                     mu_stage=self._cache[node_id]["mu"]["stage"])
        entry = gk["nodes"].setdefault(node_id, {"queried_by": []})
        entry.update(self._cache[node_id])
        entry["queried_by"].append(queried_by)
        return {"queried_by": queried_by, "node_id": node_id} | self._cache[node_id]

    def query_relational(self, gk: dict, a_id: str, b_id: str, relation: str,
                         queried_by: str, **kw) -> dict:
        """→ {queried_by, from, to, value_mm, check, pass} (M2 응답 스키마)"""
        fn = {"distance": relational.center_distance,
              "fits_inside": relational.fits_inside,
              "clearance": relational.depth_clearance,
              "gap": relational.gap}[relation]
        res = {"queried_by": queried_by, "from": a_id, "to": b_id} \
            | fn(self.nodes[a_id], self.nodes[b_id], **kw)
        gk["edges"].append(res)
        self.log(module="m3", event="relational", rel=relation)
        return res

    def query_ee(self, gk: dict, node_id: str, ee_pool: list[dict], queried_by: str,
                 reach_mm: float | None = None, crop_rgb=None) -> dict:
        """EE-conditioned: intrinsic이 없으면 자동 접지(grip_slip margin으로 마찰 게이트 연동).
        → {queried_by, node_id, ee: {ee_id: {feasible, margin, reason}}, reachability} (M2 응답 스키마)"""
        node = self.nodes[node_id]
        # grip_slip margin_fn: 가장 빡빡한 파지형 EE 기준으로 μ 민감도 게이트 구성
        grip_ees = [e for e in ee_pool if "grip_force_n" in e]
        pre = self._cache.get(node_id)
        mass0 = pre["mass_kg"] if pre else None
        margin_fn = None
        if grip_ees and mass0 is not None:
            margin_fn = grip_slip_margin_fn(mass0, min(e["grip_force_n"] for e in grip_ees))
        intrinsic = self.query_intrinsic(gk, node_id, queried_by, crop_rgb, margin_fn=margin_fn)

        verdicts = {e["ee_id"]: evaluate_ee(e, intrinsic) for e in ee_pool}
        entry = gk["nodes"][node_id]
        entry["ee"] = verdicts
        if reach_mm is not None:
            entry["reachability"] = reach_check(reach_mm, node["center_mm"])
        # margin 게이트 플래그
        for eid, v in verdicts.items():
            if abs(v["margin"]) < self.eps:
                gk["flags"]["near_threshold"].append(
                    {"node": node_id, "ee": eid, "rule": v["reason"], "queried_by": queried_by})
        self.log(module="m3", event="ee", node=node_id,
                 feasible=[k for k, v in verdicts.items() if v["feasible"]])
        return {"queried_by": queried_by, "node_id": node_id, "ee": verdicts,
                "reachability": entry.get("reachability")}

    # ── 단항 술어 2종 (Tier-1 엣지 산술 — VLM 0회) ──────

    def query_top_exposed(self, gk: dict, node_id: str, queried_by: str) -> dict:
        """상면 노출 여부: 다른 객체가 on/inside로 위를 점유하면 False."""
        blockers = [e["from"] for e in self.edges
                    if e.get("to") == node_id and e.get("type") in ("on", "inside")]
        res = {"queried_by": queried_by, "node_id": node_id, "type": "top_exposed",
               "value": not blockers, "blockers": blockers, "pass": not blockers}
        entry = gk["nodes"].setdefault(node_id, {"queried_by": []})
        entry.setdefault("predicates", {})["top_exposed"] = {
            "value": res["value"], "blockers": blockers, "queried_by": queried_by}
        entry["queried_by"].append(queried_by)
        self.log(module="m3", event="predicate", pred="top_exposed", node=node_id)
        return res

    def query_clear(self, gk: dict, region_id: str, queried_by: str) -> dict:
        """영역 비움 여부: on/inside/overlaps로 영역을 점유한 객체가 없으면 True."""
        occupants = sorted({e["from"] for e in self.edges
                            if e.get("to") == region_id
                            and e.get("type") in ("on", "inside", "overlaps")}
                           | {e["to"] for e in self.edges
                              if e.get("from") == region_id and e.get("type") == "overlaps"})
        res = {"queried_by": queried_by, "node_id": region_id, "type": "clear",
               "value": not occupants, "occupants": occupants, "pass": not occupants}
        entry = gk["nodes"].setdefault(region_id, {"queried_by": []})
        entry.setdefault("predicates", {})["clear"] = {
            "value": res["value"], "occupants": occupants, "queried_by": queried_by}
        entry["queried_by"].append(queried_by)
        self.log(module="m3", event="predicate", pred="clear", node=region_id)
        return res

    # ── 0828 신규 질의 2종 — 프로토타입 (수빈 작성, push 전 협의) ──────────
    # batch:       한 번의 액션으로 member 집합을 동시 처리할 수 있는가.
    #              안 되면 근접도/폭 기준 파티션(그룹 구성)까지 계산해 돌려준다.
    # swept_space: 액션이 지나가는 통로가 비어 있는가 (기존 clearance는 정지 간격이라 못 봄).
    # 계산은 전부 bbox 산술 근사 — 계산 수준과 G_k 반영 위치는 협의 항목.
    @staticmethod
    def _greedy_partition(members: list[dict], cap_mm: float) -> list[list[str]]:
        """주축 정렬 그리디: 폭이 cap 안에 들어오는 만큼씩 근접한 것끼리 묶는다."""
        span = lambda ax: (max(m["center_mm"][ax] for m in members)
                           - min(m["center_mm"][ax] for m in members))
        ax = 0 if span(0) >= span(1) else 1
        order = sorted(members, key=lambda m: m["center_mm"][ax])
        groups, cur = [], []
        for m in order:
            trial = cur + [m]
            if cur and relational.group_extent(trial)["value_mm"] > cap_mm:
                groups.append(cur)
                cur = [m]
            else:
                cur = trial
        if cur:
            groups.append(cur)
        return [[m["id"] for m in g] for g in groups]

    def query_batch(self, gk: dict, call: dict, queried_by: str) -> dict:
        """→ {queried_by, subgoal_id, kind, actor, feasible, checks, binding_check, partition}"""
        import os
        members = [self.nodes[i] for i in call.get("member_ids", []) if i in self.nodes]
        actor = call.get("actor") or {}
        res = {"queried_by": queried_by, "subgoal_id": gk["subgoal_id"],
               "kind": "batch", "actor": actor}
        if actor.get("type") != "object" or len(members) < 2:
            # EE 풀 주체(relocate)의 동시 파지 계산은 스펙 협의 후 — 지금은 미측정 응답
            return res | {"feasible": None, "checks": [],
                          "binding_check": None, "partition": None}
        tool = self.nodes[actor["id"]]
        safety = float(os.environ.get("TUJ_BATCH_SAFETY", "1.0"))   # 실험용 노브
        cap = min(tool["bbox_mm"][0], tool["bbox_mm"][1]) * safety
        demand = relational.group_extent(members)["value_mm"]
        margin = round(cap - demand, 1)
        part = self._greedy_partition(members, cap)
        self.log(module="m3", event="batch", actor=actor.get("id"), n_groups=len(part))
        return res | {"feasible": len(part) == 1,
                      "checks": [{"rule": "group_extent_le_tool_width",
                                  "capacity": round(cap, 1), "demand": demand,
                                  "unit": "mm", "margin": margin, "pass": bool(margin > 0)}],
                      "binding_check": "group_extent_le_tool_width",
                      "partition": part}

    def query_swept_space(self, gk: dict, call: dict, queried_by: str) -> dict:
        """→ {queried_by, subgoal_id, kind, actor, clear, margin_mm, blockers}"""
        members = [self.nodes[i] for i in call.get("member_ids", []) if i in self.nodes]
        actor = call.get("actor") or {}
        res = {"queried_by": queried_by, "subgoal_id": gk["subgoal_id"],
               "kind": "swept_space", "actor": actor}
        to = self.nodes.get(call.get("to"))
        if actor.get("type") != "object" or not members or to is None:
            return res | {"clear": None, "margin_mm": None, "blockers": []}
        tool = self.nodes[actor["id"]]
        width = min(tool["bbox_mm"][0], tool["bbox_mm"][1])
        # ignore_ids (0831, M2 협의): 형제 서브골에서 같은 목적지로 처리될 대상은
        # 실제 방해물이 아니므로 계획 모듈이 명시한 목록을 판정에서 제외한다.
        exclude = (set(call.get("member_ids", [])) | {actor["id"], call.get("to")}
                   | set(call.get("ignore_ids", [])))
        others = [n for n in self.nodes.values() if n["id"] not in exclude]
        r = relational.corridor_blockers(members, to, width, others)
        self.log(module="m3", event="swept_space", actor=actor.get("id"), clear=r["clear"])
        return res | {"clear": r["clear"], "margin_mm": r["margin_mm"],
                      "blockers": r["blockers"]}


def new_gk(subgoal_id: str) -> dict:
    return {"subgoal_id": subgoal_id, "nodes": {}, "edges": [],
            "flags": {"near_threshold": []}}
