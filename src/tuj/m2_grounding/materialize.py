"""M2 — materialize: M1 질의 → 접지 실행 → G_k에 심기 + margin 게이트.

핵심 계약:
  · pull 방식 — query된 객체만 접지 (lazy). 캐시로 중복 방지.
  · 모든 결과에 queried_by(질의한 술어 id) 각인 → M5의 역추적 라우팅 근거.
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
    def __init__(self, m0: dict, backend=None, friction: FrictionHead | None = None,
                 logger=None, eps_margin: float = 0.1,
                 remeasure_fn=None, probe_fn=None):
        """m0: m0_scene.build_m0() 결과. remeasure_fn/probe_fn: 에스컬레이션 훅
        (remeasure_fn(node_id)→(material,rms) | probe_fn(node_id)→(track_mm, dt))."""
        self.nodes = {n["id"]: n for n in m0["nodes"]}
        self.edges = list(m0.get("edges", []))         # coarse 관계 (top_exposed/clear 판정 근거)
        self.backend = backend or MockBackend()
        self.friction = friction or FrictionHead()
        self.log = logger or (lambda **kw: None)
        self.eps = eps_margin
        self.remeasure_fn = remeasure_fn
        self.probe_fn = probe_fn
        self._cache: dict[str, dict] = {}

    # ── 질의 3종 (M1의 술어가 부름) ─────────────────────

    def query_intrinsic(self, gk: dict, node_id: str, queried_by: str,
                        crop_rgb=None, margin_fn=None) -> dict:
        """→ {queried_by, node_id, geometry..., material, mass_kg, mu, ...} (M1 응답 스키마)"""
        if node_id not in self._cache:
            hooks = {}
            if margin_fn is not None:                     # 마찰 에스컬레이션 활성화
                hooks = dict(
                    margin_fn=margin_fn, eps=self.eps,
                    remeasure_fn=(lambda: self.remeasure_fn(node_id)) if self.remeasure_fn else None,
                    # TODO(M4): probe_push 프리미티브 완성 시 주석 해제 (2단 마찰 프로브)
                    # probe_fn=(lambda: self.probe_fn(node_id)) if self.probe_fn else None,
                    probe_fn=None)
            self._cache[node_id] = ground_intrinsic(
                self.nodes[node_id], crop_rgb, self.backend, self.friction, **hooks)
            self.log(module="m2", event="intrinsic", node=node_id,
                     mu_stage=self._cache[node_id]["mu"]["stage"])
        entry = gk["nodes"].setdefault(node_id, {"queried_by": []})
        entry.update(self._cache[node_id])
        entry["queried_by"].append(queried_by)
        return {"queried_by": queried_by, "node_id": node_id} | self._cache[node_id]

    def query_relational(self, gk: dict, a_id: str, b_id: str, relation: str,
                         queried_by: str, **kw) -> dict:
        """→ {queried_by, from, to, value_mm, check, pass} (M1 응답 스키마)"""
        fn = {"distance": relational.center_distance,
              "fits_inside": relational.fits_inside,
              "clearance": relational.depth_clearance,
              "gap": relational.gap}[relation]
        res = {"queried_by": queried_by, "from": a_id, "to": b_id} \
            | fn(self.nodes[a_id], self.nodes[b_id], **kw)
        gk["edges"].append(res)
        self.log(module="m2", event="relational", rel=relation)
        return res

    def query_ee(self, gk: dict, node_id: str, ee_pool: list[dict], queried_by: str,
                 reach_mm: float | None = None, crop_rgb=None) -> dict:
        """EE-conditioned: intrinsic이 없으면 자동 접지(grip_slip margin으로 마찰 게이트 연동).
        → {queried_by, node_id, ee: {ee_id: {feasible, margin, reason}}, reachability} (M1 응답 스키마)"""
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
        self.log(module="m2", event="ee", node=node_id,
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
        self.log(module="m2", event="predicate", pred="top_exposed", node=node_id)
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
        self.log(module="m2", event="predicate", pred="clear", node=region_id)
        return res


def new_gk(subgoal_id: str) -> dict:
    return {"subgoal_id": subgoal_id, "nodes": {}, "edges": [],
            "flags": {"near_threshold": []}}
