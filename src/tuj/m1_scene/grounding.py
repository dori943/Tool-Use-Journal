"""객체 단일 속성 접지 (구 M3-(a) intrinsic — M1 으로 통합).

기하(치수·표면 RMS)는 점군에서 직접, 의미·물리(material/density/mass/E)는
PropertyBackend(SiPhy 자리), 마찰 μ는 우리 신설 FrictionHead(3단 에스컬레이션).
"""
from __future__ import annotations

import numpy as np

G = 9.81  # m/s^2


# ── 기하 ──────────────────────────────────────────────

def pca_dims(points) -> dict:
    pts = np.asarray(points, dtype=np.float64)
    X = pts - pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(X.T @ X / len(X))
    ext = np.ptp(X @ evecs[:, np.argsort(evals)[::-1]], axis=0)
    minor = sorted(ext[1:])
    cyl = minor[1] > 0 and (minor[1] - minor[0]) / minor[1] < 0.15
    return {"length_mm": round(float(ext[0]), 1),
            "diameter_mm": round(float(np.mean(ext[1:])), 1) if cyl else None,
            "extents_mm": [round(float(e), 1) for e in ext],
            "height_z_mm": round(float(np.ptp(pts[:, 2])), 1),
            "cylinder_like": bool(cyl)}


def surface_rms(points, top_frac=0.12) -> float:
    """상면 패치 평면 피팅 잔차 RMS(mm) — g(RMS)·흡착 판정 공용 입력."""
    pts = np.asarray(points, dtype=np.float64)
    z = pts[:, 2]
    top = pts[z > z.max() - top_frac * max(np.ptp(z), 1e-6)]
    if len(top) < 20:
        return float("nan")
    Xc = top - top.mean(axis=0)
    normal = np.linalg.svd(Xc, full_matrices=False)[2][-1]
    return float(np.sqrt(np.mean((Xc @ normal) ** 2)))


NO_SEAL_PATCH_MM = 99.9   # seal 크기 평면 패치가 없을 때의 sentinel (흡착 불가)


def seal_patch_rms_mm(points, seal_r_mm=15.0, top_frac=0.25, n_cand=40) -> float:
    """흡착컵이 밀폐할 수 있는 평면 패치가 상면에 있는가 — 있으면 가장 평평한
    패치(반경 seal_r_mm)의 평면잔차 RMS(mm), 없으면 NO_SEAL_PATCH_MM.
    상면 후보 중심들에서 반경 내 점을 모아 평면을 피팅, seal 지름을 실제로 덮는
    패치만 인정한다. 오목한 그릇(숟가락)·둥근 물체(사과)는 seal 크기 평면이 없어
    sentinel → vac 자동 탈락. (오목도 링 지표는 접시 림에 오검출돼 폐기.)"""
    pts = np.asarray(points, dtype=np.float64)
    z = pts[:, 2]
    top = pts[z > z.max() - top_frac * max(np.ptp(z), 1e-6)]
    if len(top) < 20:
        return NO_SEAL_PATCH_MM
    rng = np.random.default_rng(0)
    cand = top[rng.choice(len(top), min(n_cand, len(top)), replace=False)][:, :2]
    best = NO_SEAL_PATCH_MM
    for c in cand:
        patch = top[np.linalg.norm(top[:, :2] - c, axis=1) <= seal_r_mm]
        if len(patch) < 10:
            continue
        if min(np.ptp(patch[:, 0]), np.ptp(patch[:, 1])) < seal_r_mm:
            continue                      # seal 지름을 못 덮는 좁은 패치는 무효
        Xc = patch - patch.mean(axis=0)
        nrm = np.linalg.svd(Xc, full_matrices=False)[2][-1]
        rms = float(np.sqrt(np.mean((Xc @ nrm) ** 2)))
        best = min(best, rms)
    return round(best, 3)


# ── SiPhy 백엔드 (material / density / mass / E) ──────

class PropertyBackend:
    def estimate(self, crop_rgb, cls_hint: str, points_mm=None) -> dict:
        """→ {material, density_kgm3, mass_kg|None, youngs_gpa|None, confidence}
        points_mm(선택): 표면 점군 — 부피적분형 백엔드(SiPhy)가 질량 직접 산출에 사용."""
        raise NotImplementedError



class MockBackend(PropertyBackend):
    """클래스→속성 표 (배관·결정로직 검증용, 결정론적)."""
    TABLE = {
        "milk_carton":  dict(material="liquid_carton", density_kgm3=1030, youngs_gpa=0.5),
        "glass_bottle": dict(material="glass", density_kgm3=2500, youngs_gpa=70),
        "hammer":       dict(material="wood", density_kgm3=600, youngs_gpa=10),
        "rod":          dict(material="wood", density_kgm3=500, youngs_gpa=10),
        "board":        dict(material="wood", density_kgm3=250, youngs_gpa=10),
        "short_hook":   dict(material="wood", density_kgm3=500, youngs_gpa=10),
    }

    def estimate(self, crop_rgb, cls_hint: str, points_mm=None) -> dict:
        d = dict(self.TABLE.get(cls_hint,
                                dict(material="unknown", density_kgm3=500, youngs_gpa=None)))
        d.setdefault("mass_kg", None)
        d["confidence"] = 0.85 if cls_hint in self.TABLE else 0.5
        return d


# ── 마찰 헤드 (우리 신설) ─────────────────────────────

class FrictionHead:
    """0단(상시): μ = μ_table[재질]×g(RMS), g=clip(1+α(RMS−RMS₀), 0.8, 1.5)
    1단(|margin|<ε): 재관측 콜백 → 재계산 / 2단: 프로브 콜백 → 활주 감속 a → μ=a/g
    ※ MU_TABLE은 추정기 테이블 — GT 카탈로그 테이블과 분리 유지."""

    MU_TABLE = {"glass": 0.40, "wood": 0.50, "metal": 0.45, "plastic": 0.35,
                "cardboard": 0.45, "liquid_carton": 0.42, "rubber": 0.90, "unknown": 0.45}
    ALPHA, RMS0, CLIP = 0.05, 1.0, (0.8, 1.5)   # α는 캘리브레이션 씬에서 조정 (GT α와 분리)
    MIN_SLIDE_MM = 50.0

    def g(self, rms_mm: float) -> float:
        if not np.isfinite(rms_mm):
            return 1.0
        return float(np.clip(1.0 + self.ALPHA * (rms_mm - self.RMS0), *self.CLIP))

    def stage0(self, material: str, rms_mm: float) -> float:
        return round(self.MU_TABLE.get(material, self.MU_TABLE["unknown"]) * self.g(rms_mm), 3)

    def probe_mu_from_track(self, positions_mm, dt_s: float):
        """자유 활주 위치 3점 이상(등간격 dt) → p(t)=p0+v0t−½at² 최소자승 → μ=a/g.
        힘·질량 불요. 정지 후 프레임(이동<2mm) 자동 트리밍, 활주<50mm면 무효."""
        P = np.asarray(positions_mm, dtype=np.float64)[:, :2] / 1000.0
        if len(P) < 3:
            return None
        steps = np.linalg.norm(np.diff(P, axis=0), axis=1) * 1000.0
        stop = np.argmax(steps < 2.0) + 1 if np.any(steps < 2.0) else len(P)
        P = P[:max(stop, 3)]
        if len(P) < 3 or np.linalg.norm(P[-1] - P[0]) * 1000.0 < self.MIN_SLIDE_MM:
            return None
        u = (P[-1] - P[0]) / np.linalg.norm(P[-1] - P[0])
        s = (P - P[0]) @ u
        t = np.arange(len(P)) * dt_s
        A = np.stack([t, -0.5 * t ** 2, np.ones_like(t)], axis=1)
        _, a, _ = np.linalg.lstsq(A, s, rcond=None)[0]
        return round(float(a / G), 3) if a > 0 else None

    def estimate(self, material: str, rms_mm: float, margin_fn=None, eps: float = 0.05,
                 remeasure_fn=None, probe_fn=None) -> dict:
        """margin_fn(mu)→결정 margin. 없거나 여유 있으면 0단 종료 (측정 lazy)."""
        mu = self.stage0(material, rms_mm)
        rec = {"mu": mu, "stage": 0, "material": material, "rms_mm": round(float(rms_mm), 2)}
        if margin_fn is None or abs(margin_fn(mu)) >= eps:
            return rec
        if remeasure_fn is not None:                              # 1단: 근접 재관측
            material, rms_mm = remeasure_fn()
            mu = self.stage0(material, rms_mm)
            rec.update(mu=mu, stage=1, material=material, rms_mm=round(float(rms_mm), 2))
            if abs(margin_fn(mu)) >= eps:
                return rec
        # ── 2단 마찰 프로브: M5 프리미티브(probe_push) 연결 전까지 비활성 ──
        # TODO(M5): src/tuj/m5_motion/README.md의 probe_push 완성 시 주석 해제
        # if probe_fn is not None:                                  # 2단: 물리 프로브
        #     track, dt = probe_fn()
        #     mu_p = self.probe_mu_from_track(track, dt)
        #     if mu_p is not None:
        #         rec.update(mu=mu_p, stage=2)
        return rec


# ── 기하 스키마 ────────────────────────────────────────
# EE 규칙이 요구하는 기하 필드. 메모리 hit이 이 필드를 못 갖추면(옛 스키마) 재계산한다.
# 새 기하 필드를 추가하면 여기에도 등록할 것 — 그래야 캐시가 자동 갱신된다.
GEOMETRY_REQUIRED = ("footprint_mm", "height_mm", "seal_patch_rms_mm", "surface_rms_mm")
# vac flatness / seal_patch / grasp footprint 등 EE 평가에 필요한 numeric 필드.
GEOMETRY_FINITE_SCALARS = ("surface_rms_mm", "seal_patch_rms_mm", "height_mm")


def _finite_number(value) -> bool:
    """EE 비교에 쓸 수 있는 finite float 인가 (None / NaN / ±Inf 제외)."""
    if value is None:
        return False
    try:
        if hasattr(value, "item") and not isinstance(value, (bytes, str)):
            try:
                value = value.item()
            except (ValueError, AttributeError):
                pass
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def geometry_from_node(node: dict) -> dict:
    """노드 점군·bbox → 기하 dict (VLM 0회, 결정론적). 메모리 hit 기하 갱신에도 사용."""
    pts = node["_points"]
    bbox = node["bbox_mm"]
    return pca_dims(pts) | {
        "surface_rms_mm": round(surface_rms(pts), 2),
        "seal_patch_rms_mm": seal_patch_rms_mm(pts),
        "footprint_mm": [round(float(bbox[0]), 1), round(float(bbox[1]), 1)],
        "height_mm": round(float(bbox[2]), 1)}


def geometry_is_current(geometry) -> bool:
    """기하 dict가 EE 평가에 필요한 키·finite numeric 값을 갖췄는가."""
    if not geometry:
        return False
    if not all(k in geometry for k in GEOMETRY_REQUIRED):
        return False
    if not all(_finite_number(geometry.get(k)) for k in GEOMETRY_FINITE_SCALARS):
        return False
    fp = geometry.get("footprint_mm")
    try:
        seq = list(fp) if fp is not None else []
    except TypeError:
        return False
    return len(seq) >= 2 and all(_finite_number(x) for x in seq[:2])


def estimate_mass_kg_from_bbox(node: dict, density_kgm3: float,
                               cylinder_like: bool = False) -> float:
    """최후 폴백: bbox 부피(m³) × density (hollow object에는 부적절)."""
    vol = float(np.prod(np.asarray(node["bbox_mm"], dtype=np.float64) / 1000.0))
    return round(vol * (np.pi / 4 if cylinder_like else 1.0) * float(density_kgm3), 3)


def estimate_mass_from_material_hypotheses(points_mm, materials_topk) -> dict | None:
    """Full SiPhy와 동일: 저장된 material hypotheses(thickness 포함) + 현재 점군 → shell mass.

    materials_topk 항목에 density_kgm3·thickness_cm range와 prob가 있어야 한다.
    불완전하면 None → 호출측이 bbox 폴백으로 떨어질 수 있다.
    """
    from tuj.m1_scene.siphy_backend import (  # local: avoid import cycle at module load
        _effective_probs, shell_mass_integral)

    if points_mm is None or len(np.asarray(points_mm)) < 3:
        return None
    if not materials_topk:
        return None

    dens_rows, thick_rows, probs = [], [], []
    for m in materials_topk:
        dens = m.get("density_kgm3")
        thick = m.get("thickness_cm")
        if dens is None or thick is None:
            return None
        dens = list(dens) if not isinstance(dens, (int, float)) else [float(dens), float(dens)]
        thick = list(thick) if not isinstance(thick, (int, float)) else [float(thick), float(thick)]
        if len(dens) < 2:
            dens = [float(dens[0]), float(dens[0])]
        if len(thick) < 2:
            thick = [float(thick[0]), float(thick[0])]
        dens_rows.append([float(dens[0]), float(dens[1])])
        thick_rows.append([float(thick[0]), float(thick[1])])
        probs.append(max(float(m.get("prob", 0.0)), 0.0))

    probs_raw = np.asarray(probs, dtype=np.float64)
    if probs_raw.sum() <= 0:
        probs_raw = np.full(len(probs_raw), 1.0 / len(probs_raw))
    else:
        probs_raw = probs_raw / probs_raw.sum()
    probs_eff, _, _, _ = _effective_probs(probs_raw)
    return shell_mass_integral(
        points_mm, probs_eff,
        np.asarray(dens_rows, dtype=np.float64),
        np.asarray(thick_rows, dtype=np.float64))


def apply_memory_hit_to_observation(node: dict, reused: dict) -> dict:
    """Memory HIT: intrinsic/material 재사용 + 현재 observation geometry/mass.

    재사용: material, density, Young's, mu(마찰계수), confidence, materials_topk, caption
    현재 관측: geometry 전부
    mass: Full SiPhy ``shell_mass_integral`` (hypotheses thickness × 현재 점군).
          thickness 정보가 없을 때만 bbox×density 폴백.
    """
    geom = geometry_from_node(node)
    density = reused["density_kgm3"]

    shell = estimate_mass_from_material_hypotheses(
        node.get("_points"), reused.get("materials_topk"))
    if shell is not None:
        mass = shell["mass_kg"]
        mass_range = shell.get("mass_range_kg")
        mass_source = "shell_mass_integral"
    else:
        mass = estimate_mass_kg_from_bbox(
            node, density, bool(geom.get("cylinder_like")))
        mass_range = None
        mass_source = "bbox_density_fallback"

    mu_src = reused.get("mu") or {}
    mu = {
        "mu": mu_src.get("mu"),
        "stage": mu_src.get("stage", 0),
        "material": mu_src.get("material"),
        # RMS는 geometry-dependent → 현재 상면 RMS로 맞춤 (μ 계수 자체는 memory)
        "rms_mm": geom.get("surface_rms_mm", mu_src.get("rms_mm")),
    }

    out = {
        "geometry": geom,
        "material": reused.get("material"),
        "density_kgm3": density,
        "mass_kg": mass,
        "youngs_gpa": reused.get("youngs_gpa"),
        "mu": mu,
        "confidence": reused.get("confidence"),
        "geometry_source": "current_observation",
        "intrinsic_source": "memory",
        "mass_source": mass_source,
    }
    if mass_range is not None:
        out["mass_range_kg"] = mass_range
    for k in ("materials_topk", "caption"):
        if k in reused:
            out[k] = reused[k]
    return out


# ── intrinsic 접지 진입점 ─────────────────────────────

def ground_intrinsic(node: dict, crop_rgb=None, backend: PropertyBackend | None = None,
                     friction: FrictionHead | None = None, **friction_hooks) -> dict:
    backend = backend or MockBackend()
    friction = friction or FrictionHead()
    pts = node["_points"]
    geom = geometry_from_node(node)
    rms = surface_rms(pts)                             # 마찰 헤드용 원시 RMS
    props = backend.estimate(crop_rgb, node["class"], points_mm=pts)

    mass = props.get("mass_kg")
    if mass is None:                                   # 백엔드 미제공 시 bbox부피×밀도 폴백
        mass = estimate_mass_kg_from_bbox(
            node, props["density_kgm3"], bool(geom.get("cylinder_like")))

    mu = friction.estimate(props["material"], rms, **friction_hooks)
    out = {"geometry": geom,
           "material": props["material"], "density_kgm3": props["density_kgm3"],
           "mass_kg": mass, "youngs_gpa": props.get("youngs_gpa"),
           "mu": mu, "confidence": props.get("confidence"),
           "geometry_source": "current_observation",
           "intrinsic_source": "backend"}
    for k in ("mass_range_kg", "materials_topk", "caption"):   # SiPhy 부가 출력 보존
        if k in props:
            out[k] = props[k]
    return out
