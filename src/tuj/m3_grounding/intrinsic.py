"""M3-(a) intrinsic 접지: 객체 단일 속성.

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


# ── intrinsic 접지 진입점 ─────────────────────────────

def ground_intrinsic(node: dict, crop_rgb=None, backend: PropertyBackend | None = None,
                     friction: FrictionHead | None = None, **friction_hooks) -> dict:
    backend = backend or MockBackend()
    friction = friction or FrictionHead()
    pts = node["_points"]
    dims = pca_dims(pts)
    rms = surface_rms(pts)
    props = backend.estimate(crop_rgb, node["class"], points_mm=pts)

    mass = props.get("mass_kg")
    if mass is None:                                   # 백엔드 미제공 시 bbox부피×밀도 폴백
        vol = float(np.prod(np.asarray(node["bbox_mm"]) / 1000.0))
        mass = round(vol * (np.pi / 4 if dims["cylinder_like"] else 1.0)
                     * props["density_kgm3"], 3)

    mu = friction.estimate(props["material"], rms, **friction_hooks)
    out = {"geometry": dims | {"surface_rms_mm": round(rms, 2)},
           "material": props["material"], "density_kgm3": props["density_kgm3"],
           "mass_kg": mass, "youngs_gpa": props.get("youngs_gpa"),
           "mu": mu, "confidence": props.get("confidence")}
    for k in ("mass_range_kg", "materials_topk", "caption"):   # SiPhy 부가 출력 보존
        if k in props:
            out[k] = props[k]
    return out
