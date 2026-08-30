"""M3-(c) EE-conditioned 접지: intrinsic 수치 × 로봇 스펙 → EE별 {feasible, margin, reason}.

규칙 5종 (임계값은 전부 ee_spec에서 — 하드코딩 없음):
  dim_lt_stroke   파지 치수 < 개구/스트로크           (2F/3F)
  mass_lt_payload 질량 < 페이로드                     (전 EE)
  grip_slip       필요 파지력 = m·g/(2μ) ≤ 최대 파지력 (2F/3F, grip_force_n 있을 때 μ가 결정에 진입)
  flatness        상면 RMS < seal 허용치               (vac)
  seal_contact    접촉면 최소 치수 ≥ seal_diameter     (vac — 뾰족·좁은 물체 자동 탈락)
margin = 지배 제약(가장 빡빡한/위반된 것)의 여유. 마찰 에스컬레이션의 margin_fn도 여기서 제공.
"""
from __future__ import annotations

import math

G = 9.81

# ee_spec 기대 형식 (configs/robot_spec.json ee_pool 각 항목):
# {"ee_id","type": "parallel_2f|underactuated_3f|vacuum",
#  "stroke_mm"|"aperture_mm", "payload_kg",
#  "grip_force_n"(2F/3F),
#  "seal_rms_tol_mm"(vac; run_m3가 flatness_tol_rms_mm → seal_rms_tol_mm 정규화),
#  "seal_diameter_mm"(vac), "pass_height_mm"}


def _check(rule, value, limit, less_than=True):
    ok = value < limit if less_than else value > limit
    slack = (limit - value) if less_than else (value - limit)
    return {"rule": rule, "value": round(float(value), 3),
            "limit": round(float(limit), 3), "pass": bool(ok), "slack": round(float(slack), 3)}


def grasp_dim_mm(geometry: dict) -> float:
    """파지 치수: 원통이면 지름, 아니면 최소 유의미 extent."""
    if geometry.get("diameter_mm") is not None:
        return geometry["diameter_mm"]
    ext = sorted(geometry["extents_mm"])
    return ext[0] if ext[0] > 5 else ext[1]           # 최소 유의미 치수


def seal_contact_dim_mm(geometry: dict) -> float:
    """vac 접촉면 최소 폭: extents 정렬 후 2번째로 큰 값 (평면 상면 가정).
    · 접시  [182,182,11]  → 182 (통과 후보)
    · 스푼자루 [180,15,5] → 15 (탈락)
    · 원통  diameter로 대체."""
    if geometry.get("diameter_mm") is not None:
        return geometry["diameter_mm"]
    return sorted(geometry["extents_mm"], reverse=True)[1]


def grip_slip_margin_fn(mass_kg: float, grip_force_n: float):
    """μ의 결정 margin 함수 (마찰 헤드 에스컬레이션 게이트에 그대로 주입).
    필요 파지력 F_req = m·g/(2μ)  →  margin(μ) = 1 − F_req/F_max  (0 근처 = 임계)."""
    def margin(mu: float) -> float:
        f_req = mass_kg * G / (2 * max(mu, 1e-6))
        return 1.0 - f_req / max(grip_force_n, 1e-6)
    return margin


def evaluate_ee(ee_spec: dict, intrinsic: dict) -> dict:
    """intrinsic = ground_intrinsic() 결과. → {feasible, margin, margin_unit, reason, checks}"""
    g, m = intrinsic["geometry"], intrinsic["mass_kg"]
    mu = intrinsic["mu"]["mu"]
    t = ee_spec["type"]
    checks, reasons, units = [], {}, {}

    if t in ("parallel_2f", "underactuated_3f"):
        limit = ee_spec.get("stroke_mm") or ee_spec.get("aperture_mm")
        dim = grasp_dim_mm(g)
        checks.append(_check("dim_lt_stroke", dim, limit))
        units["dim_lt_stroke"] = "mm"
        reasons["dim_lt_stroke"] = f"치수 {dim:.0f} vs 개구 {limit:.0f}mm"

        checks.append(_check("mass_lt_payload", m, ee_spec["payload_kg"]))
        units["mass_lt_payload"] = "kg"
        reasons["mass_lt_payload"] = f"질량 {m} vs 페이로드 {ee_spec['payload_kg']}kg"

        if "grip_force_n" in ee_spec:                 # ★ grip_slip: μ가 결정에 들어옴
            f_req = m * G / (2 * max(mu, 1e-6))
            checks.append(_check("grip_slip", f_req, ee_spec["grip_force_n"]))
            units["grip_slip"] = "N"
            reasons["grip_slip"] = (f"필요 파지력 {f_req:.1f}N(μ={mu}) "
                                    f"vs 최대 {ee_spec['grip_force_n']}N")
    elif t == "vacuum":
        rms = g["surface_rms_mm"]
        checks.append(_check("flatness", rms, ee_spec["seal_rms_tol_mm"]))
        units["flatness"] = "mm"
        reasons["flatness"] = f"상면 RMS {rms} vs 허용 {ee_spec['seal_rms_tol_mm']}"

        checks.append(_check("mass_lt_payload", m, ee_spec["payload_kg"]))
        units["mass_lt_payload"] = "kg"
        reasons["mass_lt_payload"] = f"질량 {m} vs 페이로드 {ee_spec['payload_kg']}kg"

        if "seal_diameter_mm" in ee_spec:             # ★ seal_contact: 접촉면 최소 폭
            contact = seal_contact_dim_mm(g)
            checks.append(_check("seal_contact", ee_spec["seal_diameter_mm"], contact))
            units["seal_contact"] = "mm"
            reasons["seal_contact"] = (f"접촉면 폭 {contact:.0f} vs "
                                       f"seal 직경 {ee_spec['seal_diameter_mm']:.0f}mm")
    else:
        return {"feasible": False, "margin": 0.0, "margin_unit": "",
                "reason": f"규칙 없음: {t}", "checks": []}

    fails = [c for c in checks if not c["pass"]]
    dominant = min(fails or checks, key=lambda c: c["slack"])
    return {"feasible": not fails,
            "margin": dominant["slack"],
            "margin_unit": units.get(dominant["rule"], ""),
            "reason": "; ".join(reasons[c["rule"]] for c in (fails or [dominant])),
            "checks": [{k: c[k] for k in ("rule", "value", "limit", "pass")} for c in checks]}


def reach_check(reach_mm: float, center_mm) -> dict:
    d = math.hypot(center_mm[0], center_mm[1])
    return {"distance_mm": round(d, 1), "reach_mm": reach_mm,
            "reachable": d <= reach_mm, "margin_mm": round(reach_mm - d, 1)}
