"""질량 프로브 (2단 에스컬레이션) — 리프트 테스트로 시각 추정 오판 교정 데모.

★ 현재 비활성 (M5 프리미티브 연결 전) — 실행 시 안내만 출력하고 종료.
  xfrc 직접 인가 방식은 sim 검증용 거친 버전이라, M5의 grasp-lift-replace
  (src/tuj/m5_motion/README.md) 완성 시 lift_probe()를 교체하고
  아래 main() 끝의 주석을 해제해 활성화한다.

배치: Tool-Use-Journal/scripts/probe_mass_c1_1.py
선행: python scripts/run_m3_c1_1.py --backend siphy   (시각 추정 생성)
실행: python scripts/probe_mass_c1_1.py               (기본: light/heavy 접시 프로브)
      python scripts/probe_mass_c1_1.py --update      (m3_intrinsic.json에 프로브 질량 반영)

원리: 물체에 알려진 상방력 F를 인가 → 리프트오프 후 자유비행 z(t)를 2차 피팅
      → 가속도 a → m = F / (a + g).  힘/질량 센서 불요 (마찰 프로브 μ=a/g와 동일 철학).

트리거 논의: heavy 접시는 시각 추정이 '자신 있게 틀려' margin 게이트가 안 울림.
  실전 트리거는 ① 동일 외형 쌍 감지(시각 구분 불가 후보) ② 임계 결정(vac payload)
  사전검증 정책 ③ M6 실행 실패 피드백 — 본 스크립트는 그 트리거가 울렸다고 가정한 2단.
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

TASK = "c1_1"
OUT = ROOT / "output" / TASK
G = 9.81
F_UP = 15.0        # N — 예상 최대 질량(1kg 남짓)의 무게보다 충분히 큰 상방력
LIFT_MM = 3.0      # 리프트오프 판정 (초기 대비 상승량)


def fit_mass_from_track(z_m, dt_s, f_up=F_UP, lift_mm=LIFT_MM):
    """자유비행 구간 z(t) → 2차 최소자승 → a → m = F/(a+g). (순수 함수 — 테스트용)"""
    z = np.asarray(z_m, dtype=np.float64)
    off = np.nonzero(z - z[0] > lift_mm / 1000.0)[0]   # 리프트오프 이후만 사용
    if len(off) < 8:
        return None
    i0 = off[0]
    zz = z[i0:]
    t = np.arange(len(zz)) * dt_s
    A = np.stack([np.ones_like(t), t, 0.5 * t ** 2], axis=1)
    _, _, a = np.linalg.lstsq(A, zz, rcond=None)[0]
    if a <= -G:                                        # 물리적으로 불가
        return None
    return round(float(f_up / (a + G)), 3)


def lift_probe(env, root_body: str, settle=100, steps=120):
    """sim에서 리프트 프로브 실행. (실제 로봇에선 M5의 제어된 리프트가 이 역할)"""
    sim = env.sim
    bid = sim.model.body_name2id(root_body)
    for _ in range(settle):                            # 정착
        sim.step()
    xfrc = sim.data.xfrc_applied                       # (nbody, 6)
    xfrc[bid][:] = 0.0
    xfrc[bid][2] = F_UP
    dt = float(sim.model.opt.timestep)
    zs = []
    for _ in range(steps):
        sim.step()
        zs.append(float(sim.data.body_xpos[bid][2]))
    xfrc[bid][:] = 0.0
    return fit_mass_from_track(zs, dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="프로브 질량을 m3_intrinsic.json에 반영 (stage 2 기록)")
    args = ap.parse_args()

    import os, platform
    if platform.system() == "Windows" and os.environ.get("MUJOCO_GL", "").lower() not in ("", "wgl", "glfw"):
        os.environ["MUJOCO_GL"] = "wgl"
    import environments  # noqa: F401
    import robosuite as suite
    from tuj.m3_grounding import evaluate_ee

    cache_p = OUT / "m3_intrinsic.json"
    est_cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}

    env = suite.make(env_name="C1_1_LegoSweep", robots="UR5e",
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, render_camera="agentview", ignore_done=True)
    env.reset()

    spec = json.loads((ROOT / "configs" / "robot_spec.json").read_text(encoding="utf-8"))
    ee_pool = []
    for e in spec["ee_pool"]:
        e = dict(e)
        if "flatness_tol_rms_mm" in e:
            e["seal_rms_tol_mm"] = e["flatness_tol_rms_mm"]
        ee_pool.append(e)

    # 프로브 대상: 시각 구분 불가 쌍 (트리거 ① 가정)
    targets = {"obj_plate_light_plate": env.light_plate,
               "obj_plate_heavy_plate": env.heavy_plate}
    results = {}
    for nid, obj in targets.items():
        m_probe = lift_probe(env, obj.root_body)
        env.reset()                                    # 프로브는 씬을 교란 → 복원
        gt = float(env.sim.model.body_subtreemass[env.sim.model.body_name2id(obj.root_body)])
        est = est_cache.get(nid, {})
        m_vis = est.get("mass_kg")
        line = {"mass_visual_kg": m_vis, "mass_probe_kg": m_probe,
                "mass_gt_kg": round(gt, 3), "stage": 2}
        if est and m_probe is not None:
            feas_vis = sorted(k for k, v in ((e["ee_id"], evaluate_ee(e, est)) for e in ee_pool)
                              if v["feasible"])
            feas_prb = sorted(k for k, v in ((e["ee_id"], evaluate_ee(e, est | {"mass_kg": m_probe}))
                                             for e in ee_pool) if v["feasible"])
            feas_gt = sorted(k for k, v in ((e["ee_id"], evaluate_ee(e, est | {"mass_kg": gt}))
                                            for e in ee_pool) if v["feasible"])
            line |= {"feasible_visual": feas_vis, "feasible_probe": feas_prb,
                     "feasible_gt": feas_gt,
                     "decision_fixed": feas_prb == feas_gt and feas_vis != feas_gt}
        results[nid] = line
        print(f"[probe] {nid}: 시각 {m_vis}kg -> 프로브 {m_probe}kg (GT {gt:.3f}kg)")
        if "feasible_visual" in line:
            print(f"        EE 판정  시각 {line['feasible_visual']} | "
                  f"프로브 {line['feasible_probe']} | GT {line['feasible_gt']}"
                  + ("  << 결정 교정됨" if line.get("decision_fixed") else ""))
    env.close()

    (OUT / "probe_mass.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] -> {OUT / 'probe_mass.json'}")

    if args.update and est_cache:
        for nid, r in results.items():
            if nid in est_cache and r["mass_probe_kg"] is not None:
                est_cache[nid]["mass_kg"] = r["mass_probe_kg"]
                est_cache[nid]["mass_stage"] = 2
                est_cache[nid]["mass_range_kg"] = [round(r["mass_probe_kg"] * 0.9, 3),
                                                   round(r["mass_probe_kg"] * 1.1, 3)]
        cache_p.write_text(json.dumps(est_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[update] m3_intrinsic.json에 프로브 질량 반영 (stage 2) — "
              f"eval_m3_mass_c1_1.py 재실행으로 decision_agreement 확인")


if __name__ == "__main__":
    # TODO(M5): grasp-lift-replace 프리미티브 연결 시 아래 주석 해제
    print("[probe] 질량 프로브는 M5 프리미티브(probe_lift) 연결 전까지 비활성 상태입니다.")
    print("        활성화: 이 파일 맨 아래 main() 주석 해제 (검증용 xfrc 버전)")
    # main()
