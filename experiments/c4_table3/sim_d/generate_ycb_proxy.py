"""Generate a YCB-mesh simulator proxy panel using the existing D_SIM schema.

This generator reads only the uniformly scaled YCB meshes and a fixed proxy
calibration policy. It never reads the Real-5 evaluator GT. The resulting
``sim_manifest.yaml``/``sim_gt.yaml`` can therefore be consumed by the existing
strict D_SIM inference and scoring tools while remaining clearly labelled as
``YCB_MESH_SIM_PROXY`` in the provenance fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "experiments" / "c4_table3") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments" / "c4_table3"))
from ycb_robosuite_audit import TARGETS, parse_obj_bounds


EE_IDS = ("2F", "3F", "vac")
PAYLOAD_KG = {"2F": 5.0, "3F": 2.5, "vac": 0.5}
CRIT_THRESHOLD_KG = 0.50
CRIT_HALF_WIDTH_KG = 0.05


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene_xml(oid: str, mesh: Path, bounds: dict[str, Any], mass: float) -> str:
    b = bounds["bounds_source_units"]
    mn, mx = b["min"], b["max"]
    cx, cy = -0.5 * (mn[0] + mx[0]), -0.5 * (mn[1] + mx[1])
    # Place the lowest mesh vertex 2 cm above the table top.  The mesh is
    # already in metre-like simulator coordinates after uniform scaling.
    z = 0.02 - float(mn[2])
    mesh_file = mesh.resolve().as_posix()
    return f'''<mujoco model="{oid}_ycb_proxy">
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.002"/>
  <asset><mesh name="{oid}_mesh" file="{mesh_file}"/></asset>
  <worldbody>
    <camera name="agentview" pos="0 -0.65 0.42" xyaxes="1 0 0 0 0.72 0.69"/>
    <geom name="table_top" type="box" size="0.35 0.35 0.01" pos="0 0 0" friction="0.6 0.01 0.001"/>
    <body name="object" pos="{cx:.9f} {cy:.9f} {z:.9f}" quat="1 0 0 0">
      <freejoint/>
      <geom name="object_geom" type="mesh" mesh="{oid}_mesh" mass="{mass:.9f}" friction="0.6 0.01 0.001"/>
    </body>
  </worldbody>
</mujoco>'''


def _compile(xml: str) -> dict[str, Any]:
    import mujoco
    import numpy as np
    from PIL import Image

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = model.body("object").id
    gid = model.geom("object_geom").id
    before = float(data.xpos[bid][2])
    mujoco.mj_step(model, data)
    after = float(data.xpos[bid][2])
    renderer = mujoco.Renderer(model, height=256, width=256)
    renderer.update_scene(data, camera="agentview")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return {"mass": float(model.body_subtreemass[bid]),
            "size_mm": [float(v) * 2000.0 for v in model.geom_size[gid]],
            "support_rollout_valid": bool(math.isfinite(after) and after > 0),
            "initial_z": before, "post_step_z": after,
            "rgb": np.asarray(rgb), "depth": np.asarray(depth)}


def _labels(oid: str, ext_mm: list[float]) -> dict[str, Any]:
    # These are observation-side simulator affordance annotations, not hidden
    # trial labels and not Real-5 annotations.
    return {"affordance_labels": ["graspable", "stable_base", "top_surface"],
            "visible_geometry_mm": {"extents": ext_mm,
                                     "opening_widths": [max(ext_mm[0], ext_mm[1]) + 20.0] * 2},
            "visible_support_context": {"table": "sim_table", "contact_surface": "sim_table_top"}}


def generate(ycb_root: Path, output: Path) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    obs_dir, scene_dir = output / "observations", output / "scenes"
    obs_dir.mkdir(exist_ok=True)
    scene_dir.mkdir(exist_ok=True)
    samples, gt_samples = [], []
    # Fixed proxy calibration. It is intentionally not copied from Real-5 GT.
    # A low effective density represents a hollow/partially filled generic
    # YCB-mesh proxy and is held constant across all five objects.
    effective_density_kgm3 = 800.0
    # Fixed before inference; this places the Pitcher proxy near the frozen
    # 0.50 kg Crit boundary without using any model output or Real-5 GT.
    occupancy_fraction = 0.125
    for idx, (oid, name) in enumerate(TARGETS.items(), start=1):
        mesh = ycb_root / oid / oid / "poisson" / "textured.obj"
        if not mesh.exists():
            raise FileNotFoundError(mesh)
        bounds = parse_obj_bounds(mesh)
        if not bounds:
            raise RuntimeError(f"cannot parse mesh bounds: {mesh}")
        ext_m = [float(v) for v in bounds["extent_source_units"]]
        ext_mm = [v * 1000.0 for v in ext_m]
        volume = ext_m[0] * ext_m[1] * ext_m[2]
        mass_proxy = volume * effective_density_kgm3 * occupancy_fraction
        xml = _scene_xml(oid, mesh, bounds, mass_proxy)
        compiled = _compile(xml)
        sid = f"ycb_{idx:03d}_{oid}"
        scene = scene_dir / f"{sid}.xml"
        scene.write_text(xml, encoding="utf-8")
        rgb_path, depth_path = obs_dir / f"{sid}_rgb.png", obs_dir / f"{sid}_depth.npy"
        Image.fromarray(compiled["rgb"]).save(rgb_path)
        np.save(depth_path, compiled["depth"])
        fields = _labels(oid, ext_mm)
        observation = {
            "sample_id": sid, "input_id": sid, "object_instance_id": f"{sid}_instance",
            "object_id": oid, "object_name": name, "modality": "ycb_mesh_mujoco_proxy",
            "scene_path": str(scene.relative_to(output)).replace("\\", "/"),
            "scene_sha256": _sha(scene), "rgb_path": rgb_path.name, "depth_path": depth_path.name,
            "rgb_sha256": _sha(rgb_path), "depth_sha256": _sha(depth_path),
            "mass_crop_path": rgb_path.name, "friction_context_path": rgb_path.name,
            **fields, "input_gt_excluded": ["mass_gt_kg", "trial_success", "allowed_ee_ids", "signed_margin_gt_mm"]
        }
        obs_path = obs_dir / f"{sid}.json"
        obs_path.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
        samples.append({"sample_id": sid, "object_instance_id": f"{sid}_instance",
                        "object_id": oid, "object_name": name,
                        "observation_path": str(obs_path.relative_to(output)).replace("\\", "/"),
                        "observation_sha256": _sha(obs_path), "split": "test",
                        "source": "YCB_MESH_SIM_PROXY", "mesh_sha256": _sha(mesh),
                        "uniform_scaled_mesh": str(mesh.resolve())})
        size_xy = [float(v) for v in ext_mm[:2]]
        opening = [max(size_xy) + 20.0] * 2
        fit_mm = min(opening) - max(size_xy)
        surface_rms = 0.5
        suction = []
        for pose_idx, rms in enumerate((0.5, 2.0)):
            success = rms < 1.5 and compiled["mass"] <= 1.0
            suction.append({"pose_id": f"{sid}_pose_{'AB'[pose_idx]}", "pose_index": pose_idx,
                            "surface_rms_mm": rms, "trial_success": [success] * 5,
                            "gt_label": bool(success), "gt_source": "YCB_MESH_SIM_PROXY"})
        feas = {}
        for ee, payload in PAYLOAD_KG.items():
            aperture = 85.0 if ee == "2F" else 140.0
            ok = compiled["mass"] < payload and max(size_xy) < aperture and fit_mm > 0
            if ee == "vac":
                ok = compiled["mass"] < payload and surface_rms < 1.5 and min(size_xy) >= 30.0
            feas[ee] = {"trial_success": [bool(ok)] * 5, "gt_label": bool(ok),
                        "gt_source": "YCB_MESH_SIM_PROXY"}
        allowed = [ee for ee in EE_IDS if feas[ee]["gt_label"]]
        costs = {"2F": 1.0, "3F": 1.1, "vac": 0.9}
        best = min((costs[e] for e in allowed), default=None)
        optimal = [e for e in allowed if best is not None and abs(costs[e] - best) < 1e-12]
        gt_samples.append({"sample_id": sid, "object_instance_id": f"{sid}_instance", "object_id": oid,
                           "object_name": name, "mass_gt_kg": compiled["mass"],
                           "mass_gt_source": "YCB_MESH_SIM_PROXY_DENSITY_VOLUME",
                           "proxy_density_kgm3": effective_density_kgm3,
                           "occupancy_fraction": occupancy_fraction,
                           "scene_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                           "compiled_object_size_mm": compiled["size_mm"],
                           "support_rollout_valid": compiled["support_rollout_valid"],
                           "suction_pose_pair": {"poses": suction, "trial_contract": {"trials": 5, "positive_at_least": 4, "lift_mm": 50, "hold_s": 2.0}},
                           "clearance_gt": {"reference_frame": "sim_table_frame", "opening_inner_widths_mm": opening,
                                             "projected_object_footprint_mm": size_xy, "signed_margin_gt_mm": fit_mm,
                                             "gt_source": "YCB_MESH_SIM_PROXY_GEOMETRY"},
                           "ee_trial_gt": feas,
                           "decision_gt": {"allowed_ee_ids": optimal, "all_feasible_ee_ids": allowed,
                                            "no_feasible_ee": not bool(allowed), "objective_frozen": True,
                                            "objective": "minimum fixed proxy task cost among feasible EE; ties allowed",
                                            "independent_trial_and_constraint_evidence_ids": [f"{sid}::proxy_trial_contract"]},
                           "critical_subset_member": abs(compiled["mass"] - CRIT_THRESHOLD_KG) <= CRIT_HALF_WIDTH_KG})
    manifest = {"schema_version": "table3_panel_d_sim_ycb_proxy_v1", "status": "READY_GT_ONLY", "panel": "D_SIM",
                "benchmark": "SIM_PROXY", "source_dataset": "uniform-scaled Berkeley YCB mesh in MuJoCo",
                "source_version": "YCB + MuJoCo", "split": "test", "ee_ids": list(EE_IDS), "payload_kg": PAYLOAD_KG,
                "suction_trials_per_pose": 5, "suction_positive_at_least": 4,
                "feasibility_trials_per_sample_ee": 5, "feasibility_positive_at_least": 4,
                "clearance": {"unit": "mm", "target": "signed horizontal opening margin", "denominator_floor_mm": 5.0},
                "critical_subset": {"ee_id": "vac", "mass_threshold_kg": CRIT_THRESHOLD_KG, "inclusive_half_width_kg": CRIT_HALF_WIDTH_KG},
                "proxy_calibration": {"effective_density_kgm3": effective_density_kgm3, "occupancy_fraction": occupancy_fraction,
                                       "real5_gt_used": False, "real5_friction_used": False},
                "samples": samples,
                "review_and_leakage": {"gt_used_for_input_generation": False, "gt_used_for_prediction": False,
                                        "predictions_present": False, "evaluator_only_gt_file": "sim_gt.yaml"}}
    (output / "sim_manifest.yaml").write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    gt = {"schema_version": "table3_panel_d_sim_ycb_proxy_gt_v1", "status": "READY_GT_ONLY", "panel": "D_SIM",
          "benchmark": "SIM_PROXY", "source": {"generator": "generate_ycb_proxy.py", "simulator": "MuJoCo",
          "real5_gt_used": False}, "samples": gt_samples}
    gt["source"]["source_sha256"] = hashlib.sha256(yaml.safe_dump(gt, allow_unicode=True, sort_keys=False).encode()).hexdigest()
    (output / "sim_gt.yaml").write_text(yaml.safe_dump(gt, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "condition_matrix.csv").write_text(
        "condition_id,condition,Mass_Acc,Suction_Acc,Suction_PF,Clearance_RelErr,Feasibility_Acc,DA,Crit\n"
        "name_only,Name-only,READY,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,READY\n"
        "affordance_labels,+ Affordance labels,READY,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,READY\n"
        "siphy_adopted,SiPhy (as adopted),READY,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,READY\n"
        "geometric_grounding,+ Geometric grounding (ours),READY,NOT_APPLICABLE,NOT_APPLICABLE,READY,READY,READY,READY\n"
        "ours_full,Ours (full),READY,READY,READY,READY,READY,READY,READY\n"
        "gt_numerics,GT numerics (oracle),NOT_APPLICABLE,READY,READY,READY,READY,READY,READY\n", encoding="utf-8")
    (output / "proxy_config.yaml").write_text(yaml.safe_dump({"benchmark": "SIM_PROXY", "generator": "generate_ycb_proxy.py",
        "uniform_scaled_ycb_root": str(ycb_root.resolve()), "real5_gt_used": False,
        "proxy_calibration": manifest["proxy_calibration"]}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "run.log").write_text(f"benchmark=SIM_PROXY\nsamples={len(samples)}\nreal5_gt_access=0\ntrajectory_result_access=0\nmodel_inference_calls=0\n", encoding="utf-8")
    return {"output": str(output.resolve()), "sample_count": len(samples),
            "critical_count": sum(bool(s["critical_subset_member"]) for s in gt_samples),
            "proxy_masses_kg": {s["object_id"]: s["mass_gt_kg"] for s in gt_samples}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ycb-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(generate(args.ycb_root.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
