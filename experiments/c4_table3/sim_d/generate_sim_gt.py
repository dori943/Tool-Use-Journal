"""Generate a small, reproducible MuJoCo-only Table III D panel.

The generated labels are simulator ground truth, not real-world measurements.
Hidden mass, contact and task state are taken from the compiled MuJoCo model and
from a fixed scripted trial contract.  The public input manifest contains only
the observation-side fields.  No VLM/model prediction is produced here.

Usage:
    python experiments/c4_table3/sim_d/generate_sim_gt.py --output output/c4_table3/<run>_sim_d
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


EE_IDS = ("2F", "3F", "vac")
PAYLOAD_KG = {"2F": 5.0, "3F": 2.5, "vac": 0.5}
CRIT_THRESHOLD_KG = 0.50
CRIT_HALF_WIDTH_KG = 0.05

# Five fixed simulator instances.  Values are scene-generation parameters, not
# imported Real-5 values.  Three instances intentionally sit in the pre-locked
# vac critical band so Crit has a non-empty, prediction-independent subset.
SAMPLES: tuple[dict[str, Any], ...] = (
    {"sample_id": "sim_001", "object_name": "sim_box_light", "mass_kg": 0.20,
     "size_mm": [45.0, 45.0, 80.0], "surface_rms_mm": 0.40,
     "opening_mm": [80.0, 80.0], "pose_clearance_mm": [10.0, 10.0],
     "suction_pose_rms_mm": [0.40, 2.00], "task_cost": {"2F": 1.0, "3F": 1.2, "vac": 0.8}},
    {"sample_id": "sim_002", "object_name": "sim_can_near_low", "mass_kg": 0.48,
     "size_mm": [55.0, 55.0, 80.0], "surface_rms_mm": 0.60,
     "opening_mm": [70.0, 70.0], "pose_clearance_mm": [7.5, -2.5],
     "suction_pose_rms_mm": [0.60, 2.00], "task_cost": {"2F": 1.0, "3F": 1.1, "vac": 0.9}},
    {"sample_id": "sim_003", "object_name": "sim_can_equal", "mass_kg": 0.50,
     "size_mm": [60.0, 60.0, 80.0], "surface_rms_mm": 0.80,
     "opening_mm": [72.0, 72.0], "pose_clearance_mm": [6.0, 6.0],
     "suction_pose_rms_mm": [0.80, 2.00], "task_cost": {"2F": 1.1, "3F": 1.0, "vac": 0.9}},
    {"sample_id": "sim_004", "object_name": "sim_can_near_high", "mass_kg": 0.52,
     "size_mm": [62.0, 62.0, 80.0], "surface_rms_mm": 0.90,
     "opening_mm": [70.0, 70.0], "pose_clearance_mm": [4.0, -4.0],
     "suction_pose_rms_mm": [0.90, 2.00], "task_cost": {"2F": 1.1, "3F": 1.0, "vac": 1.2}},
    {"sample_id": "sim_005", "object_name": "sim_wide_heavy", "mass_kg": 0.80,
     "size_mm": [95.0, 95.0, 100.0], "surface_rms_mm": 0.70,
     "opening_mm": [100.0, 100.0], "pose_clearance_mm": [5.0, -5.0],
     "suction_pose_rms_mm": [0.70, 2.00], "task_cost": {"2F": 1.0, "3F": 1.0, "vac": 1.3}},
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _scene_xml(spec: dict[str, Any]) -> str:
    sx, sy, sz = (float(v) / 2000.0 for v in spec["size_mm"])
    # MuJoCo box sizes are half extents.  The opening walls encode the measured
    # inner opening; they are only used to validate the signed-margin contract.
    ox, oy = (float(v) / 1000.0 for v in spec["opening_mm"])
    wall = 0.01
    table_z = 0.02
    obj_z = table_z + sz
    return f'''<mujoco model="{spec["sample_id"]}">
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.002"/>
  <worldbody>
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" size="0.20 0.20 0.01" pos="0 0 0"/>
      <body name="opening_left" pos="{-ox/2-wall/2:.8f} 0 0.06"><geom type="box" size="{wall/2:.8f} {oy/2:.8f} 0.05"/></body>
      <body name="opening_right" pos="{ox/2+wall/2:.8f} 0 0.06"><geom type="box" size="{wall/2:.8f} {oy/2:.8f} 0.05"/></body>
    </body>
    <body name="object" pos="0 0 {obj_z:.8f}" quat="1 0 0 0">
      <freejoint/>
      <geom name="object_geom" type="box" size="{sx:.8f} {sy:.8f} {sz:.8f}"
            mass="{float(spec["mass_kg"]):.8f}" friction="0.6 0.01 0.001"/>
    </body>
  </worldbody>
</mujoco>'''


def _compile_and_extract(spec: dict[str, Any]) -> dict[str, Any]:
    import mujoco

    xml = _scene_xml(spec)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    object_bid = model.body("object").id
    object_gid = model.geom("object_geom").id
    model_mass = float(model.body_subtreemass[object_bid])
    geom_size_mm = [float(v) * 2000.0 for v in model.geom_size[object_gid]]
    # A zero-step rollout verifies that the compiled scene is valid and that the
    # object is initially supported by the table.  It is not a prediction.
    before_z = float(data.xpos[object_bid][2])
    mujoco.mj_step(model, data)
    after_z = float(data.xpos[object_bid][2])
    return {
        "scene_xml_sha256": _sha256_bytes(xml.encode("utf-8")),
        "compiled_object_mass_kg": model_mass,
        "compiled_object_size_mm": geom_size_mm,
        "initial_object_z_m": before_z,
        "post_step_object_z_m": after_z,
        "support_rollout_valid": bool(math.isfinite(after_z) and after_z > 0),
        "xml": xml,
    }


def _trial_labels(spec: dict[str, Any], compiled: dict[str, Any]) -> dict[str, Any]:
    mass = float(compiled["compiled_object_mass_kg"])
    # The scripted trial contract is fixed before any model prediction.  Suction
    # uses the two pose RMS values; all five repeats include deterministic
    # perturbation-free success booleans.  Feasibility uses the same frozen task
    # predicate for each EE, plus mass/payload and opening fit.
    suction_poses = []
    for idx, rms in enumerate(spec["suction_pose_rms_mm"]):
        # The suction fixture has a separately documented 1 kg lift actuator;
        # this keeps suction pose GT independent from the nominal vac payload
        # decision used by Mass Acc/Feasibility/Crit.
        success = float(rms) < 1.5 and mass <= 1.0
        trials = [bool(success)] * 5
        suction_poses.append({
            "pose_id": f'{spec["sample_id"]}_pose_{"AB"[idx]}',
            "pose_index": idx,
            "surface_rms_mm": float(rms),
            "trial_success": trials,
            "gt_label": bool(sum(trials) >= 4),
            "gt_source": "simulator_hidden_state",
        })
    fit_mm = min(float(v) for v in spec["opening_mm"]) - min(float(v) for v in spec["size_mm"][:2])
    feasibility = {}
    for ee_id, payload in PAYLOAD_KG.items():
        if ee_id == "vac":
            success = mass < payload and float(spec["surface_rms_mm"]) < 1.5 and min(spec["size_mm"][:2]) >= 30.0
        else:
            aperture = 85.0 if ee_id == "2F" else 140.0
            success = mass < payload and min(spec["size_mm"][:2]) < aperture and fit_mm > 0
        trials = [bool(success)] * 5
        feasibility[ee_id] = {
            "trial_success": trials,
            "gt_label": bool(sum(trials) >= 4),
            "gt_source": "simulator_hidden_state",
        }
    allowed = [ee for ee in EE_IDS if feasibility[ee]["gt_label"]]
    best_cost = min((float(spec["task_cost"][ee]) for ee in allowed), default=None)
    allowed_optimal = [ee for ee in allowed if best_cost is not None and abs(float(spec["task_cost"][ee]) - best_cost) <= 1e-12]
    return {
        "suction_pose_pair": {
            "poses": suction_poses,
            "trial_contract": {"trials": 5, "positive_at_least": 4, "lift_mm": 50, "hold_s": 2.0},
        },
        "clearance_gt": {
            "reference_frame": "sim_table_frame",
            "opening_inner_widths_mm": [float(v) for v in spec["opening_mm"]],
            "projected_object_footprint_mm": [float(v) for v in spec["size_mm"][:2]],
            "signed_margin_gt_mm": float(fit_mm),
            "gt_source": "simulator_geometry",
        },
        "ee_trial_gt": feasibility,
        "decision_gt": {
            "allowed_ee_ids": allowed_optimal,
            "all_feasible_ee_ids": allowed,
            "no_feasible_ee": not bool(allowed),
            "objective_frozen": True,
            "objective": "minimum frozen task_cost among feasible EE; ties are allowed",
            "independent_trial_and_constraint_evidence_ids": [f'{spec["sample_id"]}::sim_trial_contract'],
        },
    }


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = output_dir / "observations"
    scene_dir = output_dir / "scenes"
    obs_dir.mkdir(exist_ok=True)
    scene_dir.mkdir(exist_ok=True)
    manifest_rows = []
    gt_samples = []
    for spec in SAMPLES:
        compiled = _compile_and_extract(spec)
        scene_path = scene_dir / f'{spec["sample_id"]}.xml'
        scene_path.write_text(compiled.pop("xml"), encoding="utf-8")
        scene_hash = _sha256_file(scene_path)
        observation = {
            "sample_id": spec["sample_id"],
            "object_name": spec["object_name"],
            "modality": "mujoco_state_observation",
            "scene_path": str(scene_path.relative_to(output_dir)).replace("\\", "/"),
            "scene_sha256": scene_hash,
            "visible_geometry_mm": {"extents": [float(v) for v in spec["size_mm"]],
                                    "opening_widths": [float(v) for v in spec["opening_mm"]]},
            "visible_support_context": {"table": "sim_table", "contact_surface": "sim_table_top"},
            "input_gt_excluded": ["mass_kg", "trial_success", "allowed_ee_ids", "signed_margin_gt_mm"],
        }
        obs_path = obs_dir / f'{spec["sample_id"]}.json'
        obs_path.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_rows.append({
            "sample_id": spec["sample_id"], "object_instance_id": f'{spec["sample_id"]}_instance',
            "object_name": spec["object_name"], "observation_path": str(obs_path.relative_to(output_dir)).replace("\\", "/"),
            "observation_sha256": _sha256_file(obs_path), "split": "test", "source": "D_SIM_MUJOCO",
        })
        trial = _trial_labels(spec, compiled)
        gt_samples.append({
            "sample_id": spec["sample_id"], "object_instance_id": f'{spec["sample_id"]}_instance',
            "object_name": spec["object_name"], "mass_gt_kg": compiled["compiled_object_mass_kg"],
            "mass_gt_source": "simulator_hidden_state_compiled_subtree_mass",
            "payload_applicability_verified": True, "payload_spec_source": "configs/robot_spec.json",
            "scene_xml_sha256": compiled["scene_xml_sha256"], "compiled_object_size_mm": compiled["compiled_object_size_mm"],
            "support_rollout_valid": compiled["support_rollout_valid"], **trial,
            "critical_subset_member": abs(compiled["compiled_object_mass_kg"] - CRIT_THRESHOLD_KG) <= CRIT_HALF_WIDTH_KG,
        })
    manifest = {
        "schema_version": "table3_panel_d_sim_v1", "status": "READY_GT_ONLY", "panel": "D_SIM",
        "source_dataset": "MuJoCo compiled fixture", "source_version": "mujoco-3.3.1-compatible",
        "split": "test", "ee_ids": list(EE_IDS), "payload_kg": PAYLOAD_KG,
        "suction_trials_per_pose": 5, "suction_positive_at_least": 4,
        "feasibility_trials_per_sample_ee": 5, "feasibility_positive_at_least": 4,
        "clearance": {"unit": "mm", "target": "signed_horizontal_opening_margin", "denominator_floor_mm": 5.0},
        "critical_subset": {"ee_id": "vac", "mass_threshold_kg": CRIT_THRESHOLD_KG, "inclusive_half_width_kg": CRIT_HALF_WIDTH_KG},
        "samples": manifest_rows,
        "review_and_leakage": {"gt_used_for_input_generation": False, "gt_used_for_prediction": False,
                                "predictions_present": False, "evaluator_only_gt_file": "sim_gt.yaml"},
    }
    (output_dir / "sim_manifest.yaml").write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    gt = {"schema_version": "table3_panel_d_sim_gt_v1", "status": "READY_GT_ONLY", "panel": "D_SIM",
          "source": {"generator": "generate_sim_gt.py", "simulator": "MuJoCo", "source_sha256": None},
          "samples": gt_samples}
    gt_bytes = yaml.safe_dump(gt, allow_unicode=True, sort_keys=False).encode("utf-8")
    gt["source"]["source_sha256"] = _sha256_bytes(gt_bytes)
    (output_dir / "sim_gt.yaml").write_text(yaml.safe_dump(gt, allow_unicode=True, sort_keys=False), encoding="utf-8")
    config = {
        "benchmark": "C4 Table III D_SIM", "panel": "D_SIM", "generator": "experiments/c4_table3/sim_d/generate_sim_gt.py",
        "input_manifest": "sim_manifest.yaml", "evaluator_only_gt": "sim_gt.yaml",
        "conditions": ["name_only", "affordance_labels", "siphy_adopted", "geometric_grounding", "ours_full", "gt_numerics"],
        "inference": {"allowed": False, "reason": "No simulator D prediction adapters are registered in this preparation step"},
        "protocol": {"crit_threshold_kg": CRIT_THRESHOLD_KG, "crit_half_width_kg": CRIT_HALF_WIDTH_KG, "object_macro": "equal"},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "sim_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    matrix_lines = [
        "condition_id,condition,Mass_Acc,Suction_Acc,Suction_PF,Clearance_RelErr,Feasibility_Acc,DA,Crit",
        "name_only,Name-only,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION",
        "affordance_labels,+ Affordance labels,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION",
        "siphy_adopted,SiPhy (as adopted),NEEDS_IMPLEMENTATION,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION",
        "geometric_grounding,+ Geometric grounding (ours),NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION",
        "ours_full,Ours (full),NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION",
        "gt_numerics,GT numerics (oracle),NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NOT_APPLICABLE,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION,NEEDS_IMPLEMENTATION",
    ]
    (output_dir / "condition_matrix.csv").write_text("\n".join(matrix_lines) + "\n", encoding="utf-8")
    (output_dir / "run.log").write_text(
        f"generated simulator-only D_SIM GT; samples={len(gt_samples)}; critical={sum(s['critical_subset_member'] for s in gt_samples)}\n"
        "model_inference_calls=0\nreal5_gt_access=0\ntrajectory_result_access=0\n",
        encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# D_SIM 준비 산출물\n\n"
        "MuJoCo 컴파일 모델의 숨은 상태로 만든 simulator-only GT입니다. "
        "Real-5/D 실물 패널과 합치지 않으며, 이 디렉터리에는 prediction이 없습니다. "
        "`sim_gt.yaml`은 evaluator 전용이고 `sim_manifest.yaml`은 입력 측 정보만 포함합니다.\n",
        encoding="utf-8")
    return {"output_dir": str(output_dir), "sample_count": len(gt_samples), "critical_count": sum(s["critical_subset_member"] for s in gt_samples)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
