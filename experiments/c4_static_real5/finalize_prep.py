"""Add visual QC, adapter contracts, and evaluator-only Table 5 GT to a new prep bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from tuj.m1_scene.siphy_backend import SYS_MSG  # noqa: E402
from infer import FRICTION_PROMPT  # noqa: E402
FIELDS = ["input_id", "object_id", "sequence_id", "frame_id", "rgb_path", "depth_path",
          "mass_crop_path", "friction_context_path", "mass_qc_status", "friction_qc_status",
          "warning_reason", "selection_rule_version", "input_sha256"]

# Recorded from the RGB/mass/context contact sheet before any new model prediction is made.
# These are warnings, not exclusions or per-object threshold changes.
VISUAL_WARNINGS = {
    "000003_000040": "minor table/robot fragment in mass mask",
    "000005_000040": "minor background fragment in mass mask",
    "000008_000010": "minor background fragment in mass mask",
    "000009_000040": "minor background fragment in mass mask",
    "000010_000010": "robot fragment in mass mask",
    "000010_000040": "minor background fragment in mass mask",
    "000012_000040": "minor table fragment in mass mask",
    "000015_000010": "minor table fragment in mass mask",
    "000015_000040": "visible table streak in mass mask; object remains identifiable",
    "000018_000010": "minor background fragment in mass mask",
    "000021_000040": "minor background fragment in mass mask",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", type=Path, required=True)
    args = ap.parse_args()
    run = args.prep_dir.resolve()
    manifest = run / "selected_static_inputs.csv"
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 50 or any(r["mass_qc_status"] not in ("APPROVED", "WARNING") for r in rows):
        raise ValueError("base manifest not ready for visual QC")
    for row in rows:
        warning = VISUAL_WARNINGS.get(row["input_id"])
        if warning:
            row["warning_reason"] = "; ".join(filter(None, (row["warning_reason"], warning)))
            row["mass_qc_status"] = "WARNING"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    counts = Counter(r["object_id"] for r in rows)
    warnings = [r for r in rows if r["warning_reason"]]
    report = run / "static_input_qc_report.md"
    with report.open("a", encoding="utf-8") as f:
        f.write("\n## Contact sheet 수동 검토 (GT·기존 prediction 미참조)\n\n")
        f.write("50개 원본 RGB, black-background mass crop, 객체+테이블 context를 모두 확인했다. ")
        f.write("객체 식별 불가능·심한 절단/가림·배경만 있는 crop·테이블이 보이지 않는 context는 선정본에 없었다. ")
        f.write("mask의 작은 table/robot fragment는 원본을 고치지 않고 WARNING으로만 표시했다. ")
        f.write("자동 정책에 의해 `000017` 두 번째 슬롯의 frame 40은 객체 mask가 너무 작아 frame 35로 대체되었다. ")
        f.write("`000015` 두 프레임은 객체가 보이고 표면 context가 유효하므로 유지했으며, 질량 crop의 배경 잔여만 경고한다. ")
        f.write("`000009` 두 프레임도 유지했다; 과거 궤적 품질은 이 정지 평가의 제외 근거가 아니다.\n\n")
        f.write(f"최종: 50/50 입력, 객체별 {dict(counts)}, WARNING {len(warnings)}, 프레임 교체 1, 최종 제외 0.\n\n")
        for row in warnings:
            f.write(f"- `{row['input_id']}`: {row['warning_reason']}\n")
        f.write("\nRGB/depth 동일 index·해상도 및 per-frame camera K는 확인했다. 픽셀 단위의 물리적 정합 오차 ")
        f.write("(예: reprojection residual)는 별도 실측 calibration 없이 증명하지 못한다. ")
        f.write("원본 mask는 생성하지 않았으며 M1 depth/RGB 분할 결과 overlay를 파생물로 보존했다. ")
        f.write("mask 수동 수정 없음.\n")
        f.write("\n실측 GT는 동일 EV-RealPhys 실물 5개에만 대응하며 일반 YCB 이미지로 옮기지 않는다. ")
        f.write("`source_trial_count=10`은 원 논문의 마찰 측정 활주 횟수로, 객체별 정지 이미지 10장이나 ")
        f.write("추후 모델 5회 반복과 다른 수다. Mustard Bottle, Pitcher, Bleach Cleanser의 숨겨진 ")
        f.write("모래 충전 상태는 단일 이미지에서 식별하기 어렵고 모델 입력으로 제공하지 않는다. ")
        f.write("정지 시각 마찰도 감속도 기반 물리 측정치가 아니라 시각 prior로만 해석한다.\n")
    source = json.loads((ROOT / "configs/c4_real5_gt.json").read_text(encoding="utf-8"))
    gt = {"benchmark": "C4 Static Real-5 evaluator-only",
          "source": {**source["source"], "original_gt_sha256": sha(ROOT / "configs/c4_real5_gt.json"),
                     "static_friction_interpretation": "visual prior compared with measured combined object-table kinetic coefficient",
                     "source_trial_count_meaning": "tilted-table GT sliding trials; not 10 static images or five model repeats"},
          "objects": [{"object_id": x["object_id"], "object_name": x["object_name"],
                       "mass_gt_kg": x["mass_gt_kg"], "combined_friction_gt": x["friction_gt"],
                       "source_trial_count": x["source_trial_count"]} for x in source["objects"]]}
    with (run / "evaluator_only_gt.yaml").open("x", encoding="utf-8") as f:
        yaml.safe_dump(gt, f, sort_keys=False, allow_unicode=True)
    config = {"benchmark": "C4 Static Real-5", "panel": "R", "dataset_split": "test_sliding",
              "prep_dir": str(run), "input_manifest_sha256": sha(manifest),
              "selection_policy_sha256": sha(run / "selection_policy.json"),
              "source_manifest_sha256": sha(ROOT / "data/external/kandukuri_ev_realphys/manifest.json"),
              "model": "gemini-3.6-flash", "temperature": 0.0, "requested_seed_base": 100,
              "mass_prompt_sha256": hashlib.sha256(SYS_MSG.encode()).hexdigest(),
              "friction_prompt_sha256": hashlib.sha256(FRICTION_PROMPT.encode()).hexdigest(),
              "provider_seed_supported": False, "repeat_count_full_evaluation": 5,
              "repeat_zero_smoke_excluded": True,
              "mass_object_aggregate": "median_of_10_images", "friction_object_aggregate": "median_of_10_images",
              "macro_average": "equal_weight_five_objects", "repeated_score_std_ddof": 1,
              "friction_kind": "static_visual_prior_combined_object_table",
              "legacy_rgbd_trajectory_friction": "auxiliary_only"}
    with (run / "inference_config.yaml").open("x", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    matrix = [
        ("name_only", "Name-only", "NEEDS_IMPLEMENTATION", "STRUCTURALLY_UNSUPPORTED", "train-only name prior absent", "-"),
        ("affordance_labels", "+ Affordance labels", "NEEDS_IMPLEMENTATION", "STRUCTURALLY_UNSUPPORTED", "same missing name prior; train-only labels absent", "-"),
        ("siphy_adopted", "SiPhy (as adopted)", "READY", "STRUCTURALLY_UNSUPPORTED", "siphy_static_rgbd_mass", "-"),
        ("geometric_grounding", "+ Geometric grounding (ours)", "SHARED_BACKEND", "STRUCTURALLY_UNSUPPORTED", "siphy_static_rgbd_mass", "-"),
        ("ours_full", "Ours (full: + friction, + spatial)", "SHARED_BACKEND", "READY", "siphy_static_rgbd_mass", "visual_context_combined_friction"),
        ("gt_numerics", "GT numerics (upper bound)", "STRUCTURALLY_UNSUPPORTED", "STRUCTURALLY_UNSUPPORTED", "-", "-"),
    ]
    with (run / "condition_matrix.csv").open("x", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition_id", "condition", "Mass_MnRE_adapter", "StaticVisualFriction_MAE_adapter",
                    "mass_prediction_source", "friction_prediction_source"])
        w.writerows(matrix)
    audit = """# C4 Static Real-5 조건 adapter 감사

정지 평가용 구현을 기존 Table 3 궤적 평가와 구분한다. `infer.py`는 evaluator/GT loader를 import하지 않고 숫자 GT 파일명을 런타임 파일 접근 경계에서 거부한다. `evaluate_static.py`만 prediction 완료 후 `evaluator_only_gt.yaml`을 읽는다. GT·과거 prediction·`scene_properties` 물리 값은 prompt, crop, 선택, 파싱, prior에 사용하지 않았다.

| 조건 | 실제 입력 | 추정 모듈·prompt | 질량 | 정지 시각 마찰 | downstream |
|---|---|---|---|---|---|
| Name-only | 이름만 허용 | train-only 이름 prior 미구현 | NEEDS_IMPLEMENTATION | STRUCTURALLY_UNSUPPORTED | D 패널 미연결 |
| + Affordance labels | 이름+훈련 출처 범주 레이블 허용 | 레이블 파일/출처 미확보, prior 미구현 | NEEDS_IMPLEMENTATION | STRUCTURALLY_UNSUPPORTED | D 패널 미연결 |
| SiPhy (as adopted) | black-background mass RGB crop + 동일 프레임 M1 RGB-D 점군 | 기존 `SiPhyBackend.estimate`, `SYS_MSG` | READY (`siphy_static_rgbd_mass`) | STRUCTURALLY_UNSUPPORTED | D 패널 미연결 |
| + Geometric grounding | SiPhy와 같은 정지 프레임·crop·점군 | 질량은 **동일 SiPhy prediction 공유**, 추가 M1/M3 기하는 이 두 지표의 입력 아님 | SHARED_BACKEND | STRUCTURALLY_UNSUPPORTED | M1/M3 기하 코드 존재, Table 3 공통 decision adapter 미연결 |
| Ours (full) | 위와 같은 질량 입력; 동일 프레임의 객체+접촉 테이블 context RGB | 질량 SiPhy prediction 공유; 새 `FRICTION_PROMPT`+VLM JSON | SHARED_BACKEND | READY (`visual_context_combined_friction`) | M3/M4 spatial/EE 공통 adapter 미연결 |
| GT numerics | evaluator-only 실측 숫자 | estimator 아님; downstream oracle | STRUCTURALLY_UNSUPPORTED | STRUCTURALLY_UNSUPPORTED | D 패널 GT/adapter 없음 |

질량의 SiPhy/Geometric/Ours 세 칸은 하나의 backend와 완전히 같은 입력을 쓰므로 독립 실험 또는 세 번의 API 호출로 세지 않는다. `shared_prediction_source=siphy_static_rgbd_mass` 하나를 참조한다. Ours의 static visual friction은 비디오 감속도로 측정한 값이 아니며 불확실한 시각 prior를 Table 5 실측 combined object-table μ와 비교한다. 기존 `FrictionHead.probe_mu_from_track` 결과는 auxiliary만 유지하며 새 prompt·결과에 혼합하지 않는다. 이 μ는 gripper–object contact μ가 아니므로 `evaluate_ee.grip_slip`에 넣지 않는다.

`SYS_MSG`와 `FRICTION_PROMPT`의 SHA-256은 각 raw prediction에 기록한다. 모델/temperature/requested seed/실제 provider seed 지원 여부도 각 기록에 넣는다. Gemini seed는 API에 전달하지 않는다. temperature 0을 결정성 보장으로 주장하지 않는다. 현재 설정은 `gemini-3.6-flash`, temperature 0, 전체 반복 5회이며 이번 prep에서는 smoke만 실행한다.

Affordance labels는 저장소의 train-only 자료로 확인되지 않았다. 이 실물 객체의 실측 GT나 test 영상, 사후 성공 annotation으로 만들면 평가 GT 유출이므로 adapter를 구현/READY 처리하지 않았다.
"""
    (run / "adapter_audit.md").write_text(audit, encoding="utf-8")
    with (run / "run.log").open("a", encoding="utf-8") as f:
        f.write(f"visual_qc_warning_count={len(warnings)}\nfinal_exclusions=0\n"
                f"locked_manifest_sha256={sha(manifest)}\n"
                f"evaluator_gt_sha256={sha(run / 'evaluator_only_gt.yaml')}\n"
                f"mass_prediction_source=siphy_static_rgbd_mass\n"
                f"friction_prediction_source=visual_context_combined_friction\n")
    print(f"{run}: locked 50 inputs; {len(warnings)} warnings; 1 replacement")


if __name__ == "__main__":
    main()
