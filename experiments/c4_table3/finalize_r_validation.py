"""Seal a new R-panel QC run; never infer or score a blocked cell."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/c4_table3")]
from scoring import snapshot_gt  # noqa: E402
from run import DISPLAY, METRICS  # noqa: E402
from tuj.m1_scene.siphy_backend import SYS_MSG  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: object) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.run_dir.resolve()
    if not out.is_relative_to((ROOT / "output/c4_table3").resolve()):
        raise ValueError("run directory outside C4 Table 3 output")
    required_absent = ("config_snapshot.json", "gt_snapshot.json", "condition_matrix.csv",
                       "metric_specific_manifest.csv", "mass_qc_report.md",
                       "friction_semantics_report.md", "raw_predictions.jsonl",
                       "per_object_summary.csv", "per_repeat_summary.csv",
                       "overall_summary.json", "failures.csv", "table3.csv", "table3.md", "run.log")
    if any((out / name).exists() for name in required_absent):
        raise ValueError("refuse to overwrite an existing finalized run")
    manifest_file = ROOT / "data/external/kandukuri_ev_realphys/manifest.json"
    source = json.loads(manifest_file.read_text(encoding="utf-8"))
    archive = ROOT / "data/external/kandukuri_ev_realphys/ev-realphys.tar.gz"
    if sha256(archive) != source["archive_sha256"]:
        raise ValueError("official archive SHA-256 mismatch")
    if (source["split"] != "test_sliding" or len(source["sequences"]) != 25 or
            Counter(r["object_id"] for r in source["sequences"]) !=
            Counter({"003_cracker_box": 5, "006_mustard_bottle": 5,
                     "019_pitcher_base": 5, "021_bleach_cleanser": 5, "025_mug": 5}) or
            any(not r["mapping_verified"] or "tuna" in r["object_id"].lower()
                for r in source["sequences"])):
        raise ValueError("official Real-5 mapping/split invalid")
    mass_audit_file = out / "mass_candidate_audit.json"
    friction_audit_file = out / "friction_window_audit_v3.json"
    mass_audit = json.loads(mass_audit_file.read_text(encoding="utf-8"))
    friction_audit = json.loads(friction_audit_file.read_text(encoding="utf-8"))
    if not (mass_audit["numeric_gt_used"] is False and
            mass_audit["prior_prediction_error_used"] is False and
            friction_audit["numeric_gt_used"] is False and
            friction_audit["prior_prediction_error_used"] is False):
        raise ValueError("GT or prior prediction used in input selection")
    mass_by_sid = {r["sequence_id"]: r for r in mass_audit["rows"]}
    friction_by_sid = {r["sequence_id"]: r for r in friction_audit["rows"]}
    with (out / "mass_visual_review.csv").open(encoding="utf-8-sig", newline="") as f:
        reviews = list(csv.DictReader(f))
    review_by_sid = {r["sequence_id"]: r for r in reviews}
    source_ids = {r["sequence_id"] for r in source["sequences"]}
    if (len(reviews) != 25 or len(review_by_sid) != 25 or
            set(mass_by_sid) != source_ids or set(friction_by_sid) != source_ids or
            set(review_by_sid) != source_ids):
        raise ValueError("25-sequence audit/review mismatch")
    records = []
    for row in source["sequences"]:
        sid = row["sequence_id"]
        mass = mass_by_sid[sid]
        review = review_by_sid[sid]
        friction = friction_by_sid[sid]
        status = review["mass_qc_status"]
        if status not in {"APPROVED", "RESELECTED", "EXCLUDED"}:
            raise ValueError(f"invalid mass review status {sid}")
        selected_frame, selected_artifact, selected_crop_sha = "", "", ""
        if status != "EXCLUDED":
            selected_frame = int(review["selected_mass_frame"])
            choices = mass["candidates"]
            more = out / f"{sid[-6:]}_candidate_metrics.json"
            if more.exists():
                choices += json.loads(more.read_text())
            choice = next((c for c in choices if c["frame"] == selected_frame and
                           c.get("passes_automatic")), None)
            if choice is None:
                raise ValueError(f"selected mass frame fails locked automatic QC: {sid}")
            selected_artifact = choice["artifacts"]["crop"]
            selected_crop_sha = choice["metrics"]["crop_sha256"]
            if sha256(ROOT / selected_artifact) != selected_crop_sha:
                raise ValueError(f"mass crop hash changed: {sid}")
        elif review["selected_mass_frame"]:
            raise ValueError(f"excluded mass frame must be empty: {sid}")
        if friction["semantics_status"] != "TARGET_SEMANTICS_UNVERIFIED":
            raise ValueError(f"unsupported friction semantics transition: {sid}")
        window = friction["selected_friction_window"]
        records.append({
            "panel": "R", "sequence_id": sid, "object_id": row["object_id"],
            "bop_object_id": row["bop_object_id"], "source_split": "test_sliding",
            "source_archive_sha256": source["archive_sha256"],
            "mass_qc_status": status, "mass_qc_reason": review["mass_qc_reason"],
            "selected_mass_frame": selected_frame,
            "mass_candidate_frame": mass["selected"]["frame"] if mass["selected"] else "",
            "mass_crop": selected_artifact, "mass_crop_sha256": selected_crop_sha,
            "friction_qc_status": friction["friction_qc_status"],
            "friction_qc_reason": friction["friction_qc_reason"],
            "selected_friction_window": json.dumps(window) if window else "",
            "semantics_status": friction["semantics_status"],
            "friction_window_policy_sha256": friction_audit["policy_sha256"],
        })
    manifest_fields = list(records[0])
    write_csv(out / "metric_specific_manifest.csv", manifest_fields, records)
    matrix_src = ROOT / "experiments/table3_protocol/condition_matrix.csv"
    with (out / "condition_matrix.csv").open("xb") as f:
        f.write(matrix_src.read_bytes())
    with (out / "condition_matrix.csv").open(encoding="utf-8-sig", newline="") as f:
        matrix = list(csv.DictReader(f))
    gt_src = ROOT / "configs/c4_real5_gt.json"
    snapshot_gt(gt_src, "4a0f66e54e8676abc16ca3634fed7564cc2ad2cf9da737a858540d35950433b3",
                out / "gt_snapshot.json")
    policy_v2 = ROOT / "experiments/c4_table3/r_qc_policy_v2.yaml"
    policy_v3 = ROOT / "experiments/c4_table3/r_friction_window_policy_v3.yaml"
    config = {
        "schema_version": "c4_r_validation_20260914", "run_id": out.name,
        "scope": "R panel only; D remains blocked", "dataset_manifest_sha256": sha256(manifest_file),
        "source_archive_sha256": sha256(archive), "gt_snapshot_sha256": sha256(out / "gt_snapshot.json"),
        "gt_read_scope": "evaluator snapshot only; input audit and inference schedule do not read numerics",
        "metric_specific_manifest_sha256": sha256(out / "metric_specific_manifest.csv"),
        "condition_matrix_sha256": sha256(out / "condition_matrix.csv"),
        "mass_audit_sha256": sha256(mass_audit_file),
        "mass_visual_review_sha256": sha256(out / "mass_visual_review.csv"),
        "friction_audit_v3_sha256": sha256(friction_audit_file),
        "mass_qc_policy_sha256": sha256(policy_v2),
        "friction_window_policy_sha256": sha256(policy_v3),
        "rgbd_adapter_sha256": sha256(ROOT / "experiments/c4_real5/real_rgbd.py"),
        "friction_tracker_sha256": sha256(ROOT / "experiments/c4_real5/friction_from_rgbd.py"),
        "friction_probe_sha256": sha256(ROOT / "src/tuj/m1_scene/grounding.py"),
        "siphy_backend_sha256": sha256(ROOT / "src/tuj/m1_scene/siphy_backend.py"),
        "model_requested": "gemini-3.6-flash", "model_version_observed": None,
        "temperature_requested": 0.0, "requested_seeds": [100, 101, 102, 103, 104],
        "provider_seed_supported": False, "provider_seed_sent": False,
        "actual_repeats": 0, "actual_model_api_calls": 0,
        "temperature_zero_is_deterministic": False,
        "prompt_sha256": hashlib.sha256(SYS_MSG.encode("utf-8")).hexdigest(),
        "shared_prediction_source_planned": {
            "siphy_adopted": "SiPhyBackend.estimate",
            "geometric_grounding": "same SiPhyBackend mass prediction",
            "ours_full": "same SiPhyBackend mass prediction"},
        "requested_repeat_std_ddof": 1,
    }
    write_json(out / "config_snapshot.json", config)
    mass_counts = Counter(r["mass_qc_status"] for r in records)
    friction_counts = Counter(r["friction_qc_status"] for r in records)
    by_object = defaultdict(list)
    for row in records:
        by_object[row["object_id"]].append(row)
    object_rows = []
    for oid, group in sorted(by_object.items()):
        object_rows.append({"object_id": oid, "planned_sequences": len(group),
                            "mass_approved_sequences": sum(r["mass_qc_status"] in
                                                           {"APPROVED", "RESELECTED"} for r in group),
                            "friction_approved_sequences": sum(r["friction_qc_status"] in
                                                               {"APPROVED", "RESELECTED"} for r in group),
                            "mass_prediction_count": 0, "friction_prediction_count": 0,
                            "mass_median_kg": "", "mass_relative_error": "",
                            "friction_median": "", "friction_absolute_error": ""})
    write_csv(out / "per_object_summary.csv", list(object_rows[0]), object_rows)
    write_csv(out / "per_repeat_summary.csv",
              ["repeat_id", "requested_seed", "provider_seed_sent", "model_version",
               "mass_MnRE", "friction_MAE", "coverage_mass", "coverage_friction"], [])
    with (out / "raw_predictions.jsonl").open("x", encoding="utf-8"):
        pass
    failure_rows = []
    for r in records:
        if r["mass_qc_status"] == "EXCLUDED":
            failure_rows.append({"panel": "R", "metric": "Mass_MnRE", "condition": "all",
                                 "sequence_id": r["sequence_id"], "stage": "INPUT_QC",
                                 "reason": r["mass_qc_reason"]})
        failure_rows.append({"panel": "R", "metric": "Friction_MAE", "condition": "ours_full",
                             "sequence_id": r["sequence_id"], "stage": "TARGET_SEMANTICS",
                             "reason": r["friction_qc_reason"]})
    write_csv(out / "failures.csv", ["panel", "metric", "condition", "sequence_id", "stage", "reason"],
              failure_rows)
    cells = []
    for condition in matrix:
        for metric in METRICS:
            structural = condition[metric] == "NOT_APPLICABLE"
            panel = "R" if metric in {"Mass_MnRE", "Friction_MAE"} else "D"
            if structural:
                status, reason = "-", "structurally unsupported or oracle numeric error excluded"
            elif panel == "D":
                status, reason = "BLOCKED", "independent manipulation/geometry GT and condition adapter unavailable"
            elif metric == "Mass_MnRE":
                status, reason = "BLOCKED", "mass QC 20/25; no complete 25-sequence input or verified condition adapter"
                if condition["condition_id"] in {"name_only", "affordance_labels"}:
                    reason += "; train-only prior/affordance source missing"
            else:
                status, reason = "BLOCKED", "TARGET_SEMANTICS_UNVERIFIED: table gravity level/contact/rotation not proved"
            cells.append({"condition_id": condition["condition_id"],
                          "condition": condition["condition"], "metric": metric,
                          "display": DISPLAY[metric], "panel": panel,
                          "planned_n": 25 if panel == "R" else 0,
                          "approved_input_n": (20 if metric == "Mass_MnRE" else 0) if panel == "R" else 0,
                          "matrix_status": condition[metric], "status": status,
                          "value": "", "repeat_std_ddof1": "", "reason": reason})
    write_csv(out / "table3.csv", list(cells[0]), cells)
    c_by_key = {(c["condition_id"], c["metric"]): c for c in cells}
    lines = ["# ICRA Table 3 — R 패널 재검증", "",
             "| Condition | " + " | ".join(DISPLAY[m] for m in METRICS) + " |",
             "| --- | " + " | ".join("---" for _ in METRICS) + " |"]
    for condition in matrix:
        lines.append("| " + condition["condition"] + " | " + " | ".join(
            c_by_key[(condition["condition_id"], m)]["status"] for m in METRICS) + " |")
    lines += ["", "R: 공식 EV-RealPhys test_sliding, 5객체×5시퀀스. 질량 입력 승인 20/25 "
              "(APPROVED 6, RESELECTED 14), 마찰 창 후보 25/25이나 물리 의미 승인 0/25. "
              "D: 별도 조작·기하 GT 미확보. 점수와 run 간 std는 산출하지 않음. ",
              "", "캡션: Mass MnRE는 객체별 유효 시퀀스 예측 중앙값의 상대오차를 "
              "5객체 동일 가중 평균한 무차원 값(표시 시 %)이고 Friction MAE-Real5는 같은 "
              "순서의 결합 object–table 마찰계수 절대오차다. 추후 stochastic 조건은 "
              "전체 평가 5회 점수의 mean ± 표본 std(ddof=1)를 표시한다. "
              "객체 간·시퀀스 간 변동은 별도 통계이며 반복은 독립 객체 수를 늘리지 않는다. "
              "'-'는 구조적 미지원, BLOCKED는 입력·GT·의미 게이트 미통과다. "
              "본 실행의 API 호출은 0회다.", ""]
    with (out / "table3.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    summary = {"run_id": out.name, "status": "BLOCKED_R_INPUT_AND_FRICTION_SEMANTICS",
               "ready_cells": [], "ready_cell_count": 0, "actual_model_api_calls": 0,
               "raw_prediction_count": 0, "mass_qc_counts": dict(mass_counts),
               "friction_qc_counts": dict(friction_counts),
               "mass_approved_sequences": 20, "mass_planned_sequences": 25,
               "mass_input_assessment": "PARTIAL", "official_mass_MnRE": None,
               "friction_candidate_windows": sum(bool(r["selected_friction_window"]) for r in records),
               "friction_approved_sequences": 0, "official_friction_MAE_Real5": None,
               "friction_semantics_status": "TARGET_SEMANTICS_UNVERIFIED",
               "target_semantics_mismatch": False,
               "repeats_executed": 0, "repeat_std_ddof1": None,
               "repeat_std_note": "N/A: no repeated model evaluations; object/sequence variation is not run stability",
               "cells": cells, "per_object_input_coverage": object_rows,
               "historical_predictions_or_results_reused": False}
    write_json(out / "overall_summary.json", summary)
    mass_report = ["# R 질량 입력 QC", "",
        "공식 archive와 5객체×5시퀀스 매핑을 SHA-256으로 확인했다. 질량 입력은 단일 RGB-D 프레임이며 "
        "마찰 창 상태가 질량 선택에 영향을 주지 않는다. 수치 GT·이전 prediction error는 QC 코드가 읽지 않았다.",
        "", f"고정 자동 기준({policy_v2.name})은 25개 시퀀스에서 자동 통과 후보를 찾았으나, "
        "RGB crop과 mask overlay 시각 검토에서 그 기준이 테이블/로봇 조각을 놓침을 확인했다. "
        "25개 최초 후보를 모두 검사하고 시각 불량 19개는 고정 10후보를 추가 검사했다. 동일한 '객체 전체/심한 blur·가림 없음/"
        "과도한 배경·테이블 없음/유효 깊이 비율≥0.80' 기준을 적용했다. "
        "결과는 6 APPROVED + 14 RESELECTED + 5 EXCLUDED = 25다.", "",
        "| sequence | object | mass QC | selected frame | reason |", "| --- | --- | --- | ---: | --- |"]
    for r in records:
        mass_report.append(f"| {r['sequence_id'][-6:]} | {r['object_id']} | "
                           f"{r['mass_qc_status']} | {r['selected_mass_frame'] or '—'} | "
                           f"{r['mass_qc_reason']} |")
    mass_report += ["", "000015: 15번의 테이블 띠를 비롯해 고정 10후보 모두 입력 mask가 부적합하다. "
                    "000017: 일부 프레임에서 객체가 아닌 배경만 분할되며 다른 후보도 테이블을 포함한다. "
                    "000009: 마찰 실패와 무관하게 질량 15번 crop은 승인했다.", "",
                    "원본 RGB/depth와 mask는 수정하지 않았다. 각 후보의 seed/RGB-refined/point-filtered mask, "
                    "crop, overlay와 hash는 이 run의 `original_masks/`, `crops/`, `overlays/` 및 감사 JSON에 있다. "
                    "RGB/depth 파일 번호·해상도와 보정행렬은 확인됐으나 픽셀 수준 등록 오차는 독립 검증하지 않았다. "
                    "따라서 MASS_INPUT_READY=false, 25/25 공식 점수는 PARTIAL 입력 상태이며 수치는 미산출이다.", ""]
    with (out / "mass_qc_report.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(mass_report))
    friction_report = ["# R 마찰 정의와 창 검증", "",
        "`points_from_frame`은 보정 K와 depth(m)를 이용해 3D를 만들고 1000을 곱해 mm로 반환한다. "
        "`track_scene`은 RGB-D 분할 점군의 world XY 중앙값(mm)을 만든다. "
        "`FrictionHead.probe_mu_from_track`은 mm를 1000으로 나눠 m로 변환하고 "
        "`t=np.arange(n)*dt_s`에 `s=v0*t-0.5*a*t²+c`를 적합해 양의 감속도 `a/9.81`을 반환한다. "
        "원 논문 기록상 공개 sequence는 30 Hz로 downsample되어 명목 `dt=1/30 s`를 쓴다. "
        "합성 `μ=0.2`, `v0=1m/s` 궤적에서 결과는 0.2였고 dt를 잘못 절반으로 주면 0.8이 됐다. "
        "양의 가속도 궤적은 None이다. 4개 focused unit test가 통과했다.", "",
        "이는 **수평면·외력 없음·동일 물체–테이블 접촉의 순수 병진 자유 활주**이면 "
        "Table 5의 combined object–table kinetic μ와 정의가 일치한다. 따라서 이전의 "
        "회전>10°만으로 전부 TARGET_SEMANTICS_MISMATCH라 한 판정은 지나쳤다. "
        "현재 판정은 25/25 TARGET_SEMANTICS_UNVERIFIED다. "
        "camera/world normal의 |z|≈1은 카메라 좌표계 안의 정렬이지 독립 중력계 inclinometer 검증이 아니다. "
        "RGB-D 30 Hz의 실제 timestamp jitter, push stick 접촉 종료, 회전 에너지, 충돌·접촉 손실, "
        "tracking centroid bias도 창별로 증명되지 않았다. `stage0`의 gripper-contact proxy는 사용하지 않았다.", "",
        "기존 v2 선택기는 000009에서 뒤쪽 분할 점프 때문에 실패했다. GT와 예측 오차를 보지 않고 "
        "v3 정책을 기록한 뒤 모든 25개 트랙의 점프 사이 연속 구간을 재탐색했다. "
        "000009의 [0,13]이 후보로 복구됐다(13 intervals, 487.29 mm, quadratic RMS 1.795 mm). "
        "25/25에 후보 창이 있지만 시작·끝·중간 RGB를 재검토했을 때 많은 창이 막대/손 접촉 "
        "또는 정지 후 프레임을 배제했다고 증명할 수 없었다. 특히 v3 최장 구간 규칙은 "
        "000003·000012·000013·000018에서 53–59 interval을 선택해 후반부 혼입 위험이 명백하다. "
        "그러므로 자유 활주 승인 0/25, MAE는 계산하지 않는다.", "",
        "다음 단계: 중력 기준 테이블 기울기(기존 0.5° 한계) 원자료 확인; "
        "RGB/모션으로 손·막대 이탈 프레임과 회전/충돌을 독립 라벨링; "
        "30 Hz 실제 timestamp 확인; 최대 창 길이·정지/반전·quadratic residual의 GT-blind "
        "공통 필터를 고정하고 25개 재감사한다. 입력으로 Table 5 숫자나 MoCap GT pose를 쓰지 않는다.", ""]
    with (out / "friction_semantics_report.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(friction_report))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (out / "run.log").open("x", encoding="utf-8") as f:
        f.write(f"{now} cmd={' '.join(sys.argv)}\n")
        f.write(f"{now} archive_sha256={sha256(archive)} manifest_sha256={sha256(manifest_file)}\n")
        f.write(f"{now} mass_review=6 APPROVED,14 RESELECTED,5 EXCLUDED; no mask pixel edits\n")
        f.write(f"{now} v2 friction=24 candidates,000009 failed; v3 friction=25 candidates,0 approved\n")
        f.write(f"{now} semantics=TARGET_SEMANTICS_UNVERIFIED ready_cells=0 api_calls=0\n")
    print(json.dumps({"run_dir": str(out), "mass_approved": 20,
                      "friction_approved": 0, "ready_cells": 0,
                      "api_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
