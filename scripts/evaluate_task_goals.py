"""Evaluate saved ablation runs without any model calls or simulator replay.

python scripts/evaluate_task_goals.py --baseline output/ablation/without-reasoner \
    --current output/<new-run>/without-reasoner --output output/<new-evaluation>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tuj.evaluation.saved_runs import evaluate_saved_run
from tuj.evaluation.task_goals import POLICY_VERSION, TASKS


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_comparison(baseline, current, output, seed=0):
    baseline, current, output = [Path(p).resolve() for p in (baseline, current, output)]
    if output == baseline or output == current or baseline.is_relative_to(output) or current.is_relative_to(output):
        raise ValueError("output must be a new evaluation directory, not a source directory")
    output.mkdir(parents=True, exist_ok=False)
    records, verified = {}, {}
    for label, root in (("baseline", baseline), ("current", current)):
        records[label] = []
        for task in TASKS:
            run_dir = root / task / f"seed_{seed}"
            evaluation = evaluate_saved_run(run_dir, task=task)
            rr, batch, summary = [read(run_dir / p) for p in ("run_result.json", "batch_record.json", "m5/m5_summary.json")]
            interrupted = evaluation["task_goal_status"] == "INCOMPLETE"
            execution = not interrupted and (evaluation.get("execution_status") or rr.get("m5_status") or summary.get("status")) == "SUCCESS"
            goal = evaluation["goal_success"]
            record = {"schema": "run-result-goal-reevaluation-v1", "task": task, "seed": seed,
                "exec_success": execution, "goal_success": goal, "success": goal if execution else False,
                "task_goal_status": evaluation["task_goal_status"], "policy_version": POLICY_VERSION,
                "source_directory": str(run_dir), "original_failure_code": rr.get("failure_code") or batch.get("reason_code"),
                "original_success": rr.get("success"), "failed_checks": evaluation["failed_checks"],
                "unknown_checks": evaluation["unknown_checks"], "evaluation": f"{label}/{task}/task_goal_evaluation.json"}
            write(output / label / task / "task_goal_evaluation.json", evaluation)
            write(output / label / task / "run_result.evaluated.json", record)
            records[label].append(record)
            for relative, digest in evaluation["source_sha256"].items():
                path = run_dir / relative
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"source changed during evaluation: {path}")
                verified[str(path)] = digest
    totals = {label: {"task_count": len(rows), "execution_completed": sum(r["exec_success"] for r in rows),
        "goal_evaluated": sum(r["goal_success"] is not None for r in rows),
        "goal_satisfied": sum(r["goal_success"] is True for r in rows),
        "end_to_end_success": sum(r["success"] is True for r in rows)} for label, rows in records.items()}
    write(output / "comparison.json", {"policy_version": POLICY_VERSION, "records": records, "totals": totals,
        "new_experiments": 0, "new_model_calls": 0, "post_hoc_policy": True,
        "evaluation_code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(), ROOT / "src/tuj/evaluation/task_goals.py", ROOT / "src/tuj/evaluation/saved_runs.py"]}})
    write(output / "preservation_check.json", {"all_read_sources_unchanged": True, "source_count": len(verified), "sha256": verified})
    labels = {"SATISFIED": "성공", "FAILED": "실패", "NOT_EXECUTED": "실행 전 실패", "INCOMPLETE": "중단·최종 판정 불가", "UNKNOWN": "근거 부족"}
    lines = ["# w/o Reasoner 목표 달성 재평가", "", f"판정 기준: `{POLICY_VERSION}`. 저장된 실제 관측값만 사용했고 실험 재실행·API 호출은 0회입니다.", "",
        "기존 원본 결과를 덮어쓰지 않았습니다. 논문용 사전 등록 기준이 아니라 작업 지시와 기존 동작 평가를 바탕으로 명시한 사후 기준입니다.", "",
        "| 태스크 | 이전 실행 완료 | 이전 목표 | 수정 후 실행 완료 | 수정 후 목표 |", "|---|---|---|---|---|"]
    for old, new in zip(records['baseline'], records['current']):
        lines.append(f"| {new['task']} | {'예' if old['exec_success'] else '아니오'} | {labels[old['task_goal_status']]} | {'예' if new['exec_success'] else '아니오'} | {labels[new['task_goal_status']]} |")
    lines += ["", "실행 전 실패는 파이프라인 실패이며 물리 목표를 관측해 실패로 판정한 경우와 구분합니다. 중단된 실행은 마지막 저장 상태를 최종 성공으로 사용하지 않습니다.", "",
        f"확인된 전체 성공: 이전 {totals['baseline']['end_to_end_success']}/8, 수정 후 {totals['current']['end_to_end_success']}/8. 이전 결과에 최종 상태가 없는 중단 실험이 있어 실제 물리 목표 성공률의 완전한 비교는 아닙니다.", "",
        "서로 다른 모델 응답·운영 환경을 포함하는 각 1회 결과이므로, 이 차이를 코드 수정의 인과 효과나 논문 최종 성공률로 해석할 수 없습니다.", "",
        "물체는 회전을 반영한 전체 형상이 목표 영역 안에 있어야 합니다(허용 오차 1.5mm). 지지면 간격/침투 허용치는 기존 resting 검사와 같은 5mm입니다. 중간 place 동작의 30mm 허용치보다 엄격한 과제 전체 판정입니다.", "",
        "C3_2의 ‘가지런히’는 포크와 숟가락의 긴 축 차이 15도 이내, 두 식기 형상 비중첩으로 정의했습니다. 사용자가 2026-10-25에 이 기준을 확정했으며, 두 트레이 간 동일 종류 물체 교환은 허용합니다.", "",
        "## 판정 근거", ""]
    for label in ('baseline', 'current'):
        for row in records[label]:
            if row['goal_success'] is not None:
                lines.append(f"- {label} {row['task']}: {labels[row['task_goal_status']]}; 실패 조건: {', '.join(row['failed_checks']) or '없음'}. [상세]({row['evaluation']})")
    lines += ["", f"평가에 읽은 {len(verified)}개 파일의 SHA-256이 평가 전후 동일함을 확인했습니다. `comparison.json`, 태스크별 `run_result.evaluated.json`은 새 폴더의 파생 결과입니다.", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--current', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate_comparison(args.baseline, args.current, args.output, args.seed), indent=2))


if __name__ == '__main__':
    main()
