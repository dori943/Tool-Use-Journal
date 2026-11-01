# -*- coding: utf-8 -*-
"""M5 실패를 계획·실행·실행 후 목표 검사로 분류한다.

ablation 표에서 "추론이 없어서 못 푼 것"과 "파지가 풀려서 떨어뜨린 것"을
구분하기 위한 공통 규칙. 어떤 조건(Full / w/o ...)에서든 같은 함수를 쓴다.

  failure_stage    goal      : 실제 동작 후 배치/기존 배치 유지 검사에서 미달 또는 증거 부족
                   planning  : 키프레임/IK/충돌 검사 단계에서 궤적을 만들지 못함
                   execution : 시뮬레이터에서 실제로 움직이다 실패 (접촉 상실, 미끄러짐,
                               목표 미달, 상태 발산 등)
  failure_code     메시지에서 뽑은 대표 원인 코드 (배치 검사 코드, 물리 코드 순서로 우선)
  physics_related  접촉/파지 유지 관련 코드가 포함돼 있으면 True.
                   planning 단계라도 controller preview 가 GRASP_RETENTION_FAILED 로
                   거절한 경우가 여기 해당한다 (물리 노이즈로 집계할 수 있게).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

# 접촉/파지 유지 계열 — 실행 노이즈로 집계할 코드
PHYSICS_CODES: tuple[str, ...] = (
    "GRASP_RETENTION_FAILED",
    "CONTACT_FRICTION_TRANSLATION_SLIP",
    "CONTACT_FRICTION_ORIENTATION_SLIP",
    "CONTACT_LOST_DURING_LIFT",
    "SCRIPTED_GRASP_SLIPPED",
    "SCRIPTED_GRASP_CONTACT_LOST",
    "VACUUM_ATTACHMENT_LOST",
    "VACUUM_ATTACHMENT_POSE_ERROR",
    "VACUUM_CONTACT_GATE_NOT_PASSED",
    "BILATERAL_CONTACT_GATE_NOT_PASSED",
    "THIN_HANDLE_CONTACT_GATE_NOT_PASSED",
    "HANDLE_CONTACT_NOT_STABLE",
    "THREE_FINGER_CONTACT_NOT_STABLE",
    "PRE_LIFT_CONTACT_NOT_STABLE",
    "POST_GRASP_ATTACHMENT_OBJECT_MISMATCH",
    "TRANSPORT_TOOL_HOME_RETENTION_MISMATCH",
    "PHYSICS_NUMERICAL_INSTABILITY",
    "UNEXPECTED_COLLISION",
    "FLATTEN_PHYSICAL_GOAL_NOT_SATISFIED",
    "EXECUTION_FAILED",
    "GOAL_FAILED",
    "STATE_DIVERGED",
)

# A post-execution predicate failure does not establish physical noise.
POST_EXECUTION_GOAL_CODES = (
    "CONTAINER_NOT_SETTLED",
    "PLACEMENT_GOAL_NOT_SATISFIED",
    "PLACEMENT_PRESERVATION_FAILED",
)

# 실행 단계에서 올라오는 예외 타입
EXECUTION_ERROR_TYPES: tuple[str, ...] = (
    "LivePlanExecutionError",
    "ScriptedGraspError",
    "ScriptedGraspExecutionError",
)

# 계획 단계 예외 타입
PLANNING_ERROR_TYPES: tuple[str, ...] = (
    "MotionPlanningPipelineError",
    "MotionPlanBuildError",
    "GenericMotionRunnerError",
)

_CODE_RE = re.compile(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+){1,7})\b")
_CODE_IGNORE = frozenset(
    {
        "SCRIPTED_GRASP_LIVE", "M5_MOTION_PLAN", "SCRIPTED_GRASP", "NOT_EVALUATED",
        "IN_PROGRESS", "LIVE_RUNTIME", "LIVE_EXECUTION_RUNTIME", "UTF_8",
    }
)


def extract_codes(text: str | None) -> list[str]:
    """메시지에 등장한 UPPER_SNAKE 코드를 등장 순서대로(중복 제거) 돌려준다."""
    if not text:
        return []
    seen: list[str] = []
    for code in _CODE_RE.findall(str(text)):
        if code in _CODE_IGNORE or code in seen:
            continue
        seen.append(code)
    return seen


def _last_failed_step(steps: Iterable[Mapping[str, Any]] | None) -> Mapping[str, Any] | None:
    if not steps:
        return None
    failed = [s for s in steps if isinstance(s, Mapping) and str(s.get("status", "")).upper() == "FAILED"]
    return failed[-1] if failed else None


def classify_failure(
    detail: str | None,
    *,
    error: BaseException | None = None,
    steps: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """실패 메시지(+ live manifest 의 steps)로 단계/코드를 결정한다.

    우선순위
      0. 실행 후 배치 검사 코드 → goal (물리 원인 여부는 별도 증거로 판단)
      1. 마지막 FAILED step 이 어댑터 실행까지 갔으면(`execution_status`) → execution
      2. 마지막 FAILED step 의 route 가 SCRIPTED_GRASP 이면 (live 시뮬에서 직접 파지) → execution
      3. 예외 타입이 실행 계열 → execution, 계획 계열 → planning
      4. 그 외: 물리 코드가 있으면 execution, 없으면 planning
    """
    text = str(detail or "")
    error_type = type(error).__name__ if error is not None else None
    if error is not None and str(error) and str(error) not in text:
        text = f"{text} {error}"
    step = _last_failed_step(steps)
    step_text = ""
    if step is not None:
        step_text = " ".join(
            str(step.get(k) or "") for k in ("error", "execution_status", "detail", "failure")
        )
    combined = f"{text} {step_text}"
    codes = extract_codes(combined)
    physics = [c for c in codes if c in PHYSICS_CODES]

    stage: str
    basis: str
    placement_code = next((code for code in codes if code in POST_EXECUTION_GOAL_CODES), None)
    if placement_code is not None:
        stage, basis = "goal", "post_execution_placement_check"
    elif step is not None and step.get("execution_status") not in (None, "SUCCESS"):
        stage, basis = "execution", "step.execution_status"
    elif step is not None and step.get("route") == "SCRIPTED_GRASP":
        stage, basis = "execution", "step.route=SCRIPTED_GRASP"
    elif error_type in EXECUTION_ERROR_TYPES or any(t in text for t in EXECUTION_ERROR_TYPES):
        stage, basis = "execution", "error_type"
    elif error_type in PLANNING_ERROR_TYPES or any(t in text for t in PLANNING_ERROR_TYPES):
        stage, basis = "planning", "error_type"
    elif physics:
        stage, basis = "execution", "physics_code"
    else:
        stage, basis = "planning", "default"

    code = placement_code or (physics[0] if physics else (next((c for c in codes if c not in EXECUTION_ERROR_TYPES
                                              and c not in PLANNING_ERROR_TYPES), None)))
    return {
        "failure_stage": stage,
        "failure_code": code,
        "physics_related": bool(physics),
        "failure_codes": codes[:12],
        "failure_basis": basis,
        "failed_step": (
            {
                "index": step.get("index"),
                "subgoal_id": step.get("subgoal_id"),
                "action_type": step.get("action_type"),
                "route": step.get("route"),
                "object_id": step.get("object_id"),
                "execution_status": step.get("execution_status"),
            }
            if step is not None
            else None
        ),
    }
