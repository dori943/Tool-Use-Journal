"""OpenAI LLM-backed recovery router for M6."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .prompts import build_recovery_router_instructions, build_recovery_router_text
from .recovery_config import get_recovery_model
from .recovery_router import (
    RecoveryAPIError,
    RecoveryResponseError,
    RecoveryValidationError,
    build_past_recoveries,
    validate_recovery_output,
)

logger = logging.getLogger(__name__)


class MissingOpenAIAPIKeyError(RecoveryAPIError):
    """OPENAI_API_KEY is not available to the process."""


class _OpenAIResponsesClient(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...


class _OpenAIClient(Protocol):
    @property
    def responses(self) -> _OpenAIResponsesClient: ...


class GeneratedRecoveryTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoal_id: str | None = None
    object_id: str | None = None
    property: str | None = None
    relation: str | None = None
    ee_id: str | None = None
    tool_id: str | None = None


class GeneratedRecoveryAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(min_length=1)
    target_module: str = Field(min_length=1)
    target: GeneratedRecoveryTarget


class GeneratedRecoveryRouting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    restart_from: str = Field(min_length=1)
    rerun_modules: list[str]
    invalidate: list[str] = Field(default_factory=list)


class GeneratedRecoveryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recovery_category: str = Field(min_length=1)
    action: GeneratedRecoveryAction
    routing: GeneratedRecoveryRouting


def build_openai_recovery_input(
    failure_context: dict,
    diagnosis: dict,
    decision_mode: str,
    recovery_evidence: list[dict],
) -> tuple[str, list[dict[str, Any]]]:
    return (
        build_recovery_router_instructions(),
        [
            {
                "type": "input_text",
                "text": build_recovery_router_text(
                    failure_context,
                    diagnosis,
                    decision_mode,
                    recovery_evidence,
                ),
            }
        ],
    )


def _parsed_recovery_to_output(parsed: GeneratedRecoveryDecision) -> dict:
    return {
        "recovery_category": parsed.recovery_category,
        "action": {
            "action_type": parsed.action.action_type,
            "target_module": parsed.action.target_module,
            "target": parsed.action.target.model_dump(),
            "parameters": {},
        },
        "routing": {
            "restart_from": parsed.routing.restart_from,
            "rerun_modules": list(parsed.routing.rerun_modules),
            "invalidate": list(parsed.routing.invalidate or []),
        },
    }


def _validate_router_output(
    recovery_output: dict,
    *,
    failure_context: dict,
    diagnosis: dict,
    decision_mode: str,
    recovery_evidence: list[dict],
) -> None:
    probe_recovery = {
        "decision_mode": decision_mode,
        "guidance": {
            "experience_ids": [
                item.get("experience_id")
                for item in recovery_evidence
                if item.get("experience_id") is not None
            ],
            "past_recoveries": build_past_recoveries(recovery_evidence),
            "recovery_evidence": list(recovery_evidence),
            "selection": {
                "selected_experience_ids": [
                    item.get("experience_id")
                    for item in recovery_evidence
                    if item.get("experience_id") is not None
                ],
                "selection_count": len(recovery_evidence),
                "selection_audit": [],
            },
        },
        "recovery_category": recovery_output["recovery_category"],
        "action": recovery_output["action"],
        "routing": recovery_output["routing"],
        "outcome": {"status": None, "verification_result": None},
        "metadata": {"attempt": 1, "created_at": None},
    }
    validate_recovery_output(probe_recovery)


class OpenAIRecoveryRouter:
    """OpenAI Responses API recovery router with structured output and taxonomy validation."""

    def __init__(self, model: str | None = None, client: _OpenAIClient | None = None):
        self.model = get_recovery_model(model)
        self._client = client

    def _openai_client(self) -> _OpenAIClient:
        if self._client is not None:
            return self._client
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingOpenAIAPIKeyError(
                "OPENAI_API_KEY is required for OpenAI recovery routing"
            )
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - packaging guard
            raise RecoveryAPIError(
                "install openai>=2,<3 to enable OpenAI recovery routing"
            ) from error
        self._client = OpenAI()
        return self._client

    def route(
        self,
        failure_context: dict,
        diagnosis: dict,
        decision_mode: str,
        recovery_evidence: list[dict],
    ) -> dict:
        instructions, content = build_openai_recovery_input(
            failure_context,
            diagnosis,
            decision_mode,
            recovery_evidence,
        )

        logger.debug(
            "openai recovery request backend=openai model=%s decision_mode=%s recovery_evidence_count=%s",
            self.model,
            decision_mode,
            len(recovery_evidence),
        )

        try:
            response = self._openai_client().responses.parse(
                model=self.model,
                instructions=instructions,
                input=[{"role": "user", "content": content}],
                text_format=GeneratedRecoveryDecision,
                store=False,
            )
        except MissingOpenAIAPIKeyError:
            raise
        except Exception as error:  # noqa: BLE001 - SDK error surface varies
            raise RecoveryAPIError(
                f"OpenAI Responses request failed ({type(error).__name__})"
            ) from error

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            response_id = str(getattr(response, "id", "unknown"))
            status = str(getattr(response, "status", "unknown"))
            raise RecoveryResponseError(
                f"OpenAI response {response_id!r} had no parsed output (status={status})"
            )
        if not isinstance(parsed, GeneratedRecoveryDecision):
            try:
                parsed = GeneratedRecoveryDecision.model_validate(parsed)
            except ValidationError as error:
                raise RecoveryResponseError(
                    "OpenAI response did not match the recovery decision schema"
                ) from error

        recovery_output = _parsed_recovery_to_output(parsed)
        recovery_output["past_recoveries"] = (
            build_past_recoveries(recovery_evidence)
            if decision_mode == "EXPERIENCE_GUIDED"
            else []
        )

        try:
            _validate_router_output(
                recovery_output,
                failure_context=failure_context,
                diagnosis=diagnosis,
                decision_mode=decision_mode,
                recovery_evidence=recovery_evidence,
            )
        except RecoveryValidationError as error:
            raise RecoveryResponseError(str(error)) from error

        logger.debug(
            "openai recovery result category=%s action_type=%s target_module=%s restart_from=%s",
            recovery_output["recovery_category"],
            recovery_output["action"]["action_type"],
            recovery_output["action"]["target_module"],
            recovery_output["routing"]["restart_from"],
        )
        return recovery_output
