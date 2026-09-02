"""Defensive conversion of generic pipeline state into an M6 context."""

from collections.abc import Mapping
from copy import deepcopy

from .schemas import empty_failure_context


def _merge_known(target: dict, source: Mapping) -> None:
    """Copy known keys recursively while ignoring unknown or malformed values."""
    for key, value in source.items():
        if key not in target:
            continue
        if isinstance(target[key], dict):
            if isinstance(value, Mapping):
                _merge_known(target[key], value)
        else:
            target[key] = deepcopy(value)


class FailureContextBuilder:
    """Build a temporary, JSON-compatible diagnosis working context."""

    _SECTIONS = (
        "task", "subgoal", "verification", "scene", "grounding", "task_plan",
        "motion_plan", "execution", "observation", "history",
    )

    def build(self, pipeline_state) -> dict:
        context = empty_failure_context()
        if not isinstance(pipeline_state, Mapping):
            return context

        if "failure_id" in pipeline_state:
            context["failure_id"] = deepcopy(pipeline_state["failure_id"])
        for section in self._SECTIONS:
            value = pipeline_state.get(section)
            if isinstance(value, Mapping):
                _merge_known(context[section], value)
        return context
