"""Defensive conversion of generic pipeline state into an M6 context."""

from collections.abc import Mapping
from copy import deepcopy

from .schemas import empty_failure_context, empty_m5_result


def _merge_known(target: dict, source: Mapping) -> None:
    """Copy known keys recursively while ignoring unknown or malformed values.

    Empty dict templates (e.g. ``grounding.geometry``) accept a source mapping
    wholesale so dynamic keys such as object ids are preserved. Non-empty dict
    templates still merge only already-known nested keys.
    """
    for key, value in source.items():
        if key not in target:
            continue
        if isinstance(target[key], dict):
            if isinstance(value, Mapping):
                if not target[key]:
                    target[key] = deepcopy(value)
                else:
                    _merge_known(target[key], value)
        else:
            target[key] = deepcopy(value)


class FailureContextBuilder:
    """Build a temporary, JSON-compatible diagnosis working context."""

    _SECTIONS = (
        "task",
        "subgoal",
        "scene",
        "grounding",
        "task_plan",
        "motion_plan",
        "execution",
        "m5_result",
        "observation",
        "history",
    )

    def build(self, pipeline_state) -> dict:
        context = empty_failure_context()
        if not isinstance(pipeline_state, Mapping):
            return context

        if "failure_id" in pipeline_state:
            context["failure_id"] = deepcopy(pipeline_state["failure_id"])
        for section in self._SECTIONS:
            if section not in pipeline_state:
                continue
            value = pipeline_state.get(section)
            if section == "m5_result":
                if value is None:
                    context["m5_result"] = None
                elif isinstance(value, Mapping):
                    m5_result = empty_m5_result()
                    _merge_known(m5_result, value)
                    context["m5_result"] = m5_result
                continue
            if isinstance(value, Mapping):
                _merge_known(context[section], value)
        return context
