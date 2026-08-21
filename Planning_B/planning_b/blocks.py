"""Derive EE blocks and tool blocks from the final step sequence.

Blocks are a derived, explanatory view of the selected plan — never an
optimization variable. Contiguous same-EE step ranges form EE blocks; inside
each, contiguous same-held-tool ranges form tool blocks.
"""

from __future__ import annotations

from typing import Any

from planning_b.transitions import P


def build_blocks(
    steps: list[dict[str, Any]], initial_ee: str, initial_tool: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (ee_blocks, flat tool_blocks) from serialized plan steps."""
    ee_blocks: list[dict[str, Any]] = []
    current_ee = initial_ee
    current_tool = initial_tool

    def new_tool_block(start: int) -> dict[str, Any]:
        return {
            "tool": current_tool,
            "step_start": start,
            "step_end": start,
            "subgoals": [],
        }

    def new_ee_block(start: int) -> dict[str, Any]:
        return {
            "ee": current_ee,
            "step_start": start,
            "step_end": start,
            "subgoals": [],
            "tool_blocks": [new_tool_block(start)],
        }

    if steps:
        ee_blocks.append(new_ee_block(0))

    for step in steps:
        idx = step["step_index"]
        action = step["action"]
        params = step.get("parameters") or {}

        if action in (P.ATTACH_EE, P.TERMINAL_RESTORE_EE):
            current_ee = params.get("to") or params.get("ee") or current_ee
            ee_blocks.append(new_ee_block(idx))
        elif action == P.PICK_TOOL:
            current_tool = params.get("tool")
            ee_blocks[-1]["tool_blocks"].append(new_tool_block(idx))
        elif action in (P.RETURN_TOOL, P.TERMINAL_RETURN_TOOL):
            current_tool = None
            ee_blocks[-1]["tool_blocks"].append(new_tool_block(idx))

        block = ee_blocks[-1]
        block["step_end"] = idx
        tool_block = block["tool_blocks"][-1]
        tool_block["step_end"] = idx
        if step["kind"] == "subgoal" and step.get("subgoal_id"):
            block["subgoals"].append(step["subgoal_id"])
            tool_block["subgoals"].append(step["subgoal_id"])

    # Drop empty tool blocks created by boundary primitives.
    for block in ee_blocks:
        block["tool_blocks"] = [
            tb for tb in block["tool_blocks"] if tb["subgoals"]
        ] or block["tool_blocks"][:1]

    flat_tool_blocks = [
        {**tb, "ee": block["ee"]}
        for block in ee_blocks
        for tb in block["tool_blocks"]
    ]
    return ee_blocks, flat_tool_blocks
