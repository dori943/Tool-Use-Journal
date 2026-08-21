"""Lexicographic cost vector.

Hard-constraint violations are never expressed as large costs (infeasible
edges simply do not exist), and suitability scores are never mixed into the
cost (below-threshold candidates are removed before search). Plans that
survive are compared purely by lexicographic order of this vector:

    ee_switches > tool_switches > motion_cost > execution_cost
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class CostVector:
    ee_switches: int = 0
    tool_switches: int = 0
    motion_cost: int = 0
    execution_cost: int = 0

    def __post_init__(self) -> None:
        for name in (
            "ee_switches",
            "tool_switches",
            "motion_cost",
            "execution_cost",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"CostVector.{name} must be non-negative")

    def __add__(self, other: "CostVector") -> "CostVector":
        return CostVector(
            self.ee_switches + other.ee_switches,
            self.tool_switches + other.tool_switches,
            self.motion_cost + other.motion_cost,
            self.execution_cost + other.execution_cost,
        )

    @classmethod
    def zero(cls) -> "CostVector":
        return cls()

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.ee_switches,
            self.tool_switches,
            self.motion_cost,
            self.execution_cost,
        )

    def to_dict(self, *, skip_zero: bool = False) -> dict[str, int]:
        d = {
            "ee_switches": self.ee_switches,
            "tool_switches": self.tool_switches,
            "motion_cost": self.motion_cost,
            "execution_cost": self.execution_cost,
        }
        if skip_zero:
            return {k: v for k, v in d.items() if v != 0}
        return d
