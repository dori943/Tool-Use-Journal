"""Lexicographic operational cost and search tie-breaking.

Hard-constraint violations are never expressed as large costs (infeasible
edges simply do not exist). Plans that survive are compared first by the
lexicographic order of the operational cost vector:

    ee_switches > tool_switches > motion_cost > execution_cost

If those four costs are identical, search prefers the plan with the higher
accumulated candidate suitability. The suitability term is deliberately kept
out of :class:`CostVector`, which remains the externally reported operational
cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


SUITABILITY_SCALE = 1_000_000


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


def suitability_penalty(score: float | None) -> int:
    """Convert a higher-is-better score into a stable integer search cost."""

    if score is None or not isfinite(score):
        return SUITABILITY_SCALE
    normalized = min(1.0, max(0.0, float(score)))
    return SUITABILITY_SCALE - round(normalized * SUITABILITY_SCALE)


@dataclass(frozen=True, slots=True)
class SearchPriority:
    """Internal Dijkstra priority; not part of the serialized cost vector."""

    operational_cost: CostVector
    suitability_penalty: int = 0

    def __post_init__(self) -> None:
        if self.suitability_penalty < 0:
            raise ValueError(
                "SearchPriority.suitability_penalty must be non-negative"
            )

    @classmethod
    def zero(cls) -> "SearchPriority":
        return cls(CostVector.zero())

    def advance(
        self,
        edge_cost: CostVector,
        edge_suitability_penalty: int = 0,
    ) -> "SearchPriority":
        return SearchPriority(
            operational_cost=self.operational_cost + edge_cost,
            suitability_penalty=(
                self.suitability_penalty + edge_suitability_penalty
            ),
        )

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return self.operational_cost.as_tuple() + (self.suitability_penalty,)
