"""Height field of the arm volume swept by the fixed EE rack trajectories.

EE attach, return and exchange replay stored joint trajectories; nothing
replans them around the scene.  Whatever a place puts under that sweep is
found only when the next exchange validates its path, after every earlier
subgoal has executed (c3_2: mug_a seated on tray_a 3.0 mm under
``robot0_wrist3_col`` at the shared bare home; 3F->vac failed with
PRECOMPUTED_PATH_COLLISION after 30 successful steps).

The field stores, per XY cell, the lowest point the arm surface reaches there.
An object conflicts when its top comes within the margin of that floor under
its footprint, so a flat plate may go where a mug may not.  M1 computes it in
its own frame for M2's slot layout; M5 computes it in world frame as a guard
on the seat it finally picks.  Only numpy is needed here so planning stages
can read the field without MuJoCo; see ``tuj.m5_motion.ee_path_keepout``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


# Added to the planner collision margin: covers sample and cell spacing, the
# yaw a place may still apply after seat selection, and settle drift.
PLACEMENT_BUFFER_M = 0.01
# M5 stored-path validation clearance (MotionConstraints.collision_margin_m).
DEFAULT_COLLISION_MARGIN_M = 0.005


@dataclass(frozen=True)
class SweptArmHeightField:
    """Lowest swept arm surface z per XY cell (``inf`` where the arm never is)."""

    origin_xy: np.ndarray
    cell_m: float
    floor_z: np.ndarray
    trajectory_ids: tuple[str, ...] = ()

    def lowest_z_under(
        self, center_xy: Sequence[float], half_xy: Sequence[float]
    ) -> float:
        """Lowest arm surface above the axis-aligned footprint (inf if none)."""

        center = np.asarray(center_xy, dtype=float)[:2]
        half = np.asarray(half_xy, dtype=float)[:2]
        lo = np.floor((center - half - self.origin_xy) / self.cell_m).astype(int)
        hi = np.floor((center + half - self.origin_xy) / self.cell_m).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(self.floor_z.shape) - 1)
        if np.any(hi < lo):
            return math.inf
        return float(self.floor_z[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1].min())

    def clearance(
        self,
        center_xy: Sequence[float],
        half_xy: Sequence[float],
        top_z: float,
        margin_m: float,
    ) -> float:
        """Gap from the object top to the arm, after ``margin_m`` on every side.

        Negative means the sweep would pass inside the margin of the object.
        """

        half = np.asarray(half_xy, dtype=float)[:2] + margin_m
        return self.lowest_z_under(center_xy, half) - (float(top_z) + margin_m)

    def shifted(self, offset_xy: Sequence[float]) -> "SweptArmHeightField":
        """The same field in a frame whose XY origin sits at ``offset_xy``."""

        return replace(
            self,
            origin_xy=np.asarray(self.origin_xy, dtype=float)
            - np.asarray(offset_xy, dtype=float)[:2],
        )

    def cropped(self) -> "SweptArmHeightField":
        """Drop rows/columns the arm never reaches (smaller artifact)."""

        finite = np.isfinite(self.floor_z)
        if not finite.any():
            return self
        rows, cols = np.where(finite.any(axis=1))[0], np.where(finite.any(axis=0))[0]
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        return replace(
            self,
            origin_xy=np.asarray(self.origin_xy, dtype=float)
            + np.array([r0, c0]) * self.cell_m,
            floor_z=self.floor_z[r0:r1, c0:c1].copy(),
        )


def placement_margin_m(collision_margin_m: float | None = None) -> float:
    base = DEFAULT_COLLISION_MARGIN_M if collision_margin_m is None else collision_margin_m
    return float(base) + PLACEMENT_BUFFER_M


def save_height_field(path: str | Path, field: SweptArmHeightField) -> None:
    np.savez_compressed(
        Path(path),
        origin_xy=np.asarray(field.origin_xy, dtype=float),
        cell_m=np.asarray(field.cell_m, dtype=float),
        floor_z=np.asarray(field.floor_z, dtype=np.float32),
        trajectory_ids=np.asarray(field.trajectory_ids, dtype=str),
    )


def load_height_field(path: str | Path) -> SweptArmHeightField | None:
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path) as data:
        return SweptArmHeightField(
            origin_xy=np.asarray(data["origin_xy"], dtype=float),
            cell_m=float(data["cell_m"]),
            floor_z=np.asarray(data["floor_z"], dtype=float),
            trajectory_ids=tuple(str(t) for t in data["trajectory_ids"]),
        )
