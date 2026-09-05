"""Experiment 1 v4 condition definitions (true SiPhy Stage-1 + Gemini Stage-2)."""
from __future__ import annotations

from dataclasses import dataclass


PROPERTIES = ("material", "density_kgm3", "mass_kg", "mu", "youngs_gpa")
ALL_PREDICT = PROPERTIES


@dataclass(frozen=True)
class Condition:
    id: str
    name: str
    uses_bbox: bool
    # Which SiPhy cues are passed to Gemini Stage 2 (from this condition's own SiPhy call).
    siphy_cues: tuple[str, ...]  # () | ("material",) | ("density_kgm3",) | ("material","density_kgm3")
    # Properties Gemini must newly infer (others come fixed from SiPhy when applicable).
    gemini_infer: tuple[str, ...]
    # Properties fixed from Stage-1 SiPhy into the final prediction.
    fixed_from_siphy: tuple[str, ...]
    input_factors_label: str
    # Deprecated label: no cross-condition cache. Always "none" or "per_condition_fresh".
    siphy_cache_key: str
    predict: tuple[str, ...] = ALL_PREDICT

    @property
    def uses_siphy(self) -> bool:
        return len(self.siphy_cues) > 0


CONDITIONS: dict[str, Condition] = {
    c.id: c
    for c in (
        Condition(
            id="C1",
            name="C1_BBOX",
            uses_bbox=True,
            siphy_cues=(),
            gemini_infer=ALL_PREDICT,
            fixed_from_siphy=(),
            input_factors_label="image+m1_bbox",
            siphy_cache_key="none",
        ),
        Condition(
            id="C2",
            name="C2_MATERIAL",
            uses_bbox=False,
            siphy_cues=("material",),
            gemini_infer=("density_kgm3", "mass_kg", "mu", "youngs_gpa"),
            fixed_from_siphy=("material",),
            input_factors_label="image+siphy_material",
            siphy_cache_key="per_condition_fresh",
        ),
        Condition(
            id="C3",
            name="C3_DENSITY",
            uses_bbox=False,
            siphy_cues=("density_kgm3",),
            gemini_infer=("material", "mass_kg", "mu", "youngs_gpa"),
            fixed_from_siphy=("density_kgm3",),
            input_factors_label="image+siphy_density",
            siphy_cache_key="per_condition_fresh",
        ),
        Condition(
            id="C4",
            name="C4_BBOX_MATERIAL",
            uses_bbox=True,
            siphy_cues=("material",),
            gemini_infer=("density_kgm3", "mass_kg", "mu", "youngs_gpa"),
            fixed_from_siphy=("material",),
            input_factors_label="image+m1_bbox+siphy_material",
            siphy_cache_key="per_condition_fresh",
        ),
        Condition(
            id="C5",
            name="C5_BBOX_DENSITY",
            uses_bbox=True,
            siphy_cues=("density_kgm3",),
            gemini_infer=("material", "mass_kg", "mu", "youngs_gpa"),
            fixed_from_siphy=("density_kgm3",),
            input_factors_label="image+m1_bbox+siphy_density",
            siphy_cache_key="per_condition_fresh",
        ),
        Condition(
            id="C6",
            name="C6_MATERIAL_DENSITY",
            uses_bbox=False,
            siphy_cues=("material", "density_kgm3"),
            gemini_infer=("mass_kg", "mu", "youngs_gpa"),
            fixed_from_siphy=("material", "density_kgm3"),
            input_factors_label="image+siphy_material_density",
            siphy_cache_key="per_condition_fresh",
        ),
        Condition(
            id="C7",
            name="C7_BBOX_MATERIAL_DENSITY",
            uses_bbox=True,
            siphy_cues=("material", "density_kgm3"),
            gemini_infer=("mass_kg", "mu", "youngs_gpa"),
            fixed_from_siphy=("material", "density_kgm3"),
            input_factors_label="image+m1_bbox+siphy_material_density",
            siphy_cache_key="per_condition_fresh",
        ),
    )
}
