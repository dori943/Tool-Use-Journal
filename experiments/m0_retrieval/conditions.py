"""Experiment 1 v3 condition definitions (single-call SiPhy-style sequential reasoning).

Material / Density in condition names mean *logical Stage-1 inference inside
one Gemini call*, NOT ground-truth inputs and NOT a second API / SiPhyBackend call.
"""
from __future__ import annotations

from dataclasses import dataclass


PROPERTIES = ("material", "density_kgm3", "mass_kg", "mu", "youngs_gpa")
ALL_PREDICT = PROPERTIES

# Logging tag shared by all C1–C7 (single LLM call; production SYS_MSG reused as base).
REASONING_FAMILY = "siphy_prompt_extended_single_call"


@dataclass(frozen=True)
class Condition:
    id: str
    name: str
    uses_bbox: bool
    reasoning_mode: str
    input_factors_label: str
    # Dry-run / logging descriptors (logical stages; not separate API calls)
    logical_first_inference: str  # "none" | "material" | "density" | "material + density"
    downstream_uses: str
    # none = no bbox; full = joint with image (C1); stage2 = deferred to Stage 2 (C4/C5/C7)
    bbox_stage: str
    predict: tuple[str, ...] = ALL_PREDICT
    production_siphy_prompt_reused: bool = True
    reasoning_family: str = REASONING_FAMILY

    @property
    def inputs(self) -> tuple[str, ...]:
        if self.uses_bbox:
            return ("image", "bbox")
        return ("image",)


CONDITIONS: dict[str, Condition] = {
    c.id: c
    for c in (
        Condition(
            id="C1",
            name="C1_BBOX",
            uses_bbox=True,
            reasoning_mode="bbox_guided",
            input_factors_label="image+m1_bbox",
            logical_first_inference="none",
            downstream_uses="image + M1 bbox (joint inference of all 5 properties)",
            bbox_stage="full",
        ),
        Condition(
            id="C2",
            name="C2_MATERIAL",
            uses_bbox=False,
            reasoning_mode="material_first",
            input_factors_label="image+material_first_reasoning",
            logical_first_inference="material",
            downstream_uses="image + inferred material",
            bbox_stage="none",
        ),
        Condition(
            id="C3",
            name="C3_DENSITY",
            uses_bbox=False,
            reasoning_mode="density_first",
            input_factors_label="image+density_first_reasoning",
            logical_first_inference="density",
            downstream_uses="image + inferred density",
            bbox_stage="none",
        ),
        Condition(
            id="C4",
            name="C4_BBOX_MATERIAL",
            uses_bbox=True,
            reasoning_mode="bbox_material_first",
            input_factors_label="image+m1_bbox+material_first_reasoning",
            logical_first_inference="material",
            downstream_uses="image + M1 bbox + inferred material",
            bbox_stage="stage2",
        ),
        Condition(
            id="C5",
            name="C5_BBOX_DENSITY",
            uses_bbox=True,
            reasoning_mode="bbox_density_first",
            input_factors_label="image+m1_bbox+density_first_reasoning",
            logical_first_inference="density",
            downstream_uses="image + M1 bbox + inferred density",
            bbox_stage="stage2",
        ),
        Condition(
            id="C6",
            name="C6_MATERIAL_DENSITY",
            uses_bbox=False,
            reasoning_mode="material_density_first",
            input_factors_label="image+material_density_first_reasoning",
            logical_first_inference="material + density",
            downstream_uses="image + inferred material + inferred density",
            bbox_stage="none",
        ),
        Condition(
            id="C7",
            name="C7_BBOX_MATERIAL_DENSITY",
            uses_bbox=True,
            reasoning_mode="bbox_material_density_first",
            input_factors_label="image+m1_bbox+material_density_first_reasoning",
            logical_first_inference="material + density",
            downstream_uses="image + M1 bbox + inferred material + inferred density",
            bbox_stage="stage2",
        ),
    )
}


def is_prediction_target(condition: Condition, property_name: str) -> bool:
    return property_name in condition.predict
