"""Orchestrate Experiment 1 v6: selective Stage 1 + OpenAI Stage 2 per condition."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from v6_conditions import Condition
from openai_stage2 import OpenAIStage2Runner, Stage2Result
from selective_siphy_stage1 import SiPhyStage1Result, SiPhyStage1Runner


@dataclass
class UnitResult:
    object_key: str
    condition: Condition
    prediction: dict[str, Any]
    stage1: SiPhyStage1Result | None
    stage2: Stage2Result
    siphy_cache_hit: bool  # always False after cache removal
    siphy_call_executed: bool
    siphy_model_call_count: int
    openai_model_call_count: int
    total_llm_call_count: int
    siphy_input_tokens: int | None = None
    siphy_output_tokens: int | None = None
    siphy_total_tokens: int | None = None
    openai_input_tokens: int | None = None
    openai_output_tokens: int | None = None
    openai_total_tokens: int | None = None
    # Analysis compat aliases (Stage2 tokens; same values as openai_*).
    gemini_input_tokens: int | None = None
    gemini_output_tokens: int | None = None
    gemini_total_tokens: int | None = None
    gemini_model_call_count: int = 0
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None
    total_tokens_combined: int | None = None  # alias of total_tokens
    # Common schema aliases (raw_results.csv compatibility)
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_call_count: int = 0
    provider: str | None = None
    model: str | None = None
    bbox_mm: list[float] | None = None
    crop_image_path: str | None = None
    siphy_material: str | None = None
    siphy_density_kgm3: float | None = None
    dry_run_meta: dict[str, Any] = field(default_factory=dict)


def describe_pipeline(condition: Condition) -> dict[str, Any]:
    """Static dry-run descriptors (no API)."""
    if condition.id == "C1":
        return {
            "stage1_model": "none",
            "stage1_input": "none",
            "stage1_output_used": "none",
            "stage1_cache_reuse": False,
            "stage2_model": "openai",
            "stage2_input": "Image + M1 BBox",
            "bbox_in_siphy": False,
            "bbox_in_gemini": True,
            "bbox_in_openai": True,
            "fixed_upstream_cues": [],
            "expected_siphy_calls": 0,
            "expected_gemini_calls": 1,
            "expected_openai_calls": 1,
            "expected_total_llm_calls": 1,
            "expected_token_accounting": (
                "openai Stage2 only: openai_input/output/total; siphy tokens = 0/null"
            ),
        }
    cues = list(condition.siphy_cues)
    stage1_out = " + ".join(
        "Material" if c == "material" else "Density" for c in cues
    )
    stage2_parts = ["Image"]
    if condition.uses_bbox:
        stage2_parts.append("M1 BBox")
    for c in cues:
        stage2_parts.append(
            "predicted Material" if c == "material" else "predicted Density"
        )
    return {
        "stage1_model": (
            "v6_selective_openai (material/density-only OpenAI; not full SiPhyBackend)"
        ),
        "stage1_input": "Image ONLY",
        "stage1_requested_properties": list(cues),
        "stage1_output_used": stage1_out,
        "stage1_cache_reuse": False,
        "stage2_model": "openai",
        "stage2_input": " + ".join(stage2_parts),
        "bbox_in_siphy": False,
        "bbox_in_gemini": bool(condition.uses_bbox),
        "bbox_in_openai": bool(condition.uses_bbox),
        "fixed_upstream_cues": list(condition.fixed_from_siphy),
        "expected_siphy_calls": 1,
        "expected_gemini_calls": 1,
        "expected_openai_calls": 1,
        "expected_total_llm_calls": 2,
        "expected_token_accounting": (
            "siphy_* + openai_*; total_input = siphy_in+openai_in; "
            "total_output = siphy_out+openai_out; total_tokens = siphy_total+openai_total"
        ),
    }


def _sum_optional(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


class Experiment1V6Runner:
    def __init__(
        self,
        *,
        siphy: SiPhyStage1Runner,
        openai: OpenAIStage2Runner,
        dry_run: bool = False,
    ):
        self.siphy = siphy
        self.openai = openai
        self.dry_run = dry_run

    def run_unit(
        self,
        condition: Condition,
        *,
        object_key: str,
        object_label: str,
        crop_image_path: Path,
        m1_bbox_mm: list[float] | None,
        gt_for_leak_check: dict[str, Any] | None = None,
    ) -> UnitResult:
        stage1: SiPhyStage1Result | None = None
        siphy_executed = False
        siphy_calls = 0
        fixed_cues: dict[str, Any] = {}
        siphy_material = None
        siphy_density = None

        if condition.uses_siphy:
            stage1 = self.siphy.run(
                object_key,
                condition.id,
                crop_image_path,
                siphy_cues=condition.siphy_cues,
            )
            siphy_executed = bool(stage1.siphy_call_executed)
            siphy_calls = int(stage1.siphy_model_call_count) if siphy_executed else 0
            fixed_cues = stage1.cues(condition.siphy_cues)
            siphy_material = stage1.material
            siphy_density = stage1.density_kgm3

        stage2 = self.openai.run(
            condition,
            crop_image_path=crop_image_path if crop_image_path.exists() else None,
            fixed_cues=fixed_cues,
            m1_bbox_mm=m1_bbox_mm if condition.uses_bbox else None,
            object_key=object_key,
            object_label=object_label,
            gt_for_leak_check=gt_for_leak_check,
        )

        o_calls = int(stage2.openai_model_call_count)
        if condition.uses_siphy and siphy_executed:
            s_in = stage1.siphy_input_tokens if stage1 else None
            s_out = stage1.siphy_output_tokens if stage1 else None
            s_tot = stage1.siphy_total_tokens if stage1 else None
        elif condition.uses_siphy and self.dry_run:
            s_in = s_out = s_tot = None
        else:
            # C1: no Stage1 → siphy tokens are 0 for acquisition-cost comparisons.
            s_in = 0 if not condition.uses_siphy else None
            s_out = 0 if not condition.uses_siphy else None
            s_tot = 0 if not condition.uses_siphy else None

        o_in = stage2.openai_input_tokens
        o_out = stage2.openai_output_tokens
        o_tot = stage2.openai_total_tokens

        total_in = _sum_optional(s_in, o_in)
        total_out = _sum_optional(s_out, o_out)
        total_tok = _sum_optional(s_tot, o_tot)

        meta = describe_pipeline(condition)
        meta["stage1_cache_reuse"] = False
        meta["siphy_cache_hit"] = False
        if stage1 is not None:
            meta["stage1_mode"] = stage1.stage1_mode
            meta["stage1_requested_keys"] = list(stage1.stage1_requested_keys)
            meta["stage1_youngs_gpa"] = stage1.youngs_gpa  # must be None in v6
        if self.dry_run:
            live_siphy = 1 if condition.uses_siphy else 0
            meta["expected_siphy_calls"] = live_siphy
            meta["expected_gemini_calls"] = 1
            meta["expected_openai_calls"] = 1
            meta["expected_total_llm_calls"] = live_siphy + 1
            meta["stage1_expected_call"] = live_siphy
            meta["stage1_required"] = bool(condition.uses_siphy)
        else:
            meta["expected_siphy_calls"] = siphy_calls
            meta["expected_gemini_calls"] = o_calls
            meta["expected_openai_calls"] = o_calls
            meta["expected_total_llm_calls"] = siphy_calls + o_calls
            meta["stage1_expected_call"] = siphy_calls
            meta["stage1_required"] = bool(condition.uses_siphy)

        live_siphy_count = 1 if condition.uses_siphy else 0
        if self.dry_run:
            openai_count_logged = 0
            siphy_count_logged = 0
            total_llm_logged = 0
        else:
            total_llm_logged = siphy_calls + o_calls
            openai_count_logged = o_calls
            siphy_count_logged = siphy_calls

        stage1_provider = stage1.provider if stage1 else None
        stage2_provider = getattr(stage2, "provider", "openai")
        if condition.uses_siphy:
            provider_label = f"siphy({stage1_provider or 'openai'})+openai"
        else:
            provider_label = stage2_provider or "openai"
        model_label = (
            f"siphy={stage1.model if stage1 else 'none'};openai={stage2.model}"
        )

        return UnitResult(
            object_key=object_key,
            condition=condition,
            prediction=stage2.prediction,
            stage1=stage1,
            stage2=stage2,
            siphy_cache_hit=False,
            siphy_call_executed=siphy_executed if not self.dry_run else False,
            siphy_model_call_count=siphy_count_logged,
            openai_model_call_count=openai_count_logged,
            total_llm_call_count=total_llm_logged,
            siphy_input_tokens=s_in,
            siphy_output_tokens=s_out,
            siphy_total_tokens=s_tot,
            openai_input_tokens=o_in,
            openai_output_tokens=o_out,
            openai_total_tokens=o_tot,
            gemini_input_tokens=o_in,
            gemini_output_tokens=o_out,
            gemini_total_tokens=o_tot,
            gemini_model_call_count=openai_count_logged,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_tokens=total_tok,
            total_tokens_combined=total_tok,
            input_tokens=total_in,
            output_tokens=total_out,
            model_call_count=(
                live_siphy_count + 1 if self.dry_run else total_llm_logged
            ),
            provider=provider_label,
            model=model_label,
            bbox_mm=list(m1_bbox_mm) if (condition.uses_bbox and m1_bbox_mm) else None,
            crop_image_path=str(crop_image_path),
            siphy_material=siphy_material,
            siphy_density_kgm3=siphy_density,
            dry_run_meta=meta,
        )


# Alias for any leftover imports
Experiment1V5Runner = Experiment1V6Runner
