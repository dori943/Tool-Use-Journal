"""Experiment 1 v3 - Gemini single-call SiPhy-style physical reasoning.

Prompt base reuses production SiPhy SYS_MSG wording from
``src/tuj/m3_grounding/siphy_backend.py`` (verbatim copy; not imported at runtime so
experiment never invokes OpenAI / SiPhyBackend).

- Multimodal input: production-style object crop PNG (background masked black).
- Optional textual M1 bbox_mm when condition.uses_bbox (never GT bbox).
- GT / object name / id / class never enter prompts.
- Exactly one Gemini API call per Object×Condition.
- No separate SiPhyBackend call; no local shell_mass_integral / FrictionHead predictions.
- Logical Stage 1 → Stage 2 are sequential instructions inside that one call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conditions import PROPERTIES, Condition, REASONING_FAMILY
from gemini_backend import (
    DEFAULT_GEMINI_MODEL,
    PROVIDER,
    GeminiClient,
)

# ---------------------------------------------------------------------------
# Production SiPhy SYS_MSG reuse
# Source: src/tuj/m3_grounding/siphy_backend.py  SYS_MSG  (K_MATERIALS = 5)
# ---------------------------------------------------------------------------
K_MATERIALS = 5

PRODUCTION_SIPHY_SYS_MSG = """You will be given an image of an object (background masked to black). Based on the image, give me a short (5-10 words) description of what the object is, and also %d materials that the object might be made of. For each material give: its mass density (in kg/m^3), the thickness (in cm) of that material in the object, its Young's modulus (in GPa), and, on a scale from 0 to 10, how likely it is that this object is made of that material. You may provide a range low-high of values instead of a single value for density, thickness and Young's modulus. Try to consider all the possible parts of the object. Do not include coatings like "paint" in your answer.

Format Requirement:
You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
{
    "description": description,
    "materials": [
        {"name": material1, "density_kgm3": "low-high", "thickness_cm": "low-high", "youngs_gpa": "low-high", "confidence_0_10": number},
        ...
    ]
}
Do not include any other text in your answer. Do not include unnecessary words besides the material in the material name.
""" % K_MATERIALS

# Experiment extension on top of production SYS_MSG:
# - Keep SiPhy visual / material-candidate / density / thickness / Young's style
# - Final schema = 5 evaluator fields (single numbers)
# - mass_kg and mu are LLM predictions (not local compute)
SYSTEM_PROMPT = f"""{PRODUCTION_SIPHY_SYS_MSG}

=== EXPERIMENT 1 EXTENSION (single API turn; do NOT call a second model) ===
The block above is the production SiPhy visual / material-candidate / density /
thickness / Young's reasoning style. Use that SAME style as your internal physical
thinking when the user asks for Stage-1 intermediate cues.

However, your FINAL answer for this experiment must NOT be the materials-list JSON.
Return ONLY one JSON object with this exact schema (SI units; single numbers, not ranges):
{{
  "material": string,
  "density_kgm3": number,
  "mass_kg": number,
  "mu": number,
  "youngs_gpa": number
}}

Rules:
- Complete ALL logical stages in this single response (one API turn). Do not ask for another turn.
- Always fill ALL five fields with your best estimates.
- Material: short material name only (no extra words), as in production SiPhy material names.
- density_kgm3: kg/m^3. mass_kg: kg. mu: dimensionless sliding friction coefficient.
  youngs_gpa: Young's modulus in GPa.
- You MUST predict mass_kg and mu yourself using visual evidence + physical / common-sense /
  mathematical relationships (density<->volume<->mass; material/surface friction priors;
  material Young's priors). Do NOT assume a downstream local shell-mass integral or
  friction lookup table will replace your answers.
- Prefer internally consistent physical estimates. Objects may be hollow/solid/composite;
  do not invent a single deterministic formula as always true.
- Do not treat any value as ground truth; none are provided.
- Do not output a known object category / product name / class label as an input fact.
  A brief internal description (SiPhy-style) may guide reasoning, but do not put a class
  name into the JSON and do not claim a labeled identity was given.
- When Stage-1 intermediate cues are required, fix them before Stage-2 and do not revise
  them after later cues (including bbox).
- Do not include chain-of-thought prose outside the JSON object.
"""

PRED_LABELS = {
    "material": "Material",
    "density_kgm3": "Density",
    "mass_kg": "Mass",
    "mu": "Mu",
    "youngs_gpa": "Young's modulus",
}


@dataclass
class InferenceResult:
    prediction: dict[str, Any]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    model_call_count: int
    model: str
    provider: str = PROVIDER
    prompt_user: str = ""
    prompt_system: str = SYSTEM_PROMPT
    raw_response: str | None = None
    image_used: bool = False
    crop_image_path: str | None = None
    bbox_mm_used: list[float] | None = None
    reasoning_mode: str | None = None
    reasoning_family: str = REASONING_FAMILY
    production_siphy_prompt_reused: bool = True
    bbox_stage: str | None = None
    object_name_in_prompt: bool = False
    error: str | None = None
    failure_reason: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    provided_factors: dict[str, Any] = field(default_factory=dict)
    sdk: str | None = None


def prediction_target_labels(condition: Condition) -> list[str]:
    return [PRED_LABELS[p] for p in condition.predict]


def _m1_bbox_lines(m1_bbox_mm: list[float]) -> list[str]:
    return [
        "M1 geometry cue (NOT ground truth):",
        f"bbox_mm = {list(m1_bbox_mm)}",
        "bbox_mm is the axis-aligned 3D extent [dx, dy, dz] in millimeters from production M1.",
        "It is a geometric size measurement only - not a material or density label.",
    ]


def describe_logical_stages(condition: Condition) -> dict[str, Any]:
    """Dry-run / reporting descriptors for logical Stage 1 / Stage 2 (one API call)."""
    mode = condition.reasoning_mode
    finals = ["material", "density_kgm3", "mass_kg", "mu", "youngs_gpa"]
    base = {
        "final_predicted_properties": finals,
        "expected_gemini_api_calls": 1,
        "production_siphy_prompt_reused": True,
        "bbox_stage": condition.bbox_stage,
        "reasoning_family": REASONING_FAMILY,
    }
    if mode == "bbox_guided":
        return {
            **base,
            "stage1_input": "none",
            "stage1_inference": "none",
            "stage2_input": "Image + M1 BBox",
            "bbox_allowed_stage1": False,
            "bbox_used_in_stage2": True,
            "bbox_provided": True,
        }
    if mode == "material_first":
        return {
            **base,
            "stage1_input": "Image ONLY",
            "stage1_inference": "Material",
            "stage2_input": "Image + inferred Material",
            "bbox_allowed_stage1": False,
            "bbox_used_in_stage2": False,
            "bbox_provided": False,
        }
    if mode == "density_first":
        return {
            **base,
            "stage1_input": "Image ONLY",
            "stage1_inference": "Density",
            "stage2_input": "Image + inferred Density",
            "bbox_allowed_stage1": False,
            "bbox_used_in_stage2": False,
            "bbox_provided": False,
        }
    if mode == "bbox_material_first":
        return {
            **base,
            "stage1_input": "Image ONLY",
            "stage1_inference": "Material",
            "stage2_input": "Image + M1 BBox + inferred Material",
            "bbox_allowed_stage1": False,
            "bbox_used_in_stage2": True,
            "bbox_provided": True,
        }
    if mode == "bbox_density_first":
        return {
            **base,
            "stage1_input": "Image ONLY",
            "stage1_inference": "Density",
            "stage2_input": "Image + M1 BBox + inferred Density",
            "bbox_allowed_stage1": False,
            "bbox_used_in_stage2": True,
            "bbox_provided": True,
        }
    if mode == "material_density_first":
        return {
            **base,
            "stage1_input": "Image ONLY",
            "stage1_inference": "Material + Density",
            "stage2_input": "Image + inferred Material + inferred Density",
            "bbox_allowed_stage1": False,
            "bbox_used_in_stage2": False,
            "bbox_provided": False,
        }
    if mode == "bbox_material_density_first":
        return {
            **base,
            "stage1_input": "Image ONLY",
            "stage1_inference": "Material + Density",
            "stage2_input": "Image + M1 BBox + inferred Material + inferred Density",
            "bbox_allowed_stage1": False,
            "bbox_used_in_stage2": True,
            "bbox_provided": True,
        }
    raise ValueError(f"unknown reasoning_mode: {mode}")


def build_user_prompt(
    condition: Condition,
    *,
    m1_bbox_mm: list[float] | None = None,
) -> str:
    """Build user text. NEVER accepts GT material/density/object name.

    Logical Stage 1 / Stage 2 are sequential instructions inside ONE Gemini call.
    No separate SiPhyBackend / second LLM call.

    For bbox_stage=stage2 (C4/C5/C7): Stage 1 is IMAGE ONLY; M1 bbox appears only in Stage 2.
    """
    lines: list[str] = [
        "You are given one object-crop image (background masked to black), same style as production SiPhy.",
        "Follow the system instructions: use SiPhy-style visual / material-candidate / density / "
        "Young's reasoning internally, then return the FINAL 5-field JSON only.",
        "",
        "Final JSON fields (all required): material, density_kgm3, mass_kg, mu, youngs_gpa.",
        "Predict mass_kg and mu yourself (no local mass integral / friction table after this call).",
        "",
        "Important: perform every step below in this single response. "
        "Do not request another API call or another turn.",
        "",
    ]

    mode = condition.reasoning_mode
    if condition.uses_bbox and m1_bbox_mm is None:
        raise ValueError("bbox required for this condition but m1_bbox_mm is missing")

    siphy_mat_hint = (
        f"Use the production SiPhy style: consider up to {K_MATERIALS} plausible material "
        "candidates from the image (name, density, thickness, Young's, confidence), then "
        "commit to a single most likely material for the Stage-1 cue."
    )
    siphy_mat_dens_hint = (
        f"Use the production SiPhy style: consider up to {K_MATERIALS} plausible material "
        "candidates with density / thickness / Young's / confidence from the IMAGE ONLY, "
        "then commit to a single most likely material AND a single density_kgm3 as fixed "
        "Stage-1 intermediate cues."
    )

    if mode == "bbox_guided":
        lines.extend(_m1_bbox_lines(m1_bbox_mm))  # type: ignore[arg-type]
        lines.append("")
        lines.append(
            "Reasoning instructions (single response; no Material/Density-first stage):"
        )
        lines.append(
            "Using the crop image together with bbox_mm, jointly infer material, "
            "density_kgm3, mass_kg, mu, and youngs_gpa. Apply SiPhy-style visual material "
            "and density reasoning, then extend to mass/mu with physical common sense "
            "(scale from bbox_mm + density; material friction / Young's priors)."
        )
        lines.append("Do not invent measurements beyond the image and the given bbox_mm.")
    elif mode == "material_first":
        lines.append("Reasoning instructions (logical stages inside this single response):")
        lines.append(
            "[Logical Stage 1 - SiPhy-style intermediate inference] "
            "IMAGE ONLY. Infer Material first. " + siphy_mat_hint + " "
            "Treat this material as a fixed intermediate physical estimate (not GT)."
        )
        lines.append(
            "[Logical Stage 2 - downstream physical reasoning] "
            "Using (1) the crop image and (2) the FIXED Stage-1 material, infer "
            "density_kgm3, mass_kg, mu, and youngs_gpa. Do not revise the Stage-1 material."
        )
        lines.append(
            "Put Stage-1 material and Stage-2 estimates into the same final JSON."
        )
    elif mode == "density_first":
        lines.append("Reasoning instructions (logical stages inside this single response):")
        lines.append(
            "[Logical Stage 1 - SiPhy-style intermediate inference] "
            "IMAGE ONLY. Infer density_kgm3 first using SiPhy-style density reasoning from "
            "visual appearance (and implicit material candidates if helpful). "
            "Treat Stage-1 density as a fixed intermediate estimate (not GT)."
        )
        lines.append(
            "[Logical Stage 2 - downstream physical reasoning] "
            "Using (1) the crop image and (2) the FIXED Stage-1 density, infer "
            "material, mass_kg, mu, and youngs_gpa. Do not revise the Stage-1 density."
        )
        lines.append(
            "Put Stage-1 density and Stage-2 estimates into the same final JSON."
        )
    elif mode == "bbox_material_first":
        lines.append("Reasoning instructions (logical stages inside this single response):")
        lines.append(
            "[Logical Stage 1 - SiPhy-style intermediate inference] "
            "IMAGE ONLY. Infer Material first. " + siphy_mat_hint + " "
            "Do NOT use the 3D bbox when inferring the intermediate material. "
            "Ignore any size/bbox cue until Stage 2. "
            "Treat Stage-1 material as FIXED (not GT)."
        )
        lines.append(
            "[Logical Stage 2 - downstream physical reasoning] "
            "Only now may you use bbox_mm below with the image and the FIXED Stage-1 material. "
            "Do NOT revise or re-estimate material after seeing the bbox."
        )
        lines.extend(_m1_bbox_lines(m1_bbox_mm))  # type: ignore[arg-type]
        lines.append(
            "Using (1) image, (2) bbox_mm, (3) fixed Stage-1 material, infer "
            "density_kgm3, mass_kg, mu, and youngs_gpa."
        )
        lines.append("Return one JSON with Stage-1 material + Stage-2 estimates.")
    elif mode == "bbox_density_first":
        lines.append("Reasoning instructions (logical stages inside this single response):")
        lines.append(
            "[Logical Stage 1 - SiPhy-style intermediate inference] "
            "IMAGE ONLY. Infer density_kgm3 first (SiPhy-style density reasoning). "
            "Do NOT use the 3D bbox when inferring the intermediate density. "
            "Ignore any size/bbox cue until Stage 2. "
            "Treat Stage-1 density as FIXED (not GT)."
        )
        lines.append(
            "[Logical Stage 2 - downstream physical reasoning] "
            "Only now may you use bbox_mm below with the image and the FIXED Stage-1 density. "
            "Do NOT revise or re-estimate density after seeing the bbox."
        )
        lines.extend(_m1_bbox_lines(m1_bbox_mm))  # type: ignore[arg-type]
        lines.append(
            "Using (1) image, (2) bbox_mm, (3) fixed Stage-1 density, infer "
            "material, mass_kg, mu, and youngs_gpa."
        )
        lines.append("Return one JSON with Stage-1 density + Stage-2 estimates.")
    elif mode == "material_density_first":
        lines.append("Reasoning instructions (logical stages inside this single response):")
        lines.append(
            "[Logical Stage 1 - SiPhy-style intermediate inference] "
            "IMAGE ONLY. " + siphy_mat_dens_hint
        )
        lines.append(
            "[Logical Stage 2 - downstream physical reasoning] "
            "Using (1) the crop image, (2) FIXED Stage-1 material, and "
            "(3) FIXED Stage-1 density, infer mass_kg, mu, and youngs_gpa. "
            "Do not revise material or density."
        )
        lines.append("Return one JSON with Stage-1 material+density + Stage-2 estimates.")
    elif mode == "bbox_material_density_first":
        lines.append("Reasoning instructions (logical stages inside this single response):")
        lines.append(
            "[Logical Stage 1 - SiPhy-style intermediate inference] "
            "IMAGE ONLY. " + siphy_mat_dens_hint + " "
            "Do NOT use the 3D bbox when inferring intermediate material or density. "
            "Ignore any size/bbox cue until Stage 2."
        )
        lines.append(
            "[Logical Stage 2 - downstream physical reasoning] "
            "Only now may you use bbox_mm below with the image and the FIXED Stage-1 "
            "material and density. Do NOT revise material or density after seeing the bbox."
        )
        lines.extend(_m1_bbox_lines(m1_bbox_mm))  # type: ignore[arg-type]
        lines.append(
            "Using (1) image, (2) bbox_mm, (3) fixed material, (4) fixed density, infer "
            "mass_kg, mu, and youngs_gpa."
        )
        lines.append("Return one JSON with Stage-1 material+density + Stage-2 estimates.")
    else:
        raise ValueError(f"unknown reasoning_mode: {mode}")

    lines.append("")
    lines.append("Do not use ground-truth values. Return only the final JSON object.")
    return "\n".join(lines)


def format_dry_run_log(
    condition: Condition,
    *,
    crop_path: Path | None,
    m1_bbox_mm: list[float] | None,
    prompt: str,
) -> dict[str, Any]:
    stages = describe_logical_stages(condition)
    low = prompt.lower()
    if condition.bbox_stage == "stage2":
        # Allowed only if the forbid instruction is missing (should be False after fix).
        bbox_allowed_stage1 = "do not use the 3d bbox" not in low
    else:
        bbox_allowed_stage1 = False
    return {
        "crop_image_path": str(crop_path) if crop_path else None,
        "image_attached": bool(crop_path and crop_path.exists()),
        "bbox_included": bool(condition.uses_bbox and m1_bbox_mm is not None),
        "m1_bbox_mm": list(m1_bbox_mm) if (condition.uses_bbox and m1_bbox_mm) else None,
        "condition": condition.id,
        "condition_name": condition.name,
        "reasoning_mode": condition.reasoning_mode,
        "reasoning_family": REASONING_FAMILY,
        "production_siphy_prompt_reused": True,
        "logical_first_inference": condition.logical_first_inference,
        "downstream_uses": condition.downstream_uses,
        "input_factors": condition.input_factors_label,
        "prediction_targets": list(condition.predict),
        "expected_gemini_api_calls": stages["expected_gemini_api_calls"],
        "gt_used_in_inference": False,
        "prompt_user": prompt,
        "prompt_system": SYSTEM_PROMPT,
        "logical_stages": stages,
        "bbox_provided": bool(condition.uses_bbox and m1_bbox_mm is not None),
        "bbox_allowed_during_stage1": bbox_allowed_stage1,
        "bbox_used_during_stage2": stages["bbox_used_in_stage2"],
        "bbox_stage": condition.bbox_stage,
        "bbox_available_in_stage1_reasoning_instruction": bbox_allowed_stage1,
        "bbox_used_in_stage2": stages["bbox_used_in_stage2"],
    }


def prompt_contains_object_name(prompt: str, object_key: str, object_label: str) -> bool:
    low = prompt.lower()
    return (object_key.lower() in low) or (object_label.lower() in low)


def assert_no_gt_leakage(prompt: str, gt: dict[str, Any] | None) -> list[str]:
    """Return leakage descriptions if GT values appear in the user prompt text."""
    if not gt:
        return []
    leaks: list[str] = []
    mat = gt.get("material") or {}
    if mat.get("available") and mat.get("value") is not None:
        token = str(mat["value"]).strip().lower()
        if token and (
            f"material: {token}" in prompt.lower()
            or f"material = {token}" in prompt.lower()
            or f"- material: {token}" in prompt.lower()
        ):
            leaks.append(f"material_gt={mat['value']!r}")

    def _numeric_leaks(field: str, key: str) -> None:
        entry = gt.get(key) or {}
        if not entry.get("available") or entry.get("value") is None:
            return
        val = entry["value"]
        forms = {str(val)}
        try:
            f = float(val)
            forms.update({f"{f:.1f}", f"{f:.0f}", f"{f:.6g}"})
        except (TypeError, ValueError):
            pass
        low = prompt.lower()
        for form in forms:
            if not form or form in ("0", "1"):
                continue
            if form in prompt and (
                "ground-truth" in low
                or "gt " in low
                or f"{field}:" in low and form in low
            ):
                idx = low.find(form.lower() if form.lower() in low else form)
                if idx >= 0:
                    window = low[max(0, idx - 40) : idx + 40]
                    if field in window or "ground" in window or "gt" in window:
                        leaks.append(f"{field}_gt={val!r}")
                        return

    _numeric_leaks("density", "density_kgm3")
    _numeric_leaks("mass", "mass_kg")
    _numeric_leaks("mu", "mu")
    _numeric_leaks("youngs", "youngs_gpa")

    bbox = gt.get("bbox_mm") or {}
    if bbox.get("available") and bbox.get("value") is not None:
        gt_bbox = [float(x) for x in bbox["value"]]
        if str(gt_bbox) in prompt or str([round(x, 1) for x in gt_bbox]) in prompt:
            leaks.append(f"bbox_gt={gt_bbox!r}")
    return leaks


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    out: dict[str, Any] = {k: None for k in PROPERTIES}
    if data.get("material") not in (None, ""):
        out["material"] = str(data["material"])
    for key in ("density_kgm3", "mass_kg", "mu", "youngs_gpa"):
        if data.get(key) is None or data.get(key) == "":
            continue
        out[key] = float(data[key])
    return out


def _empty_pred() -> dict[str, Any]:
    return {k: None for k in PROPERTIES}


def _result_meta(condition: Condition) -> dict[str, Any]:
    return {
        "reasoning_mode": condition.reasoning_mode,
        "reasoning_family": REASONING_FAMILY,
        "production_siphy_prompt_reused": True,
        "bbox_stage": condition.bbox_stage,
    }


class ConditionedSiPhyRunner:
    """Gemini multimodal runner. Never creates OpenAI / never reads GT into prompts."""

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        dry_run: bool = False,
        client: GeminiClient | None = None,
    ):
        self.model = model
        self.dry_run = dry_run
        self.provider = PROVIDER
        self.client = client
        if not dry_run and client is None:
            self.client = GeminiClient(model=model)

    def infer(
        self,
        condition: Condition,
        *,
        m1_bbox_mm: list[float] | None = None,
        crop_image_path: Path | None = None,
        object_key: str | None = None,
        object_label: str | None = None,
        gt_for_leak_check: dict[str, Any] | None = None,
    ) -> InferenceResult:
        missing: list[str] = []
        if crop_image_path is None or not Path(crop_image_path).exists():
            missing.append("crop_image")
        if condition.uses_bbox and m1_bbox_mm is None:
            missing.append("bbox(m1)")

        bbox_for_prompt = list(m1_bbox_mm) if (condition.uses_bbox and m1_bbox_mm) else None
        prompt = ""
        if "bbox(m1)" not in missing:
            prompt = build_user_prompt(condition, m1_bbox_mm=bbox_for_prompt)

        provided = {
            "crop_image": str(crop_image_path) if crop_image_path else None,
            "input_factors": condition.input_factors_label,
            "reasoning_mode": condition.reasoning_mode,
            "reasoning_family": REASONING_FAMILY,
            "bbox_stage": condition.bbox_stage,
        }
        if bbox_for_prompt is not None:
            provided["bbox_mm"] = bbox_for_prompt
            provided["bbox_source"] = "m1"

        name_leak = False
        if prompt and object_key and object_label:
            name_leak = prompt_contains_object_name(prompt, object_key, object_label)

        meta = _result_meta(condition)

        if missing:
            skip = f"required input unavailable: {missing}"
            if self.dry_run:
                leaks = assert_no_gt_leakage(prompt, gt_for_leak_check) if prompt else []
                if leaks:
                    skip += f" | GT_LEAK_CHECK_FAIL: {leaks}"
                skip = f"API Call: SKIPPED (dry-run); {skip}"
            return InferenceResult(
                prediction=_empty_pred(),
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                model_call_count=0,
                model=self.model,
                prompt_user=prompt,
                image_used=False,
                crop_image_path=str(crop_image_path) if crop_image_path else None,
                bbox_mm_used=bbox_for_prompt,
                object_name_in_prompt=name_leak,
                skipped=True,
                skip_reason=skip,
                provided_factors=provided,
                **meta,
            )

        if self.dry_run:
            leaks = assert_no_gt_leakage(prompt, gt_for_leak_check)
            skip = "API Call: SKIPPED (dry-run)"
            if leaks:
                skip += f" | GT_LEAK_CHECK_FAIL: {leaks}"
            return InferenceResult(
                prediction=_empty_pred(),
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                model_call_count=0,
                model=self.model,
                prompt_user=prompt,
                image_used=True,
                crop_image_path=str(crop_image_path),
                bbox_mm_used=bbox_for_prompt,
                provided_factors=provided,
                object_name_in_prompt=name_leak,
                skipped=False,
                skip_reason=skip,
                **meta,
            )

        assert self.client is not None
        try:
            resp = self.client.generate_json(
                SYSTEM_PROMPT,
                prompt,
                image_path=Path(crop_image_path),
            )
        except Exception as exc:  # noqa: BLE001
            return InferenceResult(
                prediction=_empty_pred(),
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                model_call_count=1,
                model=self.model,
                prompt_user=prompt,
                image_used=True,
                crop_image_path=str(crop_image_path),
                bbox_mm_used=bbox_for_prompt,
                error="failed",
                failure_reason=f"api_call_failed: {exc}",
                provided_factors=provided,
                object_name_in_prompt=name_leak,
                **meta,
            )

        inp = resp.usage.input_tokens
        out = resp.usage.output_tokens
        total = resp.usage.total_tokens

        try:
            pred = _parse_json_response(resp.text)
        except Exception as exc:  # noqa: BLE001
            return InferenceResult(
                prediction=_empty_pred(),
                input_tokens=inp,
                output_tokens=out,
                total_tokens=total,
                model_call_count=1,
                model=self.model,
                prompt_user=prompt,
                raw_response=resp.text,
                image_used=True,
                crop_image_path=str(crop_image_path),
                bbox_mm_used=bbox_for_prompt,
                error="failed",
                failure_reason=f"json_parse_failed: {exc}",
                provided_factors=provided,
                object_name_in_prompt=name_leak,
                sdk=resp.sdk,
                **meta,
            )

        return InferenceResult(
            prediction=pred,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            model_call_count=1,
            model=self.model,
            prompt_user=prompt,
            raw_response=resp.text,
            image_used=True,
            crop_image_path=str(crop_image_path),
            bbox_mm_used=bbox_for_prompt,
            provided_factors=provided,
            object_name_in_prompt=name_leak,
            sdk=resp.sdk,
            **meta,
        )


DEFAULT_MODEL = DEFAULT_GEMINI_MODEL


def format_input_factors_log(provided: dict[str, Any], condition: Condition) -> str:
    """Human-readable dry-run / live banner (no GT values)."""
    lines = [
        f"input_factors: {condition.input_factors_label}",
        f"reasoning_mode: {condition.reasoning_mode}",
        f"reasoning_family: {REASONING_FAMILY}",
        f"crop: {provided.get('crop_image') or 'MISSING'}",
    ]
    if condition.uses_bbox:
        bbox = provided.get("bbox_mm")
        lines.append(f"M1 bbox_mm: {bbox}" if bbox is not None else "M1 bbox_mm: MISSING")
    else:
        lines.append("M1 bbox_mm: NOT USED")
    return "\n".join(lines)
