"""Stage 2: OpenAI downstream reasoning over fixed SiPhy cues (+ optional M1 bbox)."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from v6_conditions import PROPERTIES, Condition

# Local OpenAI client (does not modify parent Gemini backends / v4 imports).
_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.append(str(_PARENT))

from openai_backend import (  # noqa: E402
    DEFAULT_OPENAI_MODEL,
    PROVIDER,
    OpenAIClient,
)


SYSTEM_PROMPT = """You estimate remaining physical properties of an unknown rigid object.

You may be given:
- an object-crop image (background masked to black), same style as production SiPhy
- optional upstream module predictions (material and/or density) from a separate SiPhy stage
- optional M1 3D bounding-box extents bbox_mm = [dx, dy, dz] in millimeters

Rules:
- Treat any supplied upstream material/density as FIXED predictions from an upstream module.
  Do NOT revise, re-estimate, or overwrite those fixed cues.
- Do not invent ground-truth values; none are provided.
- Do not assume a labeled object category / product name / class was given as an input fact.
- Predict mass_kg and mu yourself using visual evidence + physical / common-sense relationships
  (density-volume-mass; material/surface friction priors; material Young's priors).
  Do not assume a local shell-mass integral or friction lookup will replace your answers.
- Return ONLY one JSON object with exactly these fields (SI units; single numbers):
{
  "material": string,
  "density_kgm3": number,
  "mass_kg": number,
  "mu": number,
  "youngs_gpa": number
}
- For fields marked FIXED in the user message, copy the provided upstream value into the JSON.
- For fields you must infer, fill your best estimate.
- No chain-of-thought outside the JSON.
"""


@dataclass
class Stage2Result:
    prediction: dict[str, Any]
    prompt_user: str
    prompt_system: str = SYSTEM_PROMPT
    openai_input_tokens: int | None = None
    openai_output_tokens: int | None = None
    openai_total_tokens: int | None = None
    openai_model_call_count: int = 0
    openai_call_executed: bool = False
    model: str = DEFAULT_OPENAI_MODEL
    provider: str = PROVIDER
    raw_response: str | None = None
    image_used: bool = False
    object_name_in_prompt: bool = False
    error: str | None = None
    failure_reason: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    sdk: str | None = None
    fixed_cues_applied: dict[str, Any] = field(default_factory=dict)


def _m1_bbox_lines(m1_bbox_mm: list[float]) -> list[str]:
    return [
        "M1 geometry cue (NOT ground truth):",
        f"bbox_mm = {list(m1_bbox_mm)}",
        "bbox_mm is the axis-aligned 3D extent [dx, dy, dz] in millimeters from production M1.",
        "It is a geometric size measurement only - not a material or density label.",
    ]


def build_stage2_user_prompt(
    condition: Condition,
    *,
    fixed_cues: dict[str, Any],
    m1_bbox_mm: list[float] | None = None,
) -> str:
    lines: list[str] = [
        "You are given one object-crop image (background masked to black).",
        "This is Stage 2 of a two-stage pipeline. Upstream SiPhy (if any) already ran separately.",
        "Return one JSON with: material, density_kgm3, mass_kg, mu, youngs_gpa.",
        "",
    ]

    if condition.uses_bbox:
        if m1_bbox_mm is None:
            raise ValueError("bbox required for this condition but m1_bbox_mm is missing")
        lines.extend(_m1_bbox_lines(m1_bbox_mm))
        lines.append("")

    if condition.fixed_from_siphy:
        lines.append("FIXED upstream SiPhy cues (do not revise):")
        for key in condition.fixed_from_siphy:
            val = fixed_cues.get(key)
            if key == "material":
                lines.append(f"- material = {val!s}")
                lines.append("  Do not revise the provided material.")
            elif key == "density_kgm3":
                lines.append(f"- density_kgm3 = {val}")
                lines.append("  Do not revise the provided density.")
        lines.append(
            "Treat the supplied physical cue(s) as an upstream module prediction."
        )
        lines.append("")

    infer = list(condition.gemini_infer)
    lines.append(
        "Infer ONLY these fields from the image"
        + (" and bbox_mm" if condition.uses_bbox else "")
        + (" and the fixed upstream cue(s)" if condition.fixed_from_siphy else "")
        + f": {', '.join(infer)}."
    )
    if condition.fixed_from_siphy:
        lines.append(
            "In the final JSON, copy FIXED fields exactly from the upstream values above; "
            "fill the remaining fields with your estimates."
        )
    else:
        lines.append(
            "No upstream SiPhy cues for this condition. Infer all five fields from image"
            + (" + bbox_mm" if condition.uses_bbox else "")
            + "."
        )

    lines.append("")
    lines.append("Do not use ground-truth values. Return only the JSON object.")
    return "\n".join(lines)


def prompt_contains_object_name(prompt: str, object_key: str, object_label: str) -> bool:
    low = prompt.lower()
    return (object_key.lower() in low) or (object_label.lower() in low)


def assert_no_gt_leakage(prompt: str, gt: dict[str, Any] | None) -> list[str]:
    if not gt:
        return []
    leaks: list[str] = []
    mat = gt.get("material") or {}
    if mat.get("available") and mat.get("value") is not None:
        token = str(mat["value"]).strip().lower()
        if token and (
            f"material: {token}" in prompt.lower()
            or f"material = {token}" in prompt.lower()
        ):
            # Fixed SiPhy cues may coincidentally equal GT; only flag if labeled as GT.
            if "ground-truth" in prompt.lower() or "gt " in prompt.lower():
                leaks.append(f"material_gt={mat['value']!r}")

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


def apply_fixed_cues(
    prediction: dict[str, Any],
    condition: Condition,
    fixed_cues: dict[str, Any],
) -> dict[str, Any]:
    """Enforce Stage-1 SiPhy values into the final prediction for fixed fields."""
    out = dict(prediction)
    for key in condition.fixed_from_siphy:
        out[key] = fixed_cues.get(key)
    return out


class OpenAIStage2Runner:
    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        dry_run: bool = False,
        client: OpenAIClient | None = None,
    ):
        self.model = model
        self.dry_run = dry_run
        self.client = client
        if not dry_run and client is None:
            self.client = OpenAIClient(model=model)

    def run(
        self,
        condition: Condition,
        *,
        crop_image_path: Path | None,
        fixed_cues: dict[str, Any],
        m1_bbox_mm: list[float] | None = None,
        object_key: str | None = None,
        object_label: str | None = None,
        gt_for_leak_check: dict[str, Any] | None = None,
    ) -> Stage2Result:
        missing: list[str] = []
        if crop_image_path is None or not Path(crop_image_path).exists():
            missing.append("crop_image")
        if condition.uses_bbox and m1_bbox_mm is None:
            missing.append("bbox(m1)")
        for key in condition.fixed_from_siphy:
            if fixed_cues.get(key) is None:
                missing.append(f"siphy_cue:{key}")

        prompt = ""
        if "bbox(m1)" not in missing and not any(m.startswith("siphy_cue:") for m in missing):
            # Allow building prompt for dry-run even if crop missing (inspect text).
            try:
                prompt = build_stage2_user_prompt(
                    condition, fixed_cues=fixed_cues, m1_bbox_mm=m1_bbox_mm
                )
            except ValueError:
                prompt = ""

        name_leak = False
        if prompt and object_key and object_label:
            name_leak = prompt_contains_object_name(prompt, object_key, object_label)

        if missing:
            skip = f"required input unavailable: {missing}"
            if self.dry_run:
                leaks = assert_no_gt_leakage(prompt, gt_for_leak_check) if prompt else []
                if leaks:
                    skip += f" | GT_LEAK_CHECK_FAIL: {leaks}"
                skip = f"API Call: SKIPPED (dry-run); {skip}"
            return Stage2Result(
                prediction=_empty_pred(),
                prompt_user=prompt,
                skipped=True,
                skip_reason=skip,
                object_name_in_prompt=name_leak,
                fixed_cues_applied={k: fixed_cues.get(k) for k in condition.fixed_from_siphy},
                model=self.model,
            )

        if self.dry_run:
            leaks = assert_no_gt_leakage(prompt, gt_for_leak_check)
            skip = "API Call: SKIPPED (dry-run)"
            if leaks:
                skip += f" | GT_LEAK_CHECK_FAIL: {leaks}"
            # Structural dry-run prediction: fixed cues filled; others None.
            pred = apply_fixed_cues(_empty_pred(), condition, fixed_cues)
            return Stage2Result(
                prediction=pred,
                prompt_user=prompt,
                openai_model_call_count=0,
                openai_call_executed=False,
                image_used=True,
                object_name_in_prompt=name_leak,
                skipped=False,
                skip_reason=skip,
                fixed_cues_applied={k: fixed_cues.get(k) for k in condition.fixed_from_siphy},
                model=self.model,
            )

        assert self.client is not None
        try:
            resp = self.client.generate_json(
                SYSTEM_PROMPT,
                prompt,
                image_path=Path(crop_image_path),
            )
        except Exception as exc:  # noqa: BLE001
            return Stage2Result(
                prediction=_empty_pred(),
                prompt_user=prompt,
                openai_model_call_count=1,
                openai_call_executed=True,
                image_used=True,
                error="failed",
                failure_reason=f"api_call_failed: {exc}",
                object_name_in_prompt=name_leak,
                fixed_cues_applied={k: fixed_cues.get(k) for k in condition.fixed_from_siphy},
                model=self.model,
            )

        try:
            pred = _parse_json_response(resp.text)
        except Exception as exc:  # noqa: BLE001
            return Stage2Result(
                prediction=_empty_pred(),
                prompt_user=prompt,
                openai_input_tokens=resp.usage.input_tokens,
                openai_output_tokens=resp.usage.output_tokens,
                openai_total_tokens=resp.usage.total_tokens,
                openai_model_call_count=1,
                openai_call_executed=True,
                raw_response=resp.text,
                image_used=True,
                error="failed",
                failure_reason=f"json_parse_failed: {exc}",
                object_name_in_prompt=name_leak,
                sdk=resp.sdk,
                fixed_cues_applied={k: fixed_cues.get(k) for k in condition.fixed_from_siphy},
                model=self.model,
            )

        pred = apply_fixed_cues(pred, condition, fixed_cues)
        return Stage2Result(
            prediction=pred,
            prompt_user=prompt,
            openai_input_tokens=resp.usage.input_tokens,
            openai_output_tokens=resp.usage.output_tokens,
            openai_total_tokens=resp.usage.total_tokens,
            openai_model_call_count=1,
            openai_call_executed=True,
            raw_response=resp.text,
            image_used=True,
            object_name_in_prompt=name_leak,
            sdk=resp.sdk,
            fixed_cues_applied={k: fixed_cues.get(k) for k in condition.fixed_from_siphy},
            model=self.model,
        )
