"""Experiment 1 v6 Stage 1: property-selective SiPhy-style cue acquisition (OpenAI).

Same schemas/prompts as v5; provider is OpenAI instead of Gemini.
Does NOT call production ``SiPhyBackend.estimate``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai_backend import (  # noqa: E402
    DEFAULT_OPENAI_MODEL,
    PROVIDER,
    OpenAIClient,
)

MODE_MATERIAL = ("material",)
MODE_DENSITY = ("density_kgm3",)
MODE_BOTH = ("material", "density_kgm3")


def resolve_siphy_model(explicit: str | None = None) -> str:
    """CLI --model is authoritative; soft default only when omitted."""
    if explicit:
        return explicit
    return DEFAULT_OPENAI_MODEL


def static_openai_feasibility(repo_root: Path | None) -> dict[str, Any]:
    okey = bool(os.environ.get("OPENAI_API_KEY"))
    return {
        "provider": PROVIDER,
        "OPENAI_API_KEY_set": okey,
        "stage1_backend": "v6_selective_openai (not production SiPhyBackend.estimate)",
        "cls_hint_in_vlm_prompt": False,
        "bbox_in_stage1": False,
        "repo_root": str(repo_root) if repo_root else None,
        "note": (
            "v6 Stage1 requests only material and/or density via minimized JSON schema; "
            "Young's / mass / mu are never requested at Stage1. Same prompts as v5."
        ),
    }


# Identical prompt text to v5 (provider change only).
_SYS_MATERIAL_ONLY = """You will be given an image of an object (background masked to black). Based on the image, identify the single most likely material the object is made of. Do not invent a product name or class label as an input fact. Do not estimate density, mass, friction, Young's modulus, or thickness.

Format Requirement:
Return ONLY one JSON object:
{
    "material": string
}
Do not include any other keys or text.
"""

_SYS_DENSITY_ONLY = """You will be given an image of an object (background masked to black). Based on the image, estimate the object's bulk mass density in kg/m^3. Do not invent a product name or class label as an input fact. Do not estimate material name, mass, friction, Young's modulus, or thickness.

Format Requirement:
Return ONLY one JSON object:
{
    "density_kgm3": number
}
Use a single number (not a range). Do not include any other keys or text.
"""

_SYS_MATERIAL_DENSITY = """You will be given an image of an object (background masked to black). Based on the image, identify the single most likely material and estimate the object's bulk mass density in kg/m^3. Do not invent a product name or class label as an input fact. Do not estimate mass, friction, Young's modulus, or thickness.

Format Requirement:
Return ONLY one JSON object:
{
    "material": string,
    "density_kgm3": number
}
density_kgm3 must be a single number (not a range). Do not include any other keys or text.
"""

_USER_HINT = (
    "Object-crop image attached (background masked to black). "
    "Return only the JSON object required by the system message."
)

FORBIDDEN_STAGE1_KEYS = (
    "youngs_gpa",
    "mass_kg",
    "mu",
    "thickness_cm",
    "materials",
    "materials_topk",
    "mass_range_kg",
    "bbox_mm",
    "points_mm",
)


def stage1_mode_for_cues(cues: tuple[str, ...]) -> str:
    c = tuple(cues)
    if c == MODE_MATERIAL:
        return "material_only"
    if c == MODE_DENSITY:
        return "density_only"
    if c == MODE_BOTH:
        return "material_density"
    raise ValueError(f"unsupported Stage1 cue set: {cues!r}")


def system_prompt_for_cues(cues: tuple[str, ...]) -> str:
    mode = stage1_mode_for_cues(cues)
    if mode == "material_only":
        return _SYS_MATERIAL_ONLY
    if mode == "density_only":
        return _SYS_DENSITY_ONLY
    return _SYS_MATERIAL_DENSITY


@dataclass
class SiPhyStage1Result:
    object_key: str
    condition_id: str
    material: str | None
    density_kgm3: float | None
    youngs_gpa: float | None
    raw_siphy_output: dict[str, Any]
    crop_image_path: str
    model: str
    provider: str | None = None
    siphy_input_tokens: int | None = None
    siphy_output_tokens: int | None = None
    siphy_total_tokens: int | None = None
    siphy_call_executed: bool = True
    siphy_model_call_count: int = 1
    siphy_api_attempt_count: int = 0
    siphy_cache_hit: bool = False
    error: str | None = None
    stage1_mode: str | None = None
    stage1_requested_keys: tuple[str, ...] = ()
    stage1_system_prompt: str | None = None

    def cues(self, names: tuple[str, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for n in names:
            if n == "material":
                out["material"] = self.material
            elif n == "density_kgm3":
                out["density_kgm3"] = self.density_kgm3
            else:
                raise KeyError(n)
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "object_key": self.object_key,
            "condition_id": self.condition_id,
            "stage1_mode": self.stage1_mode,
            "stage1_requested_keys": list(self.stage1_requested_keys),
            "material": self.material,
            "density_kgm3": self.density_kgm3,
            "youngs_gpa": None,
            "raw_siphy_output": self.raw_siphy_output,
            "crop_image_path": self.crop_image_path,
            "model": self.model,
            "provider": self.provider,
            "token_usage": {
                "input_tokens": self.siphy_input_tokens,
                "output_tokens": self.siphy_output_tokens,
                "total_tokens": self.siphy_total_tokens,
            },
            "siphy_call_executed": self.siphy_call_executed,
            "siphy_model_call_count": self.siphy_model_call_count,
            "siphy_api_attempt_count": self.siphy_api_attempt_count,
            "siphy_cache_hit": False,
            "error": self.error,
            "stage1_system_prompt": self.stage1_system_prompt,
        }


def _parse_selective_json(text: str, cues: tuple[str, ...]) -> dict[str, Any]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Stage1 response is not a JSON object")
    for bad in FORBIDDEN_STAGE1_KEYS:
        if bad in data and data[bad] is not None:
            raise ValueError(f"Stage1 returned forbidden key {bad!r}")
    out: dict[str, Any] = {}
    if "material" in cues:
        mat = data.get("material")
        if mat in (None, ""):
            raise ValueError("material missing in Stage1 response")
        out["material"] = str(mat).strip().lower()
    if "density_kgm3" in cues:
        dens = data.get("density_kgm3")
        if dens in (None, ""):
            raise ValueError("density_kgm3 missing in Stage1 response")
        out["density_kgm3"] = float(dens)
    return out


def assert_selective_output(parsed: dict[str, Any], cues: tuple[str, ...]) -> None:
    allowed = set(cues)
    extras = set(parsed.keys()) - allowed
    if extras:
        raise ValueError(f"Stage1 parsed extras {extras}")
    for bad in ("youngs_gpa", "mass_kg", "mu"):
        if bad in parsed:
            raise ValueError(f"Stage1 must not produce {bad}")


@dataclass
class SelectiveSiPhyStage1Runner:
    out_path: Path
    dry_run: bool = False
    model: str | None = None
    repo_root: Path | None = None
    _log: dict[str, dict[str, SiPhyStage1Result]] = field(default_factory=dict)
    _client: OpenAIClient | None = None

    def __post_init__(self) -> None:
        self.model = resolve_siphy_model(self.model)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.out_path.exists():
            raw = json.loads(self.out_path.read_text(encoding="utf-8"))
            for obj_key, by_cond in (raw.get("objects") or {}).items():
                if not isinstance(by_cond, dict):
                    continue
                self._log[obj_key] = {}
                for cid, val in by_cond.items():
                    if not isinstance(val, dict):
                        continue
                    self._log[obj_key][cid] = SiPhyStage1Result(
                        object_key=obj_key,
                        condition_id=cid,
                        material=val.get("material"),
                        density_kgm3=val.get("density_kgm3"),
                        youngs_gpa=None,
                        raw_siphy_output=val.get("raw_siphy_output") or {},
                        crop_image_path=val.get("crop_image_path") or "",
                        model=val.get("model") or self.model,
                        provider=val.get("provider"),
                        siphy_input_tokens=(val.get("token_usage") or {}).get("input_tokens"),
                        siphy_output_tokens=(val.get("token_usage") or {}).get("output_tokens"),
                        siphy_total_tokens=(val.get("token_usage") or {}).get("total_tokens"),
                        siphy_call_executed=bool(val.get("siphy_call_executed", True)),
                        siphy_model_call_count=int(val.get("siphy_model_call_count") or 1),
                        siphy_api_attempt_count=int(val.get("siphy_api_attempt_count") or 0),
                        siphy_cache_hit=False,
                        error=val.get("error"),
                        stage1_mode=val.get("stage1_mode"),
                        stage1_requested_keys=tuple(val.get("stage1_requested_keys") or ()),
                        stage1_system_prompt=val.get("stage1_system_prompt"),
                    )

    def save(self) -> None:
        payload = {
            "experiment": "experiment1_v6",
            "provider": PROVIDER,
            "note": (
                "v6 SELECTIVE Stage-1 LOG via OpenAI (not production SiPhyBackend.estimate). "
                "Same cue schemas as v5. No Young's / mass / mu at Stage1."
            ),
            "objects": {
                obj: {cid: r.to_json() for cid, r in by_c.items()}
                for obj, by_c in self._log.items()
            },
        }
        self.out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _record(self, result: SiPhyStage1Result) -> None:
        self._log.setdefault(result.object_key, {})[result.condition_id] = result
        self.save()

    def _ensure_client(self) -> OpenAIClient:
        if self._client is None:
            self._client = OpenAIClient(model=self.model, max_retries=0)
        return self._client

    def run(
        self,
        object_key: str,
        condition_id: str,
        crop_image_path: Path,
        *,
        siphy_cues: tuple[str, ...],
    ) -> SiPhyStage1Result:
        mode = stage1_mode_for_cues(siphy_cues)
        sys_prompt = system_prompt_for_cues(siphy_cues)

        if self.dry_run:
            material = None
            density = None
            if "material" in siphy_cues:
                material = f"__siphy_material_{condition_id}__"
            if "density_kgm3" in siphy_cues:
                density = float(-100 - ord(condition_id[-1]))
            raw = {
                "dry_run": True,
                "condition_id": condition_id,
                "stage1_mode": mode,
                "requested_keys": list(siphy_cues),
                "system_prompt": sys_prompt,
                "user_text": _USER_HINT,
                "note": "selective Stage1 OpenAI; production SiPhyBackend.estimate NOT used",
            }
            assert_selective_output(
                {k: v for k, v in (("material", material), ("density_kgm3", density)) if v is not None},
                siphy_cues,
            )
            result = SiPhyStage1Result(
                object_key=object_key,
                condition_id=condition_id,
                material=material,
                density_kgm3=density,
                youngs_gpa=None,
                raw_siphy_output=raw,
                crop_image_path=str(crop_image_path),
                model=self.model,
                provider=PROVIDER,
                siphy_call_executed=False,
                siphy_model_call_count=0,
                siphy_api_attempt_count=0,
                stage1_mode=mode,
                stage1_requested_keys=siphy_cues,
                stage1_system_prompt=sys_prompt,
            )
            self._record(result)
            return result

        if not crop_image_path.exists():
            result = SiPhyStage1Result(
                object_key=object_key,
                condition_id=condition_id,
                material=None,
                density_kgm3=None,
                youngs_gpa=None,
                raw_siphy_output={},
                crop_image_path=str(crop_image_path),
                model=self.model,
                provider=PROVIDER,
                siphy_call_executed=False,
                siphy_model_call_count=0,
                error=f"crop missing: {crop_image_path}",
                stage1_mode=mode,
                stage1_requested_keys=siphy_cues,
                stage1_system_prompt=sys_prompt,
            )
            self._record(result)
            return result

        client = self._ensure_client()
        try:
            resp = client.generate_json(
                sys_prompt,
                _USER_HINT,
                image_path=Path(crop_image_path),
            )
            parsed = _parse_selective_json(resp.text, siphy_cues)
            assert_selective_output(parsed, siphy_cues)
        except Exception as exc:  # noqa: BLE001
            result = SiPhyStage1Result(
                object_key=object_key,
                condition_id=condition_id,
                material=None,
                density_kgm3=None,
                youngs_gpa=None,
                raw_siphy_output={"error": str(exc)},
                crop_image_path=str(crop_image_path),
                model=self.model,
                provider=PROVIDER,
                siphy_call_executed=True,
                siphy_model_call_count=1,
                siphy_api_attempt_count=1,
                error=str(exc),
                stage1_mode=mode,
                stage1_requested_keys=siphy_cues,
                stage1_system_prompt=sys_prompt,
            )
            self._record(result)
            return result

        result = SiPhyStage1Result(
            object_key=object_key,
            condition_id=condition_id,
            material=parsed.get("material"),
            density_kgm3=parsed.get("density_kgm3"),
            youngs_gpa=None,
            raw_siphy_output={
                "stage1_mode": mode,
                "requested_keys": list(siphy_cues),
                "parsed": parsed,
                "raw_text": resp.text,
            },
            crop_image_path=str(crop_image_path),
            model=self.model,
            provider=PROVIDER,
            siphy_input_tokens=resp.usage.input_tokens,
            siphy_output_tokens=resp.usage.output_tokens,
            siphy_total_tokens=resp.usage.total_tokens,
            siphy_call_executed=True,
            siphy_model_call_count=1,
            siphy_api_attempt_count=resp.usage.api_attempt_count,
            stage1_mode=mode,
            stage1_requested_keys=siphy_cues,
            stage1_system_prompt=sys_prompt,
        )
        self._record(result)
        return result


SiPhyStage1Runner = SelectiveSiPhyStage1Runner
# Alias used by CLI dry-run feasibility helper name from v5
static_gemini_only_feasibility = static_openai_feasibility
