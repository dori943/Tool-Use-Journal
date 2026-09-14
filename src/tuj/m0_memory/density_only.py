"""Image-only density inference used by M0 cross-task retrieval.

Lightweight C3 path: crop → few material hypotheses (density range + confidence)
→ same aggregation as Full SiPhy (``aggregate_density_kgm3``).

Unlike ``SiPhyBackend.estimate``, this does not ask for thickness, mass,
friction, Young's modulus, caption, or object identity — only the density cue.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from tuj.m1_scene.siphy_backend import (
    K_MATERIALS,
    SiPhyBackend,
    _parse_range,
    _to_b64_png,
    aggregate_density_kgm3,
)


DENSITY_ONLY_PROMPT = """You will be given an image of an object (background masked to black). Propose %d materials that the object might be made of. For each material give only: its mass density (in kg/m^3), and on a scale from 0 to 10 how likely it is that this object is made of that material. You may provide a range low-high of values instead of a single value for density. Do not use or invent an object name, class, caption, or product identity. Do not estimate thickness, mass, friction, or Young's modulus. Do not include coatings like "paint" in your answer.

Format Requirement:
You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
{
    "materials": [
        {"name": material1, "density_kgm3": "low-high", "confidence_0_10": number},
        ...
    ]
}
Do not include any other text in your answer. Do not include unnecessary words besides the material in the material name.
""" % K_MATERIALS


@dataclass(frozen=True)
class DensityOnlyResult:
    density_kgm3: float | None
    llm_called: bool
    token_usage: dict | None = None
    error: str | None = None
    materials_topk: list | None = None
    material_committed: bool | None = None
    top1_gap: float | None = None


def _parse_density_hypotheses(raw: dict) -> list[dict]:
    if set(raw.keys()) != {"materials"}:
        raise ValueError(f"unexpected density-only keys: {sorted(raw)}")
    mats = []
    for m in raw["materials"]:
        allowed = {"name", "density_kgm3", "confidence_0_10"}
        if not allowed.issuperset(m.keys()):
            raise ValueError(f"unexpected material keys: {sorted(m)}")
        dens = _parse_range(m["density_kgm3"])
        if not (dens[0] > 0 and dens[1] > 0):
            raise ValueError("density range must be positive")
        mats.append({
            "name": str(m["name"]).lower(),
            "density": dens,
            "confidence": max(float(m.get("confidence_0_10", 0)), 0.0),
        })
    if not mats:
        raise ValueError("materials empty")
    return mats


class DensityOnlyBackend:
    """One-call, image-only selective backend sharing production provider setup."""

    def __init__(self, api_key=None, model="gpt-4o-mini", client=None,
                 repo_root=None, seed=100):
        base = SiPhyBackend(api_key=api_key, model=model, client=client,
                            repo_root=repo_root, seed=seed)
        self.client, self.model = base.client, base.model
        self._supports_seed = base._supports_seed
        # Hypothesis JSON is larger than a single scalar; keep well below Full SiPhy.
        self._max_tokens = 400 if getattr(base, "_is_gemini", False) else 256

    def infer(self, crop_rgb) -> DensityOnlyResult:
        if crop_rgb is None:
            return DensityOnlyResult(None, False, error="crop image missing")
        kwargs = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": DENSITY_ONLY_PROMPT},
                {"role": "user", "content": [{"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{_to_b64_png(crop_rgb)}"}}]},
            ],
        }
        if self._supports_seed:
            kwargs["seed"] = 100
        try:
            response = self.client.chat.completions.create(**kwargs)
            raw = (response.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
            mats = _parse_density_hypotheses(json.loads(raw))
            agg = aggregate_density_kgm3(mats)
            usage = getattr(response, "usage", None)
            tokens = None if usage is None else {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            topk = [
                {"name": m["name"], "prob": round(float(p), 3),
                 "density_kgm3": list(m["density"])}
                for m, p in zip(mats, agg["probs_raw"])
            ]
            return DensityOnlyResult(
                agg["density_kgm3"], True, tokens,
                materials_topk=topk,
                material_committed=agg["committed"],
                top1_gap=round(agg["gap"], 3),
            )
        except Exception as exc:  # retrieval failure safely falls through to full M3
            return DensityOnlyResult(None, True, error=str(exc))
