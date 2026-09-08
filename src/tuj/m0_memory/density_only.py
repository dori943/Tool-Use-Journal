"""Image-only density inference used by M0 cross-task retrieval.

This deliberately is not ``SiPhyBackend.estimate``: retrieval needs one scalar
cue and must not pay for, or accidentally infer, the full M3 property bundle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from tuj.m3_grounding.siphy_backend import SiPhyBackend, _to_b64_png


DENSITY_ONLY_PROMPT = """You will be given an image of an object (background masked to black). Estimate only its bulk mass density in kg/m^3. Do not use or invent an object name, class, caption, or product identity. Do not estimate material, mass, friction, Young's modulus, or thickness.

Return ONLY this JSON object:
{"density_kgm3": number}
"""


@dataclass(frozen=True)
class DensityOnlyResult:
    density_kgm3: float | None
    llm_called: bool
    token_usage: dict | None = None
    error: str | None = None


class DensityOnlyBackend:
    """One-call, image-only selective backend sharing production provider setup."""

    def __init__(self, api_key=None, model="gpt-4o-mini", client=None,
                 repo_root=None, seed=100):
        base = SiPhyBackend(api_key=api_key, model=model, client=client,
                            repo_root=repo_root, seed=seed)
        self.client, self.model = base.client, base.model
        self._supports_seed = base._supports_seed
        self._max_tokens = 128

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
            parsed = json.loads(raw)
            if set(parsed) != {"density_kgm3"}:
                raise ValueError(f"unexpected density-only keys: {sorted(parsed)}")
            density = float(parsed["density_kgm3"])
            if not (density > 0):
                raise ValueError("density must be positive")
            usage = getattr(response, "usage", None)
            tokens = None if usage is None else {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            return DensityOnlyResult(density, True, tokens)
        except Exception as exc:  # retrieval failure safely falls through to full M3
            return DensityOnlyResult(None, True, error=str(exc))
