"""Stage 1: production SiPhyBackend — one independent call per condition.

Uses ``tuj.m3_grounding.siphy_backend.SiPhyBackend.estimate`` without modifying
production code. C2–C7 each execute their own SiPhy VLM call (no cross-condition
cache reuse). Results are logged per (object, condition) for analysis only.

Important:
- Image-only input to SiPhy (no M1 bbox, no GT, no object class label to the LLM).
- ``cls_hint`` is required by the PropertyBackend interface but is NOT sent to the
  VLM in ``_propose``; we always pass a neutral placeholder.
- ``points_mm`` is omitted so production ``shell_mass_integral`` is not used for
  Stage-1 cues (mass/mu final answers come from Gemini Stage 2).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tuj.m3_grounding.siphy_backend import SiPhyBackend

# Neutral interface hint - never an object class / name. Not used by SiPhy VLM prompt.
NEUTRAL_CLS_HINT = "unknown_object"


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
    siphy_input_tokens: int | None = None
    siphy_output_tokens: int | None = None
    siphy_total_tokens: int | None = None
    siphy_call_executed: bool = True
    siphy_model_call_count: int = 1
    siphy_cache_hit: bool = False  # always False; deprecated reuse path removed
    error: str | None = None

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
            "material": self.material,
            "density_kgm3": self.density_kgm3,
            "youngs_gpa": self.youngs_gpa,
            "raw_siphy_output": self.raw_siphy_output,
            "crop_image_path": self.crop_image_path,
            "model": self.model,
            "token_usage": {
                "input_tokens": self.siphy_input_tokens,
                "output_tokens": self.siphy_output_tokens,
                "total_tokens": self.siphy_total_tokens,
            },
            "siphy_call_executed": self.siphy_call_executed,
            "siphy_model_call_count": self.siphy_model_call_count,
            "siphy_cache_hit": False,
            "error": self.error,
        }


# Backward-compatible alias for older imports
SiPhyCacheEntry = SiPhyStage1Result


class _UsageCapturingClient:
    """Thin wrapper around OpenAI client to record last chat completion usage."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.last_usage: Any = None

    @property
    def chat(self) -> Any:
        return _ChatProxy(self)


class _ChatProxy:
    def __init__(self, owner: _UsageCapturingClient):
        self._owner = owner

    @property
    def completions(self) -> Any:
        return _CompletionsProxy(self._owner)


class _CompletionsProxy:
    def __init__(self, owner: _UsageCapturingClient):
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        resp = self._owner._inner.chat.completions.create(**kwargs)
        self._owner.last_usage = getattr(resp, "usage", None)
        return resp


def _usage_triplet(usage: Any) -> tuple[int | None, int | None, int | None]:
    if usage is None:
        return None, None, None
    inp = getattr(usage, "prompt_tokens", None)
    out = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    return (
        int(inp) if inp is not None else None,
        int(out) if out is not None else None,
        int(total) if total is not None else None,
    )


def _load_crop_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


@dataclass
class SiPhyStage1Runner:
    """Independent SiPhy calls per (object, condition). Log-only persistence."""

    out_path: Path
    dry_run: bool = False
    model: str = "gpt-4o-mini"
    repo_root: Path | None = None
    # object_key -> condition_id -> result  (log store, never used as inference cache)
    _log: dict[str, dict[str, SiPhyStage1Result]] = field(default_factory=dict)
    _backend: SiPhyBackend | None = None
    _usage_client: _UsageCapturingClient | None = None

    def __post_init__(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.out_path.exists():
            raw = json.loads(self.out_path.read_text(encoding="utf-8"))
            # Support new condition-keyed log; ignore legacy object-keyed cache for reuse.
            objects = raw.get("objects") or {}
            for obj_key, by_cond in objects.items():
                if not isinstance(by_cond, dict):
                    continue
                # Legacy flat object entry (no condition nesting) -> do not reload as cache
                if "material" in by_cond and "condition_id" not in by_cond and "C2" not in by_cond:
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
                        youngs_gpa=val.get("youngs_gpa"),
                        raw_siphy_output=val.get("raw_siphy_output") or {},
                        crop_image_path=val.get("crop_image_path") or "",
                        model=val.get("model") or self.model,
                        siphy_input_tokens=(val.get("token_usage") or {}).get("input_tokens"),
                        siphy_output_tokens=(val.get("token_usage") or {}).get("output_tokens"),
                        siphy_total_tokens=(val.get("token_usage") or {}).get("total_tokens"),
                        siphy_call_executed=bool(val.get("siphy_call_executed", True)),
                        siphy_model_call_count=int(val.get("siphy_model_call_count") or 1),
                        siphy_cache_hit=False,
                        error=val.get("error"),
                    )

    def save(self) -> None:
        payload = {
            "experiment": "experiment1_v4",
            "note": (
                "Condition-specific SiPhy Stage-1 LOG only (not an inference cache). "
                "Each C2-C7 unit runs its own SiPhyBackend.estimate call. "
                "mass/mu from local integrals are NOT used as final predictions."
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

    def _ensure_backend(self) -> SiPhyBackend:
        if self._backend is not None:
            return self._backend
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        root = self.repo_root
        if not key and root is not None:
            self._backend = SiPhyBackend(
                api_key=None, model=self.model, repo_root=str(root), client=None
            )
            if getattr(self._backend, "client", None) is not None:
                self._usage_client = _UsageCapturingClient(self._backend.client)
                self._backend.client = self._usage_client  # type: ignore[assignment]
            return self._backend

        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY required for live SiPhy Stage 1 "
                "(or provide repo_root with my_api_key.py)."
            )
        real = OpenAI(api_key=key)
        self._usage_client = _UsageCapturingClient(real)
        self._backend = SiPhyBackend(
            api_key=key, model=self.model, client=self._usage_client, repo_root=None
        )
        return self._backend

    def run(
        self,
        object_key: str,
        condition_id: str,
        crop_image_path: Path,
    ) -> SiPhyStage1Result:
        """Always execute a fresh SiPhy call for this condition (never reuse)."""
        if self.dry_run:
            # Unique placeholder per condition so paired conditions are NOT forced equal.
            result = SiPhyStage1Result(
                object_key=object_key,
                condition_id=condition_id,
                material=f"__siphy_material_{condition_id}__",
                density_kgm3=float(-100 - ord(condition_id[-1])),  # distinct sentinel
                youngs_gpa=None,
                raw_siphy_output={
                    "dry_run": True,
                    "condition_id": condition_id,
                    "note": "independent placeholder; SiPhyBackend.estimate not executed",
                },
                crop_image_path=str(crop_image_path),
                model=self.model,
                siphy_input_tokens=None,
                siphy_output_tokens=None,
                siphy_total_tokens=None,
                siphy_call_executed=False,
                siphy_model_call_count=0,
                siphy_cache_hit=False,
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
                siphy_call_executed=False,
                siphy_model_call_count=0,
                siphy_cache_hit=False,
                error=f"crop missing: {crop_image_path}",
            )
            self._record(result)
            return result

        backend = self._ensure_backend()
        if self._usage_client is not None:
            self._usage_client.last_usage = None
        crop = _load_crop_rgb(crop_image_path)
        props = backend.estimate(crop, NEUTRAL_CLS_HINT, points_mm=None)
        inp = out = total = None
        if self._usage_client is not None:
            inp, out, total = _usage_triplet(self._usage_client.last_usage)

        raw = json.loads(json.dumps(props, default=str))
        result = SiPhyStage1Result(
            object_key=object_key,
            condition_id=condition_id,
            material=str(props.get("material")) if props.get("material") is not None else None,
            density_kgm3=(
                float(props["density_kgm3"]) if props.get("density_kgm3") is not None else None
            ),
            youngs_gpa=(
                float(props["youngs_gpa"]) if props.get("youngs_gpa") is not None else None
            ),
            raw_siphy_output=raw,
            crop_image_path=str(crop_image_path),
            model=self.model,
            siphy_input_tokens=inp,
            siphy_output_tokens=out,
            siphy_total_tokens=total,
            siphy_call_executed=True,
            siphy_model_call_count=1,
            siphy_cache_hit=False,
        )
        self._record(result)
        return result


# Alias kept so older call sites naming "Cache" still import; behavior is non-caching.
SiPhyStage1Cache = SiPhyStage1Runner
