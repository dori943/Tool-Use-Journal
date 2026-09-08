"""C3 Density-only: material-hypothesis cue + Full SiPhy aggregation."""
from __future__ import annotations

import json

from tuj.m0_memory.density_only import (
    DENSITY_ONLY_PROMPT,
    DensityOnlyBackend,
    _parse_density_hypotheses,
)
from tuj.m1_scene.siphy_backend import TOP1_GAP, aggregate_density_kgm3


def test_c3_prompt_is_lightweight_and_identity_free():
    assert "density_kgm3" in DENSITY_ONLY_PROMPT
    assert "confidence_0_10" in DENSITY_ONLY_PROMPT
    assert "thickness" in DENSITY_ONLY_PROMPT  # forbidden, must be mentioned as do-not
    assert "Do not estimate thickness" in DENSITY_ONLY_PROMPT
    assert "Young" in DENSITY_ONLY_PROMPT
    assert "description" not in DENSITY_ONLY_PROMPT.lower()
    assert "caption" in DENSITY_ONLY_PROMPT  # forbidden wording
    assert "Do not use or invent an object name" in DENSITY_ONLY_PROMPT


def test_aggregate_density_matches_full_siphy_mixture_and_top1_commit():
    import numpy as np

    # Close confidences → mixture (gap ≤ TOP1_GAP)
    mats = [
        {"name": "wood", "density": (600.0, 800.0), "confidence": 4.0},
        {"name": "plastic", "density": (900.0, 1200.0), "confidence": 3.5},
        {"name": "metal", "density": (7500.0, 8000.0), "confidence": 2.5},
    ]
    agg = aggregate_density_kgm3(mats)
    assert agg["committed"] is False
    conf = np.array([4.0, 3.5, 2.5])
    probs = conf / conf.sum()
    expected = round(float(probs @ np.array([700.0, 1050.0, 7750.0])), 1)
    assert agg["density_kgm3"] == expected
    assert agg["gap"] <= TOP1_GAP

    # Top-1 commit when gap is large
    committed_mats = [
        {"density": (600.0, 800.0), "confidence": 9.0},
        {"density": (7500.0, 8000.0), "confidence": 1.0},
    ]
    agg2 = aggregate_density_kgm3(committed_mats)
    assert agg2["committed"] is True
    assert agg2["density_kgm3"] == 700.0


def test_parse_rejects_extra_property_fields():
    try:
        _parse_density_hypotheses({
            "materials": [{
                "name": "wood", "density_kgm3": "600-800",
                "confidence_0_10": 8, "thickness_cm": "0.2-1.0",
            }]
        })
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unexpected material keys" in str(exc)


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    def __init__(self):
        self.prompt_tokens = 1200
        self.completion_tokens = 80
        self.total_tokens = 1280


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class _Completions:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _Resp(self._content)


class _Chat:
    def __init__(self, content):
        self.completions = _Completions(content)


class _Client:
    def __init__(self, content):
        self.chat = _Chat(content)
        self.base_url = "https://api.openai.com/v1"


def test_density_only_backend_aggregates_like_siphy():
    import numpy as np

    payload = {
        "materials": [
            {"name": "Wood", "density_kgm3": "600-800", "confidence_0_10": 9},
            {"name": "Plastic", "density_kgm3": "900-1200", "confidence_0_10": 1},
        ]
    }
    client = _Client(json.dumps(payload))
    backend = DensityOnlyBackend(client=client, model="gpt-4o-mini")
    result = backend.infer(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.error is None
    assert result.llm_called is True
    assert result.density_kgm3 == 700.0  # committed top-1 wood mid
    assert result.material_committed is True
    assert result.token_usage["total_tokens"] == 1280
    assert result.materials_topk[0]["name"] == "wood"
    # Still image-only: system prompt is density-only, user has only image_url
    msgs = client.chat.completions.last_kwargs["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == DENSITY_ONLY_PROMPT
    assert all(part.get("type") == "image_url" for part in msgs[1]["content"])
