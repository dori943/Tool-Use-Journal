from __future__ import annotations

import json
from pathlib import Path

import pytest

from .adapters import AdapterError, ConditionAdapter, OracleAdapter, SharedPredictionCache, load_observation


class RecordingProvider:
    model_version = "test-provider"
    temperature = 0.0

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def predict(self, *, prompt, image_path, input_id):
        self.calls.append({"prompt": prompt, "image_path": image_path, "input_id": input_id})
        return dict(self.payload)


def _observation(tmp_path: Path):
    path = tmp_path / "obs.json"
    path.write_text(json.dumps({
        "sample_id": "sim_001", "object_name": "sim_box_light",
        "visible_geometry_mm": {"extents": [45, 45, 80], "opening_widths": [80, 80]},
        "visible_support_context": {"table": "sim_table"},
        "affordance_labels": ["side_grasp", "top_flat"],
    }), encoding="utf-8")
    return load_observation(path)


def test_name_only_prompt_and_envelope_do_not_contain_gt(tmp_path: Path):
    provider = RecordingProvider({"mass_kg": 0.2})
    result = ConditionAdapter("name_only", provider).predict(_observation(tmp_path), "Mass_Acc")
    assert result["parse_status"] == "OK"
    assert "mass_gt" not in provider.calls[0]["prompt"]
    assert result["independent_model_call"] is True


def test_affordance_labels_are_required_and_passed_as_input(tmp_path: Path):
    obs = _observation(tmp_path)
    provider = RecordingProvider({"mass_kg": 0.2})
    result = ConditionAdapter("affordance_labels", provider).predict(obs, "Mass_Acc")
    assert result["parse_status"] == "OK"
    assert "side_grasp" in provider.calls[0]["prompt"]


def test_bundle_uses_one_provider_call_and_preserves_raw_response(tmp_path: Path):
    obs = _observation(tmp_path)
    provider = RecordingProvider({"mass_kg": 0.2, "feasible_by_ee": {"2F": True, "3F": True, "vac": True}})
    rows = ConditionAdapter("geometric_grounding", provider).predict_bundle(
        obs, ["Clearance_RelErr", "Feasibility_Acc"])
    assert len(provider.calls) == 1
    assert rows[0]["parse_status"] == "FAILED"  # clearance field was absent
    assert rows[1]["parse_status"] == "OK"
    assert rows[1]["raw_response"]["feasible_by_ee"]["vac"] is True


def test_geo_and_ours_share_siphy_source(tmp_path: Path):
    obs = _observation(tmp_path)
    provider = RecordingProvider({"mass_kg": 0.2})
    cache = SharedPredictionCache()
    siphy = ConditionAdapter("siphy_adopted", provider, shared_cache=cache)
    siphy.predict(obs, "Mass_Acc")
    geo = ConditionAdapter("geometric_grounding", provider, shared_cache=cache).predict(obs, "Mass_Acc")
    ours = ConditionAdapter("ours_full", provider, shared_cache=cache).predict(obs, "Mass_Acc")
    assert geo["shared_prediction"] and ours["shared_prediction"]
    assert geo["shared_prediction_source"] == "siphy_adopted"
    assert ours["independent_model_call"] is False
    assert len(provider.calls) == 1


def test_gt_fields_in_observation_are_rejected(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"sample_id": "x", "object_name": "x", "mass_gt_kg": 1,
                                "visible_geometry_mm": {"extents": [1, 1, 1]},
                                "visible_support_context": {}}), encoding="utf-8")
    with pytest.raises(AdapterError, match="GT_LEAKAGE_DETECTED"):
        load_observation(path)


def test_oracle_is_explicit_and_not_normal_condition():
    with pytest.raises(AdapterError):
        OracleAdapter()
    oracle = OracleAdapter(oracle=True)
    # The oracle adapter requires evaluator code to provide the GT row; this is
    # intentionally a separate call path rather than a normal provider.
    assert oracle.spec.gt_channel == "evaluator_only"
