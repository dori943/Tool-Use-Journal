"""Condition adapters for the simulator-only Table III D panel.

The adapters share one input envelope and one prediction envelope.  They do
not load evaluator-only GT and they do not manufacture predictions.  A real
provider is injected at runtime; tests use a tiny recording provider only to
validate prompt/input isolation and parsing.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


CONDITIONS = (
    "name_only", "affordance_labels", "siphy_adopted",
    "geometric_grounding", "ours_full", "gt_numerics",
)
FORBIDDEN_INPUT_KEYS = {
    "mass_gt_kg", "trial_success", "allowed_ee_ids", "all_feasible_ee_ids",
    "signed_margin_gt_mm", "gt_label", "critical_subset_member",
}


class AdapterError(ValueError):
    pass


class JsonProvider(Protocol):
    """Provider contract used by real VLM/API backends and local test doubles."""

    model_version: str
    temperature: float

    def predict(self, *, prompt: str, image_path: str | None, input_id: str) -> Any:
        ...


@dataclass(frozen=True)
class ProviderResponse:
    parsed: dict[str, Any]
    raw_response: str
    request_id: str | None = None


class SharedPredictionCache:
    """In-memory provenance cache used to prevent duplicate mass calls."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}

    def put(self, row: dict[str, Any]) -> None:
        self._rows[(str(row["input_id"]), str(row["metric"]))] = dict(row)

    def get(self, input_id: str, metric: str) -> dict[str, Any] | None:
        row = self._rows.get((str(input_id), str(metric)))
        return dict(row) if row is not None else None


@dataclass(frozen=True)
class AdapterSpec:
    condition_id: str
    condition_name: str
    input_fields: tuple[str, ...]
    active_modules: tuple[str, ...]
    prediction_source: str
    supports: tuple[str, ...]
    gt_channel: str = "none"


@dataclass
class Observation:
    input_id: str
    sample_id: str
    object_name: str
    source_path: Path
    payload: dict[str, Any]

    @property
    def image_path(self) -> str | None:
        value = self.payload.get("rgb_path") or self.payload.get("mass_crop_path")
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.source_path.parent / path
        return str(path)

    @property
    def friction_context_path(self) -> str | None:
        value = self.payload.get("friction_context_path") or self.payload.get("rgb_path")
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.source_path.parent / path
        return str(path)


def load_observation(path: Path) -> Observation:
    row = json.loads(path.read_text(encoding="utf-8"))
    leaked = sorted(FORBIDDEN_INPUT_KEYS.intersection(row))
    if leaked:
        raise AdapterError(f"GT_LEAKAGE_DETECTED:{','.join(leaked)}")
    required = ("sample_id", "object_name", "visible_geometry_mm", "visible_support_context")
    missing = [key for key in required if key not in row]
    if missing:
        raise AdapterError(f"OBSERVATION_SCHEMA_MISSING:{','.join(missing)}")
    return Observation(
        input_id=str(row.get("input_id", row["sample_id"])),
        sample_id=str(row["sample_id"]),
        object_name=str(row["object_name"]),
        source_path=path,
        payload=row,
    )


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _finite(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise AdapterError("NON_NUMERIC_VALUE")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError("NON_NUMERIC_VALUE") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise AdapterError("OUT_OF_RANGE")
    return result


def _bool(value: Any) -> bool:
    if type(value) is not bool:
        raise AdapterError("NON_BOOLEAN_VALUE")
    return value


def _geometry_text(obs: Observation) -> str:
    geom = obs.payload["visible_geometry_mm"]
    extents = geom.get("extents", [])
    opening = geom.get("opening_widths", [])
    return f"visible_extents_mm={extents}; visible_opening_widths_mm={opening}"


def _make_points(obs: Observation) -> np.ndarray:
    """Build an observation-side box surface point set for the existing SiPhy API.

    This derives points only from visible geometry in the input manifest.  It
    never reads the simulator GT file or hidden mass/trial fields.
    """
    ext = np.asarray(obs.payload["visible_geometry_mm"]["extents"], dtype=float)
    if ext.shape != (3,) or not np.all(np.isfinite(ext)) or np.any(ext <= 0):
        raise AdapterError("INVALID_VISIBLE_GEOMETRY")
    half = ext / 2.0
    grid = np.linspace(-1.0, 1.0, 8)
    points = []
    for a in grid:
        for b in grid:
            points.extend([
                [half[0], a * half[1], b * half[2]], [-half[0], a * half[1], b * half[2]],
                [a * half[0], half[1], b * half[2]], [a * half[0], -half[1], b * half[2]],
                [a * half[0], b * half[1], half[2]], [a * half[0], b * half[1], -half[2]],
            ])
    return np.asarray(points, dtype=float)


SPECS = {
    "name_only": AdapterSpec("name_only", "Name-only", ("object_name", "image_path"),
                              ("name_prompt",), "name_only", ("Mass_Acc", "Crit")),
    "affordance_labels": AdapterSpec("affordance_labels", "+ Affordance labels",
                                      ("object_name", "affordance_labels", "image_path"),
                                      ("affordance_prompt",), "affordance_labels", ("Mass_Acc", "Crit")),
    "siphy_adopted": AdapterSpec("siphy_adopted", "SiPhy (as adopted)",
                                  ("object_name", "image_path", "visible_geometry_mm"),
                                  ("SiPhyBackend", "shell_mass_integral"), "siphy_adopted",
                                  ("Mass_Acc", "Crit")),
    "geometric_grounding": AdapterSpec("geometric_grounding", "+ Geometric grounding (ours)",
                                        ("object_name", "image_path", "visible_geometry_mm", "visible_support_context"),
                                        ("SiPhyBackend", "M1_geometry", "clearance_normalization"),
                                        "siphy_adopted", ("Mass_Acc", "Clearance_RelErr", "Feasibility_Acc", "DA", "Crit")),
    "ours_full": AdapterSpec("ours_full", "Ours (full)",
                              ("object_name", "image_path", "friction_context_path", "visible_geometry_mm", "visible_support_context"),
                              ("SiPhyBackend", "M1_geometry", "visual_friction", "M4_downstream"),
                              "siphy_adopted", ("Mass_Acc", "Suction_Acc", "Suction_PF", "Clearance_RelErr", "Feasibility_Acc", "DA", "Crit")),
    "gt_numerics": AdapterSpec("gt_numerics", "GT numerics (oracle)", (), ("oracle_channel_only",),
                                "oracle", ("Mass_Acc", "Suction_Acc", "Suction_PF", "Clearance_RelErr", "Feasibility_Acc", "DA", "Crit"),
                                gt_channel="evaluator_only"),
}


class ConditionAdapter:
    def __init__(self, condition_id: str, provider: JsonProvider | None = None,
                 *, shared_cache: SharedPredictionCache | None = None):
        if condition_id not in SPECS or condition_id == "gt_numerics":
            raise AdapterError("GT_NUMERICS_REQUIRES_ORACLE_ADAPTER" if condition_id == "gt_numerics" else "UNKNOWN_CONDITION")
        self.spec = SPECS[condition_id]
        self.provider = provider
        self.shared_cache = shared_cache
        self.last_prompt: str | None = None

    def build_prompt(self, obs: Observation, metric: str) -> str:
        if metric not in self.spec.supports:
            raise AdapterError(f"METRIC_UNSUPPORTED:{self.spec.condition_id}:{metric}")
        base = [
            "Return JSON only.", f"condition={self.spec.condition_id}",
            f"object_name={obs.object_name}", f"metric={metric}",
        ]
        if self.spec.condition_id == "affordance_labels":
            labels = obs.payload.get("affordance_labels")
            if not isinstance(labels, list):
                raise AdapterError("AFFORDANCE_LABELS_MISSING")
            base.append(f"affordance_labels={labels}")
        if self.spec.condition_id in {"siphy_adopted", "geometric_grounding", "ours_full"}:
            base.append(_geometry_text(obs))
        if self.spec.condition_id in {"geometric_grounding", "ours_full"}:
            base.append(f"support_context={obs.payload['visible_support_context']}")
        if self.spec.condition_id == "ours_full":
            base.append("estimate static object-table combined friction from visual context; no trajectory dynamics")
        prompt = "\n".join(base)
        self.last_prompt = prompt
        return prompt

    def predict(self, obs: Observation, metric: str) -> dict[str, Any]:
        if self.provider is None:
            if self.shared_cache is None:
                raise AdapterError("MODEL_PROVIDER_NOT_CONFIGURED")
        if metric not in self.spec.supports:
            raise AdapterError(f"METRIC_UNSUPPORTED:{self.spec.condition_id}:{metric}")
        # Mass is explicitly shared by the protocol for +Geo and Ours.  A
        # missing SiPhy source is an error, never a silent duplicate call.
        if (metric == "Mass_Acc" and self.spec.prediction_source == "siphy_adopted"
                and self.spec.condition_id != "siphy_adopted"):
            if self.shared_cache is None:
                raise AdapterError("SHARED_SIPHY_PREDICTION_REQUIRED")
            source = self.shared_cache.get(obs.input_id, metric)
            if source is None:
                raise AdapterError("SHARED_SIPHY_PREDICTION_MISSING")
            source_prediction_id = source.get("prediction_id", f'{source["condition_id"]}:{source["input_id"]}:{source["metric"]}')
            return {
                **source,
                "condition_id": self.spec.condition_id,
                "shared_prediction": True,
                "shared_prediction_source": "siphy_adopted",
                "source_prediction_id": source_prediction_id,
                "independent_model_call": False,
            }
        prompt = self.build_prompt(obs, metric)
        image = obs.friction_context_path if metric in {"Suction_Acc", "Suction_PF", "Clearance_RelErr", "Feasibility_Acc", "DA"} else obs.image_path
        response = self.provider.predict(prompt=prompt, image_path=image, input_id=obs.input_id)
        if isinstance(response, ProviderResponse):
            raw = response.parsed
            raw_response: Any = response.raw_response
            request_id = response.request_id
        else:
            raw = response
            raw_response = response
            request_id = None
        if not isinstance(raw, dict):
            raise AdapterError("INVALID_FORMAT")
        value = self._parse_value(raw, metric)
        result = {
            "input_id": obs.input_id,
            "sample_id": obs.sample_id,
            "object_id": obs.payload.get("object_instance_id", obs.sample_id),
            "condition_id": self.spec.condition_id,
            "metric": metric,
            "value": value,
            "unit": self._unit(metric),
            "raw_response": raw_response,
            "request_id": request_id,
            "parse_status": "OK",
            "failure_reason": None,
            "prompt_sha256": _prompt_hash(prompt),
            "model_version": str(getattr(self.provider, "model_version", "unknown")),
            "temperature": float(getattr(self.provider, "temperature", 0.0)),
            "shared_prediction": self.spec.prediction_source == "siphy_adopted" and self.spec.condition_id != "siphy_adopted",
            "shared_prediction_source": self.spec.prediction_source if self.spec.condition_id != "siphy_adopted" else None,
            "independent_model_call": True,
        }
        if self.shared_cache is not None and self.spec.condition_id == "siphy_adopted":
            self.shared_cache.put(result)
        return result

    def predict_bundle(self, obs: Observation, metrics: list[str]) -> list[dict[str, Any]]:
        """One provider call for all requested outputs for this input.

        This is the cost-controlled execution path for D_SIM.  Missing fields
        become per-metric failures in the runner; one response is never silently
        reused as a different metric.
        """
        if not metrics:
            return []
        unsupported = sorted(set(metrics) - set(self.spec.supports))
        if unsupported:
            raise AdapterError(f"METRIC_UNSUPPORTED:{self.spec.condition_id}:{','.join(unsupported)}")
        shared_mass: dict[str, Any] | None = None
        if "Mass_Acc" in metrics and self.spec.prediction_source == "siphy_adopted" and self.spec.condition_id != "siphy_adopted":
            if self.shared_cache is None:
                raise AdapterError("SHARED_SIPHY_PREDICTION_REQUIRED")
            shared_mass = self.shared_cache.get(obs.input_id, "Mass_Acc")
            if shared_mass is None:
                raise AdapterError("SHARED_SIPHY_PREDICTION_MISSING")
        call_metrics = [m for m in metrics if not (m == "Mass_Acc" and shared_mass is not None)]
        raw: dict[str, Any] = {}
        raw_response: Any = None
        request_id = None
        if call_metrics:
            if self.provider is None:
                raise AdapterError("MODEL_PROVIDER_NOT_CONFIGURED")
            prompt = self.build_bundle_prompt(obs, call_metrics)
            image = obs.friction_context_path if any(m in {"Suction_Acc", "Suction_PF", "Clearance_RelErr", "Feasibility_Acc", "DA"} for m in call_metrics) else obs.image_path
            response = self.provider.predict(prompt=prompt, image_path=image, input_id=obs.input_id)
            if isinstance(response, ProviderResponse):
                raw, raw_response, request_id = response.parsed, response.raw_response, response.request_id
            else:
                raw, raw_response = response, response
            if not isinstance(raw, dict):
                raise AdapterError("INVALID_FORMAT")
        rows = []
        for metric in metrics:
            if metric == "Mass_Acc" and shared_mass is not None:
                row = {**shared_mass, "condition_id": self.spec.condition_id,
                       "shared_prediction": True, "shared_prediction_source": "siphy_adopted",
                       "source_prediction_id": shared_mass.get("prediction_id", f'{shared_mass["condition_id"]}:{shared_mass["input_id"]}'),
                       "independent_model_call": False}
                rows.append(row)
                continue
            try:
                value = self._parse_value(raw, metric)
                row = self._envelope(obs, metric, value, raw_response, request_id)
            except AdapterError as error:
                row = self._failure_envelope(obs, metric, raw_response, request_id, str(error))
            rows.append(row)
        return rows

    def build_bundle_prompt(self, obs: Observation, metrics: list[str]) -> str:
        base = [self.build_prompt(obs, metrics[0]), f"requested_metrics={metrics}",
                "Return one JSON object with only the requested fields."]
        fields = []
        if "Mass_Acc" in metrics or "Crit" in metrics:
            fields.append('"mass_kg": positive number')
        if "Clearance_RelErr" in metrics:
            fields.append('"clearance_mm": finite number')
        if "Suction_Acc" in metrics or "Suction_PF" in metrics:
            fields.append('"suction_pose_predictions": {"pose_A": boolean, "pose_B": boolean}')
        if "Feasibility_Acc" in metrics:
            fields.append('"feasible_by_ee": {"2F": boolean, "3F": boolean, "vac": boolean}')
        if "DA" in metrics:
            fields.append('"selected_ee": "2F" | "3F" | "vac" | "NO_FEASIBLE_EE"')
        return "\n".join(base + ["JSON fields:", *fields])

    def _envelope(self, obs: Observation, metric: str, value: Any,
                  raw_response: Any, request_id: str | None) -> dict[str, Any]:
        prompt = self.last_prompt or ""
        return {"input_id": obs.input_id, "sample_id": obs.sample_id,
                "object_id": obs.payload.get("object_instance_id", obs.sample_id),
                "condition_id": self.spec.condition_id, "metric": metric,
                "value": value, "unit": self._unit(metric), "raw_response": raw_response,
                "request_id": request_id, "parse_status": "OK", "failure_reason": None,
                "prompt_sha256": _prompt_hash(prompt),
                "model_version": str(getattr(self.provider, "model_version", "unknown")),
                "temperature": float(getattr(self.provider, "temperature", 0.0)),
                "shared_prediction": False, "shared_prediction_source": None,
                "independent_model_call": True}

    def _failure_envelope(self, obs: Observation, metric: str, raw_response: Any,
                          request_id: str | None, reason: str) -> dict[str, Any]:
        return {"input_id": obs.input_id, "sample_id": obs.sample_id,
                "object_id": obs.payload.get("object_instance_id", obs.sample_id),
                "condition_id": self.spec.condition_id, "metric": metric,
                "value": None, "unit": self._unit(metric), "raw_response": raw_response,
                "request_id": request_id, "parse_status": "FAILED", "failure_reason": reason,
                "prompt_sha256": _prompt_hash(self.last_prompt or ""),
                "model_version": str(getattr(self.provider, "model_version", "unknown")),
                "temperature": float(getattr(self.provider, "temperature", 0.0)),
                "shared_prediction": False, "shared_prediction_source": None,
                "independent_model_call": True}

    @staticmethod
    def _unit(metric: str) -> str:
        return {"Mass_Acc": "kg", "Crit": "kg", "Clearance_RelErr": "mm",
                "Suction_Acc": "bool", "Suction_PF": "bool", "Feasibility_Acc": "bool",
                "DA": "ee_id"}[metric]

    @staticmethod
    def _parse_value(raw: dict[str, Any], metric: str) -> Any:
        if metric in {"Mass_Acc", "Crit"}:
            return _finite(raw.get("mass_kg"), positive=True)
        if metric == "Clearance_RelErr":
            return _finite(raw.get("clearance_mm"))
        if metric in {"Suction_Acc", "Suction_PF"}:
            poses = raw.get("suction_pose_predictions")
            if not isinstance(poses, dict) or set(poses) != {"pose_A", "pose_B"}:
                raise AdapterError("POSE_LEVEL_PREDICTION_MISSING")
            return {key: _bool(poses[key]) for key in ("pose_A", "pose_B")}
        if metric == "Feasibility_Acc":
            values = raw.get("feasible_by_ee")
            if not isinstance(values, dict) or set(values) != {"2F", "3F", "vac"}:
                raise AdapterError("MISSING_VALUE")
            return {key: _bool(values[key]) for key in ("2F", "3F", "vac")}
        if metric == "DA":
            value = raw.get("selected_ee")
            if value not in {"2F", "3F", "vac", "NO_FEASIBLE_EE"}:
                raise AdapterError("OUT_OF_RANGE")
            return value
        raise AdapterError(f"METRIC_UNSUPPORTED:{metric}")


class SiPhyAdapter(ConditionAdapter):
    """Adapter using the repository's existing SiPhyBackend when supplied."""

    def __init__(self, provider: JsonProvider | None = None, *, backend: Any | None = None):
        super().__init__("siphy_adopted", provider)
        self.backend = backend

    def predict_mass_with_backend(self, obs: Observation) -> dict[str, Any]:
        if self.backend is None:
            raise AdapterError("SIPHY_BACKEND_NOT_CONFIGURED")
        image = obs.image_path
        if image is None:
            raise AdapterError("RGB_INPUT_REQUIRED_FOR_SIPHY")
        props = self.backend.estimate(image, obs.object_name, points_mm=_make_points(obs))
        mass = _finite(props.get("mass_kg"), positive=True)
        return {"mass_kg": mass, "raw_backend_output": props}


class OracleAdapter:
    """Explicit evaluator-only channel; never registered as a normal provider."""

    def __init__(self, *, oracle: bool = False):
        if not oracle:
            raise AdapterError("ORACLE_CHANNEL_REQUIRES_EXPLICIT_FLAG")
        self.spec = SPECS["gt_numerics"]

    def predict(self, *, observation: Observation, gt_row: dict[str, Any], metric: str) -> dict[str, Any]:
        if metric not in self.spec.supports:
            raise AdapterError(f"METRIC_UNSUPPORTED:{metric}")
        if not isinstance(gt_row, dict) or gt_row.get("sample_id") != observation.sample_id:
            raise AdapterError("ORACLE_SAMPLE_MISMATCH")
        field = {"Mass_Acc": "mass_gt_kg", "Crit": "mass_gt_kg",
                 "Clearance_RelErr": ("clearance_gt", "signed_margin_gt_mm")}.get(metric)
        if field is None:
            raise AdapterError("ORACLE_METRIC_FIELD_NOT_REGISTERED")
        value = gt_row[field] if isinstance(field, str) else gt_row[field[0]][field[1]]
        return {"input_id": observation.input_id, "condition_id": "gt_numerics", "metric": metric,
                "value": value, "unit": ConditionAdapter._unit(metric), "oracle_channel": "evaluator_only",
                "shared_prediction": False, "independent_model_call": False}


def adapter_audit() -> list[dict[str, Any]]:
    return [{"condition_id": spec.condition_id, "condition": spec.condition_name,
             "input_fields": list(spec.input_fields), "active_modules": list(spec.active_modules),
             "prediction_source": spec.prediction_source, "supports": list(spec.supports),
             "status": "ORACLE_ISOLATED" if spec.condition_id == "gt_numerics" else "INTERFACE_IMPLEMENTED_NEEDS_PROVIDER",
             "gt_leakage": "forbidden_input_keys_rejected"} for spec in SPECS.values()]
