"""Unified, section-preserving persistence for M0 Knowledge Memory."""
from __future__ import annotations

import copy
import json
from pathlib import Path


SCHEMA_VERSION = "1.0"


class UnifiedMemoryStore:
    """Own the complete memory document and preserve sections it does not edit."""

    def __init__(self, path):
        if path is None:
            raise ValueError("unified memory path is required")
        self.path = Path(path)
        self.document = self._empty_document()
        self.load()

    @staticmethod
    def _empty_document():
        return {
            "schema_version": SCHEMA_VERSION,
            "object_knowledge": {"objects": {}},
            "failure_recovery_experience": {"experiences": []},
        }

    def load(self):
        if not self.path.exists():
            self.document = self._empty_document()
            return self.document
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("memory root must be a JSON object")
        doc = copy.deepcopy(raw)  # unknown root sections are intentionally preserved
        version = doc.get("schema_version")
        if version not in (None, SCHEMA_VERSION):
            raise ValueError(f"unsupported memory schema: {version}")
        doc["schema_version"] = SCHEMA_VERSION
        doc.setdefault("failure_recovery_experience", {"experiences": []})
        doc["failure_recovery_experience"].setdefault("experiences", [])
        if "object_knowledge" not in doc:
            doc["object_knowledge"] = {"objects": {}}
        doc["object_knowledge"].setdefault("objects", {})

        legacy = doc.pop("objects", None)
        if isinstance(legacy, dict):
            from .object_knowledge import _memory_key, props_to_entry
            target = doc["object_knowledge"]["objects"]
            for object_id, wrapper in legacy.items():
                if not isinstance(wrapper, dict) or not isinstance(wrapper.get("props"), dict):
                    continue
                key = _memory_key("legacy", object_id)
                if key in target:
                    continue
                entry = props_to_entry(
                    wrapper["props"], None, object_id,
                    source_method=wrapper.get("source", "m3"))
                meta = entry["metadata"]
                meta.update({
                    "source_task": None,
                    "episodes_seen": int(wrapper.get("episodes_seen", 1)),
                    "updated_at": wrapper.get("updated_at"),
                    "stale": bool(wrapper.get("stale", False)),
                    "stale_reason": wrapper.get("stale_reason"),
                    "stage": int(wrapper.get("stage", 0)),
                    "legacy_key": object_id,
                })
                target[key] = entry
        self.document = doc
        return self.document

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.document, ensure_ascii=False, indent=2), encoding="utf-8")
