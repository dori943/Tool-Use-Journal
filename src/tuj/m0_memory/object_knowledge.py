"""M0 Object Knowledge persistence and task-aware retrieval."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from .memory_store import UnifiedMemoryStore


SCHEMA_VERSION = "1.0"
PROVISIONAL_BBOX_RELATIVE_THRESHOLD = 0.25
PROVISIONAL_DENSITY_RELATIVE_THRESHOLD = 0.20


def _props_stage(props: dict) -> int:
    return max(int(props.get("mass_stage", 0)),
               int((props.get("mu") or {}).get("stage", 0)))


def _finite_positive(value):
    try:
        value = float(value)
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _memory_key(task_id: str, object_id: str) -> str:
    # JSON pointer-like escaping makes the separator unambiguous.
    esc = lambda s: str(s).replace("%", "%25").replace(":", "%3A")
    return f"{esc(task_id)}::{esc(object_id)}"


def props_to_entry(props: dict, task_id: str, object_id: str, *, episode=None,
                   source_method="visual", old=None) -> dict:
    geometry = props.get("geometry") or {}
    mu = props.get("mu") or {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "identity": {"object_id": object_id, "caption": props.get("caption")},
        # 0908: geometry는 통째로 보존한다. 이전엔 6개 필드만 골라 저장해 footprint_mm,
        # seal_patch_rms_mm, height_mm이 빠졌고, memory hit 뒤 EE 판정(plate vac → 2F/3F,
        # lid vac → 없음)과 flat_face(전부 false)가 첫 측정과 달라졌다 (0908 8태스크 실행).
        # 기존 memory.json 항목은 필드가 없으므로 지우고 다시 쌓아야 한다.
        "geometry": {
            **copy.deepcopy(geometry),
            "height_z_mm": geometry.get("height_z_mm", geometry.get("height_mm")),
        },
        "physical_properties": {
            "material": {"value": props.get("material"), "confidence": props.get("confidence")},
            "density_kgm3": {"value": props.get("density_kgm3")},
            "mass_kg": {"value": props.get("mass_kg"), "range": props.get("mass_range_kg")},
            "youngs_gpa": {"value": props.get("youngs_gpa")},
            "friction": {"value": mu.get("mu"), "stage": mu.get("stage"),
                         "material_basis": mu.get("material"),
                         "surface_rms_mm": mu.get("rms_mm")},
        },
        "material_hypotheses": copy.deepcopy(props.get("materials_topk") or []),
        "metadata": {
            "source_module": "M3", "source_method": source_method,
            "source_task": task_id, "source_episode": episode,
            "episodes_seen": int((old or {}).get("metadata", {}).get("episodes_seen", 0)) + 1,
            "updated_at": now, "stale": False, "stale_reason": None,
        },
    }


def entry_to_props(entry: dict) -> dict:
    p = entry["physical_properties"]
    friction = p["friction"]
    out = {
        "geometry": copy.deepcopy(entry["geometry"]),
        "material": p["material"].get("value"),
        "density_kgm3": p["density_kgm3"].get("value"),
        "mass_kg": p["mass_kg"].get("value"),
        "youngs_gpa": p["youngs_gpa"].get("value"),
        "mu": {"mu": friction.get("value"), "stage": friction.get("stage"),
               "material": friction.get("material_basis"),
               "rms_mm": friction.get("surface_rms_mm")},
        "confidence": p["material"].get("confidence"),
    }
    if p["mass_kg"].get("range") is not None:
        out["mass_range_kg"] = copy.deepcopy(p["mass_kg"]["range"])
    if entry.get("material_hypotheses"):
        out["materials_topk"] = copy.deepcopy(entry["material_hypotheses"])
    if entry.get("identity", {}).get("caption") is not None:
        out["caption"] = entry["identity"]["caption"]
    return out


class ObjectKnowledgeManager:
    def __init__(self, path=None, *, store=None,
                 bbox_relative_threshold=PROVISIONAL_BBOX_RELATIVE_THRESHOLD,
                 density_relative_threshold=PROVISIONAL_DENSITY_RELATIVE_THRESHOLD):
        self.store = store or UnifiedMemoryStore(path)
        self.path = self.store.path
        self.bbox_relative_threshold = float(bbox_relative_threshold)
        self.density_relative_threshold = float(density_relative_threshold)
        self.data = self.store.document

    @property
    def objects(self):
        return self.data["object_knowledge"]["objects"]

    def load(self):
        self.store.load()
        self.data = self.store.document
        return self.data

    def save(self):
        self.store.save()

    def update_entry(self, task_id, object_id, props, *, episode=None,
                     source_method="visual", stage=None):
        key = _memory_key(task_id, object_id)
        old = self.objects.get(key)
        new_stage = _props_stage(props) if stage is None else int(stage)
        old_stage = int((old or {}).get("metadata", {}).get("stage", 0))
        if old is not None and new_stage < old_stage:
            old["metadata"]["episodes_seen"] = int(old["metadata"].get("episodes_seen", 0)) + 1
            return old
        self.objects[key] = props_to_entry(
            props, task_id, object_id, episode=episode,
            source_method=source_method, old=old)
        self.objects[key]["metadata"]["stage"] = new_stage
        return self.objects[key]

    # Public convenience name for direct ObjectKnowledgeManager users.
    update = update_entry

    def update_from_m3(self, m3, task_id, *, episode=None):
        raw = json.loads(Path(m3).read_text(encoding="utf-8")) if isinstance(m3, (str, Path)) else m3
        found, skipped = {}, []
        for response in raw.get("responses", []):
            object_id = response.get("node_id")
            if not object_id or not isinstance(response.get("geometry"), dict):
                continue
            required = ("material", "density_kgm3", "mass_kg", "youngs_gpa", "mu")
            if not any(k in response for k in required):
                skipped.append(object_id); continue
            found[object_id] = response
        for object_id, response in found.items():
            self.update_entry(task_id, object_id, response, episode=episode)
        self.save()
        return {"updated": sorted(found), "skipped": sorted(set(skipped))}

    def lookup_exact(self, task_id, object_id):
        for key, entry in self.objects.items():
            if (entry.get("identity", {}).get("object_id") == object_id
                    and entry.get("metadata", {}).get("source_task") == task_id):
                if entry.get("metadata", {}).get("stale"):
                    return None, key, "STALE_MEMORY"
                return copy.deepcopy(entry), key, None
        return None, None, None

    @staticmethod
    def _bbox_difference(query_bbox, memory_extents):
        if not query_bbox or not memory_extents or len(query_bbox) != 3 or len(memory_extents) != 3:
            return None
        q = [_finite_positive(x) for x in query_bbox]
        m = [_finite_positive(x) for x in memory_extents]
        if any(x is None for x in q + m):
            return None          # 비정상 extents 항목은 후보에서 제외 (sorted 전에 걸러야 함)
        q, m = sorted(q), sorted(m)
        axis_diffs = [abs(a - b) / max(abs(b), 1e-9) for a, b in zip(q, m)]
        return max(axis_diffs), axis_diffs

    def filter_bbox(self, task_id, bbox_mm):
        candidates = []
        for key, entry in self.objects.items():
            meta = entry.get("metadata", {})
            if meta.get("source_task") == task_id or meta.get("stale"):
                continue
            diff = self._bbox_difference(bbox_mm, entry.get("geometry", {}).get("extents_mm"))
            if diff is None:
                continue
            maximum, axes = diff
            if maximum <= self.bbox_relative_threshold:
                candidates.append({"memory_entry_key": key,
                    "source_task": meta.get("source_task"),
                    "object_id": entry.get("identity", {}).get("object_id"),
                    "bbox_max_relative_difference": maximum,
                    "bbox_axis_relative_differences": axes})
        return sorted(candidates, key=lambda x: x["bbox_max_relative_difference"])

    def retrieve_by_density(self, candidates, query_density):
        qd = _finite_positive(query_density)
        ranked = []
        if qd is None:
            return None, ranked
        for candidate in candidates:
            entry = self.objects[candidate["memory_entry_key"]]
            md = _finite_positive(entry["physical_properties"]["density_kgm3"].get("value"))
            if md is None:
                continue
            ranked.append({**candidate, "density_kgm3": md,
                           "relative_difference": abs(qd - md) / max(abs(md), 1e-9)})
        ranked.sort(key=lambda x: x["relative_difference"])
        return (ranked[0] if ranked else None), ranked

    def lookup_or_retrieve(self, task_id, object_id, bbox_mm, crop_rgb, density_infer):
        debug = {"query_task": task_id, "query_object_id": object_id,
                 "bbox_threshold": self.bbox_relative_threshold,
                 "density_threshold": self.density_relative_threshold,
                 "threshold_status": "provisional", "c3_llm_called": False,
                 "c3_token_usage": None, "full_m3_called": False,
                 "full_m3_skipped": False}
        exact, key, exact_reason = self.lookup_exact(task_id, object_id)
        if exact is not None:
            return entry_to_props(exact), debug | {"lookup_type": "intra_task_exact",
                "best_match": {"memory_entry_key": key, "source_task": task_id,
                               "object_id": object_id}, "result": "HIT", "full_m3_skipped": True}
        candidates = self.filter_bbox(task_id, bbox_mm)
        debug |= {"lookup_type": "cross_task", "bbox_candidates": candidates}
        if not candidates:
            stale_bbox_match = any(
                entry.get("metadata", {}).get("source_task") != task_id
                and entry.get("metadata", {}).get("stale")
                and (diff := self._bbox_difference(
                    bbox_mm, entry.get("geometry", {}).get("extents_mm"))) is not None
                and diff[0] <= self.bbox_relative_threshold
                for entry in self.objects.values())
            reason = exact_reason or ("STALE_MEMORY" if stale_bbox_match else "NO_BBOX_CANDIDATE")
            return None, debug | {"density_candidates": [], "best_match": None,
                                  "result": "MISS", "miss_reason": reason}
        try:
            density_result = density_infer(crop_rgb)
        except Exception as exc:  # caller/backend errors are a safe retrieval miss
            return None, debug | {"density_candidates": [], "best_match": None,
                "result": "MISS", "miss_reason": "DENSITY_INFERENCE_FAILED",
                "density_error": str(exc)}
        debug |= {"c3_llm_called": density_result.llm_called,
                  "c3_token_usage": density_result.token_usage,
                  "query_density_kgm3": density_result.density_kgm3}
        if density_result.density_kgm3 is None:
            return None, debug | {"density_candidates": [], "best_match": None,
                "result": "MISS", "miss_reason": "DENSITY_INFERENCE_FAILED",
                "density_error": density_result.error}
        best, ranked = self.retrieve_by_density(candidates, density_result.density_kgm3)
        debug["density_candidates"] = ranked
        if best is None:
            return None, debug | {"best_match": None, "result": "MISS",
                                  "miss_reason": "NO_VALID_DENSITY"}
        debug["best_match"] = {k: best[k] for k in
                               ("memory_entry_key", "source_task", "object_id")}
        if best["relative_difference"] > self.density_relative_threshold:
            return None, debug | {"result": "MISS", "miss_reason": "DENSITY_THRESHOLD_EXCEEDED"}
        return entry_to_props(self.objects[best["memory_entry_key"]]), debug | {
            "result": "HIT", "full_m3_skipped": True}
