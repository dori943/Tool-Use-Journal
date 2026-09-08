"""Backward-compatible M3 facade over the unified M0 memory store."""
from __future__ import annotations

from tuj.m0_memory.object_knowledge import ObjectKnowledgeManager, entry_to_props


def _stage(props: dict) -> int:
    """엔트리의 측정 단계 (질량/마찰 중 높은 쪽)."""
    return max(int(props.get("mass_stage", 0)),
               int(props.get("mu", {}).get("stage", 0)))


class PropertyMemory(ObjectKnowledgeManager):
    """Retain the production API while persisting unified Object Knowledge."""

    def __init__(self, path, *, task_id=None, source_episode=None,
                 bbox_relative_threshold=0.25, density_relative_threshold=0.20):
        super().__init__(path, bbox_relative_threshold=bbox_relative_threshold,
                         density_relative_threshold=density_relative_threshold)
        self.task_id = task_id
        self.source_episode = source_episode

    def lookup(self, node_id: str, task_id=None) -> dict | None:
        query_task = self.task_id if task_id is None else task_id
        if query_task is not None:
            entry, _, _ = self.lookup_exact(query_task, node_id)
            return entry_to_props(entry) if entry is not None else None
        matches = [entry for entry in self.objects.values()
                   if entry.get("identity", {}).get("object_id") == node_id
                   and not entry.get("metadata", {}).get("stale")]
        return entry_to_props(matches[0]) if len(matches) == 1 else None

    def update(self, cache: dict, source: str = "m3", *, task_id=None,
               source_episode=None) -> dict:
        """런 종료 시 캐시 병합. → {"hits": 재사용됐던 수, "new": 신규, "upgraded": 승격}"""
        stats = {"new": 0, "upgraded": 0, "kept": 0}
        query_task = self.task_id if task_id is None else task_id
        if query_task is None:
            raise ValueError("task_id is required to update unified Object Knowledge")
        episode = self.source_episode if source_episode is None else source_episode
        for nid, raw_props in cache.items():
            props = {k: v for k, v in raw_props.items() if not k.startswith("_")}
            old, _, _ = self.lookup_exact(query_task, nid)
            old_stage = int((old or {}).get("metadata", {}).get("stage", 0))
            new_stage = _stage(props)
            self.update_entry(query_task, nid, props, episode=episode,
                              source_method=source, stage=new_stage)
            if old is None:
                stats["new"] += 1
            elif new_stage > old_stage:
                stats["upgraded"] += 1
            else:
                stats["kept"] += 1
        return stats

    def invalidate(self, node_id: str, reason: str = "", *, task_id=None):
        query_task = self.task_id if task_id is None else task_id
        matches = [entry for entry in self.objects.values()
                   if entry.get("identity", {}).get("object_id") == node_id
                   and (query_task is None
                        or entry.get("metadata", {}).get("source_task") == query_task)]
        if len(matches) == 1:
            matches[0]["metadata"].update(stale=True, stale_reason=reason)
