"""M2 물성 메모리 — 객체별 intrinsic 추론을 에피소드 간 지속·재사용.

원리: 물성(재질·질량·μ·치수)은 pose 불변 → 한 번 접지하면 재사용 가능.
  · 조회는 노드 id 키 lookup (우리 스케일에선 이것이 곧 retrieval — 벡터 RAG 불요.
    novel object 유사도 전이가 필요해지면 그때 시그니처 검색으로 확장)
  · 병합 규칙: stage 높은 쪽 우선 (probe 2 > 재관측 1 > 시각 0), 같으면 최신
  · 효과: 에피소드 반복 시 VLM 콜 → 0 수렴, M1/M3 컨텍스트에 컴팩트 요약만

무효화 주의: 내용물이 변할 수 있는 객체(컨테이너류)는 관련 이벤트(붓기/담기)
후 해당 엔트리를 invalidate() 할 것 — M5/실행 루프의 책임.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _stage(props: dict) -> int:
    """엔트리의 측정 단계 (질량/마찰 중 높은 쪽)."""
    return max(int(props.get("mass_stage", 0)),
               int(props.get("mu", {}).get("stage", 0)))


class PropertyMemory:
    """{node_id: {"props": intrinsic dict, "stage", "episodes_seen", "updated_at"}}"""

    def __init__(self, path):
        self.path = Path(path)
        self.data: dict = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8")).get("objects", {})

    def lookup(self, node_id: str) -> dict | None:
        e = self.data.get(node_id)
        return dict(e["props"]) if e and not e.get("stale") else None

    def update(self, cache: dict, source: str = "m2") -> dict:
        """런 종료 시 캐시 병합. → {"hits": 재사용됐던 수, "new": 신규, "upgraded": 승격}"""
        stats = {"new": 0, "upgraded": 0, "kept": 0}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for nid, props in cache.items():
            props = {k: v for k, v in props.items() if not k.startswith("_")}
            old = self.data.get(nid)
            if old is None:
                self.data[nid] = {"props": props, "stage": _stage(props),
                                  "source": source, "episodes_seen": 1, "updated_at": now}
                stats["new"] += 1
            elif _stage(props) >= old["stage"]:
                self.data[nid] = {"props": props, "stage": _stage(props), "source": source,
                                  "episodes_seen": old["episodes_seen"] + 1, "updated_at": now}
                stats["upgraded" if _stage(props) > old["stage"] else "kept"] += 1
            else:                                      # 낮은 단계 측정으론 강등 금지
                old["episodes_seen"] += 1
                stats["kept"] += 1
        return stats

    def invalidate(self, node_id: str, reason: str = ""):
        if node_id in self.data:
            self.data[node_id]["stale"] = True
            self.data[node_id]["stale_reason"] = reason

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"objects": self.data}, ensure_ascii=False, indent=2),
            encoding="utf-8")
