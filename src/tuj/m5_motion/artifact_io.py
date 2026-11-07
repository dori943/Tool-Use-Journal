"""Durable on-disk artifacts for large M5 world snapshots."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# Dense mesh dumps regularly exceed 30–70 MB with indent=2 and trip Windows
# ``OSError: [Errno 22]`` under AV/OneDrive locks. Planning/eval keep poses,
# dimensions, and ``collision_points_m``; triangle soups are re-captured from
# the live env when needed.
_WORLD_DUMP_OMIT_OBJECT_KEYS = frozenset({"collision_surface_triangles_m"})


def world_snapshot_json(world: Any, *, indent: int | None = None) -> str:
    """Serialize a world snapshot without the heaviest mesh payloads."""

    if hasattr(world, "model_dump"):
        payload = world.model_dump(mode="json")
    elif isinstance(world, dict):
        payload = world
    else:
        raise TypeError(f"unsupported world snapshot type: {type(world)!r}")
    objects = payload.get("objects")
    if isinstance(objects, dict):
        for record in objects.values():
            if isinstance(record, dict):
                for key in _WORLD_DUMP_OMIT_OBJECT_KEYS:
                    record.pop(key, None)
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def write_text_atomic(path: Path | str, text: str, *, retries: int = 8) -> None:
    """Write UTF-8 text via a same-directory temp file, then replace.

    Retries absorb transient Windows locks (Errno 22 / sharing violations) that
    show up when a previous multi-ten MB ``initial_world.json`` is still open.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = text if text.endswith("\n") else f"{text}\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    last_error: OSError | None = None
    for attempt in range(max(1, int(retries))):
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, destination)
            return
        except OSError as error:
            last_error = error
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
    assert last_error is not None
    raise last_error
